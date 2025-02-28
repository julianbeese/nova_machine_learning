import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

def preprocess_data(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()

    # Convert column names to lowercase and replace spaces with underscores
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('.', '')

    # Extract year from production year if it's a date
    if 'prod_year' in data.columns:
        if data['prod_year'].dtype == 'object':
            try:
                data['prod_year'] = pd.to_datetime(data['prod_year']).dt.year
            except:
                pass

    # Calculate car age (current year - production year)
    current_year = 2025  # Using the current year from the system date
    if 'prod_year' in data.columns:
        data['car_age'] = current_year - data['prod_year']

    # Handle engine volume (extract numeric part)
    if 'engine_volume' in data.columns and data['engine_volume'].dtype == 'object':
        data['engine_volume'] = data['engine_volume'].str.extract(r'(\d+\.?\d*)').astype(float)

    # Convert categorical yes/no to binary
    binary_cols = ['leather_interior']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)

    # Identify numerical and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove('price')  # Remove target variable

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Remove outliers using IQR
    for col in numeric_cols + ['price']:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]

    # Split data
    X = data.drop('price', axis=1)
    y = data['price']

    # Apply log transformation to target if the distribution is skewed
    if y.skew() > 1:
        y = np.log1p(y)  # log(1+x) to handle zeros
        print("Applied log transformation to price due to skewness")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols):
    # Numeric features pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Full pipeline with Ridge regression (a regularized version of linear regression)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    return pipeline


def hyperparameter_tuning(pipeline, X_train, y_train):
    # Define parameter grid for Ridge, Lasso, and ElasticNet
    param_grid = [
        {
            'regressor': [ElasticNet()],
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'regressor__max_iter': [1000, 3000]
        },

    ]

    # Create CV object
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1  # Use all processors
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Print results
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.4f}".format(-grid_search.best_score_))

    return grid_search


def evaluate_model(model, X_test, y_test, log_transformed=False):
    # Make predictions
    y_pred = model.predict(X_test)

    # If we applied log transformation to the target, convert predictions back
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    # Feature importance for linear models
    if hasattr(model[-1], 'coef_'):
        # Get feature names after one-hot encoding
        feature_names = []
        for transformer_name, transformer, column_names in model[0].transformers_:
            if transformer_name == 'num':
                feature_names.extend([(name, '') for name in column_names])
            else:
                # For categorical features, get all categories
                if hasattr(transformer[-1], 'get_feature_names_out'):
                    cat_features = transformer[-1].get_feature_names_out(column_names)
                    feature_names.extend([(name.split('_')[0], '_'.join(name.split('_')[1:]))
                                          for name in cat_features])

        # Get coefficients
        coefficients = model[-1].coef_

        # Create a DataFrame
        feature_importance = pd.DataFrame({
            'Feature': [f"{cat} {val}" if val else cat for cat, val in feature_names],
            'Importance': np.abs(coefficients)
        })

        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.show()

    return mse, rmse, mae, r2


def main(df):

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = preprocess_data(df)
    log_transformed = False
    if y_train.skew() < 0.5:  # Check if log transformation was applied
        log_transformed = True

    # Step 3: Build pipeline
    pipeline = build_pipeline(numeric_cols, categorical_cols)

    # Step 4: Hyperparameter tuning
    best_model = hyperparameter_tuning(pipeline, X_train, y_train)

    # Step 5: Evaluate model
    metrics = evaluate_model(best_model.best_estimator_, X_test, y_test, log_transformed)

    # Return the best model
    return best_model.best_estimator_, metrics


# Run the pipeline
if __name__ == "__main__":
    # Assuming df is already loaded
    # df = pd.read_csv('your_car_data.csv')
    best_model, metrics = main(df)

    # Save the trained model
    import joblib

    joblib.dump(best_model, 'car_price_prediction_model.pkl')
    print("Model saved as 'car_price_prediction_model.pkl'")

    # How to use the saved model for prediction
    print("\nExample of using the model for prediction:")
    print("loaded_model = joblib.load('car_price_prediction_model.pkl')")
    print("new_data = pd.DataFrame({...})  # New car data")
    print("predicted_price = loaded_model.predict(new_data)")