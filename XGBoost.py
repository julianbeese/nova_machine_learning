import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def preprocess_data(df):
    """
    Preprocess the input DataFrame for regression modeling.

    Steps:
      - Create a copy of the DataFrame.
      - Standardize column names.
      - Drop unnecessary columns ('id' and 'levy').
      - Clean columns (e.g., 'mileage', 'engine_volume') and convert data types.
      - Handle categorical columns by label encoding.
      - Replace specific values in 'doors'.
      - Compute additional features (e.g., car age from 'prod_year').
      - Remove outliers using the IQR method.
      - Split the data into features (X) and target (y) and then into training and testing sets.
      - Apply log transformation to target if it is highly skewed.

    Parameters:
      df (pd.DataFrame): The unprocessed DataFrame.

    Returns:
      X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()

    # Standardize column names: lowercase, strip spaces, replace spaces with underscores, and remove dots
    data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('.', '')

    # Convert 'prod_year' to a year if it's in a date format
    if 'prod_year' in data.columns:
        if data['prod_year'].dtype == 'object':
            try:
                data['prod_year'] = pd.to_datetime(data['prod_year']).dt.year
            except Exception:
                pass

    # Calculate car age using production year (if available)
    current_year = 2025  # Update as needed
    if 'prod_year' in data.columns:
        data['car_age'] = current_year - data['prod_year']

    # Handle the 'engine_volume' column: extract numeric part if stored as string
    if 'engine_volume' in data.columns and data['engine_volume'].dtype == 'object':
        data['engine_volume'] = data['engine_volume'].str.extract(r'(\d+\.?\d*)').astype(float)

    # Convert specific categorical yes/no columns to binary (e.g., 'leather_interior')
    binary_cols = ['leather_interior']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)

    # Label encode any remaining categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Identify numerical columns (excluding the target 'price')
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')

    # Remove outliers using the IQR method for numerical columns and the target 'price'
    for col in numeric_cols + ['price']:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]

    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Apply log transformation to the target variable if it is highly skewed
    if y.skew() > 1:
        y = np.log1p(y)
        print("Applied log transformation to 'price' due to skewness.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def run_xgb_regression(df):
    """
    Run the XGBoost regression pipeline with preprocessing, standard scaling, 
    and hyperparameter tuning using GridSearchCV.

    Parameters:
      df (pd.DataFrame): The unprocessed DataFrame.

    Returns:
      dict: A dictionary containing:
            - best_model: The best estimator from hyperparameter tuning.
            - mse: Mean Squared Error on the test set.
            - mae: Mean Absolute Error on the test set.
            - r_squared: R2 score on the test set.
            - grid_search: The GridSearchCV object.
    """
    # Preprocess the data
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = preprocess_data(df)

    # Create a pipeline that applies StandardScaler followed by the XGBoost regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(random_state=42))
    ])

    # Define a hyperparameter grid for tuning the XGBoost regressor
    param_grid = {
        'xgb__n_estimators': [500, 1200],
        'xgb__max_depth': [6, 10],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__min_child_weight': [1, 3, 5],
        'xgb__subsample': [0.6, 0.8],
        'xgb__colsample_bytree': [0.8, 1.0],
        'xgb__reg_lambda': [0.3, 1.0]
    }

    # Set up GridSearchCV to perform hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=3,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    # Retrieve the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Print the best hyperparameters and evaluation metrics
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score (neg MSE):", grid_search.best_score_)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r_squared}")

    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot: Predicted vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'best_model': best_model,
        'mse': mse,
        'mae': mae,
        'r_squared': r_squared,
        'grid_search': grid_search
    }
