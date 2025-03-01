"""
Module for implementing the Ridge model in the car price prediction project.

This module contains specific functions for training and optimizing
the Ridge regression model.
"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.model.evaluate_model import evaluate_model


def train_ridge(X_train, y_train, X_test, y_test, preprocessor, param_grid=None, log_transformed=False):
    """
    Trains a Ridge regression model with hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        preprocessor: Preprocessing pipeline (ColumnTransformer)
        param_grid: Parameter grid for GridSearchCV (optional)
        log_transformed: Whether the target values were logarithmically transformed

    Returns:
        tuple: (best_model, metrics) - Best model and evaluation metrics
    """
    if param_grid is None:
        param_grid = {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(random_state=42))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Training the Ridge model with Grid Search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters for Ridge:")
    print(grid_search.best_params_)
    print(f"Best CV score (neg. MSE): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, log_transformed)

    return best_model, metrics


def run_ridge_pipeline(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed=False):
    """
    Executes the complete pipeline for the Ridge model.

    This function creates the preprocessor, trains the Ridge model,
    and evaluates it.

    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        numeric_cols: List of numeric columns
        categorical_cols: List of categorical columns
        log_transformed: Whether the target values were logarithmically transformed

    Returns:
        tuple: (best_model, metrics) - Best model and its metrics
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return train_ridge(X_train, y_train, X_test, y_test, preprocessor, log_transformed=log_transformed)