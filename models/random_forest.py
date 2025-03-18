"""
Module for implementing the Random Forest model in the car price prediction project.

This module contains specific functions for training and optimizing
the Random Forest regression model.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.model.evaluate_model import evaluate_model


def train_random_forest(X_train, y_train, X_test, y_test, preprocessor, param_grid=None, log_transformed=False):
    """
    Trains a Random Forest regression model with hyperparameter optimization.

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
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2']
        }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    print("Training the Random Forest model with Grid Search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters for Random Forest:")
    print(grid_search.best_params_)
    print(f"Best CV score (neg. MSE): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, log_transformed)

    # Plot feature importance if available
    try:
        if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
            # Get feature names after preprocessing (this is complex with ColumnTransformer)
            # Just use indices for now
            feature_importances = best_model.named_steps['regressor'].feature_importances_
            indices = np.argsort(feature_importances)[::-1]

    except Exception as e:
        print(f"Warning: Could not plot feature importances: {e}")

    return best_model, metrics


def run_random_forest_pipeline(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed=False):
    """
    Executes the complete pipeline for the Random Forest model.

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
    # Create preprocessor
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

    # Set up parameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['sqrt', 'log2']
    }

    return train_random_forest(X_train, y_train, X_test, y_test, preprocessor, param_grid, log_transformed)