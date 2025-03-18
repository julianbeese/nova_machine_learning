"""
Module for implementing the XGBoost model in the car price prediction project.

This module contains specific functions for training and optimizing
the XGBoost regression model.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from src.model.evaluate_model import evaluate_model


def train_xgboost(X_train, y_train, X_test, y_test, param_grid=None, log_transformed=False):
    """
    Trains an XGBoost regression model with hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        param_grid: Parameter grid for GridSearchCV (optional)
        log_transformed: Whether the target values were logarithmically transformed

    Returns:
        tuple: (best_model, metrics) - Best model and evaluation metrics
    """
    if param_grid is None:
        param_grid = {
            'xgb__n_estimators': [100, 500],
            'xgb__max_depth': [3, 6, 10],
            'xgb__learning_rate': [0.01, 0.1],
            'xgb__min_child_weight': [1, 3],
            'xgb__subsample': [0.8],
            'xgb__colsample_bytree': [0.8],
            'xgb__reg_lambda': [1.0]
        }

    # Create preprocessing steps for numeric and categorical features
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a preprocessor that applies the appropriate transformer to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create a pipeline with preprocessing and XGBoost
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(random_state=42))
    ])

    print("Training the XGBoost model with Grid Search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters for XGBoost:")
    print(grid_search.best_params_)
    print(f"Best CV score (neg. MSE): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, log_transformed)

    # Feature importance visualization for XGBoost (if accessible)
    try:
        if hasattr(best_model.named_steps['xgb'], 'feature_importances_'):
            # This will give us a list of feature names after preprocessing
            # But it's not straightforward to map back to original features with OHE
            # This is a simplified approach
            importances = best_model.named_steps['xgb'].feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title('XGBoost Feature Importances')
            plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
            plt.xticks(range(min(20, len(indices))), range(min(20, len(indices))), rotation=90)
            plt.tight_layout()
            plt.savefig('results/xgboost_feature_importance.png')
            plt.close()
    except Exception as e:
        print(f"Could not plot feature importances: {e}")

    return best_model, metrics


def run_xgboost_pipeline(X_train, y_train, X_test, y_test, log_transformed=False):
    """
    Executes the complete pipeline for the XGBoost model.

    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        log_transformed: Whether the target values were logarithmically transformed

    Returns:
        tuple: (best_model, metrics) - Best model and its metrics
    """
    param_grid = {
        'xgb__n_estimators': [500, 1200],
        'xgb__max_depth': [6, 10],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__min_child_weight': [1, 3, 5],
        'xgb__subsample': [0.6, 0.8],
        'xgb__colsample_bytree': [0.8, 1.0],
        'xgb__reg_lambda': [0.3, 1.0]
    }

    return train_xgboost(X_train, y_train, X_test, y_test, param_grid, log_transformed)