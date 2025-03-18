"""
Module for model training and hyperparameter tuning in the car price prediction project.

This module contains additional helper functions for model training
that have not been moved to specialized modules yet.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet


def build_pipeline(numeric_cols, categorical_cols, model_type='ridge'):
    """
    Builds a pipeline with a preprocessor and the specified model type.

    Args:
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        model_type (str): Type of model to use (default: 'ridge')

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline
    """
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

    if model_type == 'ridge':
        regressor = Ridge(random_state=42)
    elif model_type == 'lasso':
        regressor = Lasso(random_state=42)
    elif model_type == 'elastic_net':
        regressor = ElasticNet(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    return pipeline


def hyperparameter_tuning(pipeline, X_train, y_train, param_grid=None):
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Model pipeline to tune
        X_train: Training features
        y_train: Training target values
        param_grid (dict): Parameter grid for GridSearchCV

    Returns:
        sklearn.model_selection.GridSearchCV: The fitted grid search object
    """
    # Default parameter grid if none provided
    if param_grid is None:
        # Get the regressor type from the pipeline
        regressor_type = pipeline.named_steps['regressor'].__class__.__name__

        if regressor_type == 'Ridge':
            param_grid = {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            }
        elif regressor_type == 'Lasso':
            param_grid = {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'regressor__max_iter': [1000, 3000, 5000]
            }
        elif regressor_type == 'ElasticNet':
            param_grid = {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'regressor__max_iter': [1000, 3000]
            }
        else:
            raise ValueError(f"No default parameter grid for regressor type: {regressor_type}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.4f}".format(-grid_search.best_score_))

    return grid_search


