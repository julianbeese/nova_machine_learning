"""
Module for training models in the car price prediction project.

This module contains functions for training various ML models.
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.model.evaluate_model import evaluate_model


def build_pipeline(numeric_cols, categorical_cols, model_type='ridge'):
    """
    Creates an ML pipeline with preprocessor and the desired model.

    Args:
        numeric_cols (list): List of numeric columns
        categorical_cols (list): List of categorical columns
        model_type (str): Type of model to create

    Returns:
        sklearn.pipeline.Pipeline: The created pipeline
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
        model = Ridge(random_state=42)
    elif model_type == 'lasso':
        model = Lasso(random_state=42)
    elif model_type == 'elastic_net':
        model = ElasticNet(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline


def get_param_grid(model_type, config):
    """
    Creates a parameter grid for hyperparameter optimization based on the model type.

    Args:
        model_type (str): Type of model
        config (dict): Configuration values from config/model_config.yaml

    Returns:
        dict: Parameter grid for GridSearchCV
    """
    # Default values if no configuration is provided
    if model_type == 'ridge':
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0] if not config.get('ridge') else [
                config['ridge'].get('alpha', 1.0)],
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'] if not config.get('ridge') else [
                config['ridge'].get('solver', 'auto')]
        }
    elif model_type == 'lasso':
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0] if not config.get('lasso') else [
                config['lasso'].get('alpha', 0.1)],
            'regressor__max_iter': [1000, 3000, 5000] if not config.get('lasso') else [
                config['lasso'].get('max_iter', 1000)]
        }
    elif model_type == 'elastic_net':
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0] if not config.get('elastic_net') else [
                config['elastic_net'].get('alpha', 0.1)],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9] if not config.get('elastic_net') else [
                config['elastic_net'].get('l1_ratio', 0.5)],
            'regressor__max_iter': [1000, 3000] if not config.get('elastic_net') else [
                config['elastic_net'].get('max_iter', 1000)]
        }
    elif model_type == 'random_forest':
        return {
            'regressor__n_estimators': [10, 50, 100] if not config.get('random_forest') else [
                config['random_forest'].get('n_estimators', 100)],
            'regressor__max_depth': [3, 5, 10] if not config.get('random_forest') else [
                config['random_forest'].get('max_depth', 10)],
            'regressor__min_samples_split': [2, 5, 10] if not config.get('random_forest') else [
                config['random_forest'].get('min_samples_split', 2)],
            'regressor__min_samples_leaf': [1, 2, 4] if not config.get('random_forest') else [
                config['random_forest'].get('min_samples_leaf', 1)]
        }
    elif model_type == 'xgboost':
        return {
            'regressor__n_estimators': [10, 50, 100] if not config.get('xgboost') else [
                config['xgboost'].get('n_estimators', 100)],
            'regressor__max_depth': [3, 5, 10] if not config.get('xgboost') else [
                config['xgboost'].get('max_depth', 5)],
            'regressor__learning_rate': [0.01, 0.1, 0.5] if not config.get('xgboost') else [
                config['xgboost'].get('learning_rate', 0.1)]
        }
    else:
        return {}


def train_model(X_train, y_train, X_test, y_test, model_type, config, log_transformed=False):
    """
    Trains a single model with hyperparameter optimization.

    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target values
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target values
        model_type (str): Type of model to train
        config (dict): Configuration values
        log_transformed (bool): Whether the target values were logarithmically transformed

    Returns:
        tuple: (best_model, metrics) - Best model and evaluation metrics
    """
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    pipeline = build_pipeline(numeric_cols, categorical_cols, model_type)

    param_grid = get_param_grid(model_type, config)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    print(f"Training the {model_type} model with Grid Search...")
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")

    metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, log_transformed, plot=False)

    return grid_search.best_estimator_, metrics


def train_models(X_train, y_train, X_test, y_test, model_type='all', config=None, log_transformed=False):
    """
    Trains one or more models and selects the best one.

    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target values
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target values
        model_type (str): Type of model to train ('all' or specific type)
        config (dict): Configuration values
        log_transformed (bool): Whether the target values were logarithmically transformed

    Returns:
        tuple: (best_model, best_metrics) - Best model and its metrics
    """
    if config is None:
        config = {}

    if model_type == 'all':
        print("Training all models...")
        models = {}
        metrics = {}

        for m_type in ['ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost']:
            print(f"\n===== Training the {m_type} model =====")
            model, model_metrics = train_model(
                X_train, y_train, X_test, y_test,
                model_type=m_type,
                config=config,
                log_transformed=log_transformed
            )
            models[m_type] = model
            metrics[m_type] = model_metrics

        best_model_type = min(metrics, key=lambda k: metrics[k][1])
        best_model = models[best_model_type]
        best_metrics = metrics[best_model_type]

        print(f"\nBest model: {best_model_type} with RMSE: {best_metrics[1]:.2f}")

        import joblib
        import os
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.pkl')

        return best_model, best_metrics
    else:
        return train_model(X_train, y_train, X_test, y_test, model_type, config, log_transformed)