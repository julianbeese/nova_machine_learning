"""
Module for training different regression models for car price prediction.

This module provides functions to train different types of regression models
without generating comparison reports.
"""
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Import model-specific modules
# Import from __init__.py file
from models.elastic_net import run_elastic_net_pipeline, train_elastic_net
# Import from lasso.py and ridge.py
from models.lasso import train_lasso, run_lasso_pipeline
from models.ridge import train_ridge, run_ridge_pipeline
# Import random_forest.py
from models.random_forest import run_random_forest_pipeline
# Import from xgboost.py
from models.xgboost import run_xgboost_pipeline


def train_models(X_train, y_train, X_test, y_test, model_type='all', config=None, log_transformed=False):
    """
    Train one or more regression models.

    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        model_type: Type of model to train ('all', 'ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost')
        config: Configuration dictionary for model parameters
        log_transformed: Whether target values were log-transformed

    Returns:
        tuple: (best_model, best_metrics) - The best model and its performance metrics
    """
    # Create directory for trained models if it doesn't exist
    os.makedirs('models/trained', exist_ok=True)

    # Get numerical and categorical columns from X_train
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Dictionary to store all trained models and their performance metrics
    all_models = {}
    all_metrics = {}

    if model_type in ['all', 'ridge']:
        print("\n\n=== Training Ridge Regression Model ===")
        ridge_model, ridge_metrics = run_ridge_pipeline(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed
        )
        all_models['ridge'] = ridge_model
        all_metrics['ridge'] = ridge_metrics

        # Save the model
        joblib.dump(ridge_model, 'models/trained/ridge_model.pkl')
        print("Ridge model saved to models/trained/ridge_model.pkl")

    if model_type in ['all', 'lasso']:
        print("\n\n=== Training Lasso Regression Model ===")
        lasso_model, lasso_metrics = run_lasso_pipeline(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed
        )
        all_models['lasso'] = lasso_model
        all_metrics['lasso'] = lasso_metrics

        # Save the model
        joblib.dump(lasso_model, 'models/trained/lasso_model.pkl')
        print("Lasso model saved to models/trained/lasso_model.pkl")

    if model_type in ['all', 'elastic_net']:
        print("\n\n=== Training Elastic Net Regression Model ===")
        elastic_net_model, elastic_net_metrics = run_elastic_net_pipeline(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed
        )
        all_models['elastic_net'] = elastic_net_model
        all_metrics['elastic_net'] = elastic_net_metrics

        # Save the model
        joblib.dump(elastic_net_model, 'models/trained/elastic_net_model.pkl')
        print("Elastic Net model saved to models/trained/elastic_net_model.pkl")

    if model_type in ['all', 'random_forest']:
        print("\n\n=== Training Random Forest Regression Model ===")
        random_forest_model, random_forest_metrics = run_random_forest_pipeline(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, log_transformed
        )
        all_models['random_forest'] = random_forest_model
        all_metrics['random_forest'] = random_forest_metrics

        # Save the model
        joblib.dump(random_forest_model, 'models/trained/random_forest_model.pkl')
        print("Random Forest model saved to models/trained/random_forest_model.pkl")

    if model_type in ['all', 'xgboost']:
        print("\n\n=== Training XGBoost Regression Model ===")
        xgboost_model, xgboost_metrics = run_xgboost_pipeline(
            X_train, y_train, X_test, y_test, log_transformed
        )
        all_models['xgboost'] = xgboost_model
        all_metrics['xgboost'] = xgboost_metrics

        # Save the model
        joblib.dump(xgboost_model, 'models/trained/xgboost_model.pkl')
        print("XGBoost model saved to models/trained/xgboost_model.pkl")

    # If requested a specific model, return that model and its metrics
    if model_type != 'all':
        return all_models[model_type], all_metrics[model_type]

    # Find the best model based on R²
    best_model_name = max(all_metrics, key=lambda x: all_metrics[x][3])  # R² is at index 3
    best_model = all_models[best_model_name]
    best_metrics = all_metrics[best_model_name]

    # Display basic information about the best model
    print(f"\n\n=== Best Model: {best_model_name.upper()} ===")
    print(f"MSE: {best_metrics[0]:.2f}")
    print(f"RMSE: {best_metrics[1]:.2f}")
    print(f"MAE: {best_metrics[2]:.2f}")
    print(f"R²: {best_metrics[3]:.4f}")

    # Save the best model
    joblib.dump(best_model, 'models/trained/best_model.pkl')
    print("Best model saved to models/trained/best_model.pkl")

    return best_model, best_metrics