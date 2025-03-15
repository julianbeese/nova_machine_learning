"""
Main entry point for the car price prediction project.

This script serves as the central control for the ML pipeline for car price prediction.
It can train, evaluate, and use different models for predictions.
"""
import argparse
import os
import pandas as pd
import joblib
import yaml
import warnings
import numpy as np

from src.data.preprocess import preprocess_data
from src.model.train_model import train_models
from src.model.evaluate_model import evaluate_model, evaluate_all_models


def load_config():
    config_path = 'config/model_config.yaml'
    if not os.path.exists(config_path):
        config = {
            'ridge': {'alpha': 1.0, 'solver': 'auto'},
            'lasso': {'alpha': 0.1, 'max_iter': 1000},
            'elastic_net': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000},
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }

        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
    else:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

    return config


def run_pipeline(mode='train', model_type='all', input_file=None, output_file=None, log_transform=False):
    """
    Executes the complete ML pipeline.

    Args:
        mode (str): 'train', 'evaluate', or 'predict'
        model_type (str): 'all', 'ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost', 'catboost', or 'best'
        input_file (str): Path to the input file (CSV)
        output_file (str): Path to save the trained model
        log_transform (bool): Whether logarithmic transformation of target values should be performed
    """
    config = load_config()

    if input_file is None or not os.path.exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist or was not specified.")

    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)

    if output_file is None:
        os.makedirs('models', exist_ok=True)
        output_file = f'models/{model_type}_model.pkl'

    if mode in ['train', 'evaluate']:
        print("Preprocessing the data...")
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = preprocess_data(
            data, log_transform, prediction_mode=False
        )

        if mode == 'train':
            print(f"Starting training of the model: {model_type}")
            best_model, metrics = train_models(
                X_train, y_train, X_test, y_test,
                model_type=model_type,
                config=config,
                log_transformed=log_transform
            )

            print(f"Saving trained model to {output_file}")
            joblib.dump(best_model, output_file)

            print("\nTraining results:")
            print(f"MSE: {metrics[0]:.2f}")
            print(f"RMSE: {metrics[1]:.2f}")
            print(f"MAE: {metrics[2]:.2f}")
            print(f"R²: {metrics[3]:.4f}")

        elif mode == 'evaluate':
            if model_type == 'all':
                # Evaluate all available models and create a comparison report
                print("Evaluating all available models...")
                report_path = evaluate_all_models(X_test, y_test, log_transformed=log_transform)
                print(f"\nComparison report created at: {report_path}")
            else:
                # Evaluate a single model
                if model_type == 'best':
                    model_file = 'models/best_model.pkl'
                else:
                    # Check in both locations
                    if os.path.exists(f'models/trained/{model_type}_model.pkl'):
                        model_file = f'models/trained/{model_type}_model.pkl'
                    else:
                        model_file = f'models/{model_type}_model.pkl'

                if not os.path.exists(model_file):
                    raise ValueError(f"Model {model_file} not found. Please train a model first.")

                print(f"Loading model from {model_file}")
                loaded_model = joblib.load(model_file)

                print(f"Evaluating model...")
                metrics = evaluate_model(loaded_model, X_test, y_test, log_transform, plot=False, save_report=True, model_name=model_type)

    elif mode == 'predict':
        print("Preprocessing data for prediction...")
        X_pred, _, _, _, numeric_cols, categorical_cols = preprocess_data(
            data, log_transform, prediction_mode=True
        )

        if model_type == 'best':
            model_file = 'models/best_model.pkl'
        else:
            # Check in both locations
            if os.path.exists(f'models/trained/{model_type}_model.pkl'):
                model_file = f'models/trained/{model_type}_model.pkl'
            else:
                model_file = f'models/{model_type}_model.pkl'

        if not os.path.exists(model_file):
            raise ValueError(f"Model {model_file} not found. Please train a model first.")

        print(f"Loading model from {model_file}")
        loaded_model = joblib.load(model_file)

        print("Creating predictions...")
        predictions = loaded_model.predict(X_pred)

        if log_transform:
            predictions = np.expm1(predictions)

        result_df = X_pred.copy()
        result_df['predicted_price'] = predictions

        results_file = 'results/predictions.csv'
        os.makedirs('results', exist_ok=True)
        result_df.to_csv(results_file, index=False)

        print(f"\nPredictions successfully created and saved to {results_file}")
        print(f"\nFirst 5 predictions:")
        print(result_df[['predicted_price']].head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Car Price ML Pipeline')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='Pipeline mode: train, evaluate, or predict')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost', 'best'],
                        help='Model to use')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the CSV input file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--log-transform', action='store_true',
                        help='Perform logarithmic transformation of target values')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    run_pipeline(
        mode=args.mode,
        model_type=args.model,
        input_file=args.input,
        output_file=args.output,
        log_transform=args.log_transform
    )
