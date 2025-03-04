"""
Module for model evaluation in the car price prediction project.

This module contains functions for evaluating trained models
and creating reports in Markdown format.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

from xgboost.testing.data import joblib


def evaluate_model(model, X_test, y_test, log_transformed=False, plot=True, save_report=False, model_name=None):
    """
    Evaluates a trained model with various metrics.

    Args:
        model: Trained model (sklearn-compatible)
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Actual target values
        log_transformed (bool): Whether the target values were log-transformed
        plot (bool): Whether to display plots
        save_report (bool): Whether to create and save a report
        model_name (str): Name of the model for the report (optional)

    Returns:
        tuple: (mse, rmse, mae, r2) - Evaluation metrics
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_dir = None
    if save_report:
        if model_name is None:
            model_name = "unknown_model"

        report_dir = f"results/reports/{model_name}_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)

        report_file = os.path.join(report_dir, "report.md")

        report_content = f"""# Model Evaluation Report: {model_name}

Created on: {datetime.now().strftime("%d.%m.%Y, %H:%M:%S")}

## Performance Metrics

"""

    y_pred = model.predict(X_test)

    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test_original = np.expm1(y_test)
    else:
        y_test_original = y_test

    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    print("Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

    if save_report:
        report_content += f"""- **MSE (Mean Squared Error)**: {mse:.2f}
- **RMSE (Root Mean Squared Error)**: {rmse:.2f}
- **MAE (Mean Absolute Error)**: {mae:.2f}
- **R² (Coefficient of Determination)**: {r2:.4f}

## Visualizations

"""

    prediction_stats = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_pred,
        'Difference': y_test_original - y_pred,
        'Absolute Error': np.abs(y_test_original - y_pred)
    })

    stats_summary = prediction_stats.describe()

    if save_report:
        report_content += """### Statistical Summary

| Statistic | Actual | Predicted | Difference | Absolute Error |
|-----------|--------|-----------|------------|----------------|
"""

        for idx, row in stats_summary.iterrows():
            report_content += f"| {idx} | {row['Actual']:.2f} | {row['Predicted']:.2f} | {row['Difference']:.2f} | {row['Absolute Error']:.2f} |\n"

        report_content += "\n"

    if plot or save_report:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_original, y_pred, alpha=0.5)
        plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs. Predicted Prices')
        plt.tight_layout()

        if save_report:
            actual_vs_predicted_file = os.path.join(report_dir, "actual_vs_predicted.png")
            plt.savefig(actual_vs_predicted_file, dpi=300, bbox_inches="tight")

            report_content += """### Actual vs. Predicted Prices

![Actual vs. Predicted Prices](actual_vs_predicted.png)

"""

        if plot:
            plt.show()

        if save_report and not plot:
            plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, y_test_original - y_pred, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()

        if save_report:
            residuals_file = os.path.join(report_dir, "residuals.png")
            plt.savefig(residuals_file, dpi=300, bbox_inches="tight")

            report_content += """### Residual Plot

![Residual Plot](residuals.png)

"""

        if plot:
            plt.show()

        if save_report and not plot:
            plt.close()

        plt.figure(figsize=(10, 6))
        residuals = y_test_original - y_pred
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()

        if save_report:
            histogram_file = os.path.join(report_dir, "residuals_histogram.png")
            plt.savefig(histogram_file, dpi=300, bbox_inches="tight")

            report_content += """### Distribution of Residuals

![Distribution of Residuals](residuals_histogram.png)

"""

        if plot:
            plt.show()

        if save_report and not plot:
            plt.close()

        if hasattr(model[-1], 'coef_'):
            try:
                feature_names = []
                for transformer_name, transformer, column_names in model[0].transformers_:
                    if transformer_name == 'num':
                        feature_names.extend([(name, '') for name in column_names])
                    else:
                        if hasattr(transformer[-1], 'get_feature_names_out'):
                            cat_features = transformer[-1].get_feature_names_out(column_names)
                            feature_names.extend([(name.split('_')[0], '_'.join(name.split('_')[1:]))
                                                  for name in cat_features])

                coefficients = model[-1].coef_

                feature_importance = pd.DataFrame({
                    'Feature': [f"{cat} {val}" if val else cat for cat, val in feature_names],
                    'Importance': np.abs(coefficients)
                })

                feature_importance = feature_importance.sort_values('Importance', ascending=False)

                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()

                if save_report:
                    importance_file = os.path.join(report_dir, "feature_importance.png")
                    plt.savefig(importance_file, dpi=300, bbox_inches="tight")

                    report_content += """### Feature Importance

![Feature Importance](feature_importance.png)

#### Top 20 Features by Importance

| Feature | Importance |
|---------|------------|
"""

                    for _, row in feature_importance.head(20).iterrows():
                        report_content += f"| {row['Feature']} | {row['Importance']:.6f} |\n"

                    report_content += "\n"

                    csv_file = os.path.join(report_dir, "feature_importance.csv")
                    feature_importance.to_csv(csv_file, index=False)

                if plot:
                    plt.show()

                if save_report and not plot:
                    plt.close()

            except Exception as e:
                error_msg = f"Could not create feature importance: {e}"
                print(error_msg)
                if save_report:
                    report_content += f"### Feature Importance\n\n{error_msg}\n\n"

    if save_report:
        prediction_csv = os.path.join(report_dir, "predictions.csv")
        prediction_stats.to_csv(prediction_csv, index=False)

        report_content += f"""## Summary

The model **{model_name}** achieves an R² value of **{r2:.4f}** and an RMSE of **{rmse:.2f}**.

The complete prediction data has been saved in `predictions.csv`.

Evaluation date: {datetime.now().strftime("%d.%m.%Y, %H:%M:%S")}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\nReport has been created and saved at {report_file}")

    return mse, rmse, mae, r2


def compare_models(models_dict, X_test, y_test, log_transformed=False, save_report=False):
    """
    Compares multiple models using various metrics.

    Args:
        models_dict (dict): Dictionary with model names as keys and trained models as values
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Actual target values
        log_transformed (bool): Whether the target values were log-transformed
        save_report (bool): Whether to create and save a comparison report

    Returns:
        pandas.DataFrame: DataFrame with model comparison
    """
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_dir = None
    if save_report:
        report_dir = f"results/reports/model_comparison_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)

    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")

        if save_report:
            model_report_dir = os.path.join(report_dir, model_name)
            os.makedirs(model_report_dir, exist_ok=True)

            mse, rmse, mae, r2 = evaluate_model(
                model, X_test, y_test,
                log_transformed=log_transformed,
                plot=False,
                save_report=True,
                model_name=model_name
            )
        else:
            mse, rmse, mae, r2 = evaluate_model(
                model, X_test, y_test,
                log_transformed=log_transformed,
                plot=False
            )

        results.append({
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values('RMSE')

    print("\nModel Comparison:")
    print(comparison_df)

    if save_report:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='RMSE', data=comparison_df)
        plt.title('RMSE Model Comparison (lower is better)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        comparison_file = os.path.join(report_dir, "model_comparison_rmse.png")
        plt.savefig(comparison_file, dpi=300, bbox_inches="tight")
        plt.close()

        metrics = ['MSE', 'MAE', 'R²']
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            if metric == 'R²':
                sns.barplot(x='Model', y=metric, data=comparison_df.sort_values(metric, ascending=False))
                plt.title(f'{metric} Model Comparison (higher is better)')
            else:
                sns.barplot(x='Model', y=metric, data=comparison_df.sort_values(metric))
                plt.title(f'{metric} Model Comparison (lower is better)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            metric_file = os.path.join(report_dir, f"model_comparison_{metric.replace('²', '2')}.png")
            plt.savefig(metric_file, dpi=300, bbox_inches="tight")
            plt.close()

        csv_file = os.path.join(report_dir, "model_comparison.csv")
        comparison_df.to_csv(csv_file, index=False)

        report_file = os.path.join(report_dir, "model_comparison_report.md")

        report_content = f"""# Model Comparison Report

Created on: {datetime.now().strftime("%d.%m.%Y, %H:%M:%S")}

## Model Performance Comparison

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
"""

        best_model = comparison_df.iloc[0]['Model']

        for _, row in comparison_df.iterrows():
            report_content += f"| {row['Model']} | {row['MSE']:.2f} | {row['RMSE']:.2f} | {row['MAE']:.2f} | {row['R²']:.4f} |\n"

        report_content += "\n## Comparison Charts\n\n"

        metrics_display = {
            'rmse': 'RMSE (Root Mean Squared Error)',
            'MSE': 'MSE (Mean Squared Error)',
            'MAE': 'MAE (Mean Absolute Error)',
            'R2': 'R² (Coefficient of Determination)'
        }

        for metric, display_name in metrics_display.items():
            metric_name = metric.replace('2', '²')
            file_name = f"model_comparison_{metric}.png"
            if os.path.exists(os.path.join(report_dir, file_name)):
                report_content += f"### {display_name}\n\n![{metric_name} Comparison]({file_name})\n\n"

        report_content += "## Detailed Model Reports\n\n"
        report_content += "Here are links to detailed reports for each model:\n\n"

        for model_name in models_dict.keys():
            model_report_path = os.path.join(model_name, "report.md")
            report_content += f"- [{model_name}]({model_report_path})\n"

        report_content += "\n## Summary\n\n"

        report_content += f"""Based on the RMSE value, **{best_model}** is the best model with an RMSE of **{comparison_df.iloc[0]['RMSE']:.2f}** and an R² value of **{comparison_df.iloc[0]['R²']:.4f}**.

The complete comparison data has been saved in [model_comparison.csv](model_comparison.csv).

Evaluation date: {datetime.now().strftime("%d.%m.%Y, %H:%M:%S")}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\nComparison report has been created and saved at {report_file}")

    return comparison_df


def generate_report(models_dict, X_test, y_test, log_transformed=False):
    """
    Convenient function to generate a comprehensive report for multiple models.

    Args:
        models_dict (dict): Dictionary with model names as keys and trained models as values
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Actual target values
        log_transformed (bool): Whether the target values were log-transformed

    Returns:
        str: Path to the report directory
    """
    # Create directory for reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"results/reports/full_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)

    print(f"Creating comprehensive report in directory {report_dir}...")

    comparison_df = compare_models(
        models_dict,
        X_test,
        y_test,
        log_transformed=log_transformed,
        save_report=True
    )

    return f"results/reports/model_comparison_{timestamp}/model_comparison_report.md"


def evaluate_all_models(X_test, y_test, log_transformed=False):
    """
    Evaluates all available models and creates a comparison report.

    Args:
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Actual target values
        log_transformed (bool): Whether the target values were log-transformed

    Returns:
        str: Path to the comparison report
    """
    # Define the models to look for
    model_types = ['ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost']

    # Dictionary to store available models
    models_dict = {}

    # Check which models are available and load them
    models_dir = 'models/trained'
    if not os.path.exists(models_dir):
        models_dir = 'models'  # Fallback to main models directory

    for model_type in model_types:
        model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
        print(f"DEBUG: Checking for model at {model_path}")
        if os.path.exists(model_path):
            try:
                print(f"Loading model: {model_type}")
                model = joblib.load(model_path)
                models_dict[model_type] = model
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")

    # Also check for best model
    best_model_path = os.path.join('models', 'best_model.pkl')
    if os.path.exists(best_model_path):
        try:
            print("Loading best model")
            model = joblib.load(best_model_path)
            models_dict['best'] = model
        except Exception as e:
            print(f"Error loading best model: {e}")

    if not models_dict:
        raise ValueError("No trained models found. Please train models first.")

    # Generate the comparison report
    print(f"Evaluating {len(models_dict)} models: {', '.join(models_dict.keys())}")
    return generate_report(models_dict, X_test, y_test, log_transformed)