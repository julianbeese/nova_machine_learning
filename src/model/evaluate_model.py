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
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime


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
        # Add explanation for the Actual vs. Predicted plot
        actual_vs_predicted_explanation = """**Actual vs. Predicted Prices**: This scatter plot shows how well the model's predictions match the actual prices. 
Points that fall on the red dashed line indicate perfect predictions. 
Points above the line represent underestimations (predicted < actual), while points below represent overestimations (predicted > actual).
The closer the points are to the line, the more accurate the model."""

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

"""
            report_content += actual_vs_predicted_explanation + "\n\n"
            report_content += f"![Actual vs. Predicted Prices](actual_vs_predicted.png)\n\n"

        if plot:
            plt.show()

        if save_report and not plot:
            plt.close()

        # Add explanation for the Residual Plot
        residual_plot_explanation = """**Residual Plot**: This plot shows the prediction errors (residuals) against the predicted values. 
The red dashed line at y=0 represents perfect predictions. 
Ideally, points should be randomly scattered around this line with no clear pattern.
Patterns in this plot can indicate areas where the model consistently over or underestimates prices."""

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

"""
            report_content += residual_plot_explanation + "\n\n"
            report_content += f"![Residual Plot](residuals.png)\n\n"

        if plot:
            plt.show()

        if save_report and not plot:
            plt.close()

        # Add explanation for the Distribution of Residuals
        residual_distribution_explanation = """**Distribution of Residuals**: This histogram shows the distribution of prediction errors (residuals).
Ideally, residuals should be normally distributed around zero (the red dashed line).
A symmetrical bell-shaped curve centered at zero indicates unbiased predictions.
Skewness in this distribution suggests the model may be consistently over or underestimating prices for certain ranges."""

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

"""
            report_content += residual_distribution_explanation + "\n\n"
            report_content += f"![Distribution of Residuals](residuals_histogram.png)\n\n"

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

                # Add explanation for the Feature Importance plot
                feature_importance_explanation = """**Feature Importance**: This bar chart shows the top features that influence the model's predictions the most.
Features are ranked by their absolute coefficient values, which indicate how strongly each feature affects the predicted price.
Higher values indicate stronger influence on the price prediction.
Understanding these key factors can help identify which car attributes most affect resale value."""

                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()

                if save_report:
                    importance_file = os.path.join(report_dir, "feature_importance.png")
                    plt.savefig(importance_file, dpi=300, bbox_inches="tight")

                    report_content += """### Feature Importance

"""
                    report_content += feature_importance_explanation + "\n\n"
                    report_content += f"![Feature Importance](feature_importance.png)\n\n"

                    report_content += """#### Top 20 Features by Importance

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