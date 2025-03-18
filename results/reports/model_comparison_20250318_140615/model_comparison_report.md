# Model Comparison Report

Created on: 18.03.2025, 14:06:15

## Model Performance Comparison

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| xgboost | 46738256.77 | 6836.54 | 4302.15 | 0.7854 |
| random_forest | 46943205.94 | 6851.51 | 4055.64 | 0.7845 |
| ridge | 107155659.80 | 10351.60 | 7511.17 | 0.5080 |
| elastic_net | 107300488.40 | 10358.59 | 7654.83 | 0.5073 |
| lasso | 107527599.16 | 10369.55 | 7526.13 | 0.5063 |

## Comparison Charts

### RMSE (Root Mean Squared Error)

**RMSE (Root Mean Squared Error)**: This chart compares the RMSE values across different models.
RMSE measures the average magnitude of prediction errors in the original units (euros for car prices).
Lower values indicate better model performance as they represent smaller prediction errors.
RMSE is particularly sensitive to large errors, as it squares the differences before averaging.

![rmse Comparison](model_comparison_rmse.png)

### MSE (Mean Squared Error)

**MSE (Mean Squared Error)**: This chart compares the MSE values across different models.
MSE is calculated by taking the average of squared differences between predicted and actual values.
Lower values indicate better model performance.
MSE heavily penalizes larger errors due to the squaring operation, making it useful for identifying models that avoid large mistakes.

![MSE Comparison](model_comparison_MSE.png)

### MAE (Mean Absolute Error)

**MAE (Mean Absolute Error)**: This chart compares the MAE values across different models.
MAE measures the average absolute difference between predicted and actual car prices in euros.
Lower values indicate better model performance.
Unlike MSE/RMSE, MAE treats all error magnitudes linearly (no squaring), making it less sensitive to outliers.

![MAE Comparison](model_comparison_MAE.png)

### R² (Coefficient of Determination)

**R² (Coefficient of Determination)**: This chart compares the R² values across different models.
R² represents the proportion of variance in the car prices that is predictable from the features.
Higher values (closer to 1.0) indicate better model performance.
A value of 0.75 means the model explains 75% of the variance in car prices, while a value of 0 would mean the model provides no better predictions than simply using the mean price.

![R² Comparison](model_comparison_R2.png)


## Summary

Based on the RMSE value, **xgboost** is the best model with an RMSE of **6836.54** and an R² value of **0.7854**.

The complete comparison data has been saved in [model_comparison.csv](model_comparison.csv).

Evaluation date: 18.03.2025, 14:06:15
