"""
Module for data preprocessing in the car price prediction project.

This module contains functions for preparing and transforming the raw data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df, log_transform=False, prediction_mode=False):
    """
    Preprocessing of data for car price prediction.

    Args:
        df (pandas.DataFrame): Input DataFrame with the raw data
        log_transform (bool): Whether a logarithmic transformation of the target values should be performed
        prediction_mode (bool): Whether the function is called in prediction mode (no split into training/test data)

    Returns:
        When prediction_mode=False:
            tuple: (X_train, X_test, y_train, y_test, numeric_cols, categorical_cols)
                - X_train, X_test: Training and test features
                - y_train, y_test: Training and test target values
                - numeric_cols: List of numeric columns
                - categorical_cols: List of categorical columns

        When prediction_mode=True:
            tuple: (X, None, None, None, numeric_cols, categorical_cols)
                - X: Preprocessed features for prediction
                - numeric_cols: List of numeric columns
                - categorical_cols: List of categorical columns
    """
    data = df.copy()
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('.', '')

    if 'prod_year' in data.columns:
        if data['prod_year'].dtype == 'object':
            try:
                data['prod_year'] = pd.to_datetime(data['prod_year']).dt.year
            except:
                pass

    current_year = 2025
    if 'prod_year' in data.columns:
        data['car_age'] = current_year - data['prod_year']

    if 'engine_volume' in data.columns and data['engine_volume'].dtype == 'object':
        data['engine_volume'] = data['engine_volume'].str.extract(r'(\d+\.?\d*)').astype(float)

    binary_cols = ['leather_interior']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'price' in numeric_cols and not prediction_mode:
        numeric_cols.remove('price')

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    if not prediction_mode:
        for col in numeric_cols + ['price']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                data = data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]

    if prediction_mode:
        X = data
        return X, None, None, None, numeric_cols, categorical_cols
    else:
        X = data.drop('price', axis=1)
        y = data['price']

        if log_transform or y.skew() > 1:
            y = np.log1p(y)
            print("Logarithmic transformation of target values performed")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols