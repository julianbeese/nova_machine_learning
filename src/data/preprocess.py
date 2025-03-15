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

    # Handling production year (convert to year if in object format)
    if 'prod_year' in data.columns:
        if data['prod_year'].dtype == 'object':
            try:
                data['prod_year'] = pd.to_datetime(data['prod_year']).dt.year
            except:
                pass

    current_year = 2025
    if 'prod_year' in data.columns:
        data['car_age'] = current_year - data['prod_year']

    # Handling engine volume if it's in object type
    if 'engine_volume' in data.columns and data['engine_volume'].dtype == 'object':
        data['engine_volume'] = data['engine_volume'].str.extract(r'(\d+\.?\d*)').astype(float)



    # change doors column
    data["doors"] = data["doors"].replace({'04-May': 5, '>5': 5, '02-Mar': 3})

    # Binary columns mapping (e.g., leather interior)
    binary_cols = ['leather_interior']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)

    # drop irrelevant columns
    data.drop(columns=['id', 'levy'], inplace=True)

    # Ensure 'mileage' is numeric
    if 'mileage' in data.columns:
        # Remove the ' km' suffix and convert to numeric
        data['mileage'] = data['mileage'].str.replace(' km', '', regex=False)  # Remove the ' km' part
        data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')  # Convert to numeric, non-numeric values become NaN
        data['mileage'].fillna(data['mileage'].median(), inplace=True)  # Handle missing values by replacing with median

    # Interaction Terms
    if 'car_age' in data.columns and 'mileage' in data.columns:
        data["age_mileage_interaction"] = data["car_age"] * data["mileage"]  # Older cars with higher mileage

    if 'prod_year' in data.columns and 'manufacturer' in data.columns:
        data["manufacturer_year_interaction"] = data["prod_year"].astype(str) + "_" + data["manufacturer"]  # Combining brand & year

    # List of numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove 'price' from numeric columns for prediction mode
    if 'price' in numeric_cols and not prediction_mode:
        numeric_cols.remove('price')

    # List of categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    if not prediction_mode:
        # Outlier removal using IQR method
        print(f"Data before outlier removal: {data.shape[0]} rows")
        for col in numeric_cols + ['price']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                data = data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]
        print(f"Data after outlier removal: {data.shape[0]} rows")

    if prediction_mode:
        X = data
        return X, None, None, None, numeric_cols, categorical_cols
    else:
        X = data.drop('price', axis=1)
        y = data['price']

        if log_transform or y.skew() > 1:
            y = np.log1p(y)  # Log transform price if needed
            print("Logarithmic transformation of target values performed")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
