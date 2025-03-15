import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load the trained model (update path if necessary)
MODEL_PATH = 'models/trained/best_model.pkl'

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    model = None
else:
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully.")
        print(f"Model loaded successfully: {type(model)}")  # Debugging output
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None


def preprocess_input(data):
    """
    Preprocess the input data to match the model's expected format.
    """
    # Convert leather interior to binary
    data['leather_interior'] = 1 if str(data['leather_interior']).lower() in ['yes', '1', 'true'] else 0

    # Convert engine volume to float
    try:
        data['engine_volume'] = float(data['engine_volume'])
    except ValueError:
        data['engine_volume'] = np.nan

    # Convert mileage to integer
    try:
        data['mileage'] = int(data['mileage'])
    except ValueError:
        data['mileage'] = np.nan

    # Convert production year to integer
    try:
        data['prod_year'] = int(data['prod_year'])
    except ValueError:
        data['prod_year'] = np.nan

    # Convert cylinders to integer
    try:
        data['cylinders'] = int(data['cylinders'])
    except ValueError:
        data['cylinders'] = np.nan

    # Add engineered features
    current_year = 2025
    data['car_age'] = current_year - data['prod_year'] if 'prod_year' in data else np.nan
    data['age_mileage_interaction'] = data['car_age'] * data[
        'mileage'] if 'car_age' in data and 'mileage' in data else np.nan
    data[
        'manufacturer_year_interaction'] = f"{data['prod_year']}_{data['manufacturer']}" if 'prod_year' in data and 'manufacturer' in data else "UNKNOWN"

    # Create DataFrame for model input
    df = pd.DataFrame([data])
    return df


st.title("Car Price Prediction App")
st.write("Enter the car details to get a predicted price.")

# Input fields
manufacturer = st.text_input("Manufacturer")
car_model = st.text_input("Model")
prod_year = st.number_input("Production Year", min_value=1939, max_value=2020, value=2000, step=1)
category = st.selectbox("Category",
                        ['Hatchback', 'Sedan', 'Jeep', 'Coupe', 'Universal', 'Microbus', 'Pickup', 'Minivan',
                         'Cabriolet', 'Goods wagon', 'Limousine'])
leather_interior = st.radio("Leather Interior", ("Yes", "No"))
fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG', 'Hybrid', 'CNG', 'Plug-in Hybrid', 'Hydrogen'])
engine_volume = st.number_input("Engine Volume", min_value=0.0, step=0.1)
mileage = st.number_input("Mileage", min_value=0, step=1000)
cylinders = st.number_input("Cylinders", min_value=1, max_value=20, step=1)
gear_box_type = st.selectbox("Gear Box Type", ['Manual', 'Tiptronic', 'Automatic', 'Variator'])
drive_wheels = st.selectbox("Drive Wheels", ['Front', '4x4', 'Rear'])
doors = st.radio("Doors", [3, 5])
wheel = st.radio("Wheel", ['Left wheel', 'Right-hand drive'])
color = st.text_input("Color")
airbags = st.number_input("Airbags", min_value=0, max_value=20, step=1)

if st.button("Predict Price"):
    if model is None:
        st.error("Model is not loaded. Please check the error messages above.")
    else:
        # Collect input data
        input_data = {
            'manufacturer': manufacturer.upper() if manufacturer else "UNKNOWN",
            'model': car_model if car_model else "UNKNOWN",
            'prod_year': int(prod_year) if prod_year else 2000,
            'category': category if category else "UNKNOWN",
            'leather_interior': 1 if leather_interior.lower() == "yes" else 0,
            'fuel_type': fuel_type if fuel_type else "UNKNOWN",
            'engine_volume': float(engine_volume) if engine_volume else 1.5,
            'mileage': int(mileage) if mileage else 50000,
            'cylinders': int(cylinders) if cylinders else 4,
            'gear_box_type': gear_box_type if gear_box_type else "UNKNOWN",
            'drive_wheels': drive_wheels if drive_wheels else "UNKNOWN",
            'doors': int(doors) if doors else 5,
            'wheel': wheel if wheel else "UNKNOWN",
            'color': color if color else "UNKNOWN",
            'airbags': int(airbags) if airbags else 2,
        }

        # Preprocess input
        processed_data = preprocess_input(input_data)

        try:
            # Predict
            predicted_price = model.predict(processed_data)[0]
            # Display result
            st.success(f"Estimated Car Price: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            print(f"Prediction Error: {e}")  # Debugging output