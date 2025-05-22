import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading trained ML models

# Load trained models (Ensure the model files exist)
crop_model = joblib.load("crop_model.pkl")  # Update with actual model filename
fertilizer_model = joblib.load("fertilizer_model.pkl")  # Update with actual model filename

# Streamlit App Title
st.title("🌱 Crop & Fertilizer Recommendation System")

# Sidebar Navigation
st.sidebar.header("Choose an Option")
option = st.sidebar.selectbox("Select:", ["Crop Recommendation", "Fertilizer Recommendation"])

if option == "Crop Recommendation":
    st.subheader("🌾 Recommend the Best Crop for Your Soil")
    
    # User Inputs
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=100, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    if st.button("Predict Crop"):
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        crop_prediction = crop_model.predict(features)[0]
        st.success(f"🌱 Recommended Crop: **{crop_prediction}**")

elif option == "Fertilizer Recommendation":
    st.subheader("🧪 Recommend the Best Fertilizer for Your Crop")
    
    # User Inputs
    crop_type = st.selectbox("Select Crop Type", ["Rice", "Wheat", "Maize", "Barley", "Soybean"])
    nitrogen = st.number_input("Nitrogen Level", min_value=0, max_value=100, value=50)
    phosphorus = st.number_input("Phosphorus Level", min_value=0, max_value=100, value=50)
    potassium = st.number_input("Potassium Level", min_value=0, max_value=100, value=50)

    if st.button("Predict Fertilizer"):
        features = np.array([[nitrogen, phosphorus, potassium]])  # Modify based on model input
        fertilizer_prediction = fertilizer_model.predict(features)[0]
        st.success(f"🧪 Recommended Fertilizer: **{fertilizer_prediction}**")

# Run using: `streamlit run app.py`
