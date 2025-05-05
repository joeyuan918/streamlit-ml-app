
import streamlit as st
import joblib
import numpy as np

# Load model and frequency encoding mappings
model = joblib.load("xgboost_model.pkl")
driver_freq = joblib.load("driver_freq.pkl")
delivery_freq = joblib.load("delivery_freq.pkl")

st.title("Delivery Acceptance Predictor")

# User inputs
driver_id = st.number_input("Driver ID", step=1)
delivery_id = st.number_input("Delivery ID", step=1)
pickup_time = st.number_input("Time To Pickup Location(sec)", step=1)
offer_amount = st.number_input("Offer Amount ($)", step=1.0)
offer_time = st.number_input("Offer Time (Offer - Delivery End (sec))", step=1)
delivery_time = st.number_input("Delivery Time (End - Start (sec))", step=1)

if st.button("Predict"):
    # Frequency encode IDs
    driver_encoded = driver_freq.get(driver_id, 0)
    delivery_encoded = delivery_freq.get(delivery_id, 0)

    # Construct feature vector in the same order as training
    features = np.array([[pickup_time, offer_amount, offer_time, delivery_time, driver_encoded, delivery_encoded]])

    # Make prediction
    prediction = model.predict(features)[0]
    st.write(f"Prediction: {int(prediction)} (0 = Reject, 1 = Accept)")
