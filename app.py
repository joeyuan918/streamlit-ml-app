
import streamlit as st
import joblib
import numpy as np

# Load model and frequency encoding mappings
loaded = joblib.load("xgboost_model_with_thresh.pkl")
model = loaded["xgboost_model"]
threshold = loaded["threshold"]
driver_freq = joblib.load("driver_freq.pkl")
delivery_freq = joblib.load("delivery_freq.pkl")

st.title("Delivery Acceptance Predictor")

# User inputs
driver_id = st.number_input("Driver ID", step=1)
delivery_id = st.number_input("Delivery ID", step=1)
pickup_time = st.number_input("Time To Pickup Location (0 - 1300 sec)", step=1)
offer_amount = st.number_input("Offer Amount ($4 - $12)", step=1.0)
offer_time = st.number_input("Offer Time (Offer to Delivery End (300 - 2000 sec))", step=1)
delivery_time = st.number_input("Delivery Time (Start to End (500 - 1500 sec))", step=1)

if st.button("Predict"):
    # Frequency encode IDs
    driver_encoded = driver_freq.get(driver_id, 0)
    delivery_encoded = delivery_freq.get(delivery_id, 0)

    # Construct feature vector in the same order as training
    features = np.array([[pickup_time, offer_amount, offer_time, delivery_time, driver_encoded, delivery_encoded]])

    # Make prediction
    prediction = model.predict(features)[0]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > threshold).astype(int)
    st.write(f"Prediction: {int(pred)} (0 = Reject, 1 = Accept)")
