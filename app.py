
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model, frequency encoding mappings, and top driver/delivery ID's
loaded = joblib.load("xgboost_model_with_thresh.pkl")
model = loaded["xgboost_model"]
threshold = loaded["threshold"]
driver_freq = joblib.load("driver_freq.pkl")
delivery_freq = joblib.load("delivery_freq.pkl")
top_driver_ids = joblib.load("top_driver_ids.pkl")
top_delivery_ids = joblib.load("top_delivery_ids.pkl")

# Add 'Other' option
driver_options = top_driver_ids + ["Other"]
delivery_options = top_delivery_ids + ["Other"]

st.title("Delivery Acceptance Predictor")

# User inputs
# Driver ID selection
driver_choice = st.selectbox("Select Driver ID", driver_options)
if driver_choice == "Other":
    driver_id = st.text_input("Enter Driver ID")
else:
    driver_id = driver_choice

# Delivery ID selection
delivery_choice = st.selectbox("Select Delivery ID", delivery_options)
if delivery_choice == "Other":
    delivery_id = st.text_input("Enter Delivery ID")
else:
    delivery_id = delivery_choice

pickup_time = st.number_input("Time To Pickup Location (0 - 1300 sec)", step=1)
offer_amount = st.number_input("Offer Amount (4 - 12 dollars)", step=0.01)
offer_time = st.number_input("Offer Time (Offer to Delivery End (300 - 2000 sec))", step=1)
delivery_time = st.number_input("Delivery Time (Start to End (500 - 1500 sec))", step=1)

if st.button("Predict"):
    # Frequency encode IDs
    driver_encoded = driver_freq.get(driver_id, 0)
    delivery_encoded = delivery_freq.get(delivery_id, 0)

    # Construct feature vector in the same order as training
    features = np.array([[pickup_time, offer_amount, offer_time, delivery_time, driver_encoded, delivery_encoded]])
    
    # Create input DataFrame
    input_df = pd.DataFrame([{
        "pickup_time": pickup_time,
        "offer_amount": offer_amount,
        "offer_time": offer_time,
        "delivery_time": delivery_time,
        "driver_freq": driver_encoded,
        "delivery_freq": delivery_encoded
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba > threshold).astype(int)
    st.write(f"Prediction: {int(pred)} (0 = Reject, 1 = Accept)")
