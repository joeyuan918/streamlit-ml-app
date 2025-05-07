
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

pickup_time = st.number_input("Time To Pickup Location (0 - 1300 sec)", min_value=0, max_value=1300, step=1)
offer_amount = st.number_input("Offer Amount (4 - 12 dollars)", min_value=4, max_value=12, step=0.01)
offer_time = st.number_input("Offer to Delivery Start Time (-360 - 1000 sec)", min_value=-360, max_value=1000, step=1)
delivery_time = st.number_input("Delivery Time (Start to End (540 - 1500 sec))", min_value=540, max_value=1500, step=1)

if st.button("Predict"):
    # Frequency encode IDs
    driver_encoded = driver_freq.get(driver_id, 0)
    delivery_encoded = delivery_freq.get(delivery_id, 0)

    # Construct feature vector in the same order as training
    features = np.array([[pickup_time, offer_amount, offer_time, delivery_time, driver_encoded, delivery_encoded]])
    
    # Create input DataFrame
    input_df = pd.DataFrame([{
        "TimeToPickupLocation": pickup_time,
        "OfferAmount": offer_amount,
        "OfferTime": offer_time,
        "DeliveryTime": delivery_time,
        "DriverId_FreqEnc": driver_encoded,
        "DeliveryId_FreqEnc": delivery_encoded
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba > threshold).astype(int)
    st.write(f"Prediction: {int(pred)} (0 = Reject, 1 = Accept)")
