# Delivery Offer Acceptance Predictor 🛵📦

A Streamlit web app that predicts whether a delivery driver will accept an offer, using a machine learning model trained on historical delivery data.

---

## 🚀 Live Demo

👉 [Click here to try the live app](https://YOUR-STREAMLIT-URL.streamlit.app)

---

## 🧠 Model Overview

This app uses a binary classifier (XGBoost) trained on:

- **Driver ID** (frequency-encoded)
- **Delivery ID** (frequency-encoded)
- **Pickup Time**
- **Offer Time**
- **Delivery Time**
- **Offer Amount**

---

## 🛠 Features

- Real-time predictions based on user input
- Clean Streamlit UI
- Frequency encoding using pre-computed mappings
- Easy deployment with Streamlit Cloud

---

## 📦 Tech Stack
- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- SciPy
- Matplotlib & Seaborn
- Streamlit

---

## 🗃️ Project Structure

```plaintext
delivery-acceptance-app/
├── app.py                  # Streamlit application
├── xgboost_model.pkl       # Trained ML model
├── driver_freq.pkl         # Frequency encoder for Driver ID
├── delivery_freq.pkl       # Frequency encoder for Delivery ID
├── requirements.txt        # Python dependencies
└── README.md               # Project info
```

---

## 🚀 How to Run

1. Clone the repo  
2. Run `pip install -r requirements.txt`  
3. Launch: `streamlit run app.py`
