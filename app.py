import streamlit as st
import pickle
import numpy as np

# Load model and feature list
model = pickle.load(open('models/delivery_model.pkl', 'rb'))
features = pickle.load(open('models/feature_columns.pkl', 'rb'))

st.title("🛒 Order Delivery Time Predictor")

# Inputs
payment_value = st.number_input("Payment Value (R$)", min_value=0.0, value=100.0)
purchase_dayofweek = st.selectbox("Purchase Day of Week (0=Mon ... 6=Sun)", list(range(7)))
purchase_hour = st.slider("Purchase Hour", 0, 23, 12)

# Predict button
if st.button("Predict Delivery Time"):
    input_data = np.array([[payment_value, purchase_dayofweek, purchase_hour]])
    prediction = model.predict(input_data)[0]
    st.success(f"⏱️ Predicted Delivery Time: {int(prediction)} days")
