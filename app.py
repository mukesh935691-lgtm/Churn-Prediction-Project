import pandas as pd
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgbmodel.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Telecom Customer Churn Prediction")

st.write("This app predicts probability of customer churn  based on their details.")

# Sidebar feature inputs
st.sidebar.header("Customer Details")

account_length = st.sidebar.number_input("Account Length (days)", min_value=0, max_value=500, value=100)
voice_mail_messages = st.sidebar.number_input("Voice Mail Messages", min_value=0, max_value=100, value=0)
day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, value=400.0)
evening_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, value=150.0)
night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, value=150.0)
international_mins = st.sidebar.number_input("International Minutes", min_value=0.0, value=10.0)

customer_service_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, max_value=20, value=1)
day_calls = st.sidebar.number_input("Day Calls", min_value=0, value=100)
evening_calls = st.sidebar.number_input("Evening Calls", min_value=0, value=100)
night_calls = st.sidebar.number_input("Night Calls", min_value=0, value=100)
international_calls = st.sidebar.number_input("International Calls", min_value=0, value=5)

voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan (Yes=1, No=0)", [0,1])
international_plan = st.sidebar.selectbox("International Plan (Yes=1, No=0)", [0,1])

day_charge= st.sidebar.number_input("Day Charge", min_value=0, max_value=500, value=100)
evening_charge= st.sidebar.number_input("Evening_charge", min_value=0, max_value=500,value=100)
night_charge= st.sidebar.number_input("Night Charge", min_value=0, max_value=500, value=100)
international_charge= st.sidebar.number_input("International Charge", min_value=0, max_value=500, value=100)

# Derived features
total_charge = day_charge + evening_charge + night_charge + international_charge
Total_min = day_mins + evening_mins + night_mins + international_mins
Total_calls = day_calls + evening_calls + night_calls + international_calls

# Build input dataframe
input_data = pd.DataFrame({
    "account_length": [account_length],
    "voice_mail_messages": [voice_mail_messages],
    "day_mins": [day_mins],
    "evening_mins": [evening_mins],
    "night_mins": [night_mins],
    "international_mins": [international_mins],
    "customer_service_calls": [customer_service_calls],
    "day_calls": [day_calls],
    "evening_calls": [evening_calls],
    "night_calls": [night_calls],
    "international_calls": [international_calls],
    "voice_mail_plan": [voice_mail_plan],
    "international_plan": [international_plan],
    "day_charge": [day_charge],
    "evening_charge": [evening_charge],
    "night_charge": [night_charge],
    "international_charge": [international_charge],
    "total_charge": [total_charge],
    "Total_min": [Total_min],
    "Total_calls": [Total_calls]
})

expected_features = [
    'account_length', 'voice_mail_plan', 'voice_mail_messages', 'day_mins',
    'evening_mins', 'night_mins', 'international_mins',
    'customer_service_calls', 'international_plan', 'day_calls',
    'day_charge', 'evening_calls', 'evening_charge', 'night_calls',
    'night_charge', 'international_calls', 'international_charge',
    'total_charge', 'Total_min', 'Total_calls'
]

input_data = input_data[expected_features]

# Scale input
features_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    if prediction == 1:
        st.error(f"The customer is likely to churn. Probability: {prob:.2f}")
    else:
        st.success(f"The customer is not likely to churn. Probability: {prob:.2f}")
