# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('xgb_bank_marketing_model.joblib')

st.title("Bank Term Deposit Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
job = st.selectbox("Job", ["admin.", "technician", "blue-collar", "retired", "services", "management", "unemployed", "entrepreneur", "self-employed", "housemaid", "student", "unknown"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Credit in default?", ["no", "yes"])
balance = st.number_input("Average yearly balance (€)")
housing = st.selectbox("Housing loan?", ["no", "yes"])
loan = st.selectbox("Personal loan?", ["no", "yes"])

# Convert input to DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Subscribed ✅" if prediction == 1 else "Not Subscribed ❌"
    st.success(f"The model predicts: **{result}**")
