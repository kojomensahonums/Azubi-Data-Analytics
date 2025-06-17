# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load('bank_marketing.joblib')
model_columns = joblib.load('model_columns.joblib')
robust_scaler = joblib.load('robust_scaler.joblib')
minmax_scaler = joblib.load('minmax_scaler.joblib')
robust_cols = joblib.load('robust_columns.joblib')
minmax_cols = joblib.load('minmax_columns.joblib')

# Streamlit app
st.title("📊 Term Deposit Subscription Predictor")
st.write("Enter client information below to predict subscription likelihood.")

# Form input
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'self-employed', 'student', 'unknown'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['secondary', 'tertiary', 'primary', 'unknown'])
    default = st.selectbox("Credit in default?", ['no', 'yes'])
    balance = st.number_input("Account Balance", value=1000)
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
    loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])
    contact = st.selectbox("Contact Communication", ['cellular', 'telephone', 'unknown'])
    day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    duration = st.number_input("Last Contact Duration (seconds)", value=120)
    campaign = st.number_input("Number of Contacts During Campaign", value=1)
    pdays = st.number_input("Days Since Last Contact", value=-1)
    previous = st.number_input("Number of Contacts Before This Campaign", value=0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ['unknown', 'other', 'failure', 'success'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create dataframe from input
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])

    # Derived features
    input_df['pdays_group'] = input_df['pdays'].apply(lambda x: 'Not contacted' if x < 0 else 'Contacted')

    # One-hot encoding
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'pdays_group']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    # Align with model columns
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Apply scaling
    if robust_cols:
        input_encoded[robust_cols] = robust_scaler.transform(input_encoded[robust_cols])
    if minmax_cols:
        input_encoded[minmax_cols] = minmax_scaler.transform(input_encoded[minmax_cols])

    # Predict
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    # Output
    if pred == 1:
        st.success(f"✅ The client is likely to subscribe. (Probability: {prob:.2%})")
    else:
        st.warning(f"⚠️ The client is unlikely to subscribe. (Probability: {prob:.2%})")
