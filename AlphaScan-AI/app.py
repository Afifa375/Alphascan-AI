import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# --- Streamlit app title
st.title("AlphaScan AI - Thalassemia Prediction")

# --- Get base directory
BASE_DIR = os.path.dirname(__file__)

# --- CSV path
csv_path = os.path.join(BASE_DIR, "alphanorm.csv")

# --- Load CSV with error handling
try:
    df = pd.read_csv(csv_path)
    st.success("✅ CSV loaded successfully!")
    st.dataframe(df.head())  # show first 5 rows
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# --- Load pre-trained models
try:
    rf = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- User input for prediction
st.header("Enter patient data to predict phenotype:")

def user_input_features():
    data = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Scale input
input_scaled = scaler.transform(input_df)

# --- Prediction buttons
if st.button("Predict with RandomForest"):
    pred = rf.predict(input_scaled)[0]
    st.write(f"Predicted phenotype: **{'Normal' if pred==0 else 'Alpha Carrier'}**")

if st.button("Predict with XGBoost"):
    pred = xgb.predict(input_scaled)[0]
    st.write(f"Predicted phenotype: **{'Normal' if pred==0 else 'Alpha Carrier'}**")

# --- Optional: show CSV summary
if st.checkbox("Show dataset summary"):
    st.write(df.describe())
