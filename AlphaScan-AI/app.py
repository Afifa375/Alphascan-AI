import streamlit as st
import pandas as pd
import pickle
import os

# ---- Safe model loading ----
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'model.pkl' is in the same folder as app.py.")
    st.stop()

# ---- App Title ----
st.title("AlphaScan AI - Thalassemia Prediction")

# ---- CSV Upload ----
st.subheader("Upload CSV File for Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())

        # Automatically detect columns (exclude non-numeric if needed)
        feature_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(feature_cols) == 0:
            st.error("No numeric columns found in CSV for prediction.")
        else:
            st.write(f"Detected feature columns: {feature_cols}")
            X = data[feature_cols]
            predictions = model.predict(X)
            data['Prediction'] = predictions
            st.write("Predictions:", data)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ---- Single Input Prediction ----
st.subheader("Or Predict Single Input")
try:
    # Dynamically create inputs for numeric features (optional: you can fix the features)
    input_data = []
    if 'feature_cols' in locals() and feature_cols:
        for col in feature_cols:
            value = st.number_input(f"{col}", value=0.0)
            input_data.append(value)
        if st.button("Predict Single Input"):
            pred = model.predict([input_data])
            st.success(f"Prediction: {pred[0]}")
except Exception as e:
    st.error(f"Error predicting single input: {e}")
