import streamlit as st
import pandas as pd
import pickle
import os

st.title("AlphaScan AI - Thalassemia Prediction")

# Check if model file exists
model_path = "model.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ Model file not found. Please make sure 'model.pkl' is in the same folder as app.py.")
else:
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df)

            # Make predictions
            predictions = model.predict(df)
            st.write("Predictions:")
            st.write(predictions)
        except Exception as e:
            st.error(f"Error processing file: {e}")
