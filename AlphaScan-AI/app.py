import streamlit as st
import pickle
import pandas as pd

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Make sure 'model.pkl' is in the same folder as app.py.")
    st.stop()

st.title("AlphaScan AI - Thalassemia Prediction")

# Example: assume your model expects 5 features
num_features = 5  
inputs = []

# Dynamically take inputs based on expected number of features
for i in range(num_features):
    value = st.text_input(f"Enter value for Feature {i+1}")
    inputs.append(value)

# Convert inputs to numeric values safely
try:
    inputs = [float(x) for x in inputs]
except ValueError:
    st.warning("Please enter valid numeric values for all features.")
    st.stop()

# Make prediction only if all inputs are filled
if all(inputs):
    # Convert to DataFrame as model expects
    input_df = pd.DataFrame([inputs], columns=[f"Feature{i+1}" for i in range(num_features)])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Result: {prediction}")
else:
    st.info("Please fill in all the feature values.")
