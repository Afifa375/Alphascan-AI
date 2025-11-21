import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# --- App title
st.set_page_config(page_title="AlphaScan AI", layout="wide")
st.title("🧬 AlphaScan AI - Thalassemia Prediction")

# --- Base directory
BASE_DIR = os.path.dirname(__file__)

# --- Load CSV
csv_path = os.path.join(BASE_DIR, "alphanorm.csv")
try:
    df = pd.read_csv(csv_path)
    st.success("✅ CSV loaded successfully!")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# --- Preprocessing
df = df.dropna().reset_index(drop=True)
df['phenotype'] = df['phenotype'].str.strip().str.lower().map({
    'normal':0, 'alpha carrier':1, 'alpha_carrier':1, 'alpha-carrier':1, 'carrier':1
}).astype(int)

if 'sex' in df.columns:
    df['sex'] = df['sex'].str.lower().map({'male':1,'female':0}).fillna(0).astype(int)

# --- Select numeric features only
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols.remove('phenotype')  # remove target
X = df[feature_cols]
y = df['phenotype']

# --- Load pre-trained models
try:
    rf = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- User input section
st.header("Enter Patient Data for Prediction")

def user_input_features():
    data = {}
    # Arrange input fields in 2 columns for better layout
    cols = st.columns(2)
    for i, col_name in enumerate(feature_cols):
        col_widget = cols[i % 2]
        # Special case for sex column as dropdown
        if col_name == 'sex':
            data[col_name] = col_widget.selectbox("Sex", options=[1,0], format_func=lambda x: "Male" if x==1 else "Female")
        else:
            data[col_name] = col_widget.number_input(col_name, value=float(df[col_name].mean()))
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Scale input
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error scaling input: {e}")
    st.stop()

# --- Predictions
def predict_model(model, input_scaled):
    pred = model.predict(input_scaled)
    if len(pred) > 0:
        return 'Normal' if pred[0]==0 else 'Alpha Carrier'
    else:
        return "Prediction error"

st.subheader("Predictions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict with RandomForest"):
        result = predict_model(rf, input_scaled)
        st.success(f"RandomForest Prediction: **{result}**")

with col2:
    if st.button("Predict with XGBoost"):
        result = predict_model(xgb, input_scaled)
        st.success(f"XGBoost Prediction: **{result}**")

# --- Optional: show dataset summary
if st.checkbox("Show dataset summary"):
    st.subheader("Dataset Summary")
    st.dataframe(df.describe())
