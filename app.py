import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# --- Get the base directory of the script
BASE_DIR = os.path.dirname(__file__)

# --- Load data
csv_path = os.path.join(BASE_DIR, "alphanorm.csv")  # CSV in the same folder as app.py
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")
df = pd.read_csv(csv_path)

# --- Basic preprocessing
df = df.dropna().reset_index(drop=True)
df['phenotype'] = df['phenotype'].str.strip().str.lower().map({
    'normal':0, 'alpha carrier':1, 'alpha_carrier':1, 'alpha-carrier':1, 'carrier':1
}).astype(int)

# --- Numeric features only
if 'sex' in df.columns:
    df['sex'] = df['sex'].str.lower().map({'male':1,'female':0}).fillna(0).astype(int)

X = df.select_dtypes(include=[np.number]).drop(columns=['phenotype'])
y = df['phenotype']

# --- Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# --- SMOTE balancing
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# --- Train RandomForest & XGB
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

rf.fit(X_train_bal, y_train_bal)
xgb.fit(X_train_bal, y_train_bal)

# --- Evaluate
for name, model in [("RandomForest", rf), ("XGBoost", xgb)]:
    pred = model.predict(X_test)
    print(f"\n{name} test accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# --- Save artifacts
rf_path = os.path.join(BASE_DIR, "rf_model.pkl")
xgb_path = os.path.join(BASE_DIR, "xgb_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

joblib.dump(rf, rf_path)
joblib.dump(xgb, xgb_path)
joblib.dump(scaler, scaler_path)
print("✅ Models & scaler saved.")
