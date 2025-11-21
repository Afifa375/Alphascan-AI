# create_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

# Step 1: Load your CSV
# Replace 'your_data.csv' with the actual file name
data = pd.read_csv("your_data.csv")

# Step 2: Prepare features and target
# Replace 'Diagnosis' with your actual target column name
X = data.drop(columns=["Diagnosis"])
y = data["Diagnosis"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create a pipeline with preprocessing + model
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Step 5: Train the model
pipeline.fit(X_train, y_train)

# Step 6: Save the trained model as 'model.pkl'
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully as 'model.pkl'!")
