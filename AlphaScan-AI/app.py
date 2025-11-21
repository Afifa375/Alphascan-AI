import csv
import statistics
import pickle  # if you are loading a trained ML model
def load_and_clean_csv(file_path):
    """
    Load CSV safely and handle missing values.
    Numeric columns are filled with mean, Phenotype with 'Unknown'.
    """
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        
        for row_num, row in enumerate(reader, start=2):
            if not row:
                continue
            # fill missing columns
            row_filled = []
            for i in range(len(headers)):
                if i >= len(row) or row[i].strip() == '':
                    row_filled.append(None)
                else:
                    row_filled.append(row[i].strip())
            data.append(row_filled)

    # Fill missing numeric values
    for col_index in range(len(headers)-1):  # assuming last column is 'Phenotype'
        numeric_vals = [float(row[col_index]) for row in data if row[col_index] is not None]
        col_mean = statistics.mean(numeric_vals)
        for row in data:
            if row[col_index] is None:
                row[col_index] = col_mean
            else:
                row[col_index] = float(row[col_index])

    # Fill missing Phenotype
    for row in data:
        row[-1] = row[-1] if row[-1] is not None else "Unknown"

    return headers, data
# Load your trained model (replace 'model.pkl' with your actual model file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
file_path = r"C:\Users\Ayesha\Desktop\project\data.csv"
headers, data = load_and_clean_csv(file_path)

for i, row in enumerate(data, start=1):
    features = row[:-1]  # all columns except 'Phenotype'
    prediction = model.predict([features])
    print(f"Row {i} Prediction: {prediction[0]}")
