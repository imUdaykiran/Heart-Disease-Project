# Import necessary libraries
import pandas as pd              # For data manipulation
import json                      # For saving metrics as JSON
import joblib                    # For saving model and scaler
from sklearn.model_selection import train_test_split  # To split data into training and testing
from sklearn.preprocessing import StandardScaler      # To normalize the features
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation
from xgboost import XGBClassifier # The ML algorithm used (XGBoost Classifier)
import warnings                  # To suppress warnings

# Ignore all warnings for cleaner output
warnings.filterwarnings("ignore")

# ------------------- Load and Preprocess Dataset -------------------

# Load dataset from CSV file
df = pd.read_csv("dataset.csv")

# Drop the 'id' column as it is not a useful feature
df = df.drop(columns=["id"])

# Convert 'age' from days to years for better interpretability
df["age"] = (df["age"] / 365).astype(int)

# ------------------- Feature Engineering -------------------

# Calculate Body Mass Index (BMI)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# Calculate pulse pressure (systolic - diastolic)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

# ------------------- Remove Outliers -------------------

# Keep only reasonable systolic blood pressure values
df = df[(df["ap_hi"] >= 90) & (df["ap_hi"] <= 200)]

# Keep only reasonable diastolic blood pressure values
df = df[(df["ap_lo"] >= 60) & (df["ap_lo"] <= 120)]

# Keep only reasonable height values (in cm)
df = df[(df["height"] >= 130) & (df["height"] <= 210)]

# Keep only reasonable weight values (in kg)
df = df[(df["weight"] >= 40) & (df["weight"] <= 180)]

# ------------------- Define Features and Target -------------------

# All columns except 'cardio' are features
X = df.drop("cardio", axis=1)

# Target variable: whether the person has cardiovascular disease (1) or not (0)
y = df["cardio"]

# ------------------- Split Data -------------------

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- Feature Scaling -------------------

# Initialize standard scaler to normalize data
scaler = StandardScaler()

# Fit scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)

# ------------------- Train XGBoost Model -------------------

# Initialize XGBoost Classifier with parameters:
# - n_estimators: number of trees (200)
# - max_depth: maximum depth of trees (5)
# - learning_rate: step size shrinkage (0.1)
# - random_state: for reproducibility (42)
# - algorithm == XGBClassifier
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

# Train (fit) the model on scaled training data
model.fit(X_train_scaled, y_train)

# ------------------- Model Evaluation -------------------

# Predict target values on the scaled test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report as a dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Print results
print("✅ Accuracy:", round(accuracy * 100, 2), "%\n")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ------------------- Save Model and Scaler -------------------

# Save the trained model to a file
joblib.dump(model, "heart_disease_model.pkl")

# Save the scaler to ensure the same transformation is used in production
joblib.dump(scaler, "scaler.pkl")

print("💾 Model and scaler saved successfully.")

# ------------------- Save Evaluation Metrics -------------------

# Create a dictionary of metrics
metrics = {
    "accuracy": f"{round(accuracy * 100, 2)}%",
    "classification_report": report_dict
}

# Save metrics to a JSON file
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("📝 Accuracy and classification report saved to model_metrics.json")
