import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define the path to save the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models/naive_bayes_model.pkl")

# Ensure the directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load the Pima Indians Diabetes dataset from OpenML
diabetes = fetch_openml(name="diabetes", version=1)
X = diabetes.data  # The 8 features
y = diabetes.target  # The target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"Model trained and saved at {MODEL_PATH}")

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Print additional details
print(f"\nNumber of test instances: {len(y_test)}")
print(f"Number of training instances: {len(y_train)}")
