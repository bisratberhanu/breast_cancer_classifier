import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Simple Logistic Regression class
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

# Step 1: Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 1]]  # Use only sepal length (0) and sepal width (1)
y = iris.target

# Filter to two classes: Iris-setosa (0) vs. Iris-versicolor (1)
mask = y < 2  # Exclude class 2 (Iris-virginica)
X = X[mask]
y = y[mask]

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the model
model = SimpleLogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

# Step 4: Test the model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Step 5: Evaluate
train_accuracy = np.mean(train_pred == y_train) * 100
test_accuracy = np.mean(test_pred == y_test) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")
print("\nTest Predictions:", test_pred)
print("Actual Test Labels:", y_test)

# Step 6: Predict a new sample
new_sample = np.array([[5.5, 3.0]])  # Example: sepal length=5.5, sepal width=3.0
new_pred = model.predict(new_sample)
print(f"\nNew Sample {new_sample} Prediction: {'Versicolor' if new_pred[0] == 1 else 'Setosa'}")