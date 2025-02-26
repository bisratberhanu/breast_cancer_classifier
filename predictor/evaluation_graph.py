from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Load the dataset
diabetes = fetch_openml(name="diabetes", version=1, as_frame=True)
X = diabetes.data
y = diabetes.target

# Rename columns for clarity
column_mapping = {
    'preg': 'Pregnancies',
    'plas': 'Glucose',
    'pres': 'BloodPressure',
    'skin': 'SkinThickness',
    'insu': 'Insulin',
    'mass': 'BMI',
    'pedi': 'DiabetesPedigreeFunction',
    'age': 'Age'
}
X = X.rename(columns=column_mapping)
X['Outcome'] = y.map({'tested_positive': 1, 'tested_negative': 0})

# Handle missing values (zeros in certain columns)
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    X[col] = X[col].replace(0, X[col].median())

# 1. Descriptive statistics
print("Descriptive Statistics by Outcome:\n")
for col in X.columns[:-1]:
    print(f"Feature: {col}")
    print(X.groupby('Outcome')[col].describe())
    print()

# 2. Class distribution
print("\nClass Distribution:\n", X['Outcome'].value_counts())

# 3. Skewness of each feature
print("\nSkewness of each feature:\n")
for col in X.columns[:-1]:
    print(f"{col}: {skew(X[col])}")

# 4. Correlation matrix
plt.figure(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 5. Histogram for each feature by Outcome
for col in X.columns[:-1]:
    sns.histplot(data=X, x=col, hue='Outcome', multiple='stack', bins=20)
    plt.title(f'{col} Distribution by Outcome')
    plt.show()

# 6. Box plots for each feature by Outcome
for col in X.columns[:-1]:
    sns.boxplot(x='Outcome', y=col, data=X)
    plt.title(f'{col} by Outcome')
    plt.show()

# 7. Pairplot for selected features
selected_features = ['Glucose', 'BMI', 'Age']
sns.pairplot(X, hue='Outcome', vars=selected_features)
plt.show()

# Specific scatter plot
plt.figure()
sns.scatterplot(data=X, x='Glucose', y='BMI', hue='Outcome')
plt.title('Glucose vs. BMI by Outcome')
plt.show()