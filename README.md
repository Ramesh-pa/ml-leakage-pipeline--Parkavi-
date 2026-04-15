# ml-leakage-pipeline--Parkavi-
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("Mean Accuracy:", scores.mean())
print("Std Dev:", scores.std())
