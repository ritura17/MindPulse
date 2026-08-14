import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# MINDPULSE - MODEL EVALUATION
# ==========================================================

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

data_path = "data/processed/mental_health_lifestyle_cleaned.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("MINDPULSE - MODEL EVALUATION")
print("=" * 60)


# ----------------------------------------------------------
# Features and Target
# ----------------------------------------------------------

X = df.drop("Stress Level", axis=1)
y = df["Stress Level"]


# ----------------------------------------------------------
# Feature Types
# ----------------------------------------------------------

categorical_features = [
    "Gender",
    "Exercise Level",
    "Diet Type"
]

numerical_features = [
    "Age",
    "Sleep Hours",
    "Work Hours per Week",
    "Screen Time per Day (Hours)",
    "Social Interaction Score",
    "Happiness Score"
]


# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# ----------------------------------------------------------
# Train/Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------------------------------------
# Train
# ----------------------------------------------------------

model.fit(X_train, y_train)


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = model.predict(X_test)


# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("\nModel Performance")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")
print("-" * 40)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

print("\nConfusion Matrix")
print("-" * 40)

cm = confusion_matrix(y_test, y_pred)

print(cm)