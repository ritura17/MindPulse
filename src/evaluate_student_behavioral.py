import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
# MINDPULSE - BEHAVIORAL STUDENT STRESS MODEL EVALUATION
# ==========================================================

data_path = "data/raw/student_stress.csv"

df = pd.read_csv(data_path)

print("=" * 65)
print("MINDPULSE - BEHAVIORAL STUDENT STRESS MODEL EVALUATION")
print("=" * 65)


# ----------------------------------------------------------
# Remove direct symptom / mental-health variables
# ----------------------------------------------------------

excluded_features = [
    "anxiety_level",
    "depression",
    "headache"
]

X = df.drop(
    columns=excluded_features + ["stress_level"]
)

y = df["stress_level"]


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
# Model
# ----------------------------------------------------------

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
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
# Metrics
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
# Results
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

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)