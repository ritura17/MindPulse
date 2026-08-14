import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# MINDPULSE - FINAL STUDENT STRESS MODEL
# ==========================================================

print("=" * 70)
print("MINDPULSE - FINAL STUDENT STRESS MODEL")
print("=" * 70)


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

data_path = "data/raw/student_stress.csv"

df = pd.read_csv(data_path)


# ----------------------------------------------------------
# Features
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


print("\nFeatures Used:")
print(X.columns.tolist())

print("\nNumber of Features:")
print(X.shape[1])

print("\nTarget:")
print(y.name)


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


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ----------------------------------------------------------
# Final Model
# ----------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ----------------------------------------------------------
# Train
# ----------------------------------------------------------

print("\nTraining Final Logistic Regression Model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

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


print("\nFinal Model Performance")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

model_path = "models/student_stress_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel Saved Successfully!")
print(f"Location: {model_path}")


# ----------------------------------------------------------
# Save Feature Names
# ----------------------------------------------------------

feature_path = "models/student_features.pkl"

joblib.dump(
    X.columns.tolist(),
    feature_path
)

print(f"Features Saved: {feature_path}")


print("\n" + "=" * 70)
print("FINAL STUDENT MODEL READY")
print("=" * 70)