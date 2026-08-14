import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==========================================================
# MINDPULSE - BEHAVIORAL STUDENT STRESS MODEL
# ==========================================================

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

data_path = "data/raw/student_stress.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("MINDPULSE - BEHAVIORAL STUDENT STRESS MODEL")
print("=" * 60)


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


print("\nExcluded Features:")
print(excluded_features)

print("\nFeatures Used:")
print(X.columns.tolist())

print("\nNumber of Features:")
print(X.shape[1])


# ----------------------------------------------------------
# Target
# ----------------------------------------------------------

print("\nTarget:")
print(y.name)

print("\nTarget Distribution:")
print(y.value_counts().sort_index())


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
# Model Pipeline
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

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])

print("\nFirst 20 Actual Values:")
print(y_test.iloc[:20].values)