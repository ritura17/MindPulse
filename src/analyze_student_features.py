import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==========================================================
# MINDPULSE - STUDENT FEATURE ANALYSIS
# ==========================================================

print("=" * 70)
print("MINDPULSE - STUDENT FEATURE IMPORTANCE")
print("=" * 70)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv(
    "data/raw/student_stress.csv"
)

# ----------------------------------------------------------
# Remove direct symptom variables
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
# Train Logistic Regression
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

model.fit(X_train, y_train)


# ----------------------------------------------------------
# Extract Coefficients
# ----------------------------------------------------------

classifier = model.named_steps["classifier"]

coefficients = classifier.coef_

classes = classifier.classes_

# Average absolute coefficient across classes
importance = abs(coefficients).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ----------------------------------------------------------
# Display
# ----------------------------------------------------------

print("\nFeature Importance Ranking")
print("-" * 50)

print(
    feature_importance.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)


# ----------------------------------------------------------
# Coefficients by Class
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("Coefficients by Stress Class")
print("=" * 70)

coefficient_table = pd.DataFrame(
    coefficients.T,
    index=X.columns,
    columns=[
        f"Class_{c}"
        for c in classes
    ]
)

coefficient_table["Average_Absolute"] = (
    coefficient_table.abs().mean(axis=1)
)

coefficient_table = coefficient_table.sort_values(
    by="Average_Absolute",
    ascending=False
)

print(
    coefficient_table.round(4)
)