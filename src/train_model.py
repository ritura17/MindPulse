import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ==========================================================
# MINDPULSE - MODEL TRAINING
# ==========================================================

# ----------------------------------------------------------
# Load cleaned dataset
# ----------------------------------------------------------

data_path = "data/processed/mental_health_lifestyle_cleaned.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("MINDPULSE - MODEL TRAINING")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)


# ----------------------------------------------------------
# Separate Features and Target
# ----------------------------------------------------------

X = df.drop("Stress Level", axis=1)

y = df["Stress Level"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nTarget Distribution:")
print(y.value_counts())


# ----------------------------------------------------------
# Identify column types
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
# Create Model Pipeline
# ----------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
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

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)