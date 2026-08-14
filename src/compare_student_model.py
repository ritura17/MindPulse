import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# MINDPULSE - STUDENT MODEL COMPARISON
# ==========================================================

print("=" * 70)
print("MINDPULSE - STUDENT STRESS MODEL COMPARISON")
print("=" * 70)


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

data_path = "data/raw/student_stress.csv"

df = pd.read_csv(data_path)


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


print("\nExcluded Features:")
print(excluded_features)

print("\nNumber of Features:")
print(X.shape[1])


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


# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000)
            )
        ]
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ==========================================================
# TRAIN AND EVALUATE
# ==========================================================

results = []

for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {name}...")

    model.fit(X_train, y_train)

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Training completed.")


# ==========================================================
# RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# ==========================================================
# BEST MODEL
# ==========================================================

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model    : {best_model['Model']}"
)

print(
    f"Accuracy : {best_model['Accuracy']:.4f}"
)

print(
    f"F1 Score : {best_model['F1 Score']:.4f}"
)