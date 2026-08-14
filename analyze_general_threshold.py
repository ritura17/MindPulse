import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# MINDPULSE - GENERAL USER THRESHOLD ANALYSIS
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER THRESHOLD ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. FEATURES
# ------------------------------------------------------------

FEATURES = [
    "Age",
    "Gender",
    "Income_Level",
    "Employment_Status",
    "Work_Hours_Per_Week",
    "Job_Satisfaction",
    "Work_Stress_Level",
    "Work_Life_Balance",
    "Exercise_Per_Week",
    "Sleep_Hours_Night",
    "Screen_Time_Hours_Day",
    "Social_Media_Hours_Day",
    "Hobby_Time_Hours_Week",
    "Financial_Stress",
    "Social_Support",
    "Close_Friends_Count",
    "Feel_Understood",
    "Loneliness"
]

TARGET = "Has_Mental_Health_Issue"


X = df[FEATURES]
y = df[TARGET]


# ------------------------------------------------------------
# 3. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------

numerical_features = [
    "Age",
    "Work_Hours_Per_Week",
    "Job_Satisfaction",
    "Work_Stress_Level",
    "Work_Life_Balance",
    "Sleep_Hours_Night",
    "Screen_Time_Hours_Day",
    "Social_Media_Hours_Day",
    "Hobby_Time_Hours_Week",
    "Financial_Stress",
    "Social_Support",
    "Close_Friends_Count",
    "Feel_Understood",
    "Loneliness"
]

categorical_features = [
    "Gender",
    "Income_Level",
    "Employment_Status",
    "Exercise_Per_Week"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 5. LOGISTIC REGRESSION
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# ------------------------------------------------------------
# 6. PROBABILITIES
# ------------------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]


print("\nROC-AUC:")
print(round(roc_auc_score(y_test, probabilities), 4))

print("\nPR-AUC:")
print(round(average_precision_score(y_test, probabilities), 4))


# ------------------------------------------------------------
# 7. THRESHOLD ANALYSIS
# ------------------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []


print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)


for threshold in thresholds:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced Accuracy": balanced_accuracy
    })


results_df = pd.DataFrame(results)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 8. BEST THRESHOLD
# ------------------------------------------------------------

best = results_df.loc[
    results_df["Balanced Accuracy"].idxmax()
]


print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Threshold          : "
    f"{best['Threshold']:.2f}"
)

print(
    f"Accuracy           : "
    f"{best['Accuracy']:.4f}"
)

print(
    f"Precision          : "
    f"{best['Precision']:.4f}"
)

print(
    f"Recall             : "
    f"{best['Recall']:.4f}"
)

print(
    f"F1 Score           : "
    f"{best['F1']:.4f}"
)

print(
    f"Balanced Accuracy  : "
    f"{best['Balanced Accuracy']:.4f}"
)


# ------------------------------------------------------------
# 9. CONFUSION MATRIX
# ------------------------------------------------------------

best_threshold = best["Threshold"]

best_predictions = (
    probabilities >= best_threshold
).astype(int)


print("\n" + "=" * 70)
print("CONFUSION MATRIX AT BEST THRESHOLD")
print("=" * 70)

print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)


# ------------------------------------------------------------
# 10. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)