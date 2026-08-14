import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ============================================================
# MINDPULSE - GENERAL USER MODEL EVALUATION
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER MODEL EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. FEATURES AND TARGET
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
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


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
# 5. MODELS
# ------------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(f"MODEL: {model_name}")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

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

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nPerformance")
    print("----------------------------------------")

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")

    print("\nConfusion Matrix")
    print("----------------------------------------")
    print(cm)

    print("\nClassification Report")
    print("----------------------------------------")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Balanced Accuracy": balanced_accuracy
    })


# ------------------------------------------------------------
# 7. MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GENERAL USER MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 8. BEST MODEL
# ------------------------------------------------------------

best_model = results_df.sort_values(
    by="Balanced Accuracy",
    ascending=False
).iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL BASED ON BALANCED ACCURACY")
print("=" * 70)

print(f"Model            : {best_model['Model']}")
print(f"Accuracy         : {best_model['Accuracy']:.4f}")
print(f"F1 Score         : {best_model['F1 Score']:.4f}")
print(f"ROC-AUC          : {best_model['ROC-AUC']:.4f}")
print(f"PR-AUC           : {best_model['PR-AUC']:.4f}")
print(
    f"Balanced Accuracy: "
    f"{best_model['Balanced Accuracy']:.4f}"
)


print("\n" + "=" * 70)
print("GENERAL USER MODEL EVALUATION COMPLETED")
print("=" * 70)