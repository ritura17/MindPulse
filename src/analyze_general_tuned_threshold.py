# ==============================================================
# MINDPULSE - GENERAL USER TUNED MODEL THRESHOLD ANALYSIS
# ==============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ==============================================================
# 1. PATHS
# ==============================================================

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

MODEL_PATH = "models/general_user_tuned_model.pkl"

FEATURES_PATH = "models/general_user_tuned_features.pkl"


# ==============================================================
# 2. LOAD DATA
# ==============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER TUNED MODEL THRESHOLD ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ==============================================================
# 3. LOAD MODEL AND FEATURES
# ==============================================================

model = joblib.load(MODEL_PATH)

features = joblib.load(FEATURES_PATH)

TARGET = "Has_Mental_Health_Issue"

X = df[features]
y = df[TARGET]


# ==============================================================
# 4. TRAIN TEST SPLIT
# ==============================================================

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


# ==============================================================
# 5. CHECK WHETHER MODEL IS ALREADY TRAINED
# ==============================================================
# The tuned model was already fitted by GridSearchCV.
# Therefore, we only use it to predict probabilities.

print("\nUsing tuned model:")
print(type(model).__name__)


# ==============================================================
# 6. PREDICT PROBABILITIES
# ==============================================================

y_probability = model.predict_proba(X_test)[:, 1]


# ==============================================================
# 7. ROC-AUC AND PR-AUC
# ==============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\nROC-AUC:")
print(f"{roc_auc:.4f}")

print("\nPR-AUC:")
print(f"{pr_auc:.4f}")


# ==============================================================
# 8. THRESHOLD ANALYSIS
# ==============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
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
        "Threshold": round(threshold, 2),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Balanced Accuracy": round(
            balanced_accuracy,
            4
        )
    })


# ==============================================================
# 9. RESULTS TABLE
# ==============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("TUNED RANDOM FOREST THRESHOLD COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ==============================================================
# 10. FIND BEST THRESHOLD
# ==============================================================

best_index = results_df[
    "Balanced Accuracy"
].idxmax()

best_row = results_df.loc[best_index]

best_threshold = float(
    best_row["Threshold"]
)


# ==============================================================
# 11. BEST THRESHOLD
# ==============================================================

print("\n" + "=" * 70)
print("BEST THRESHOLD FOR TUNED MODEL")
print("=" * 70)

print(
    f"Threshold          : "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy           : "
    f"{best_row['Accuracy']:.4f}"
)

print(
    f"Precision          : "
    f"{best_row['Precision']:.4f}"
)

print(
    f"Recall             : "
    f"{best_row['Recall']:.4f}"
)

print(
    f"F1 Score           : "
    f"{best_row['F1']:.4f}"
)

print(
    f"Balanced Accuracy  : "
    f"{best_row['Balanced Accuracy']:.4f}"
)


# ==============================================================
# 12. CONFUSION MATRIX
# ==============================================================

best_predictions = (
    y_probability >= best_threshold
).astype(int)


cm = confusion_matrix(
    y_test,
    best_predictions
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX AT BEST THRESHOLD")
print("=" * 70)

print(cm)


# ==============================================================
# 13. CONFUSION MATRIX DETAILS
# ==============================================================

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives  :", tn)
print("False Positives :", fp)
print("False Negatives :", fn)
print("True Positives  :", tp)


# ==============================================================
# 14. SAVE THRESHOLD RESULTS
# ==============================================================

RESULTS_PATH = (
    "models/general_user_tuned_threshold_results.csv"
)

THRESHOLD_PATH = (
    "models/general_user_tuned_threshold.pkl"
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)

joblib.dump(
    best_threshold,
    THRESHOLD_PATH
)


# ==============================================================
# 15. FINAL INFORMATION
# ==============================================================

print("\n" + "=" * 70)
print("TUNED MODEL THRESHOLD ANALYSIS SAVED")
print("=" * 70)

print(
    f"Results Location   : "
    f"{RESULTS_PATH}"
)

print(
    f"Threshold Location : "
    f"{THRESHOLD_PATH}"
)

print("\n" + "=" * 70)
print("GENERAL USER TUNED THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)