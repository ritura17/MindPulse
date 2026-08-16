# ============================================================
# MINDPULSE - GENERAL USER SELECTED FEATURE MODEL
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER SELECTED FEATURE MODEL")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "Has_Mental_Health_Issue"


print("\nTarget Distribution:")
print(df[TARGET].value_counts())

print("\nTarget Percentage:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# FEATURE GROUPS
# ============================================================

ALL_FEATURES = [
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


TOP_6_FEATURES = [
    "Work_Stress_Level",
    "Social_Support",
    "Financial_Stress",
    "Loneliness",
    "Job_Satisfaction",
    "Work_Hours_Per_Week"
]


TOP_4_FEATURES = [
    "Work_Stress_Level",
    "Social_Support",
    "Financial_Stress",
    "Loneliness"
]


# ============================================================
# CHECK FEATURES
# ============================================================

for feature in ALL_FEATURES:
    if feature not in df.columns:
        raise ValueError(f"Feature not found in dataset: {feature}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

y = df[TARGET]

X_train_all, X_test_all, y_train, y_test = train_test_split(
    df[ALL_FEATURES],
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train_all.shape)

print("\nTesting Data Shape:")
print(X_test_all.shape)


# ============================================================
# FUNCTION FOR MODEL TRAINING
# ============================================================

def train_and_evaluate(
    feature_list,
    model_name,
    model
):

    print("\n" + "-" * 60)
    print(f"FEATURE SET: {model_name}")
    print("-" * 60)

    print("\nFeatures:")
    print(feature_list)

    X_train = X_train_all[feature_list]
    X_test = X_test_all[feature_list]

    # --------------------------------------------------------
    # Identify numerical and categorical features
    # --------------------------------------------------------

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    if hasattr(pipeline, "predict_proba"):

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

    else:

        roc_auc = 0.0

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nModel Performance")
    print("----------------------------------------")

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report")
    print("----------------------------------------")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("Confusion Matrix")
    print("----------------------------------------")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    return {
        "Feature Set": model_name,
        "Model": model.__class__.__name__,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Balanced Accuracy": balanced_acc,
        "Pipeline": pipeline
    }


# ============================================================
# MODELS
# ============================================================

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
    )
}


# ============================================================
# TRAIN ALL COMBINATIONS
# ============================================================

results = []


feature_sets = {

    "All 18 Features": ALL_FEATURES,

    "Top 6 Features": TOP_6_FEATURES,

    "Top 4 Features": TOP_4_FEATURES
}


for feature_set_name, feature_list in feature_sets.items():

    for model_name, model in models.items():

        result = train_and_evaluate(
            feature_list,
            f"{feature_set_name} - {model_name}",
            model
        )

        results.append(result)


# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

comparison = results_df[
    [
        "Feature Set",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Balanced Accuracy"
    ]
].copy()


print("\n")
print("=" * 70)
print("MINDPULSE - SELECTED FEATURE MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================
#
# For this highly imbalanced dataset, use ROC-AUC and
# balanced accuracy rather than ordinary accuracy alone.
# ============================================================

best_index = results_df[
    "Balanced Accuracy"
].idxmax()

best_result = results_df.loc[
    best_index
]

best_pipeline = best_result["Pipeline"]


print("\n")
print("=" * 70)
print("BEST SELECTED FEATURE MODEL")
print("=" * 70)

print(
    f"Feature Set      : {best_result['Feature Set']}"
)

print(
    f"Model            : {best_result['Model']}"
)

print(
    f"Accuracy         : {best_result['Accuracy']:.4f}"
)

print(
    f"F1 Score         : {best_result['F1 Score']:.4f}"
)

print(
    f"ROC-AUC          : {best_result['ROC-AUC']:.4f}"
)

print(
    f"Balanced Accuracy: {best_result['Balanced Accuracy']:.4f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "general_user_selected_model.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "general_user_selected_features.pkl"
)


joblib.dump(
    best_pipeline,
    MODEL_PATH
)


# Save the actual features used
best_feature_set_name = best_result[
    "Feature Set"
]

if "Top 4" in best_feature_set_name:

    selected_features = TOP_4_FEATURES

elif "Top 6" in best_feature_set_name:

    selected_features = TOP_6_FEATURES

else:

    selected_features = ALL_FEATURES


joblib.dump(
    selected_features,
    FEATURE_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("SELECTED GENERAL USER MODEL SAVED")
print("=" * 70)

print(
    f"Model Location   : {MODEL_PATH}"
)

print(
    f"Features Location : {FEATURE_PATH}"
)

print("\nFeatures Saved:")

for feature in selected_features:
    print(f"- {feature}")

print("\n")
print("=" * 70)
print("GENERAL USER SELECTED MODEL TRAINING COMPLETED")
print("=" * 70)