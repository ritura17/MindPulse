# ============================================================
# MINDPULSE - GENERAL USER SMOTE MODEL TRAINING
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER SMOTE MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

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
# FEATURES
# ============================================================

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


print("\nFeatures Used:")
print(FEATURES)

print("\nNumber of Features:")
print(len(FEATURES))


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print("\nERROR - Missing Features:")
    print(missing_features)

    raise ValueError(
        "Some required features are missing from the dataset."
    )


# ============================================================
# X AND Y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

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


print("\nTraining Target Distribution BEFORE SMOTE:")
print(y_train.value_counts())


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# PREPROCESSOR
# ============================================================

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


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model_name, model):

    print("\n")
    print("-" * 60)
    print(f"Training {model_name}...")
    print("-" * 60)

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=5
                )
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

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

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

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nModel Performance")
    print("----------------------------------------")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1 Score          : {f1:.4f}"
    )

    print(
        f"ROC-AUC           : {roc_auc:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

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
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Balanced Accuracy": balanced_accuracy,
        "Pipeline": pipeline
    }


# ============================================================
# TRAIN ALL MODELS
# ============================================================

results = []

for model_name, model in models.items():

    result = train_model(
        model_name,
        model
    )

    results.append(result)


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)


comparison = results_df[
    [
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
print("SMOTE GENERAL USER MODEL COMPARISON")
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
# We prioritize ROC-AUC because this dataset is heavily
# imbalanced. Balanced Accuracy is also considered.
# ============================================================

best_index = results_df[
    "ROC-AUC"
].idxmax()

best_result = results_df.loc[
    best_index
]

best_pipeline = best_result[
    "Pipeline"
]


# ============================================================
# BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("BEST SMOTE GENERAL USER MODEL")
print("=" * 70)

print(
    f"Model            : {best_result['Model']}"
)

print(
    f"Accuracy         : {best_result['Accuracy']:.4f}"
)

print(
    f"Precision        : {best_result['Precision']:.4f}"
)

print(
    f"Recall           : {best_result['Recall']:.4f}"
)

print(
    f"F1 Score         : {best_result['F1 Score']:.4f}"
)

print(
    f"ROC-AUC          : {best_result['ROC-AUC']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_result['Balanced Accuracy']:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "general_user_smote_model.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "general_user_smote_features.pkl"
)


joblib.dump(
    best_pipeline,
    MODEL_PATH
)


joblib.dump(
    FEATURES,
    FEATURE_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("SMOTE GENERAL USER MODEL SAVED")
print("=" * 70)

print(
    f"Model Location   : {MODEL_PATH}"
)

print(
    f"Features Location: {FEATURE_PATH}"
)

print("\nFeatures Saved:")

for feature in FEATURES:
    print(f"- {feature}")


print("\n")
print("=" * 70)
print("GENERAL USER SMOTE TRAINING COMPLETED")
print("=" * 70)