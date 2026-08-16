# ============================================================
# MINDPULSE - GENERAL USER MODEL IMPROVEMENT
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

MODEL_PATH = "models/general_user_improved_model.pkl"
FEATURE_PATH = "models/general_user_improved_features.pkl"


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

TARGET = "Has_Mental_Health_Issue"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER MODEL IMPROVEMENT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# CHECK DATA
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


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


# ============================================================
# FEATURE TYPES
# ============================================================

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


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression - Balanced": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Random Forest - Balanced": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    )
}


# ============================================================
# TRAINING + EVALUATION
# ============================================================

results = []

best_model = None
best_model_name = None
best_balanced_accuracy = -1


for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {name}...")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nModel Performance")
    print("-" * 40)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")

    print("\nConfusion Matrix")
    print("-" * 40)
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print("-" * 40)
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Balanced Accuracy": balanced_acc
    })

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    if balanced_acc > best_balanced_accuracy:

        best_balanced_accuracy = balanced_acc
        best_model = pipeline
        best_model_name = name


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("IMPROVED GENERAL USER MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_row = results_df[
    results_df["Model"] == best_model_name
].iloc[0]

print("\n" + "=" * 70)
print("BEST IMPROVED GENERAL USER MODEL")
print("=" * 70)

print(f"Model             : {best_model_name}")
print(f"Accuracy          : {best_row['Accuracy']:.4f}")
print(f"Precision         : {best_row['Precision']:.4f}")
print(f"Recall            : {best_row['Recall']:.4f}")
print(f"F1 Score          : {best_row['F1 Score']:.4f}")
print(f"ROC-AUC           : {best_row['ROC-AUC']:.4f}")
print(f"PR-AUC            : {best_row['PR-AUC']:.4f}")
print(f"Balanced Accuracy : {best_row['Balanced Accuracy']:.4f}")


# ============================================================
# CROSS-VALIDATION OF BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring="balanced_accuracy",
    n_jobs=-1
)

print("\nBalanced Accuracy CV Scores:")
print(cv_scores.round(4))

print(f"\nMean CV Balanced Accuracy : {cv_scores.mean():.4f}")
print(f"Std CV Balanced Accuracy  : {cv_scores.std():.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    MODEL_PATH
)

joblib.dump(
    FEATURES,
    FEATURE_PATH
)

print("\n" + "=" * 70)
print("IMPROVED GENERAL USER MODEL SAVED")
print("=" * 70)

print(f"Model Location    : {MODEL_PATH}")
print(f"Features Location : {FEATURE_PATH}")

print("\nFeatures Saved:")

for feature in FEATURES:
    print(f"- {feature}")


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("GENERAL USER MODEL IMPROVEMENT COMPLETED")
print("=" * 70)