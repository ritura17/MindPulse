# ==============================================================
# MINDPULSE - GENERAL USER HYPERPARAMETER TUNING
# ==============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)


# ==============================================================
# 1. PATHS
# ==============================================================

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================
# 2. LOAD DATA
# ==============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER HYPERPARAMETER TUNING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ==============================================================
# 3. FEATURES
# ==============================================================

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


# ==============================================================
# 4. TARGET DISTRIBUTION
# ==============================================================

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print(y.value_counts(normalize=True).mul(100).round(2))


# ==============================================================
# 5. TRAIN TEST SPLIT
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
# 6. FEATURE TYPES
# ==============================================================

NUMERICAL_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "Gender",
    "Income_Level",
    "Employment_Status",
    "Exercise_Per_Week"
]


# ==============================================================
# 7. PREPROCESSOR
# ==============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERICAL_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        )
    ]
)


# ==============================================================
# 8. CROSS VALIDATION
# ==============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==============================================================
# 9. MODEL CONFIGURATIONS
# ==============================================================

models_and_grids = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000
        ),
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [
                "balanced",
                None
            ]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [
                "balanced",
                "balanced_subsample"
            ]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            random_state=42
        ),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3]
        }
    )
}


# ==============================================================
# 10. TUNING
# ==============================================================

results = []

best_model = None
best_model_name = None
best_score = -1


for model_name, (model, param_grid) in models_and_grids.items():

    print("\n" + "=" * 70)
    print(f"TUNING {model_name.upper()}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest CV Balanced Accuracy:")
    print(round(search.best_score_, 4))


    # ==========================================================
    # TEST PERFORMANCE
    # ==========================================================

    y_pred = search.predict(X_test)

    if hasattr(search, "predict_proba"):
        y_probability = search.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )
    else:
        roc_auc = 0.0


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


    print("\nTest Performance")
    print("-" * 40)
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )


    results.append({
        "Model": model_name,
        "CV Balanced Accuracy": search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Balanced Accuracy": balanced_accuracy,
        "Best Parameters": str(search.best_params_)
    })


    # ==========================================================
    # SELECT BEST MODEL
    # ==========================================================

    if balanced_accuracy > best_score:

        best_score = balanced_accuracy
        best_model = search.best_estimator_
        best_model_name = model_name


# ==============================================================
# 11. RESULTS
# ==============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GENERAL USER HYPERPARAMETER TUNING COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "Model",
            "CV Balanced Accuracy",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Balanced Accuracy"
        ]
    ].to_string(index=False)
)


# ==============================================================
# 12. BEST MODEL
# ==============================================================

print("\n" + "=" * 70)
print("BEST TUNED GENERAL USER MODEL")
print("=" * 70)

print(f"Model             : {best_model_name}")
print(f"Balanced Accuracy : {best_score:.4f}")


# ==============================================================
# 13. SAVE MODEL
# ==============================================================

model_path = os.path.join(
    MODEL_DIR,
    "general_user_tuned_model.pkl"
)

features_path = os.path.join(
    MODEL_DIR,
    "general_user_tuned_features.pkl"
)

results_path = os.path.join(
    MODEL_DIR,
    "general_user_tuning_results.csv"
)


joblib.dump(
    best_model,
    model_path
)

joblib.dump(
    FEATURES,
    features_path
)

results_df.to_csv(
    results_path,
    index=False
)


# ==============================================================
# 14. FINAL INFORMATION
# ==============================================================

print("\n" + "=" * 70)
print("GENERAL USER TUNED MODEL SAVED")
print("=" * 70)

print(f"Model Location    : {model_path}")
print(f"Features Location : {features_path}")
print(f"Results Location  : {results_path}")

print("\nFeatures Saved:")

for feature in FEATURES:
    print(f"- {feature}")

print("\n" + "=" * 70)
print("GENERAL USER HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)