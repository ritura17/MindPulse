import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# MINDPULSE - BALANCED GENERAL USER MODEL TRAINING
# ============================================================

print("=" * 70)
print("MINDPULSE - BALANCED GENERAL USER MODEL TRAINING")
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

features = [
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

target = "Has_Mental_Health_Issue"

X = df[features]
y = df[target]

print("\nFeatures Used:")
print(features)

print("\nTarget:")
print(target)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print(y.value_counts(normalize=True).mul(100).round(2))

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

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ------------------------------------------------------------
# 4. CATEGORICAL / NUMERICAL FEATURES
# ------------------------------------------------------------

categorical_features = [
    "Gender",
    "Income_Level",
    "Employment_Status",
    "Exercise_Per_Week"
]

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

# ------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------

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
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 6. MODELS
# ------------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=8
    ),

    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=300,
        random_state=42,
        max_depth=12
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []

best_model = None
best_model_name = None
best_f1 = -1

# ------------------------------------------------------------
# 7. TRAIN MODELS
# ------------------------------------------------------------

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

    # Gradient Boosting does not have class_weight.
    # Use sample weights to balance classes.
    if name == "Gradient Boosting":

        class_counts = y_train.value_counts()

        weight_0 = len(y_train) / (
            2 * class_counts[0]
        )

        weight_1 = len(y_train) / (
            2 * class_counts[1]
        )

        sample_weights = y_train.map({
            0: weight_0,
            1: weight_1
        }).values

        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights
        )

    else:
        pipeline.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # METRICS
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
        y_probability
    )

    print("\nModel Performance")
    print("----------------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report")
    print("----------------------------------------")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("----------------------------------------")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    # F1 is preferred because the dataset is highly imbalanced.
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BALANCED GENERAL USER MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)

# ============================================================
# 9. BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST BALANCED GENERAL USER MODEL")
print("=" * 70)

print(f"Model    : {best_model_name}")
print(f"F1 Score : {best_f1:.4f}")

# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/general_user_balanced_model.pkl"
)

joblib.dump(
    features,
    "models/general_user_balanced_features.pkl"
)

print("\nModel Saved Successfully!")
print(
    "Location: models/general_user_balanced_model.pkl"
)

print("Feature Information Saved!")
print(
    "Location: models/general_user_balanced_features.pkl"
)

print("\n" + "=" * 70)
print("BALANCED GENERAL USER MODEL TRAINING COMPLETED")
print("=" * 70)