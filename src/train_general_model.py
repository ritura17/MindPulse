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
# MINDPULSE - GENERAL USER MODEL TRAINING
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER MODEL TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

file_path = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. Behavioral / Lifestyle Features
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

print("\nFeatures Used:")
print(features)

print("\nTarget:")
print(target)

# ------------------------------------------------------------
# 3. Prepare X and y
# ------------------------------------------------------------

X = df[features].copy()
y = df[target].copy()

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 4. Identify Feature Types
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
# 5. Preprocessing
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
                drop="first"
            ),
            categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 6. Train-Test Split
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
# 7. Models
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=6
    ),

    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=42,
        max_depth=10
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ------------------------------------------------------------
# 8. Train and Evaluate Models
# ------------------------------------------------------------

results = []

best_model = None
best_model_name = None
best_f1 = 0

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {model_name}...")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_probability = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probability)
    else:
        roc_auc = 0

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

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

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

    # Select best model using F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = model_name

# ------------------------------------------------------------
# 9. Model Comparison
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GENERAL USER MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)

# ------------------------------------------------------------
# 10. Best Model
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST GENERAL USER MODEL")
print("=" * 70)

print(f"Model    : {best_model_name}")
print(f"F1 Score : {best_f1:.4f}")

# ------------------------------------------------------------
# 11. Save Best Model
# ------------------------------------------------------------

model_path = "models/general_user_model.pkl"

joblib.dump(best_model, model_path)

print("\nModel Saved Successfully!")
print(f"Location: {model_path}")

# ------------------------------------------------------------
# 12. Save Feature Information
# ------------------------------------------------------------

feature_info = {
    "features": features,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "target": target
}

joblib.dump(
    feature_info,
    "models/general_user_features.pkl"
)

print("Feature Information Saved!")
print("Location: models/general_user_features.pkl")

# ------------------------------------------------------------
# 13. Completion
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERAL USER MODEL TRAINING COMPLETED")
print("=" * 70)