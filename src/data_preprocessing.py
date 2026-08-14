import pandas as pd

# ==========================================================
# MINDPULSE - DATA PREPROCESSING
# ==========================================================

# Load dataset
mental_df = pd.read_csv(
    "data/raw/mental_health_lifestyle.csv"
)

print("=" * 60)
print("MENTAL HEALTH & LIFESTYLE DATA")
print("=" * 60)

# ----------------------------------------------------------
# Missing values
# ----------------------------------------------------------

print("\nMissing Values:")
print(mental_df.isnull().sum())

# ----------------------------------------------------------
# Categorical columns
# ----------------------------------------------------------

categorical_columns = [
    "Country",
    "Gender",
    "Exercise Level",
    "Diet Type",
    "Stress Level",
    "Mental Health Condition"
]

print("\nCategorical Columns:")
print(categorical_columns)

# ----------------------------------------------------------
# Category counts
# ----------------------------------------------------------

for column in categorical_columns:
    print("\n" + "-" * 50)
    print(column)
    print("-" * 50)

    print(mental_df[column].value_counts(dropna=False))