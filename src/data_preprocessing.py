import pandas as pd

import pandas as pd

# ==========================================================
# MINDPULSE - DATA PREPROCESSING
# ==========================================================

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

input_path = "data/raw/mental_health_lifestyle.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("MINDPULSE DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)


# ----------------------------------------------------------
# Select features and target
# ----------------------------------------------------------

selected_columns = [
    "Age",
    "Gender",
    "Exercise Level",
    "Diet Type",
    "Sleep Hours",
    "Work Hours per Week",
    "Screen Time per Day (Hours)",
    "Social Interaction Score",
    "Happiness Score",
    "Stress Level"
]

df = df[selected_columns].copy()


# ----------------------------------------------------------
# Check missing values
# ----------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ----------------------------------------------------------
# Remove rows with missing values
# ----------------------------------------------------------

df = df.dropna()

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ----------------------------------------------------------
# Save processed dataset
# ----------------------------------------------------------

output_path = "data/processed/mental_health_lifestyle_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nProcessed Dataset Saved:")
print(output_path)


# ----------------------------------------------------------
# Final preview
# ----------------------------------------------------------

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())