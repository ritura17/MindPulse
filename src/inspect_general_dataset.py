import pandas as pd

print("=" * 70)
print("MINDPULSE - GENERAL USER DATASET INSPECTION")
print("=" * 70)

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

file_path = "data/raw/mental_health_risk_dataset.csv"

df = pd.read_csv(file_path)

# ----------------------------------------------------------
# Basic information
# ----------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ----------------------------------------------------------
# First 5 rows
# ----------------------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------------------------------------
# Data types
# ----------------------------------------------------------

print("\nData Types:")
print(df.dtypes)

# ----------------------------------------------------------
# Missing values
# ----------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------------------------------
# Duplicate rows
# ----------------------------------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ----------------------------------------------------------
# Unique values
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"\n{column}:")
    
    unique_values = df[column].unique()
    
    if len(unique_values) <= 20:
        print(unique_values)
    else:
        print(f"{len(unique_values)} unique values")

# ----------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe(include="all").T)

# ----------------------------------------------------------
# Save inspection-friendly copy
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)