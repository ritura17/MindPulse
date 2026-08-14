import pandas as pd

# ==========================================================
# MINDPULSE - DATASET INSPECTION
# ==========================================================

# ----------------------------------------------------------
# Load datasets
# ----------------------------------------------------------

student_df = pd.read_csv("data/raw/student_stress.csv")
mental_df = pd.read_csv("data/raw/mental_health_lifestyle.csv")


# ----------------------------------------------------------
# Student Stress Dataset
# ----------------------------------------------------------

print("=" * 60)
print("STUDENT STRESS DATASET")
print("=" * 60)

print("\nShape:")
print(student_df.shape)

print("\nColumns:")
print(student_df.columns.tolist())

print("\nFirst 5 rows:")
print(student_df.head())

print("\nData Types:")
print(student_df.dtypes)

print("\nMissing Values:")
print(student_df.isnull().sum())


# ----------------------------------------------------------
# Mental Health & Lifestyle Dataset
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("MENTAL HEALTH & LIFESTYLE DATASET")
print("=" * 60)

print("\nShape:")
print(mental_df.shape)

print("\nColumns:")
print(mental_df.columns.tolist())

print("\nFirst 5 rows:")
print(mental_df.head())

print("\nData Types:")
print(mental_df.dtypes)

print("\nMissing Values:")
print(mental_df.isnull().sum())