import pandas as pd

# ==========================================================
# MINDPULSE - EXPLORATORY DATA ANALYSIS
# ==========================================================

student_df = pd.read_csv("data/raw/student_stress.csv")
mental_df = pd.read_csv("data/raw/mental_health_lifestyle.csv")


# ==========================================================
# STUDENT STRESS DATASET
# ==========================================================

print("=" * 70)
print("STUDENT STRESS DATASET - VALUE ANALYSIS")
print("=" * 70)

print("\nUnique values in each column:")

for column in student_df.columns:
    print(f"\n{column}:")
    print(student_df[column].unique())

print("\nStress Level Distribution:")
print(student_df["stress_level"].value_counts().sort_index())

print("\nDescriptive Statistics:")
print(student_df.describe())


# ==========================================================
# MENTAL HEALTH & LIFESTYLE DATASET
# ==========================================================

print("\n" + "=" * 70)
print("MENTAL HEALTH & LIFESTYLE DATASET - VALUE ANALYSIS")
print("=" * 70)

print("\nUnique values in categorical columns:")

categorical_columns = [
    "Country",
    "Gender",
    "Exercise Level",
    "Diet Type",
    "Stress Level",
    "Mental Health Condition"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(mental_df[column].unique())

print("\nStress Level Distribution:")
print(mental_df["Stress Level"].value_counts())

print("\nDescriptive Statistics:")
print(mental_df.describe(include="all"))