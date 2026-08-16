import pandas as pd
import numpy as np

# ============================================================
# MINDPULSE - GENERAL USER FEATURE ANALYSIS
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER FEATURE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

DATA_PATH = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

df = pd.read_csv(DATA_PATH)

target = "Has_Mental_Health_Issue"

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

# ------------------------------------------------------------
# 3. TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df[target].value_counts())

print("\nTarget Percentage:")
print(
    df[target]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 4. NUMERICAL FEATURES
# ------------------------------------------------------------

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
# 5. CLASS-WISE MEAN VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE NUMERICAL VALUES BY MENTAL HEALTH STATUS")
print("=" * 70)

mean_values = df.groupby(target)[numerical_features].mean()

print(
    mean_values.round(2).to_string()
)

# ------------------------------------------------------------
# 6. DIFFERENCE BETWEEN CLASSES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ABSOLUTE DIFFERENCE BETWEEN CLASS MEANS")
print("=" * 70)

class_0_mean = df[df[target] == 0][numerical_features].mean()
class_1_mean = df[df[target] == 1][numerical_features].mean()

difference = (
    class_1_mean - class_0_mean
).abs().sort_values(ascending=False)

print(
    difference.round(3).to_string()
)

# ------------------------------------------------------------
# 7. CORRELATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH MENTAL HEALTH TARGET")
print("=" * 70)

correlation = (
    df[numerical_features + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(
    correlation.round(4).to_string()
)

# ------------------------------------------------------------
# 8. ABSOLUTE CORRELATION RANKING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ABSOLUTE CORRELATION RANKING")
print("=" * 70)

absolute_correlation = (
    correlation.abs()
    .sort_values(ascending=False)
)

print(
    absolute_correlation.round(4).to_string()
)

# ------------------------------------------------------------
# 9. CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = [
    "Gender",
    "Income_Level",
    "Employment_Status",
    "Exercise_Per_Week"
]

for feature in categorical_features:

    print("\n" + "=" * 70)
    print(f"{feature.upper()} VS MENTAL HEALTH")
    print("=" * 70)

    percentage_table = pd.crosstab(
        df[feature],
        df[target],
        normalize="index"
    ) * 100

    percentage_table.columns = [
        "No Issue (%)",
        "Mental Health Issue (%)"
    ]

    print(
        percentage_table.round(2).to_string()
    )

# ------------------------------------------------------------
# 10. BEHAVIORAL RISK GROUPS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEHAVIORAL RISK GROUP ANALYSIS")
print("=" * 70)

# High work stress
high_work_stress = df["Work_Stress_Level"] >= 7

print("\nHigh Work Stress (>= 7):")

print(
    df.loc[
        high_work_stress,
        target
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# High financial stress
high_financial_stress = df["Financial_Stress"] >= 7

print("\nHigh Financial Stress (>= 7):")

print(
    df.loc[
        high_financial_stress,
        target
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# High loneliness
high_loneliness = df["Loneliness"] >= 7

print("\nHigh Loneliness (>= 7):")

print(
    df.loc[
        high_loneliness,
        target
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Low social support
low_social_support = df["Social_Support"] <= 3

print("\nLow Social Support (<= 3):")

print(
    df.loc[
        low_social_support,
        target
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# High social media usage
high_social_media = df["Social_Media_Hours_Day"] >= 7

print("\nHigh Social Media Usage (>= 7 hours):")

print(
    df.loc[
        high_social_media,
        target
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Low sleep
low_sleep = df["Sleep_Hours_Night"] < 6

print("\nLow Sleep (< 6 hours):")

print(
    df.loc[
        low_sleep,
        target
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 11. COMBINED RISK SCORE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MINDPULSE BEHAVIORAL RISK SCORE")
print("=" * 70)

df["Behavioral_Risk_Score"] = (
    (df["Work_Stress_Level"] >= 7).astype(int)
    + (df["Financial_Stress"] >= 7).astype(int)
    + (df["Loneliness"] >= 7).astype(int)
    + (df["Social_Support"] <= 3).astype(int)
    + (df["Social_Media_Hours_Day"] >= 7).astype(int)
    + (df["Sleep_Hours_Night"] < 6).astype(int)
)

print("\nRisk Score Distribution:")

print(
    df["Behavioral_Risk_Score"]
    .value_counts()
    .sort_index()
)

print("\nMental Health Rate by Risk Score:")

risk_analysis = (
    df.groupby("Behavioral_Risk_Score")[target]
    .agg(
        Count="count",
        Mental_Health_Rate="mean"
    )
)

risk_analysis["Mental_Health_Rate"] = (
    risk_analysis["Mental_Health_Rate"] * 100
)

print(
    risk_analysis.round(2).to_string()
)

# ------------------------------------------------------------
# 12. FEATURE IMPORTANCE SCREENING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SCREENING")
print("=" * 70)

feature_screening = pd.DataFrame({
    "Feature": absolute_correlation.index,
    "Correlation": correlation[
        absolute_correlation.index
    ].values,
    "Absolute_Correlation": absolute_correlation.values
})

print(
    feature_screening.round(4).to_string(index=False)
)

# ------------------------------------------------------------
# 13. RECOMMENDED FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RECOMMENDED FEATURES FOR NEXT MODEL")
print("=" * 70)

recommended = absolute_correlation[
    absolute_correlation >= 0.03
].index.tolist()

print("\nFeatures with absolute correlation >= 0.03:")

for feature in recommended:
    print(f"- {feature}")

# ------------------------------------------------------------
# 14. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERAL USER FEATURE ANALYSIS COMPLETED")
print("=" * 70)