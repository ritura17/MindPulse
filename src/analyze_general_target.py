import pandas as pd

# ============================================================
# MINDPULSE - GENERAL USER TARGET ANALYSIS
# ============================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER TARGET ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

file_path = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. Target Distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["Has_Mental_Health_Issue"].value_counts())
print("\nPercentage:")
print(
    df["Has_Mental_Health_Issue"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 3. Average Numerical Values by Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE NUMERICAL VALUES BY MENTAL HEALTH STATUS")
print("=" * 70)

numerical_features = [
    "Age",
    "Work_Hours_Per_Week",
    "Job_Satisfaction",
    "Work_Stress_Level",
    "Work_Life_Balance",
    "Sleep_Hours_Night",
    "Caffeine_Drinks_Day",
    "Screen_Time_Hours_Day",
    "Social_Media_Hours_Day",
    "Hobby_Time_Hours_Week",
    "Financial_Stress",
    "Social_Support",
    "Close_Friends_Count",
    "Feel_Understood",
    "Loneliness"
]

print(
    df.groupby("Has_Mental_Health_Issue")[numerical_features]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 4. Correlation with Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH MENTAL HEALTH TARGET")
print("=" * 70)

correlation_data = df[numerical_features + ["Has_Mental_Health_Issue"]]

correlations = (
    correlation_data
    .corr(numeric_only=True)["Has_Mental_Health_Issue"]
    .drop("Has_Mental_Health_Issue")
    .sort_values(key=abs, ascending=False)
)

print(correlations.round(3))

# ------------------------------------------------------------
# 5. Employment Status vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EMPLOYMENT STATUS VS MENTAL HEALTH")
print("=" * 70)

employment_table = pd.crosstab(
    df["Employment_Status"],
    df["Has_Mental_Health_Issue"],
    normalize="index"
) * 100

print(employment_table.round(2))

# ------------------------------------------------------------
# 6. Income Level vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INCOME LEVEL VS MENTAL HEALTH")
print("=" * 70)

income_table = pd.crosstab(
    df["Income_Level"],
    df["Has_Mental_Health_Issue"],
    normalize="index"
) * 100

print(income_table.round(2))

# ------------------------------------------------------------
# 7. Work Stress vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WORK STRESS LEVEL VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Work_Stress_Level"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 8. Work-Life Balance vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WORK-LIFE BALANCE VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Work_Life_Balance"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 9. Sleep vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SLEEP HOURS VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Sleep_Hours_Night"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 10. Screen Time vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SCREEN TIME VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Screen_Time_Hours_Day"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 11. Social Media Time vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SOCIAL MEDIA TIME VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Social_Media_Hours_Day"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 12. Exercise vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXERCISE VS MENTAL HEALTH")
print("=" * 70)

exercise_table = pd.crosstab(
    df["Exercise_Per_Week"],
    df["Has_Mental_Health_Issue"],
    normalize="index"
) * 100

print(exercise_table.round(2))

# ------------------------------------------------------------
# 13. Hobby Time vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HOBBY TIME VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Hobby_Time_Hours_Week"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 14. Financial Stress vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINANCIAL STRESS VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Financial_Stress"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 15. Social Support vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SOCIAL SUPPORT VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Social_Support"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 16. Loneliness vs Target
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LONELINESS VS MENTAL HEALTH")
print("=" * 70)

print(
    df.groupby("Has_Mental_Health_Issue")["Loneliness"]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 17. Important Features for MindPulse
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MINDPULSE BEHAVIORAL FEATURES")
print("=" * 70)

behavioral_features = [
    "Age",
    "Employment_Status",
    "Income_Level",
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

print("\nFeatures selected for further analysis:")
for feature in behavioral_features:
    print("-", feature)

# ------------------------------------------------------------
# 18. Completion
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERAL USER TARGET ANALYSIS COMPLETED")
print("=" * 70)