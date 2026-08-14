import pandas as pd

# ==========================================================
# MINDPULSE - TARGET RELATIONSHIP ANALYSIS
# ==========================================================

df = pd.read_csv(
    "data/processed/mental_health_lifestyle_cleaned.csv"
)

print("=" * 70)
print("MINDPULSE - STRESS TARGET ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------
# Stress distribution
# ----------------------------------------------------------

print("\nStress Level Distribution:")
print(df["Stress Level"].value_counts())

# ----------------------------------------------------------
# Numerical features
# ----------------------------------------------------------

numerical_features = [
    "Age",
    "Sleep Hours",
    "Work Hours per Week",
    "Screen Time per Day (Hours)",
    "Social Interaction Score",
    "Happiness Score"
]

print("\nAverage feature values by Stress Level:")
print(
    df.groupby("Stress Level")[numerical_features]
    .mean()
    .round(2)
)

# ----------------------------------------------------------
# Categorical features
# ----------------------------------------------------------

categorical_features = [
    "Gender",
    "Exercise Level",
    "Diet Type"
]

for column in categorical_features:

    print("\n" + "=" * 60)
    print(f"{column} vs Stress Level")
    print("=" * 60)

    table = pd.crosstab(
        df[column],
        df["Stress Level"],
        normalize="index"
    ) * 100

    print(table.round(2))

# ----------------------------------------------------------
# Numerical correlations
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Correlation with Stress Level")
print("=" * 60)

# Convert target temporarily to numbers
stress_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

df["Stress_Numeric"] = df["Stress Level"].map(
    stress_mapping
)

correlations = (
    df[numerical_features + ["Stress_Numeric"]]
    .corr()["Stress_Numeric"]
    .sort_values()
)

print(correlations)