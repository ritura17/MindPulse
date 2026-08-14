import pandas as pd

# ==========================================================
# MINDPULSE - GENERAL USER FEATURE ANALYSIS
# ==========================================================

print("=" * 70)
print("MINDPULSE - GENERAL USER FEATURE ANALYSIS")
print("=" * 70)


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv(
    "data/raw/mental_health_lifestyle.csv"
)


# ----------------------------------------------------------
# Remove unnecessary target-related information
# ----------------------------------------------------------

target = "Stress Level"


features = [
    "Age",
    "Gender",
    "Exercise Level",
    "Diet Type",
    "Sleep Hours",
    "Work Hours per Week",
    "Screen Time per Day (Hours)",
    "Social Interaction Score",
    "Happiness Score"
]


# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFeatures Used:")
print(features)

print("\nTarget:")
print(target)


# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

print("\nMissing Values:")
print(df[features + [target]].isnull().sum())


# ----------------------------------------------------------
# Numerical Features
# ----------------------------------------------------------

numerical_features = [
    "Age",
    "Sleep Hours",
    "Work Hours per Week",
    "Screen Time per Day (Hours)",
    "Social Interaction Score",
    "Happiness Score"
]


# ----------------------------------------------------------
# Average Values by Stress Level
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE NUMERICAL VALUES BY STRESS LEVEL")
print("=" * 70)

print(
    df.groupby(target)[numerical_features]
    .mean()
    .round(2)
)


# ----------------------------------------------------------
# Standard Deviation by Stress Level
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("STANDARD DEVIATION BY STRESS LEVEL")
print("=" * 70)

print(
    df.groupby(target)[numerical_features]
    .std()
    .round(2)
)


# ----------------------------------------------------------
# Minimum Values
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("MINIMUM VALUES BY STRESS LEVEL")
print("=" * 70)

print(
    df.groupby(target)[numerical_features]
    .min()
)


# ----------------------------------------------------------
# Maximum Values
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("MAXIMUM VALUES BY STRESS LEVEL")
print("=" * 70)

print(
    df.groupby(target)[numerical_features]
    .max()
)


# ----------------------------------------------------------
# Categorical Features
# ----------------------------------------------------------

categorical_features = [
    "Gender",
    "Exercise Level",
    "Diet Type"
]


for feature in categorical_features:

    print("\n" + "=" * 70)
    print(f"{feature.upper()} VS STRESS LEVEL")
    print("=" * 70)

    table = pd.crosstab(
        df[feature],
        df[target],
        normalize="index"
    ) * 100

    print(
        table.round(2)
    )


# ----------------------------------------------------------
# Correlation
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH NUMERICAL FEATURES")
print("=" * 70)


stress_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}


df["Stress_Numeric"] = (
    df[target]
    .map(stress_mapping)
)


correlation = (
    df[numerical_features + ["Stress_Numeric"]]
    .corr()["Stress_Numeric"]
    .drop("Stress_Numeric")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print(
    correlation.round(3)
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)