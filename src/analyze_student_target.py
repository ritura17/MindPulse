import pandas as pd

# ==========================================================
# MINDPULSE - STUDENT STRESS TARGET ANALYSIS
# ==========================================================

df = pd.read_csv(
    "data/raw/student_stress.csv"
)

print("=" * 70)
print("MINDPULSE - STUDENT STRESS TARGET ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------
# Stress distribution
# ----------------------------------------------------------

print("\nStress Level Distribution:")
print(
    df["stress_level"].value_counts().sort_index()
)

# ----------------------------------------------------------
# Numerical feature relationships
# ----------------------------------------------------------

features = [
    "anxiety_level",
    "self_esteem",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying"
]

# ----------------------------------------------------------
# Mean values by stress level
# ----------------------------------------------------------

print("\nAverage Feature Values by Stress Level:")
print(
    df.groupby("stress_level")[features]
    .mean()
    .round(2)
)

# ----------------------------------------------------------
# Correlation with stress
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("Correlation with Stress Level")
print("=" * 70)

correlations = (
    df[features + ["stress_level"]]
    .corr()["stress_level"]
    .drop("stress_level")
    .sort_values()
)

print(
    correlations.round(3)
)

# ----------------------------------------------------------
# Top relationships
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("Absolute Correlation Ranking")
print("=" * 70)

absolute_correlations = correlations.abs().sort_values(
    ascending=False
)

print(
    absolute_correlations.round(3)
)