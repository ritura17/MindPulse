import pandas as pd

# ==========================================================
# MINDPULSE - STUDENT FEATURE VALUE INSPECTION
# ==========================================================

df = pd.read_csv(
    "data/raw/student_stress.csv"
)

print("=" * 70)
print("MINDPULSE - STUDENT FEATURE VALUE INSPECTION")
print("=" * 70)

features = [
    "self_esteem",
    "mental_health_history",
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

for feature in features:

    print("\n" + "-" * 60)
    print(feature)

    print("Unique values:")
    print(sorted(df[feature].unique()))

    print("Value counts:")
    print(
        df[feature]
        .value_counts()
        .sort_index()
        .to_string()
    )