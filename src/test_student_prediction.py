import pandas as pd
import joblib

# ==========================================================
# MINDPULSE - TEST SAVED STUDENT MODEL
# ==========================================================

print("=" * 70)
print("MINDPULSE - TEST SAVED STUDENT MODEL")
print("=" * 70)


# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

model = joblib.load(
    "models/student_stress_model.pkl"
)

features = joblib.load(
    "models/student_features.pkl"
)

print("\nModel loaded successfully!")

print("\nFeatures expected by model:")
print(features)


# ----------------------------------------------------------
# Create Example User
# ----------------------------------------------------------

example_user = {
    "self_esteem": 15,
    "mental_health_history": 0,
    "blood_pressure": 2,
    "sleep_quality": 2,
    "breathing_problem": 2,
    "noise_level": 3,
    "living_conditions": 3,
    "safety": 3,
    "basic_needs": 3,
    "academic_performance": 2,
    "study_load": 4,
    "teacher_student_relationship": 3,
    "future_career_concerns": 4,
    "social_support": 1,
    "peer_pressure": 4,
    "extracurricular_activities": 2,
    "bullying": 4
}


# ----------------------------------------------------------
# Convert to DataFrame
# ----------------------------------------------------------

input_data = pd.DataFrame(
    [example_user],
    columns=features
)


print("\nUser Input:")
print(input_data)


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

prediction = model.predict(
    input_data
)[0]

probabilities = model.predict_proba(
    input_data
)[0]


# ----------------------------------------------------------
# Result
# ----------------------------------------------------------

print("\nPrediction:")
print(prediction)

print("\nPrediction Probabilities:")

for stress_class, probability in zip(
    model.classes_,
    probabilities
):
    print(
        f"Stress Level {stress_class}: "
        f"{probability:.2%}"
    )


print("\n" + "=" * 70)
print("PREDICTION TEST COMPLETED")
print("=" * 70)