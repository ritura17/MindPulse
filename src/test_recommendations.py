from recommendations import generate_recommendations


# ==========================================================
# MINDPULSE - TEST RECOMMENDATION ENGINE
# ==========================================================

user_data = {

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


recommendations = generate_recommendations(user_data)


print("=" * 70)
print("MINDPULSE - RECOMMENDATION TEST")
print("=" * 70)

print("\nRecommendations:\n")

for recommendation in recommendations:
    print("•", recommendation)