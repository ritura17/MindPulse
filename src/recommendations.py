# ==========================================================
# MINDPULSE - RECOMMENDATION ENGINE
# ==========================================================


def generate_recommendations(user_data):

    recommendations = []

    # ------------------------------------------------------
    # Study Load
    # ------------------------------------------------------

    if user_data["study_load"] >= 4:
        recommendations.append(
            "Your reported study load is high. "
            "Consider breaking study sessions into smaller "
            "sessions with regular breaks."
        )

    # ------------------------------------------------------
    # Sleep Quality
    # ------------------------------------------------------

    if user_data["sleep_quality"] <= 2:
        recommendations.append(
            "Your reported sleep quality is relatively low. "
            "Try to maintain a consistent sleep schedule "
            "and allow enough time for rest."
        )

    # ------------------------------------------------------
    # Social Support
    # ------------------------------------------------------

    if user_data["social_support"] <= 1:
        recommendations.append(
            "Your reported social support is relatively low. "
            "Consider spending time with supportive friends, "
            "family members, classmates, or mentors."
        )

    # ------------------------------------------------------
    # Peer Pressure
    # ------------------------------------------------------

    if user_data["peer_pressure"] >= 4:
        recommendations.append(
            "You reported relatively high peer pressure. "
            "Try to set personal boundaries and focus on "
            "goals that are important to you."
        )

    # ------------------------------------------------------
    # Bullying
    # ------------------------------------------------------

    if user_data["bullying"] >= 4:
        recommendations.append(
            "You reported a high level of bullying-related "
            "pressure. Consider talking to a trusted person, "
            "teacher, counselor, or another appropriate "
            "support person."
        )

    # ------------------------------------------------------
    # Career Concerns
    # ------------------------------------------------------

    if user_data["future_career_concerns"] >= 4:
        recommendations.append(
            "Your reported career concerns are relatively high. "
            "Consider breaking your career goals into smaller "
            "steps and discussing them with a mentor or "
            "career advisor."
        )

    # ------------------------------------------------------
    # Academic Performance
    # ------------------------------------------------------

    if user_data["academic_performance"] <= 1:
        recommendations.append(
            "If you're struggling academically, consider asking "
            "a teacher, mentor, or classmate for help and "
            "creating a manageable study plan."
        )

    # ------------------------------------------------------
    # Self Esteem
    # ------------------------------------------------------

    if user_data["self_esteem"] <= 10:
        recommendations.append(
            "Your reported self-esteem score is relatively low. "
            "Try setting small achievable goals and recognizing "
            "your progress."
        )

    # ------------------------------------------------------
    # Noise Level
    # ------------------------------------------------------

    if user_data["noise_level"] >= 4:
        recommendations.append(
            "Your reported noise level is relatively high. "
            "If possible, try studying or resting in a quieter "
            "environment."
        )

    # ------------------------------------------------------
    # Default
    # ------------------------------------------------------

    if not recommendations:
        recommendations.append(
            "Continue maintaining healthy study, sleep, "
            "and social habits."
        )

    return recommendations