import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# MINDPULSE
# Mental Health & Stress Analysis
# ============================================================

st.set_page_config(
    page_title="MindPulse",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #666;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #f5f7fa;
    border-left: 5px solid #4CAF50;
    margin-bottom: 20px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #f5f7fa;
    text-align: center;
    margin-top: 20px;
}

.suggestion-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f8f9fa;
    margin: 8px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_student_model():

    model_path = MODELS_DIR / "student_stress_model.pkl"
    features_path = MODELS_DIR / "student_features.pkl"

    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)

    features = None
    if features_path.exists():
        features = joblib.load(features_path)

    return model, features


@st.cache_resource
def load_general_model():

    model_path = MODELS_DIR / "general_user_tuned_model.pkl"
    features_path = MODELS_DIR / "general_user_tuned_features.pkl"

    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)

    features = None
    if features_path.exists():
        features = joblib.load(features_path)

    return model, features


student_model, student_features = load_student_model()
general_model, general_features = load_general_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_probability(model, data):
    """
    Safely extract probability from sklearn model/pipeline.
    Fixes numpy ndarray formatting issue.
    """

    try:

        probabilities = model.predict_proba(data)

        probability = np.asarray(probabilities)

        if probability.ndim == 2:
            probability = probability[0, 1]

        elif probability.ndim == 1:
            probability = probability[0]

        probability = float(probability)

        return probability

    except Exception:
        return None


def explain_slider(label, explanation, key):

    st.markdown(f"**{label}**")

    st.caption(explanation)

    return st.slider(
        label,
        min_value=0,
        max_value=10,
        value=None,
        step=1,
        key=key,
        label_visibility="collapsed"
    )


def add_suggestion(suggestions, title, message):

    suggestions.append(
        f"**{title}:** {message}"
    )


# ============================================================
# HOMEPAGE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 MindPulse</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A simple self-assessment tool for understanding stress and mental well-being</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="info-box">

### 🌱 What is MindPulse?

MindPulse is an interactive application that helps users understand
possible patterns related to **stress, lifestyle, social well-being,
academic pressure and mental well-being**.

You answer questions about your daily life, habits and experiences.
MindPulse then analyzes your responses using machine-learning models
and provides an easy-to-understand result along with practical
lifestyle suggestions.

### 🎯 Who can use MindPulse?

- 🎓 Students
- 👨‍💼 Working professionals
- 👤 General users

### 💡 What does MindPulse provide?

- Stress / mental-health related prediction
- Lifestyle factor analysis
- Behavioral risk indicators
- Personalized suggestions
- Easy-to-understand results

> ⚠️ MindPulse is an educational/self-awareness tool and is **not a
> medical diagnosis or a replacement for a qualified mental-health
> professional.**

</div>
""", unsafe_allow_html=True)


# ============================================================
# USER TYPE
# ============================================================

st.markdown(
    '<div class="section-title">👤 Choose Your Profile</div>',
    unsafe_allow_html=True
)

user_type = st.radio(
    "Please select one option:",
    ["Student", "General User"],
    index=None,
    horizontal=True
)


if user_type is None:

    st.info("👆 Please select **Student** or **General User** to continue.")

    st.stop()


# ============================================================
# ============================================================
# STUDENT SECTION
# ============================================================
# ============================================================

if user_type == "Student":

    st.markdown(
        '<div class="section-title">🎓 Student Stress Analysis</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Please answer the questions based on your current situation. "
        "For factor questions, 0 means very low and 10 means very high."
    )


    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=13,
            max_value=100,
            value=None,
            step=1,
            placeholder="Enter your age"
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Select gender", "Male", "Female"]
        )


    # --------------------------------------------------------
    # SLEEP & PHYSICAL FACTORS
    # --------------------------------------------------------

    st.subheader("😴 Sleep & Physical Factors")

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Example: 7.5"
    )

    st.caption(
        "Sleep Hours = approximately how many hours you normally sleep per day."
    )


    blood_pressure = explain_slider(
        "Blood Pressure",
        "How concerned are you about your blood pressure? "
        "0 = no concern, 10 = very high concern.",
        "student_blood_pressure"
    )

    breathing_problem = explain_slider(
        "Breathing Problem",
        "How frequently do you experience breathing difficulty or breathing discomfort? "
        "0 = never, 10 = very frequently.",
        "student_breathing"
    )

    noise_level = explain_slider(
        "Noise Level",
        "How much does noise around you disturb your study, sleep or concentration? "
        "0 = not disturbing, 10 = extremely disturbing.",
        "student_noise"
    )


    # --------------------------------------------------------
    # LIVING ENVIRONMENT
    # --------------------------------------------------------

    st.subheader("🏠 Living Environment")

    living_conditions = explain_slider(
        "Living Conditions",
        "How comfortable and suitable is your living environment? "
        "0 = very poor/uncomfortable, 10 = very comfortable.",
        "student_living"
    )

    safety = explain_slider(
        "Safety",
        "How safe and secure do you feel in your living/study environment? "
        "0 = very unsafe, 10 = completely safe.",
        "student_safety"
    )

    basic_needs = explain_slider(
        "Basic Needs",
        "How well are your basic needs such as food, housing and essential resources being met? "
        "0 = not met, 10 = completely met.",
        "student_basic_needs"
    )


    # --------------------------------------------------------
    # MENTAL & ACADEMIC FACTORS
    # --------------------------------------------------------

    st.subheader("🧠 Mental & Academic Factors")

    st.caption(
        "Rate each factor from 0 to 10. "
        "These questions describe your current experience."
    )

    self_esteem = explain_slider(
        "Self-Esteem",
        "How positively do you feel about yourself and your abilities? "
        "0 = very low, 10 = very high.",
        "student_self_esteem"
    )

    mental_health_history = st.selectbox(
        "Mental Health History",
        ["Select", "Yes", "No"]
    )

    st.caption(
        "Mental Health History = whether you have previously experienced "
        "a diagnosed or significant mental-health problem."
    )

    academic_performance = explain_slider(
        "Academic Performance",
        "How satisfied are you with your academic performance? "
        "0 = very dissatisfied, 10 = very satisfied.",
        "student_academic"
    )

    study_load = explain_slider(
        "Study Load",
        "How heavy or demanding is your current academic workload? "
        "0 = very light, 10 = extremely heavy.",
        "student_study_load"
    )

    teacher_student_relationship = explain_slider(
        "Teacher-Student Relationship",
        "How positive and supportive is your relationship with teachers/faculty? "
        "0 = very poor, 10 = excellent.",
        "student_teacher"
    )

    future_career_concerns = explain_slider(
        "Future Career Concerns",
        "How worried are you about your future career, job or professional life? "
        "0 = not worried, 10 = extremely worried.",
        "student_career"
    )


    # --------------------------------------------------------
    # SOCIAL FACTORS
    # --------------------------------------------------------

    st.subheader("👥 Social Factors")

    social_support = explain_slider(
        "Social Support",
        "How much support do you receive from family, friends or people around you? "
        "0 = no support, 10 = very strong support.",
        "student_social_support"
    )

    peer_pressure = explain_slider(
        "Peer Pressure",
        "How much pressure do you feel from friends, classmates or peers? "
        "0 = none, 10 = extremely high.",
        "student_peer_pressure"
    )

    extracurricular_activities = explain_slider(
        "Extracurricular Activities",
        "How actively do you participate in sports, clubs, hobbies or other activities outside academics? "
        "0 = never, 10 = very active.",
        "student_extra"
    )

    bullying = explain_slider(
        "Bullying",
        "How much bullying, harassment or unwanted negative behavior do you experience? "
        "0 = none, 10 = extremely high.",
        "student_bullying"
    )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    st.markdown("---")

    analyze_student = st.button(
        "🔍 Analyze My Student Profile",
        type="primary",
        use_container_width=True
    )


    if analyze_student:

        # Validation

        required_values = [
            age,
            sleep_hours,
            blood_pressure,
            breathing_problem,
            noise_level,
            living_conditions,
            safety,
            basic_needs,
            self_esteem,
            academic_performance,
            study_load,
            teacher_student_relationship,
            future_career_concerns,
            social_support,
            peer_pressure,
            extracurricular_activities,
            bullying
        ]

        if any(value is None for value in required_values):

            st.error(
                "⚠️ Please complete all fields before analyzing."
            )
            st.stop()

        if gender == "Select gender":

            st.error("⚠️ Please select your gender.")
            st.stop()

        if mental_health_history == "Select":

            st.error("⚠️ Please select your mental health history.")
            st.stop()


        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        history_value = (
            1 if mental_health_history == "Yes" else 0
        )

        input_data = pd.DataFrame([{

            "self_esteem": self_esteem,
            "mental_health_history": history_value,
            "blood_pressure": blood_pressure,
            "sleep_quality": sleep_hours,
            "breathing_problem": breathing_problem,
            "noise_level": noise_level,
            "living_conditions": living_conditions,
            "safety": safety,
            "basic_needs": basic_needs,
            "academic_performance": academic_performance,
            "study_load": study_load,
            "teacher_student_relationship": teacher_student_relationship,
            "future_career_concerns": future_career_concerns,
            "social_support": social_support,
            "peer_pressure": peer_pressure,
            "extracurricular_activities": extracurricular_activities,
            "bullying": bullying

        }])


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if student_model is None:

            st.error(
                "Student model was not found. "
                "Please check the models folder."
            )

        else:

            try:

                prediction = student_model.predict(input_data)[0]

                probability = safe_probability(
                    student_model,
                    input_data
                )

                st.markdown("---")
                st.subheader("📊 Your Result")

                if probability is not None:

                    probability_percent = probability * 100

                    st.metric(
                        "Estimated Risk Probability",
                        f"{probability_percent:.1f}%"
                    )

                if prediction == 1:

                    st.error(
                        "⚠️ The model indicates a higher stress/mental-health risk pattern."
                    )

                else:

                    st.success(
                        "✅ The model indicates a lower stress/mental-health risk pattern."
                    )


                # ------------------------------------------------
                # STUDENT SUGGESTIONS
                # ------------------------------------------------

                suggestions = []

                if study_load >= 7:

                    add_suggestion(
                        suggestions,
                        "Study Load",
                        "Your study load appears high. Try breaking large tasks "
                        "into smaller daily goals and take short breaks between study sessions."
                    )

                if future_career_concerns >= 7:

                    add_suggestion(
                        suggestions,
                        "Career Concerns",
                        "If career uncertainty is stressing you, focus on one "
                        "small career goal at a time such as improving a skill, "
                        "building a project or preparing your resume."
                    )

                if peer_pressure >= 7:

                    add_suggestion(
                        suggestions,
                        "Peer Pressure",
                        "Try to make decisions based on your own goals rather "
                        "than constantly comparing yourself with others."
                    )

                if bullying >= 4:

                    add_suggestion(
                        suggestions,
                        "Bullying",
                        "If you are experiencing bullying or harassment, "
                        "consider talking to a trusted teacher, family member, "
                        "counselor or another trusted person."
                    )

                if social_support <= 3:

                    add_suggestion(
                        suggestions,
                        "Social Support",
                        "Try connecting with a trusted friend, family member, "
                        "teacher or mentor instead of dealing with everything alone."
                    )

                if sleep_hours < 6:

                    add_suggestion(
                        suggestions,
                        "Sleep",
                        "Your sleep duration is quite low. Try maintaining a "
                        "consistent sleep schedule and reducing phone/social-media "
                        "use before bedtime."
                    )

                if screen := False:
                    pass

                if breathing_problem >= 7:

                    add_suggestion(
                        suggestions,
                        "Breathing Problems",
                        "If breathing problems are frequent or severe, consider "
                        "speaking with a qualified healthcare professional."
                    )

                if noise_level >= 7:

                    add_suggestion(
                        suggestions,
                        "Noise",
                        "Try using a quieter study environment, headphones/earplugs "
                        "where appropriate, or a library/study room."
                    )

                if self_esteem <= 3:

                    add_suggestion(
                        suggestions,
                        "Self-Esteem",
                        "Try focusing on small achievements and progress instead "
                        "of comparing yourself with others."
                    )

                if extracurricular_activities <= 2:

                    add_suggestion(
                        suggestions,
                        "Activities",
                        "Consider adding enjoyable physical or social activities "
                        "such as walking, outdoor games, sports or hobbies."
                    )

                if suggestions:

                    st.subheader("💡 Suggestions For You")

                    for suggestion in suggestions:

                        st.markdown(
                            f'<div class="suggestion-box">{suggestion}</div>',
                            unsafe_allow_html=True
                        )

                else:

                    st.success(
                        "🌱 Your current responses do not indicate major behavioral "
                        "risk areas. Keep maintaining healthy study, sleep, physical "
                        "activity and social habits."
                    )


            except Exception as e:

                st.error(
                    "Unable to process the student prediction."
                )

                st.code(str(e))


# ============================================================
# ============================================================
# GENERAL USER SECTION
# ============================================================
# ============================================================

if user_type == "General User":

    st.markdown(
        '<div class="section-title">👤 General User Mental Well-being Analysis</div>',
        unsafe_allow_html=True
    )

    st.info(
        "For factor questions, use 0–10. "
        "0 generally represents very low/none and 10 represents very high/strong."
    )


    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=13,
            max_value=100,
            value=None,
            step=1,
            placeholder="Enter your age"
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Select gender",
                "Male",
                "Female",
                "Non-binary",
                "Prefer not to say"
            ]
        )


    income_level = st.selectbox(
        "Income Level",
        ["Select", "Low", "Middle", "High"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        [
            "Select",
            "Full-time",
            "Part-time",
            "Self-employed",
            "Student",
            "Unemployed"
        ]
    )

    exercise_per_week = st.selectbox(
        "Exercise Per Week",
        [
            "Select",
            "Never",
            "1-2 times",
            "3-4 times",
            "5+ times"
        ]
    )


    # --------------------------------------------------------
    # WORK & LIFESTYLE
    # --------------------------------------------------------

    st.subheader("💼 Work & Lifestyle")

    work_hours = st.number_input(
        "Work Hours Per Week",
        min_value=0.0,
        max_value=100.0,
        value=None,
        step=1.0,
        placeholder="Example: 40"
    )

    st.caption(
        "Approximately how many hours do you spend working/studying each week?"
    )


    job_satisfaction = explain_slider(
        "Job Satisfaction",
        "How satisfied are you with your current job/study situation? "
        "0 = extremely dissatisfied, 10 = extremely satisfied.",
        "general_job_satisfaction"
    )

    work_stress = explain_slider(
        "Work Stress Level",
        "How stressful is your work/study workload? "
        "0 = no stress, 10 = extremely stressful.",
        "general_work_stress"
    )

    work_life_balance = explain_slider(
        "Work-Life Balance",
        "How well do you balance work/study with personal life? "
        "0 = very poor balance, 10 = excellent balance.",
        "general_work_balance"
    )


    # --------------------------------------------------------
    # SLEEP & SCREEN TIME
    # --------------------------------------------------------

    st.subheader("😴 Sleep & Digital Habits")

    sleep_hours = st.number_input(
        "Sleep Hours Per Night",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Example: 7"
    )

    st.caption(
        "Enter the approximate number of hours you sleep per night."
    )


    screen_time = st.number_input(
        "Screen Time Per Day (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Example: 6"
    )

    st.caption(
        "Total daily time spent using phones, computers, tablets and other screens."
    )


    social_media = st.number_input(
        "Social Media Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Example: 3"
    )

    st.caption(
        "Approximately how many hours per day do you spend on social-media platforms?"
    )


    hobby_time = st.number_input(
        "Hobby Time Per Week",
        min_value=0.0,
        max_value=168.0,
        value=None,
        step=1.0,
        placeholder="Example: 5"
    )

    st.caption(
        "Hours per week spent on hobbies, sports, creative activities or relaxation."
    )


    # --------------------------------------------------------
    # FINANCIAL & SOCIAL
    # --------------------------------------------------------

    st.subheader("💰 Financial & Social Well-being")

    financial_stress = explain_slider(
        "Financial Stress",
        "How much stress do financial responsibilities cause you? "
        "0 = no financial stress, 10 = extremely high financial stress.",
        "general_financial"
    )

    social_support = explain_slider(
        "Social Support",
        "How much support do you receive from family, friends or others? "
        "0 = no support, 10 = very strong support.",
        "general_social_support"
    )

    close_friends = explain_slider(
        "Close Friends Count",
        "How many close and trusted friendships do you feel you have? "
        "0 = none, 10 = many close friends.",
        "general_friends"
    )

    feel_understood = explain_slider(
        "Feel Understood",
        "How understood do you feel by the people close to you? "
        "0 = not understood at all, 10 = completely understood.",
        "general_understood"
    )

    loneliness = explain_slider(
        "Loneliness",
        "How lonely do you currently feel? "
        "0 = not lonely, 10 = extremely lonely.",
        "general_loneliness"
    )


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    st.markdown("---")

    analyze_general = st.button(
        "🔍 Analyze My Profile",
        type="primary",
        use_container_width=True
    )


    if analyze_general:

        required_values = [
            age,
            work_hours,
            job_satisfaction,
            work_stress,
            work_life_balance,
            sleep_hours,
            screen_time,
            social_media,
            hobby_time,
            financial_stress,
            social_support,
            close_friends,
            feel_understood,
            loneliness
        ]

        if any(value is None for value in required_values):

            st.error(
                "⚠️ Please complete all fields before analyzing."
            )
            st.stop()


        if gender == "Select gender":

            st.error("⚠️ Please select your gender.")
            st.stop()


        if income_level == "Select":

            st.error("⚠️ Please select your income level.")
            st.stop()


        if employment_status == "Select":

            st.error("⚠️ Please select your employment status.")
            st.stop()


        if exercise_per_week == "Select":

            st.error("⚠️ Please select your exercise frequency.")
            st.stop()


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "Age": age,
            "Gender": gender,
            "Income_Level": income_level,
            "Employment_Status": employment_status,
            "Work_Hours_Per_Week": work_hours,
            "Job_Satisfaction": job_satisfaction,
            "Work_Stress_Level": work_stress,
            "Work_Life_Balance": work_life_balance,
            "Exercise_Per_Week": exercise_per_week,
            "Sleep_Hours_Night": sleep_hours,
            "Screen_Time_Hours_Day": screen_time,
            "Social_Media_Hours_Day": social_media,
            "Hobby_Time_Hours_Week": hobby_time,
            "Financial_Stress": financial_stress,
            "Social_Support": social_support,
            "Close_Friends_Count": close_friends,
            "Feel_Understood": feel_understood,
            "Loneliness": loneliness

        }])


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if general_model is None:

            st.error(
                "General user model was not found. "
                "Please check the models folder."
            )

        else:

            try:

                prediction = general_model.predict(input_data)[0]

                probability = safe_probability(
                    general_model,
                    input_data
                )

                st.markdown("---")
                st.subheader("📊 Your Result")


                if probability is not None:

                    probability_percent = probability * 100

                    st.metric(
                        "Estimated Mental Health Risk Probability",
                        f"{probability_percent:.1f}%"
                    )


                if prediction == 1:

                    st.warning(
                        "⚠️ The model indicates a higher mental-health risk pattern."
                    )

                else:

                    st.success(
                        "✅ The model indicates a lower mental-health risk pattern."
                    )


                # ------------------------------------------------
                # PERSONALIZED SUGGESTIONS
                # ------------------------------------------------

                suggestions = []


                if work_stress >= 7:

                    add_suggestion(
                        suggestions,
                        "Work Stress",
                        "Your work/study stress appears high. Try dividing "
                        "large tasks into smaller goals and taking short breaks."
                    )


                if social_media >= 5:

                    add_suggestion(
                        suggestions,
                        "Social Media",
                        "Consider reducing unnecessary social-media use. "
                        "Try replacing some screen time with outdoor activities, "
                        "sports, exercise or time with friends."
                    )


                if screen_time >= 8:

                    add_suggestion(
                        suggestions,
                        "Screen Time",
                        "Your daily screen time is high. Try taking regular "
                        "screen breaks and spending more time on physical or "
                        "outdoor activities."
                    )


                if sleep_hours < 6:

                    add_suggestion(
                        suggestions,
                        "Sleep",
                        "You are getting relatively little sleep. Try maintaining "
                        "a consistent sleep schedule and reducing phone use before bed."
                    )


                if financial_stress >= 7:

                    add_suggestion(
                        suggestions,
                        "Financial Stress",
                        "Financial pressure appears high. Consider creating a "
                        "simple budget, prioritizing essential expenses and "
                        "talking with someone you trust about your concerns."
                    )


                if loneliness >= 7:

                    add_suggestion(
                        suggestions,
                        "Loneliness",
                        "You may benefit from more social connection. Try "
                        "spending time with trusted friends/family or joining "
                        "a club, sport or community activity."
                    )


                if social_support <= 3:

                    add_suggestion(
                        suggestions,
                        "Social Support",
                        "Your perceived social support is low. Consider "
                        "connecting with someone you trust and sharing how you feel."
                    )


                if exercise_per_week == "Never":

                    add_suggestion(
                        suggestions,
                        "Physical Activity",
                        "Consider starting with a short walk, stretching, "
                        "home exercise or an outdoor game. Even small amounts "
                        "of physical activity can be a useful healthy habit."
                    )


                if hobby_time < 2:

                    add_suggestion(
                        suggestions,
                        "Hobbies",
                        "Try setting aside some time each week for a hobby, "
                        "creative activity, sport or something you genuinely enjoy."
                    )


                if work_life_balance <= 3:

                    add_suggestion(
                        suggestions,
                        "Work-Life Balance",
                        "Your work-life balance appears low. Try setting "
                        "clear boundaries between work/study time and personal time."
                    )


                if job_satisfaction <= 3:

                    add_suggestion(
                        suggestions,
                        "Job Satisfaction",
                        "Think about which parts of your work/study situation "
                        "are causing dissatisfaction and whether small changes "
                        "could improve your daily experience."
                    )


                if feel_understood <= 3:

                    add_suggestion(
                        suggestions,
                        "Feeling Understood",
                        "Consider talking openly with someone you trust. "
                        "Feeling heard and understood can make difficult periods "
                        "easier to manage."
                    )


                if close_friends <= 2:

                    add_suggestion(
                        suggestions,
                        "Social Connection",
                        "Try gradually building social connections through "
                        "sports, hobbies, classes, clubs or community activities."
                    )


                # ------------------------------------------------
                # DISPLAY SUGGESTIONS
                # ------------------------------------------------

                if suggestions:

                    st.subheader("💡 Suggestions Based on Your Answers")

                    for suggestion in suggestions:

                        st.markdown(
                            f'<div class="suggestion-box">{suggestion}</div>',
                            unsafe_allow_html=True
                        )

                else:

                    st.success(
                        "🌱 Your responses do not show major behavioral risk areas. "
                        "Keep maintaining healthy sleep, physical activity, "
                        "social connection and work-life balance."
                    )


            except Exception as e:

                st.error(
                    "Unable to process the general-user prediction."
                )

                st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🧠 MindPulse is an educational and self-awareness application. "
    "Its predictions should not be interpreted as a medical diagnosis."
)