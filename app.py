import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# MINDPULSE - MENTAL HEALTH & STRESS ANALYSIS
# ============================================================

st.set_page_config(
    page_title="MindPulse",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

STUDENT_MODEL_PATH = os.path.join(
    MODEL_DIR, "student_stress_model.pkl"
)

STUDENT_FEATURES_PATH = os.path.join(
    MODEL_DIR, "student_features.pkl"
)

GENERAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "general_user_tuned_model.pkl"
)

GENERAL_FEATURES_PATH = os.path.join(
    MODEL_DIR, "general_user_tuned_features.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR, "general_user_tuned_threshold.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None

    return joblib.load(path)


@st.cache_resource
def load_features(path):
    if not os.path.exists(path):
        return None

    return joblib.load(path)


# ============================================================
# LOAD STUDENT MODEL
# ============================================================

student_model = load_model(STUDENT_MODEL_PATH)
student_features = load_features(STUDENT_FEATURES_PATH)


# ============================================================
# LOAD GENERAL USER MODEL
# ============================================================

general_model = load_model(GENERAL_MODEL_PATH)
general_features = load_features(GENERAL_FEATURES_PATH)


# ============================================================
# LOAD GENERAL USER THRESHOLD
# ============================================================

general_threshold = 0.50

if os.path.exists(THRESHOLD_PATH):
    try:
        loaded_threshold = joblib.load(THRESHOLD_PATH)

        if isinstance(loaded_threshold, (int, float)):
            general_threshold = float(loaded_threshold)

        elif isinstance(loaded_threshold, dict):
            if "threshold" in loaded_threshold:
                general_threshold = float(
                    loaded_threshold["threshold"]
                )

    except Exception:
        general_threshold = 0.50


# ============================================================
# HEADER
# ============================================================

st.title("🧠 MindPulse")

st.markdown(
    """
    ### Mental Health & Stress Analysis Platform

    MindPulse analyzes behavioral and lifestyle information
    to estimate stress or mental-health risk.

    **Choose your profile below to begin.**
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 MindPulse")

analysis_type = st.sidebar.radio(
    "Select Analysis",
    [
        "👨‍🎓 Student Stress Analysis",
        "👤 General User Analysis"
    ]
)


# ============================================================
# STUDENT SECTION
# ============================================================

if analysis_type == "👨‍🎓 Student Stress Analysis":

    st.header("👨‍🎓 Student Stress Analysis")

    st.info(
        "Enter your academic, lifestyle and social information "
        "to estimate your student stress level."
    )

    if student_model is None or student_features is None:

        st.error(
            "Student model or feature file was not found."
        )

        st.write(
            "Expected files:"
        )

        st.code(
            """
models/student_stress_model.pkl
models/student_features.pkl
            """
        )

        st.stop()

    # --------------------------------------------------------
    # STUDENT INPUTS
    # --------------------------------------------------------

    st.subheader("📚 Academic Information")

    col1, col2 = st.columns(2)

    with col1:

        academic_performance = st.slider(
            "Academic Performance",
            min_value=0,
            max_value=10,
            value=5
        )

        study_load = st.slider(
            "Study Load",
            min_value=0,
            max_value=10,
            value=5
        )

        future_career_concerns = st.slider(
            "Future Career Concerns",
            min_value=0,
            max_value=10,
            value=5
        )

        teacher_student_relationship = st.slider(
            "Teacher-Student Relationship",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:

        peer_pressure = st.slider(
            "Peer Pressure",
            min_value=0,
            max_value=10,
            value=5
        )

        extracurricular_activities = st.slider(
            "Extracurricular Activities",
            min_value=0,
            max_value=10,
            value=5
        )

        bullying = st.slider(
            "Bullying",
            min_value=0,
            max_value=10,
            value=0
        )

        self_esteem = st.slider(
            "Self Esteem",
            min_value=0,
            max_value=10,
            value=5
        )

    # --------------------------------------------------------

    st.subheader("🧠 Mental & Physical Wellbeing")

    col1, col2 = st.columns(2)

    with col1:

        mental_health_history = st.selectbox(
            "Mental Health History",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

        sleep_quality = st.slider(
            "Sleep Quality",
            min_value=0,
            max_value=10,
            value=5
        )

        breathing_problem = st.slider(
            "Breathing Problem",
            min_value=0,
            max_value=10,
            value=0
        )

        blood_pressure = st.slider(
            "Blood Pressure",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:

        noise_level = st.slider(
            "Noise Level",
            min_value=0,
            max_value=10,
            value=5
        )

        living_conditions = st.slider(
            "Living Conditions",
            min_value=0,
            max_value=10,
            value=5
        )

        safety = st.slider(
            "Safety",
            min_value=0,
            max_value=10,
            value=5
        )

        basic_needs = st.slider(
            "Basic Needs",
            min_value=0,
            max_value=10,
            value=5
        )

    # --------------------------------------------------------

    st.subheader("🤝 Social Support")

    social_support = st.slider(
        "Social Support",
        min_value=0,
        max_value=10,
        value=5
    )

    # ========================================================
    # STUDENT PREDICTION
    # ========================================================

    st.divider()

    if st.button(
        "🔍 Analyze Student Stress",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # CREATE INPUT DATA
            # ------------------------------------------------

            student_input = pd.DataFrame({

                "self_esteem": [
                    self_esteem
                ],

                "mental_health_history": [
                    mental_health_history
                ],

                "blood_pressure": [
                    blood_pressure
                ],

                "sleep_quality": [
                    sleep_quality
                ],

                "breathing_problem": [
                    breathing_problem
                ],

                "noise_level": [
                    noise_level
                ],

                "living_conditions": [
                    living_conditions
                ],

                "safety": [
                    safety
                ],

                "basic_needs": [
                    basic_needs
                ],

                "academic_performance": [
                    academic_performance
                ],

                "study_load": [
                    study_load
                ],

                "teacher_student_relationship": [
                    teacher_student_relationship
                ],

                "future_career_concerns": [
                    future_career_concerns
                ],

                "social_support": [
                    social_support
                ],

                "peer_pressure": [
                    peer_pressure
                ],

                "extracurricular_activities": [
                    extracurricular_activities
                ],

                "bullying": [
                    bullying
                ]
            })

            # ------------------------------------------------
            # CHECK REQUIRED FEATURES
            # ------------------------------------------------

            if isinstance(student_features, list):

                missing_features = [
                    feature
                    for feature in student_features
                    if feature not in student_input.columns
                ]

                if missing_features:

                    st.error(
                        "Student model expects features that are "
                        "not present in the app input."
                    )

                    st.write(
                        "Missing features:"
                    )

                    st.write(missing_features)

                    st.stop()

                # Arrange columns in exact model order

                student_input = student_input[
                    student_features
                ]

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = student_model.predict(
                student_input
            )[0]

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader("📊 Student Stress Result")

            # Handle numeric labels

            if prediction == 0:

                st.success(
                    "🟢 Low Stress Level"
                )

                st.write(
                    "The model estimates a relatively low "
                    "stress level based on the information provided."
                )

            elif prediction == 1:

                st.warning(
                    "🟡 Moderate Stress Level"
                )

                st.write(
                    "The model estimates a moderate stress level."
                )

            elif prediction == 2:

                st.error(
                    "🔴 High Stress Level"
                )

                st.write(
                    "The model estimates a high stress level."
                )

            else:

                st.info(
                    f"Predicted Stress Class: {prediction}"
                )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                student_model,
                "predict_proba"
            ):

                probabilities = (
                    student_model.predict_proba(
                        student_input
                    )[0]
                )

                st.subheader(
                    "Prediction Probability"
                )

                probability_df = pd.DataFrame(
                    {
                        "Stress Level": [
                            f"Class {i}"
                            for i in range(
                                len(probabilities)
                            )
                        ],
                        "Probability": [
                            f"{p * 100:.2f}%"
                            for p in probabilities
                        ]
                    }
                )

                st.table(
                    probability_df
                )

        except Exception as e:

            st.error(
                "Student prediction failed."
            )

            st.exception(e)


# ============================================================
# GENERAL USER SECTION
# ============================================================

else:

    st.header("👤 General User Mental Health Analysis")

    st.info(
        "Enter general lifestyle and behavioral information "
        "to estimate the likelihood of a mental health issue."
    )

    if general_model is None or general_features is None:

        st.error(
            "General user model or feature file was not found."
        )

        st.code(
            """
models/general_user_tuned_model.pkl
models/general_user_tuned_features.pkl
            """
        )

        st.stop()

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=25
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Non-binary",
                "Prefer not to say"
            ]
        )

        income_level = st.selectbox(
            "Income Level",
            [
                "Low",
                "Middle",
                "High"
            ]
        )

    with col2:

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Full-time",
                "Part-time",
                "Self-employed",
                "Student",
                "Unemployed"
            ]
        )

        work_hours = st.slider(
            "Work Hours Per Week",
            0,
            100,
            40
        )

        job_satisfaction = st.slider(
            "Job Satisfaction",
            0,
            10,
            5
        )

    # --------------------------------------------------------
    # WORK & LIFESTYLE
    # --------------------------------------------------------

    st.subheader("💼 Work & Lifestyle")

    col1, col2 = st.columns(2)

    with col1:

        work_stress = st.slider(
            "Work Stress Level",
            0,
            10,
            5
        )

        work_life_balance = st.slider(
            "Work-Life Balance",
            0,
            10,
            5
        )

        exercise = st.selectbox(
            "Exercise Per Week",
            [
                "Never",
                "1-2 times",
                "3-4 times",
                "5+ times"
            ]
        )

        sleep_hours = st.slider(
            "Sleep Hours Per Night",
            0.0,
            15.0,
            7.0,
            0.1
        )

    with col2:

        screen_time = st.slider(
            "Screen Time Per Day (Hours)",
            0.0,
            24.0,
            7.0,
            0.1
        )

        social_media = st.slider(
            "Social Media Per Day (Hours)",
            0.0,
            24.0,
            3.0,
            0.1
        )

        hobby_time = st.slider(
            "Hobby Time Per Week (Hours)",
            0.0,
            50.0,
            5.0,
            0.1
        )

    # --------------------------------------------------------
    # SOCIAL & FINANCIAL
    # --------------------------------------------------------

    st.subheader("🤝 Social & Financial Wellbeing")

    col1, col2 = st.columns(2)

    with col1:

        financial_stress = st.slider(
            "Financial Stress",
            0,
            10,
            5
        )

        social_support = st.slider(
            "Social Support",
            0,
            10,
            5
        )

        close_friends = st.number_input(
            "Close Friends Count",
            min_value=0,
            max_value=50,
            value=4
        )

    with col2:

        feel_understood = st.slider(
            "Feel Understood",
            0,
            10,
            5
        )

        loneliness = st.slider(
            "Loneliness",
            0,
            10,
            5
        )

    # ========================================================
    # GENERAL USER PREDICTION
    # ========================================================

    st.divider()

    if st.button(
        "🔍 Analyze Mental Health Risk",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # CREATE INPUT DATA
            # ------------------------------------------------

            general_input = pd.DataFrame({

                "Age": [age],

                "Gender": [gender],

                "Income_Level": [income_level],

                "Employment_Status": [
                    employment_status
                ],

                "Work_Hours_Per_Week": [
                    work_hours
                ],

                "Job_Satisfaction": [
                    job_satisfaction
                ],

                "Work_Stress_Level": [
                    work_stress
                ],

                "Work_Life_Balance": [
                    work_life_balance
                ],

                "Exercise_Per_Week": [
                    exercise
                ],

                "Sleep_Hours_Night": [
                    sleep_hours
                ],

                "Screen_Time_Hours_Day": [
                    screen_time
                ],

                "Social_Media_Hours_Day": [
                    social_media
                ],

                "Hobby_Time_Hours_Week": [
                    hobby_time
                ],

                "Financial_Stress": [
                    financial_stress
                ],

                "Social_Support": [
                    social_support
                ],

                "Close_Friends_Count": [
                    close_friends
                ],

                "Feel_Understood": [
                    feel_understood
                ],

                "Loneliness": [
                    loneliness
                ]
            })

            # ------------------------------------------------
            # CHECK FEATURES
            # ------------------------------------------------

            if isinstance(
                general_features,
                list
            ):

                missing_features = [
                    feature
                    for feature in general_features
                    if feature not in general_input.columns
                ]

                if missing_features:

                    st.error(
                        "General user model expects "
                        "features that are missing."
                    )

                    st.write(
                        missing_features
                    )

                    st.stop()

                general_input = general_input[
                    general_features
                ]

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            if hasattr(
                general_model,
                "predict_proba"
            ):

                probabilities = (
                    general_model.predict_proba(
                        general_input
                    )[0]
                )

                # Probability of class 1

                if len(probabilities) == 2:

                    mental_health_probability = (
                        probabilities[1]
                    )

                else:

                    mental_health_probability = (
                        max(probabilities)
                    )

                prediction = (
                    1
                    if mental_health_probability
                    >= general_threshold
                    else 0
                )

            else:

                prediction = general_model.predict(
                    general_input
                )[0]

                mental_health_probability = (
                    float(prediction)
                )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader(
                "📊 Mental Health Risk Result"
            )

            probability_percentage = (
                mental_health_probability * 100
            )

            st.metric(
                "Estimated Risk Probability",
                f"{probability_percentage:.2f}%"
            )

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if probability_percentage < 30:

                st.success(
                    "🟢 Lower Estimated Risk"
                )

            elif probability_percentage < 60:

                st.warning(
                    "🟡 Moderate Estimated Risk"
                )

            else:

                st.error(
                    "🔴 Higher Estimated Risk"
                )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            if prediction == 1:

                st.error(
                    "⚠️ Model Prediction: "
                    "Mental Health Issue Likely"
                )

            else:

                st.success(
                    "✅ Model Prediction: "
                    "Mental Health Issue Less Likely"
                )

            # ------------------------------------------------
            # IMPORTANT DISCLAIMER
            # ------------------------------------------------

            st.info(
                """
                **Important:** MindPulse provides a machine-learning
                based estimate for educational and analytical purposes.
                It is not a medical diagnosis. If you are concerned
                about your mental health, consider speaking with a
                qualified mental-health professional.
                """
            )

        except Exception as e:

            st.error(
                "General user prediction failed."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MindPulse | Machine Learning Mental Health & Stress Analysis"
)