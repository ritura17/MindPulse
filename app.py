import streamlit as st
import pandas as pd
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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 MindPulse</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Mental Health & Behavioral Risk Analysis</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATHS
# ============================================================

GENERAL_MODEL_PATH = "models/general_user_tuned_model.pkl"
GENERAL_FEATURES_PATH = "models/general_user_tuned_features.pkl"
GENERAL_THRESHOLD_PATH = "models/general_user_tuned_threshold.pkl"

STUDENT_MODEL_PATH = "models/student_stress_model.pkl"
STUDENT_FEATURES_PATH = "models/student_features.pkl"


# ============================================================
# LOAD GENERAL USER MODEL
# ============================================================

@st.cache_resource
def load_general_model():

    model = joblib.load(GENERAL_MODEL_PATH)
    features = joblib.load(GENERAL_FEATURES_PATH)
    threshold = joblib.load(GENERAL_THRESHOLD_PATH)

    return model, features, threshold


# ============================================================
# LOAD STUDENT MODEL
# ============================================================

@st.cache_resource
def load_student_model():

    model = joblib.load(STUDENT_MODEL_PATH)
    features = joblib.load(STUDENT_FEATURES_PATH)

    return model, features


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 MindPulse")

user_type = st.sidebar.radio(
    "Select User Type",
    [
        "General User",
        "Student"
    ]
)


# ============================================================
# GENERAL USER
# ============================================================

if user_type == "General User":

    st.header("👤 General User Mental Health Analysis")

    st.info(
        "Enter your lifestyle, work and social information. "
        "MindPulse will estimate your mental-health risk based "
        "on the trained machine-learning model."
    )

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
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

        work_hours = st.number_input(
            "Work Hours Per Week",
            min_value=0.0,
            max_value=100.0,
            value=40.0
        )


    # --------------------------------------------------------
    # WORK & LIFESTYLE
    # --------------------------------------------------------

    st.subheader("💼 Work & Lifestyle")

    col1, col2, col3 = st.columns(3)

    with col1:

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1.0,
            10.0,
            5.0
        )

        work_stress = st.slider(
            "Work Stress Level",
            1.0,
            10.0,
            5.0
        )

    with col2:

        work_life_balance = st.slider(
            "Work-Life Balance",
            1.0,
            10.0,
            5.0
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

    with col3:

        sleep_hours = st.number_input(
            "Sleep Hours Per Night",
            min_value=0.0,
            max_value=24.0,
            value=7.0
        )

        screen_time = st.number_input(
            "Screen Time Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=6.0
        )


    # --------------------------------------------------------
    # DIGITAL & PERSONAL HABITS
    # --------------------------------------------------------

    st.subheader("📱 Digital & Personal Habits")

    col1, col2, col3 = st.columns(3)

    with col1:

        social_media = st.number_input(
            "Social Media Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=3.0
        )

        hobby_time = st.number_input(
            "Hobby Time Hours Per Week",
            min_value=0.0,
            max_value=100.0,
            value=5.0
        )

    with col2:

        financial_stress = st.slider(
            "Financial Stress",
            1.0,
            10.0,
            5.0
        )

        social_support = st.slider(
            "Social Support",
            1.0,
            10.0,
            5.0
        )

    with col3:

        close_friends = st.number_input(
            "Close Friends Count",
            min_value=0,
            max_value=100,
            value=4
        )

        feel_understood = st.slider(
            "Feel Understood",
            1.0,
            10.0,
            5.0
        )


    # --------------------------------------------------------
    # LONELINESS
    # --------------------------------------------------------

    st.subheader("🤝 Social Well-being")

    loneliness = st.slider(
        "Loneliness Level",
        1.0,
        10.0,
        5.0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()

    predict_button = st.button(
        "🔍 Analyze My Mental Health",
        use_container_width=True
    )


    if predict_button:

        try:

            model, feature_names, threshold = load_general_model()

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "Age": [age],

                "Gender": [gender],

                "Income_Level": [income_level],

                "Employment_Status": [employment_status],

                "Work_Hours_Per_Week": [work_hours],

                "Job_Satisfaction": [job_satisfaction],

                "Work_Stress_Level": [work_stress],

                "Work_Life_Balance": [work_life_balance],

                "Exercise_Per_Week": [exercise],

                "Sleep_Hours_Night": [sleep_hours],

                "Screen_Time_Hours_Day": [screen_time],

                "Social_Media_Hours_Day": [social_media],

                "Hobby_Time_Hours_Week": [hobby_time],

                "Financial_Stress": [financial_stress],

                "Social_Support": [social_support],

                "Close_Friends_Count": [close_friends],

                "Feel_Understood": [feel_understood],

                "Loneliness": [loneliness]
            })


            # ------------------------------------------------
            # ENSURE CORRECT FEATURE ORDER
            # ------------------------------------------------

            input_data = input_data[feature_names]


            # ------------------------------------------------
            # PREDICT PROBABILITY
            # ------------------------------------------------

            probability = model.predict_proba(
                input_data
            )[0][1]


            # ------------------------------------------------
            # APPLY SAVED THRESHOLD
            # ------------------------------------------------

            prediction = int(
                probability >= threshold
            )


            probability_percent = probability * 100


            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.subheader("📊 MindPulse Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Mental Health Issue Probability",
                    f"{probability_percent:.2f}%"
                )

            with col2:

                st.metric(
                    "Decision Threshold",
                    f"{threshold:.2f}"
                )


            # ------------------------------------------------
            # RESULT MESSAGE
            # ------------------------------------------------

            if prediction == 1:

                st.error(
                    "⚠️ Higher Mental Health Risk Detected"
                )

                st.write(
                    "The model estimates a higher likelihood "
                    "of a mental-health issue based on the "
                    "information provided."
                )

            else:

                st.success(
                    "✅ Lower Mental Health Risk Detected"
                )

                st.write(
                    "The model estimates a lower likelihood "
                    "of a mental-health issue based on the "
                    "information provided."
                )


            # ------------------------------------------------
            # BEHAVIORAL FACTORS
            # ------------------------------------------------

            st.subheader("🔎 Behavioral Indicators")

            risk_factors = []

            if work_stress >= 7:
                risk_factors.append(
                    "High work stress"
                )

            if financial_stress >= 7:
                risk_factors.append(
                    "High financial stress"
                )

            if loneliness >= 7:
                risk_factors.append(
                    "High loneliness"
                )

            if social_support <= 3:
                risk_factors.append(
                    "Low social support"
                )

            if sleep_hours < 6:
                risk_factors.append(
                    "Low sleep duration"
                )

            if social_media >= 7:
                risk_factors.append(
                    "High social media usage"
                )


            if risk_factors:

                for factor in risk_factors:

                    st.warning(
                        f"• {factor}"
                    )

            else:

                st.success(
                    "No major behavioral risk indicators "
                    "were detected from the entered values."
                )


            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.info(
                "⚠️ MindPulse is an educational machine-learning "
                "project and is not a medical diagnostic tool. "
                "The prediction should not be used as a substitute "
                "for professional medical advice."
            )


        except Exception as e:

            st.error(
                "Unable to make prediction."
            )

            st.exception(e)


# ============================================================
# STUDENT
# ============================================================

else:

    st.header("🎓 Student Stress Analysis")

    st.info(
        "This section uses the student stress model "
        "developed earlier in the MindPulse project."
    )

    try:

        student_model, student_features = load_student_model()

        st.write(
            "Student model loaded successfully."
        )

        st.write(
            "Expected features:"
        )

        st.write(student_features)

        st.warning(
            "The student input interface will be added after "
            "we verify the exact feature names stored in "
            "student_features.pkl."
        )

    except Exception as e:

        st.error(
            "Student model could not be loaded."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MindPulse • Machine Learning Based Mental Health "
    "and Behavioral Risk Analysis"
)