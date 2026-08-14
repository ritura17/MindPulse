import streamlit as st

# ==========================================================
# MINDPULSE - STRESS & LIFESTYLE ANALYZER
# ==========================================================

st.set_page_config(
    page_title="MindPulse",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

st.title("🧠 MindPulse")
st.subheader("Lifestyle & Digital Habit Analyzer")

st.write(
    "Analyze your daily habits and lifestyle to estimate "
    "stress, anxiety, and headache risk."
)

st.info(
    "⚠️ This application is for educational and wellness purposes. "
    "It does not provide medical diagnosis."
)

# ----------------------------------------------------------
# Personal Information
# ----------------------------------------------------------

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=13,
        max_value=100,
        value=20
    )

with col2:
    employment = st.selectbox(
        "Current Status",
        [
            "Student",
            "Employed",
            "Self-employed",
            "Unemployed"
        ]
    )

# ----------------------------------------------------------
# Study / Work
# ----------------------------------------------------------

st.header("📚 Study & Work")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

with col2:
    work_hours = st.number_input(
        "Work Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=0.0,
        step=0.5
    )

with col3:
    breaks = st.number_input(
        "Breaks per Day",
        min_value=0,
        max_value=30,
        value=3
    )

# ----------------------------------------------------------
# Sleep & Physical Activity
# ----------------------------------------------------------

st.header("😴 Sleep & Physical Activity")

col1, col2, col3 = st.columns(3)

with col1:
    sleep_hours = st.number_input(
        "Sleep Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

with col2:
    exercise_hours = st.number_input(
        "Exercise Hours per Day",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.5
    )

with col3:
    water = st.number_input(
        "Water Intake (Litres)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    )

# ----------------------------------------------------------
# Digital Habits
# ----------------------------------------------------------

st.header("📱 Digital Habits")

col1, col2, col3 = st.columns(3)

with col1:
    screen_time = st.number_input(
        "Total Screen Time (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5
    )

with col2:
    social_media = st.number_input(
        "Social Media Usage (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

with col3:
    late_night_phone = st.selectbox(
        "Late-Night Phone Usage",
        ["Never", "Sometimes", "Frequently"]
    )

# ----------------------------------------------------------
# Analyze Button
# ----------------------------------------------------------

st.divider()

if st.button("🔍 Analyze My Habits", use_container_width=True):

    st.success("Your information has been collected successfully!")

    st.write("### 📊 Your Current Inputs")

    data = {
        "Age": age,
        "Status": employment,
        "Study Hours": study_hours,
        "Work Hours": work_hours,
        "Breaks": breaks,
        "Sleep Hours": sleep_hours,
        "Exercise Hours": exercise_hours,
        "Water Intake": water,
        "Screen Time": screen_time,
        "Social Media": social_media,
        "Late-Night Phone Usage": late_night_phone
    }

    st.json(data)

    st.warning(
        "🤖 ML prediction will be added in the next stages."
    )