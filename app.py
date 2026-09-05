import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("trained_model.pkl")


model = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter the student's details to predict their final grade."
)

st.divider()


# --------------------------------------------------
# Student Information
# --------------------------------------------------

st.subheader("👨‍🎓 Student Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    study_time_hours = st.slider(
        "Study Time (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    previous_grade = st.slider(
        "Previous Grade",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


with col2:

    parental_education = st.selectbox(
        "Parental Education",
        [
            "High School",
            "Bachelor's Degree",
            "Masters",
            "PhD"
        ]
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    attendance_percent = st.slider(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )


# --------------------------------------------------
# Academic Information
# --------------------------------------------------

st.subheader("📚 Academic Information")

final_exam_score = st.slider(
    "Final Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)


# --------------------------------------------------
# Student Activities
# --------------------------------------------------

st.subheader("🌐 Activities & Access")

col1, col2, col3 = st.columns(3)

with col1:

    internet_access = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )


with col2:

    extracurricular_activities = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )


with col3:

    part_time_job = st.selectbox(
        "Part-Time Job",
        ["Yes", "No"]
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Final Grade",
    use_container_width=True
):

    # --------------------------------------------------
    # Convert Yes/No values to 1/0
    # --------------------------------------------------

    internet_access_encoded = (
        1 if internet_access == "Yes" else 0
    )

    extracurricular_activities_encoded = (
        1 if extracurricular_activities == "Yes" else 0
    )

    part_time_job_encoded = (
        1 if part_time_job == "Yes" else 0
    )


    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    input_data = pd.DataFrame({

        "study_time_hours": [study_time_hours],

        "gender": [gender],

        "previous_grade": [previous_grade],

        "parental_education": [parental_education],

        "internet_access": [internet_access_encoded],

        "extracurricular_activities": [
            extracurricular_activities_encoded
        ],

        "part_time_job": [part_time_job_encoded],

        "sleep_hours": [sleep_hours],

        "attendance_percent": [attendance_percent],

        "final_exam_score": [final_exam_score]

    })


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------
    # Grade Mapping
    # --------------------------------------------------

    grade_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "F"
    }


    # --------------------------------------------------
    # Convert Prediction to Grade
    # --------------------------------------------------

    predicted_grade = grade_mapping[int(prediction)]


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success(
        f"🎯 Predicted Final Grade: {predicted_grade}"
    )