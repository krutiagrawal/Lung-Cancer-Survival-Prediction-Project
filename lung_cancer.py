import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import requests


if 'model' not in st.session_state:
    st.session_state.model = joblib.load("lung_cancer_model.pkl")
if 'scaler' not in st.session_state:
    st.session_state.scaler = joblib.load("scaler.pkl")


@st.cache_data
def get_countries():
    response = requests.get("https://restcountries.com/v3.1/all")
    countries = sorted([country["name"]["common"] for country in response.json()])
    return countries


st.title("Lung Cancer Survival Prediction")


st.header("Enter Patient Details")
age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.selectbox("Gender", ("Male", "Female"))
country = st.selectbox("Country", get_countries())
diagnosis_date = st.date_input("Diagnosis Date")
cancer_stage = st.selectbox("Cancer Stage", ("Stage I", "Stage II", "Stage III", "Stage IV"))
family_history = st.selectbox("Family History of Cancer", ("Yes", "No"))
smoking_status = st.selectbox("Smoking Status", ("Current smoker", "Former smoker", "Never smoked", "Passive smoker"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
cholesterol_level = st.number_input("Cholesterol Level", min_value=60, max_value=300, value=180)
hypertension = st.selectbox("Hypertension", ("Yes", "No"))
asthma = st.selectbox("Asthma", ("Yes", "No"))
cirrhosis = st.selectbox("Cirrhosis", ("Yes", "No"))
other_cancer = st.selectbox("Other Cancer", ("Yes", "No"))
treatment_type = st.selectbox("Treatment Type", ("Surgery", "Chemotherapy", "Radiation", "Combined"))
end_treatment_date = st.date_input("End Treatment Date")


treatment_duration = (end_treatment_date - diagnosis_date).days


if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        'age': [age],
        'gender': [0 if gender == "Male" else 1],
        'cancer_stage': [0 if cancer_stage == "Stage I" else 1 if cancer_stage == "Stage II" else 2 if cancer_stage == "Stage III" else 3],
        'family_history': [1 if family_history == "Yes" else 0],
        'smoking_status': [0 if smoking_status == "Current smoker" else 1 if smoking_status == "Former smoker" else 2 if smoking_status == "Never smoked" else 3],
        'bmi': [bmi],
        'cholesterol_level': [cholesterol_level],
        'hypertension': [1 if hypertension == "Yes" else 0],
        'asthma': [1 if asthma == "Yes" else 0],
        'cirrhosis': [1 if cirrhosis == "Yes" else 0],
        'other_cancer': [1 if other_cancer == "Yes" else 0],
        'treatment_type': [0 if treatment_type == "Surgery" else 1 if treatment_type == "Chemotherapy" else 2 if treatment_type == "Radiation" else 3],
        'treatment_duration': [treatment_duration]
    })


    input_data_scaled = st.session_state.scaler.transform(input_data)


    prediction_proba = st.session_state.model.predict_proba(input_data_scaled)


    threshold = 0.6

    if prediction_proba[0][1] >= threshold:
        survival_status = "Chances of Survival are High"
    else:
        survival_status = "Chances of Survival are Low"


    st.subheader("Prediction")
    st.write(survival_status)
    st.write(f"Probability of Survival: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Not Surviving: {prediction_proba[0][0]:.2f}")
