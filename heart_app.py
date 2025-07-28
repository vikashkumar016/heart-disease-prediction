import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💓 Heart Disease Prediction App")
st.write("Enter patient details below to check the likelihood of heart disease.")

# Input fields
age = st.slider("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

# Categorical selections
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
thal = st.selectbox("Thalassemia", [1, 2, 3])

# One-hot encode manually
cp_1, cp_2, cp_3 = int(cp == 1), int(cp == 2), int(cp == 3)
restecg_1, restecg_2 = int(restecg == 1), int(restecg == 2)
slope_1, slope_2 = int(slope == 1), int(slope == 2)
thal_1, thal_2, thal_3 = int(thal == 1), int(thal == 2), int(thal == 3)

# Prepare input for prediction
input_data = np.array([[ 
    age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca,
    cp_1, cp_2, cp_3,
    restecg_1, restecg_2,
    slope_1, slope_2,
    thal_1, thal_2, thal_3
]])

# Prediction button
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("❗ The person is likely to have heart disease.")
    else:
        st.success("✅ The person is unlikely to have heart disease.")
