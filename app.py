import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_disease_model.pkl")
st.title("Heart Disease Predictor")

age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0
chol = st.number_input("Cholesterol")
cp = st.selectbox("Chest pain type", [0, 1, 2, 3])
restecg = st.selectbox("Rest ECG", [0, 1, 2])

thalach = st.number_input("Maximum Heart Rate")
trestbps = st.number_input("Resting Blood Pressure")

fbs = st.selectbox("Fasting Blood Sugar ", [0, 1])
exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("Old Peak", value=0.0)

slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])

thal = st.selectbox("Thal", [0, 1, 2, 3])
# st.write("Age:", age)
# st.write("Sex:", sex)
# st.write("Cholesterol:", chol)


if st.button("Predict"):
    features = np.array(
        [
            [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ]
    )

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
