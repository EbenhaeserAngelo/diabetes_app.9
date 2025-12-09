import streamlit as st 
import pandas as pd 
import joblib 
import matplotlib.pyplot as plt

#load model
model = joblib.load('/home/ebeanski/Streamlit/Diabetes_app/diabetes_model.pkl')

#App title and intro
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.write("Fill in the details below to predict the likelihood of diabetes.")

#Input form
with st.form("Prediction Form"):
    st.subheader("Enter Patient Details")
    
    col1,col2 =st.columns(2)
    
    with col1:
        pregnancies = st.slider("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.slider("Glucose Level", min_value=0, max_value=200, value=100)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.slider("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.slider("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.selectbox("Age group",options=[i for i in range(18,81,1)], index=2)
        
        
    submitted = st.form_submit_button("Predict")
    if submitted:
        #Prepare input data for prediction
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        #Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction_proba = round(prediction_proba, 2)
        
        #Display result
        if prediction[0] == 1:
            st.error(f"The model predicts that the patient is likely to have diabetes. (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"The model predicts that the patient is unlikely to have diabetes. (Probability: {1 - prediction_proba:.2f})")
        st.balloons()
        st.snow()
        st.success("Thank you for using the Diabetes Prediction App!")

import time
progress = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress.progress(i + 1)
    
with st.spinner("Predicting..."):
    time.sleep(2)
st.success("Prediction complete!")
