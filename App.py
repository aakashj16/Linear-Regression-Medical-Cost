import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit

# Load scaler and Ridge Regression model
scaler = joblib.load('scaler.joblib')
model = joblib.load('Ridge Regression Model.joblib')

def predict_medical_cost(age, sex, bmi, children, smoker, region):
    try:
        # Basic input validation
        if age < 0 or age > 120:
            raise ValueError("Age must be between 0 and 120.")
        if bmi < 10 or bmi > 50:
            raise ValueError("BMI must be between 10 and 50.")
        if children < 0 or children > 10:
            raise ValueError("Number of children must be between 0 and 10.")

        input_data = pd.DataFrame({
            'age': [age],
            'sex_male': [1 if sex == 'male' else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northeast': [1 if region == 'northeast' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        })

        scaled_input_data = scaler.transform(input_data)
        predicted_log_charges = model.predict(scaled_input_data)
        predicted_charges = np.exp(predicted_log_charges)

        return predicted_charges[0]

    except ValueError as e:
        print(f"ValueError occurred: {e}")
        return None

# Streamlit app interface
streamlit.title('Medical Cost Prediction')

# User inputs
age = streamlit.number_input('Age', min_value=0, max_value=120, value=18)
sex = streamlit.selectbox('Sex', ['male', 'female'])
bmi = streamlit.number_input('BMI', min_value=10, max_value=50, value=25)
children = streamlit.number_input('Children', min_value=0, max_value=10, value=0)
smoker = streamlit.selectbox('Smoker', ['yes', 'no'])
region = streamlit.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

if streamlit.button('Predict'):
    predict_charges = predict_medical_cost(age, sex, bmi, children, smoker, region)
    if predict_charges is not None:
        streamlit.write(f'Predicted Medical Cost: ${round(predict_charges, 2)}')
    else:
        streamlit.write("Prediction failed. Please check the input values.")
