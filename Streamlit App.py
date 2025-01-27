# importing libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit

# loading scaler and ridge regression model
scaler = joblib.load('scaler.joblib')
model = joblib.load('Ridge Regression Model.joblib')

# defining functions to make predictions
def predict_medical_cost(age, sex, bmi, children, smoker, region):
    try:
        age = int(age)
        bmi = float(bmi)
        children = int(children)

        input_data = pd.DataFrame({
            'age':[age],
            'sex_male':[1 if sex == 'male' else 0],
            'bmi':[bmi],
            'children':[children],
            'smoker_yes':[1 if smoker == 'yes' else 0],
            'region_northeast':[1 if region == 'northeast' else 0],
            'region_northwest':[1 if region == 'northwest' else 0],
            'region_southeast':[1 if region == 'southeast' else 0],
            'region_southwest':[1 if region == 'southwest' else 0]
        })

        scaled_input_data = scaler.transform(input_data)
        
        predicted_log_charges = model.predict(scaled_input_data)

        predicted_charges = np.exp(predicted_log_charges)

        return predicted_charges[0]

    except Exception as e:
        print('Error! Please try again.')

# streamlit app interface
streamlit.title('Medical Cost Prediction')

# user inputs
age = streamlit.number_input('Age', min_value = 0, max_value = 120, value = 18)
sex = streamlit.selectbox('Sex', ['male', 'female'])
bmi = streamlit.number_input('BMI', min_value = 10, max_value = 50, value = 25)
children = streamlit.number_input('Children', min_value = 0, max_value = 10, value = 0)
smoker = streamlit.selectbox('Smoker', ['yes', 'no'])
region = streamlit.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

if streamlit.button('Predict'):
    predict_charges = predict_medical_cost(age, sex, bmi, children, smoker, region)
    streamlit.write(f'Predicted Medical Cost: ${round(predict_charges, 2)}')