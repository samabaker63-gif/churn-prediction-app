import streamlit as st
import joblib
import pandas as pd

# Load the saved files
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
le_geo = joblib.load('le_geo.pkl')
le_gen = joblib.load('le_gen.pkl')

st.title("Customer Churn Prediction App")

# User inputs
credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", le_geo.classes_)
gender = st.selectbox("Gender", le_gen.classes_)
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 200000.0, 0.0)
num_of_products = st.number_input("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict"):
    # Prepare data for prediction
    input_data = pd.DataFrame([[credit_score, le_geo.transform([geography])[0], 
                                le_gen.transform([gender])[0], age, tenure, balance, 
                                num_of_products, has_cr_card, is_active_member, salary]],
                              columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    # Display result
    if prediction[0] == 1:
        st.error("The customer is likely to churn!")
    else:
        st.success("The customer is likely to stay.")
