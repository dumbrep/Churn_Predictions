import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scalling.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff6f61;
        text-align: center;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: grey;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title of the app
st.title('Customer Churn Prediction')

# Input fields grouped using columns for better layout
with st.form("customer_form"):
    st.header("Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider("Age", 18, 100)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850)

    with col2:
        balance = st.number_input('Balance', min_value=0.0)
        estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
        tenure = st.slider('Tenure', 0, 10)
        num_of_products = st.slider('Number of Products', 1, 4)

    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    submitted = st.form_submit_button("Submit")

# Prepare input data once the form is submitted
if submitted:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': label_encoder_gender.transform([[gender][0]]),
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out())

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict([input_data_scaled])
    prediction_proba = prediction[0][0]

    # Output section
    st.header("Prediction Result")
    if prediction_proba > 0.5:
        st.success(f"The Customer is likely to churn ({prediction_proba:.2%} probability).")
    else:
        st.success(f"The Customer is not likely to churn ({prediction_proba:.2%} probability).")

    

# Footer with additional info
st.markdown(
    """
    <div class="footer">
        <p>Customer Churn Prediction App - Powered by Artificial Neural Network</p>
    </div>
    """, 
    unsafe_allow_html=True
)
