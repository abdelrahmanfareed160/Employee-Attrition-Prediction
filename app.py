import joblib  # to load your model
import streamlit as st

import pandas as pd
from sklearn import preprocessing

import os
os.environ["STREAMLIT_SERVER_PORT"] = "8000"
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"

# Load your trained model
model = joblib.load('ada_model.pkl')

# Title
st.title("Employee Attrition Predictor")
st.write("Predict if an employee is likely to leave based on key factors.")

# Sidebar inputs
st.sidebar.header("Enter Employee Details:")

# Personal Info
st.sidebar.subheader("🔹 Personal Info")
age = st.sidebar.slider("Age", 18, 60, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 0)

# Job Info
st.sidebar.subheader("🔹 Job Info")
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])
company_tenure = st.sidebar.number_input("Company Tenure (Years)", 0, 50, 5)
company_reputation = st.sidebar.slider("Company Reputation (1-5)", 1, 5, 3)

# Work Factors
st.sidebar.subheader("🔹 Work Factors")
work_life_balance = st.sidebar.selectbox("Work-Life Balance", [1, 2, 3, 4],
                                         format_func=lambda x: {1: "Bad", 2: "Average", 3: "Good", 4: "Excellent"}[x])
job_satisfaction = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
performance_rating = st.sidebar.selectbox("Performance Rating", [1, 2, 3, 4])
promotions = st.sidebar.number_input("Number of Promotions", 0, 10, 0)
overtime = st.sidebar.checkbox("Overtime")
remote_work = st.sidebar.checkbox("Remote Work")
distance_from_home = st.sidebar.slider("Distance from Home (km)", 0, 100, 10)

# Education
st.sidebar.subheader("🔹 Education")
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor’s Degree", "Master’s", "PhD"])

# Process inputs into dataframe
# Updated preprocess_inputs function


def preprocess_inputs():
    # Initialize the label encoder
    le = preprocessing.LabelEncoder()

    # Collect user inputs into a DataFrame
    df = pd.DataFrame({
        'Age': [age],
        'Years at Company': [years_at_company],
        'Work-Life Balance': [work_life_balance],
        'Job Satisfaction': [job_satisfaction],
        'Performance Rating': [performance_rating],
        'Number of Promotions': [promotions],
        'Overtime': [1 if overtime else 0],
        'Distance from Home': [distance_from_home],
        'Marital Status': [marital_status],
        'Number of Dependents': [dependents],
        'Job Level': [job_level],
        'Company Size': [company_size],
        'Company Tenure': [company_tenure],
        'Innovation Opportunities': [3],  # Placeholder for this feature
        'Company Reputation': [company_reputation],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Education Level_Bachelor’s Degree': [1 if education_level == "Bachelor’s Degree" else 0],
        'Education Level_PhD': [1 if education_level == "PhD" else 0],
        'Remote Work_Yes': [1 if remote_work else 0]
    })

    # List of categorical columns to be label-encoded (excluding already one-hot encoded columns)
    cat_col = ['Marital Status', 'Company Size', 'Job Level']
    for col in cat_col:
        df[col] = le.fit_transform(df[col])

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Gender_Male", "Job Level", "Education Level_Bachelor’s Degree",
                                     "Education Level_PhD", "Remote Work_Yes"], drop_first=True)

    # Add engineered features (same as you did during training)
    df['Tenure_Per_Dependents'] = df['Years at Company'] / (df['Number of Dependents'] + 1)
    df['Tenure_Per_Promotion'] = df['Years at Company'] / (df['Number of Promotions'] + 1)
    df['Performance_Dependents_Ratio'] = df['Performance Rating'] / (df['Number of Dependents'] + 1)
    df['Performance_Tenure_Ratio'] = df['Performance Rating'] / (df['Years at Company'] + 1)
    df['Satisfaction_Tenure_Ratio'] = df['Job Satisfaction'] / (df['Years at Company'] + 1)
    df['WorkLife_Satisfaction_Interaction'] = df['Work-Life Balance'] * df['Job Satisfaction']
    df['WorkLife_Tenure_Ratio'] = df['Work-Life Balance'] / (df['Years at Company'] + 1)
    df['Age_Company_Ratio'] = df['Years at Company'] / (df['Age'] + 1)
    df['Age_Tenure_Ratio'] = df['Years at Company'] / (df['Age'] + 1)

    # Define model's expected feature names (training columns)
    model_columns = [
        'Age', 'Years at Company', 'Work-Life Balance', 'Job Satisfaction',
        'Performance Rating', 'Number of Promotions', 'Overtime', 'Distance from Home',
        'Marital Status', 'Number of Dependents', 'Job Level', 'Company Size',
        'Company Tenure', 'Innovation Opportunities', 'Company Reputation', 'Gender_Male',
        'Education Level_Bachelor’s Degree', 'Education Level_PhD', 'Remote Work_Yes',
        'Tenure_Per_Dependents', 'Tenure_Per_Promotion', 'Performance_Dependents_Ratio',
        'Performance_Tenure_Ratio', 'Satisfaction_Tenure_Ratio',
        'WorkLife_Satisfaction_Interaction', 'WorkLife_Tenure_Ratio',
        'Age_Company_Ratio', 'Age_Tenure_Ratio'
    ]

    # Align input data to match the model columns (if any features are missing)
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

# Predict button
if st.button("🚀 Predict Attrition Risk"):
    with st.spinner("Predicting..."):
        input_df = preprocess_inputs()
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100  # probability of attrition

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk — This employee is likely to leave.\nChance of leaving: {prob:.1f}%")
    else:
        st.success(f"✅ Low Attrition Risk — This employee is likely to stay.\nChance of leaving: {prob:.1f}%")


# Define variables (replace with your desired names and location)
RESOURCE_GROUP_NAME="AttritionPredictorRG"      # Choose a name for your resource group
LOCATION="eastus"                             # Choose an Azure region (e.g., eastus, westeurope, northeurope)
ACR_NAME="myattritionacr12345"                # Choose a GLOBALLY UNIQUE name for your ACR
APP_SERVICE_PLAN_NAME="AttritionPredictorPlan"  # Choose a name for your App Service Plan
WEB_APP_NAME="attrition-predictor-app-CSED26" # Choose a GLOBALLY UNIQUE name for your Web App
DOCKER_IMAGE_NAME="attrition-predictor"       # Name for your Docker image
DOCKER_IMAGE_TAG="latest"                     # Tag for your Docker image
