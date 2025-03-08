import streamlit as st
import requests
import pandas as pd
import numpy as np

# URL of the FastAPI backend
API_URL = "http://localhost:8000"

# Define the retraining form
st.header("Model Retraining")

with st.form("retrain_form"):
    st.write("Adjust model hyperparameters")

    n_estimators = st.number_input(
        "Number of estimators", min_value=50, max_value=500, value=100
    )
    max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=10)
    random_state = st.number_input("Random state", min_value=0, max_value=100, value=42)

    retrain_button = st.form_submit_button("Retrain Model")

    if retrain_button:
        retrain_payload = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }

        response = requests.post(f"{API_URL}/retrain", json=retrain_payload)

        if response.status_code == 200:
            st.success("Model retrained successfully!")
            st.json(response.json())
        else:
            st.error(f"Error: {response.text}")

# Define the prediction form
st.header("Churn Prediction")

with st.form("predict_form"):
    st.write("Enter input data for prediction")

    # Input fields for prediction
    Account_Length = st.number_input(
        "Account Length", min_value=0, max_value=500, value=100
    )
    Area_Code = st.number_input("Area Code", min_value=0, max_value=1000, value=415)
    Customer_Service_Calls = st.number_input(
        "Customer Service Calls", min_value=0, max_value=50, value=5
    )

    # Map International_Plan and Voicemail_Plan to integers
    International_Plan = st.selectbox(
        "International Plan", ["IN", "US"]
    )  # Example values
    Voicemail_Plan = st.selectbox("Voicemail Plan", ["Yes", "No"])  # Example values

    International_Plan = (
        1 if International_Plan == "IN" else 0
    )  # Map "IN" to 1 and "US" to 0
    Voicemail_Plan = 1 if Voicemail_Plan == "Yes" else 0  # Map "Yes" to 1 and "No" to 0

    Number_of_Voicemail_Messages = st.number_input(
        "Number of Voicemail Messages", min_value=0, max_value=100, value=10
    )
    Total_Day_Calls = st.number_input(
        "Total Day Calls", min_value=0, max_value=1000, value=100
    )
    Total_Day_Charge = st.number_input(
        "Total Day Charge", min_value=0.0, max_value=100.0, value=20.5
    )
    Total_Day_Minutes = st.number_input(
        "Total Day Minutes", min_value=0.0, max_value=200.0, value=50.0
    )
    Total_Night_Calls = st.number_input(
        "Total Night Calls", min_value=0, max_value=1000, value=100
    )
    Total_Night_Charge = st.number_input(
        "Total Night Charge", min_value=0.0, max_value=100.0, value=10.5
    )
    Total_Night_Minutes = st.number_input(
        "Total Night Minutes", min_value=0.0, max_value=200.0, value=50.0
    )
    Total_Evening_Calls = st.number_input(
        "Total Evening Calls", min_value=0, max_value=1000, value=100
    )
    Total_Evening_Charge = st.number_input(
        "Total Evening Charge", min_value=0.0, max_value=100.0, value=15.5
    )
    Total_Evening_Minutes = st.number_input(
        "Total Evening Minutes", min_value=0.0, max_value=200.0, value=60.0
    )
    International_Calls = st.number_input(
        "International Calls", min_value=0, max_value=1000, value=50
    )
    Extra_Feature_1 = st.number_input(
        "Extra Feature 1", min_value=0.0, max_value=100.0, value=5.0
    )
    Extra_Feature_2 = st.number_input(
        "Extra Feature 2", min_value=0.0, max_value=100.0, value=5.0
    )
    Extra_Feature_3 = st.number_input(
        "Extra Feature 3", min_value=0.0, max_value=100.0, value=5.0
    )

    predict_button = st.form_submit_button("Predict")

    if predict_button:
        predict_payload = {
            "Account_Length": Account_Length,
            "Area_Code": Area_Code,
            "Customer_Service_Calls": Customer_Service_Calls,
            "International_Plan": International_Plan,
            "Voicemail_Plan": Voicemail_Plan,
            "Number_of_Voicemail_Messages": Number_of_Voicemail_Messages,
            "Total_Day_Calls": Total_Day_Calls,
            "Total_Day_Charge": Total_Day_Charge,
            "Total_Day_Minutes": Total_Day_Minutes,
            "Total_Night_Calls": Total_Night_Calls,
            "Total_Night_Charge": Total_Night_Charge,
            "Total_Night_Minutes": Total_Night_Minutes,
            "Total_Evening_Calls": Total_Evening_Calls,
            "Total_Evening_Charge": Total_Evening_Charge,
            "Total_Evening_Minutes": Total_Evening_Minutes,
            "International_Calls": International_Calls,
            "Extra_Feature_1": Extra_Feature_1,
            "Extra_Feature_2": Extra_Feature_2,
            "Extra_Feature_3": Extra_Feature_3,
        }

        response = requests.post(f"{API_URL}/predict", json=predict_payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.subheader("Features used for prediction:")
            st.json(result["features"])
        else:
            st.error(f"Error: {response.text}")
