import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fare_model.pkl")
columns = joblib.load("model_columns.pkl")
opts = joblib.load("options.pkl")

def predict_fare(airline, journey_date, source, destination):
    row = pd.DataFrame([{
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Journey_Day": journey_date.day,
        "Journey_Month": journey_date.month,
        "Journey_Weekday": journey_date.weekday(),
    }])
    row = pd.get_dummies(row, columns=["Airline", "Source", "Destination"])
    row = row.reindex(columns=columns, fill_value=0)
    return float(model.predict(row)[0])

st.title("✈️ Flight Fare Predictor")

airline = st.selectbox("Airline", opts["airlines"])
source = st.selectbox("From", opts["sources"])
destination = st.selectbox("To", opts["destinations"])
date = st.date_input("Journey date")

if st.button("Predict fare"):
    price = predict_fare(airline, date, source, destination)
    st.success(f"Estimated fare: Rs. {price:,.0f}")