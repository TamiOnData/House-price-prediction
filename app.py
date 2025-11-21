import streamlit as st
import pickle
import numpy as np

linear_model = pickle.load(open("model.pkl", "rb"))
ridge_model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("House Price Prediction App")
st.write("Predict house prices using Linear Regression or Ridge Regression.")

model_choice = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Ridge Regression"]
)

longitude = st.number_input("Longitude", value=0.0)
latitude = st.number_input("Latitude", value=0.0)
housing_median_age = st.number_input("Housing Median Age", value=0.0)
total_rooms = st.number_input("Total Rooms", value=0.0)
total_bedrooms = st.number_input("Total Bedrooms", value=0.0)
population = st.number_input("Population", value=0.0)
households = st.number_input("Households", value=0.0)
median_income = st.number_input("Median Income", value=0.0)

features = np.array([[
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    population, households, median_income
]])

if st.button("Predict Price"):
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(features)[0]
    else:
        scaled_features = scaler.transform(features)
        prediction = ridge_model.predict(scaled_features)[0]

    st.success(f"Predicted House Price: ${prediction:,.2f}")