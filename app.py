
import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st

try:
    with open("model.pkl", "rb") as f:
        model = pk.load(f)
    st.header("Car Price Prediction Model")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

df = pd.read_csv("Cardetails.csv")

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip(' ')

df['name'] = df['name'].apply(get_brand_name)

brand_list = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel']


name = st.selectbox("Select Car Brand", df['name'].unique())

year = st.slider("Car Manufactured Year", 1994, 2024, 2015)

km_driven = st.slider("No of Kms Driven", 11, 200000, 50000)

fuel = st.selectbox("Fuel Type", df['fuel'].unique())

seller_type = st.selectbox("Seller Type", df['seller_type'].unique())

transmission = st.selectbox("Transmission Type", df['transmission'].unique())

owner = st.selectbox("Ownership", df['owner'].unique())

mileage = st.slider("Car Mileage", 10, 40, 18)

engine = st.slider("Engine CC", 700, 5000, 1200)

max_power = st.slider("Max Power", 0, 200, 80)

seats = st.slider("No of seats", 5, 10, 5)

if st.button("Predict"):
    input_data_model = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner,
                                       mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
                 'mileage', 'engine', 'max_power', 'seats'])

    st.write("Raw input:")
    st.write(input_data_model)

    input_data_model['owner'] = input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5])

    input_data_model['fuel'] = input_data_model['fuel'].map({
        'Diesel': 1,
        'Petrol': 2,
        'LPG': 3,
        'CNG': 4
    })

    input_data_model['seller_type'] = input_data_model['seller_type'].map({
        'Individual': 1,
        'Dealer': 2,
        'Trustmark Dealer': 3
    })

    input_data_model['transmission'] = input_data_model['transmission'].replace(
        ['Manual', 'Automatic'], [1, 2])

    input_data_model['name'] = input_data_model['name'].replace(brand_list, list(range(1, len(brand_list) + 1)))

    st.write("Encoded input sent to the model:")
    st.write(input_data_model)

    car_price = model.predict(input_data_model)[0]


    car_price = max(0, car_price)

    st.markdown(f"### Predicted car price: ₹{car_price:,.2f}")

    



    