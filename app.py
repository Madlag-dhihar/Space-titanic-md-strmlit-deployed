import streamlit as st
import pickle
import pandas as pd

st.title("ASG 04 MD - Shafi - Spaceship Titanic Model Deployment")


model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("preprocessor.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))


HomePlanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
VIP = st.selectbox("VIP", [True, False])

Age = st.number_input("Age", value=30)

RoomService = st.number_input("RoomService", value=0)
FoodCourt = st.number_input("FoodCourt", value=0)
ShoppingMall = st.number_input("ShoppingMall", value=0)
Spa = st.number_input("Spa", value=0)
VRDeck = st.number_input("VRDeck", value=0)

Deck = st.selectbox("Deck", ["A","B","C","D","E","F","G"])
Side = st.selectbox("Side", ["P","S"])

Cabin_num = st.number_input("Cabin Number", value=100)

Group_size = st.number_input("Group Size", value=1)
Family_size = st.number_input("Family Size", value=1)

if st.button("Predict"):

    
    TotalSpending = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck

    HasSpending = 1 if TotalSpending > 0 else 0
    NoSpending = 1 if TotalSpending == 0 else 0

    Age_missing = 0
    CryoSleep_missing = 0

    Solo = 1 if Group_size == 1 else 0


    if Age <= 12:
        Age_group = "Child"
    elif Age <= 18:
        Age_group = "Teen"
    elif Age <= 30:
        Age_group = "Young_Adult"
    elif Age <= 50:
        Age_group = "Adult"
    else:
        Age_group = "Senior"


    RoomService_ratio = RoomService / (TotalSpending + 1)
    FoodCourt_ratio = FoodCourt / (TotalSpending + 1)
    ShoppingMall_ratio = ShoppingMall / (TotalSpending + 1)
    Spa_ratio = Spa / (TotalSpending + 1)
    VRDeck_ratio = VRDeck / (TotalSpending + 1)

 
    data = pd.DataFrame([{
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep,
        "Destination": Destination,
        "VIP": VIP,
        "Deck": Deck,
        "Side": Side,
        "Age_group": Age_group,

        "Age": Age,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall": ShoppingMall,
        "Spa": Spa,
        "VRDeck": VRDeck,
        "Cabin_num": Cabin_num,
        "Group_size": Group_size,
        "Solo": Solo,
        "Family_size": Family_size,
        "TotalSpending": TotalSpending,
        "HasSpending": HasSpending,
        "NoSpending": NoSpending,
        "Age_missing": Age_missing,
        "CryoSleep_missing": CryoSleep_missing,

        "RoomService_ratio": RoomService_ratio,
        "FoodCourt_ratio": FoodCourt_ratio,
        "ShoppingMall_ratio": ShoppingMall_ratio,
        "Spa_ratio": Spa_ratio,
        "VRDeck_ratio": VRDeck_ratio
    }])

 

    for col, encoder in label_encoders.items():
        data[col] = encoder.transform(data[col].astype(str))

 
    data = data[feature_columns]

   

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("Passenger was Transported ")
    else:
        st.error("Passenger was NOT Transported ")