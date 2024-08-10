import streamlit as st
import pickle
import pandas as pd

xgboost_classifier_path = "models/xgboost_classifier.pkl"
preprocessor_model_path = "models/preprocessor.pkl"

try:
    with open(preprocessor_model_path, "rb") as preprocessor_file:
        preprocessor_model = pickle.load(preprocessor_file)
    with open(xgboost_classifier_path, "rb") as xgboost_file:
        xgboost_classifier_model = pickle.load(xgboost_file)
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

st.title("Holiday Package Prediction")

age = st.slider("Age", 18, 100, value=30)

typeofcontact = st.radio(
    "Type of Contact", options=["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox("City Tier", options=[1, 2, 3])

durationofpitch = st.slider("Duration of Pitch (minutes)", 0, 60, value=30)

occupation = st.selectbox(
    "Occupation",
    options=["Salaried", "Free Lancer", "Small Business", "Large Business"],
)

gender = st.radio("Gender", options=["Male", "Female"])

numberoffollowups = st.number_input(
    "Number of Follow-ups", min_value=0.0, step=1.0, value=2.0
)

productpitched = st.selectbox(
    "Product Pitched",
    options=["Deluxe", "Basic", "Standard", "Super Deluxe", "King"],
)

preferredpropertystar = st.select_slider(
    "Preferred Property Star", options=[3.0, 4.0, 5.0], value=4.0
)

maritalstatus = st.radio(
    "Marital Status", options=["Unmarried", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips", min_value=0.0, step=1.0, value=1.0
)

passport = st.radio("Passport", options=["Yes", "No"])

pitchsatisfactionscores = st.select_slider(
    "Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], value=3
)

owncar = st.radio("Own Car", options=["Yes", "No"])

designation = st.selectbox(
    "Designation", options=["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income (in currency)", min_value=0.0, step=1000.0, value=50000.0
)

totalvisiting = st.number_input(
    "Total Number of People Visiting", min_value=0, step=1, value=1
)

if st.button("Predict"):
    try:
        input_data = [
            [
                age,
                typeofcontact,
                citytier,
                durationofpitch,
                occupation,
                gender,
                numberoffollowups,
                productpitched,
                preferredpropertystar,
                maritalstatus,
                numberoftrips,
                1 if passport == "Yes" else 0,
                pitchsatisfactionscores,
                1 if owncar == "Yes" else 0,
                designation,
                monthlyincome,
                totalvisiting,
            ]
        ]

        columns = [
            "Age",
            "TypeofContact",
            "CityTier",
            "DurationOfPitch",
            "Occupation",
            "Gender",
            "NumberOfFollowups",
            "ProductPitched",
            "PreferredPropertyStar",
            "MaritalStatus",
            "NumberOfTrips",
            "Passport",
            "PitchSatisfactionScore",
            "OwnCar",
            "Designation",
            "MonthlyIncome",
            "TotalVisiting",
        ]

        input_data_transformed = preprocessor_model.transform(
            pd.DataFrame(input_data, columns=columns)
        )

        prediction = xgboost_classifier_model.predict(input_data_transformed)[0]

        if prediction == 1:
            st.success("The customer is likely to purchase the travel package.")
        else:
            st.warning("The customer is unlikely to purchase the travel package.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
