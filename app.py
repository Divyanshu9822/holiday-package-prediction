from flask import Flask, render_template, request
import pickle
from config import Config
import pandas as pd
import os

app = Flask(__name__)
app.config.from_object(Config)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            typeofcontact = request.form["typeofcontact"]
            citytier = int(request.form["citytier"])
            durationofpitch = float(request.form["durationofpitch"])
            occupation = request.form["occupation"]
            gender = request.form["gender"]
            numberoffollowups = float(request.form["numberoffollowups"])
            productpitched = request.form["productpitched"]
            preferredpropertystar = float(request.form["preferredpropertystar"])
            maritalstatus = request.form["maritalstatus"]
            numberoftrips = float(request.form["numberoftrips"])
            passport = int(request.form["passport"])
            pitchsatisfactionscores = int(request.form["pitchsatisfactionscores"])
            owncar = int(request.form["owncar"])
            designation = request.form["designation"]
            monthlyincome = float(request.form["monthlyincome"])
            totalvisiting = float(request.form["totalvisiting"])

            xgboost_classfier_path = os.path.join(
                os.path.dirname(__file__), "models/xgboost_classifier.pkl"
            )
            preprocessor_model_path = os.path.join(
                os.path.dirname(__file__), "models/preprocessor.pkl"
            )
            print(f"XGBoost Classifier Path: {xgboost_classfier_path}")
            print(f"Preprocessor Path: {preprocessor_model_path}")
            try:
                with open(preprocessor_model_path, "rb") as preprocessor_file:
                    preprocessor_model = pickle.load(preprocessor_file)
                with open(xgboost_classfier_path, "rb") as xgboost_file:
                    xgboost_classfier_model = pickle.load(xgboost_file)
            except FileNotFoundError as fnf_error:
                print(f"File not found: {fnf_error}")
                return render_template(
                    "index.html", prediction="Error: Model file not found."
                )
            except pickle.UnpicklingError as pickle_error:
                print(f"Error unpickling file: {pickle_error}")
                return render_template(
                    "index.html", prediction="Error: Corrupted model file."
                )
            except Exception as e:
                print(f"Error loading models: {e}")
                return render_template(
                    "index.html", prediction="Error: Unable to load models."
                )

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
                    passport,
                    pitchsatisfactionscores,
                    owncar,
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

            prediction = xgboost_classfier_model.predict(input_data_transformed)[0]

            print("Received input values:")
            print(f"Age: {age}")
            print(f"Type of Contact: {typeofcontact}")
            print(f"City Tier: {citytier}")
            print(f"Duration of Pitch: {durationofpitch}")
            print(f"Occupation: {occupation}")
            print(f"Gender: {gender}")
            print(f"Number of Follow-ups: {numberoffollowups}")
            print(f"Product Pitched: {productpitched}")
            print(f"Preferred Property Star: {preferredpropertystar}")
            print(f"Marital Status: {maritalstatus}")
            print(f"Number of Trips: {numberoftrips}")
            print(f"Passport: {passport}")
            print(f"Pitch Satisfaction Score: {pitchsatisfactionscores}")
            print(f"Own Car: {owncar}")
            print(f"Designation: {designation}")
            print(f"Monthly Income: {monthlyincome}")
            print(f"Total Visiting: {totalvisiting}")
            print(f"Prediction: {prediction}")

            if prediction == 1:
                prediction_message = (
                    "The customer is likely to purchase the travel package."
                )
            else:
                prediction_message = (
                    "The customer is unlikely to purchase the travel package."
                )

            print(f"Prediction: {prediction}")
            print(f"Prediction Message: {prediction_message}")

            return render_template("index.html", prediction=prediction_message)

        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template(
                "index.html", prediction="Error: Unable to make prediction."
            )
    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
