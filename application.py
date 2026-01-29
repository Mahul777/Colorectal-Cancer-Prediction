# Importing the required libraries
import joblib
import pandas as pd
from flask import Flask, render_template, request
import numpy as np

# Creating the Flask application
app = Flask(__name__)

# Importing the model, scaler, features, label_encoders, and ohe_encoders from the processed folder
model = joblib.load("artifacts/processed/model.pkl")
scaler = joblib.load("artifacts/processed/scaler.pkl")
features = joblib.load("artifacts/processed/features.pkl")
label_encoders = joblib.load("artifacts/processed/label_encoders.pkl")
ohe_encoders = joblib.load("artifacts/processed/ohe_encoders.pkl")

# Defining the route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Defining the route for the form submission
@app.route("/predict", methods=["POST"])
def predict():
    # Getting the input values from the form
    healthcare_costs = request.form.get("healthcare_costs")
    tumor_size = request.form.get("tumor_size")
    treatment_type = request.form.get("treatment_type").strip()  # Stripping spaces
    diet_risk = request.form.get("diet_risk").strip()  # Stripping spaces
    diabetes = request.form.get("diabetes").strip()  # Stripping spaces
    age = request.form.get("age")
    survival_5_years = request.form.get("survival_5_years").strip()  # Stripping spaces
    mortality_rate = request.form.get("mortality_rate")

    # Checking if all the input values are present
    if not all(value for value in [healthcare_costs, tumor_size, treatment_type, diet_risk, diabetes, age, survival_5_years, mortality_rate]):
        return "Please fill in all the required fields."

    try:
        # Converting the input values to the appropriate data types
        healthcare_costs = float(healthcare_costs)
        tumor_size = float(tumor_size)
        treatment_type = str(treatment_type).strip()  # Ensure it's a string and remove extra spaces
        diet_risk = int(diet_risk)  # Ensure it's an integer
        diabetes = int(diabetes)  # Ensure it's an integer
        age = float(age)
        survival_5_years = int(survival_5_years)  # Ensure it's an integer
        mortality_rate = float(mortality_rate)
    except ValueError as e:
        return f"Error converting input values: {e}"

    # Checking if the input values are valid
    if not (0 <= healthcare_costs <= 1000000):
        return "Invalid Healthcare Costs. Please enter a valid amount."

    if not (0 <= tumor_size <= 1000):
        return "Invalid Tumor Size. Please enter a valid size in mm."

    if not (0 <= age <= 120):
        return "Invalid Age. Please enter a valid age between 0 and 120."

    if not (0 <= mortality_rate <= 1000):
        return "Invalid Mortality Rate. Please enter a valid mortality rate."

    # Print available categories for treatment_type (for debugging)
    print("Treatment_Type categories:", ohe_encoders["Treatment_Type"].categories_)

    # Strip spaces from the encoder categories to match input
    stripped_categories = [category.strip() for category in ohe_encoders["Treatment_Type"].categories_[0]]

    # Check if the treatment_type exists in the stripped categories
    if treatment_type not in stripped_categories:
        return "Invalid Treatment Type. Please select a valid treatment."

    # Apply one-hot encoding to the categorical features
    try:
        # We now use the stripped categories to check and encode
        treatment_type_encoded = ohe_encoders["Treatment_Type"].transform([[stripped_categories.index(treatment_type)]])  # Corrected the key to 'Treatment_Type'
    except KeyError:
        return "Error: 'Treatment_Type' key not found in encoder."
    
    try:
        if diet_risk not in ohe_encoders["Diet_Risk"].categories_[0]:
            return "Invalid Diet Risk. Please select a valid option."
        diet_risk_encoded = ohe_encoders["Diet_Risk"].transform([[diet_risk]])  # Corrected the key to 'Diet_Risk'
    except KeyError:
        return "Error: 'Diet_Risk' key not found in encoder."
    
    try:
        if diabetes not in ohe_encoders["Diabetes"].categories_[0]:
            return "Invalid Diabetes status. Please select a valid option."
        diabetes_encoded = ohe_encoders["Diabetes"].transform([[diabetes]])  # Corrected the key to 'Diabetes'
    except KeyError:
        return "Error: 'Diabetes' key not found in encoder."
    
    try:
        if survival_5_years not in ohe_encoders["Survival_5_years"].categories_[0]:
            return "Invalid Survival 5 years status. Please select a valid option."
        survival_5_years_encoded = ohe_encoders["Survival_5_years"].transform([[survival_5_years]])  # Corrected the key to 'Survival_5_years'
    except KeyError:
        return "Error: 'Survival_5_years' key not found in encoder."

    # Concatenating all the features (numerical + one-hot encoded categorical features)
    X_raw = pd.DataFrame(
        [[healthcare_costs, tumor_size, age, mortality_rate]],
        columns=['healthcare_costs', 'tumor_size', 'age', 'mortality_rate']
    )

    # Combine the numerical features and the encoded categorical features into one dataframe
    X = pd.concat([
        X_raw,
        pd.DataFrame(treatment_type_encoded, columns=ohe_encoders["Treatment_Type"].categories_[0]),
        pd.DataFrame(diet_risk_encoded, columns=ohe_encoders["Diet_Risk"].categories_[0]),
        pd.DataFrame(diabetes_encoded, columns=ohe_encoders["Diabetes"].categories_[0]),
        pd.DataFrame(survival_5_years_encoded, columns=ohe_encoders["Survival_5_years"].categories_[0])
    ], axis=1)

    # Now apply the scaler to the full feature set
    try:
        X_scaled = scaler.transform(X)
    except ValueError as e:
        return f"Error in scaling the data: {e}"

    # Predicting the outcome using the model
    try:
        y_pred = model.predict(X_scaled)
    except Exception as e:
        return f"Error in prediction: {e}"

    # Converting the predicted probability to a binary classification
    y_pred = y_pred.round()

    # Encoding the predicted binary classification to a string
    prediction = label_encoders["Survival_Prediction"].inverse_transform(y_pred)

    # Returning the prediction to the user
    return render_template("index.html", prediction=prediction[0])

# Running the Flask application
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)







