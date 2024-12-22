from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model and column transformer
with open('model_files/model_log.pkl', 'rb') as model_file:
    model_log = pickle.load(model_file)

with open('model_files/column_transformer.pkl', 'rb') as transformer_file:
    clmn = pickle.load(transformer_file)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect the data from the form
            shipment_id = request.form.get("shipment_id", "")  # Not used in prediction
            origin = request.form.get("origin", "")
            destination = request.form.get("destination", "")
            shipment_date = request.form["shipment_date"]
            planned_delivery_date = request.form["planned_delivery_date"]
            vehicle_type = request.form["vehicle_type"]
            distance = float(request.form["distance"])
            weather_conditions = request.form["weather_conditions"]
            traffic_conditions = request.form["traffic_conditions"]

            # Convert the date strings to datetime format
            shipment_date = pd.to_datetime(shipment_date, errors='coerce')
            planned_delivery_date = pd.to_datetime(planned_delivery_date, errors='coerce')

            # Validate date conversion
            if pd.isnull(shipment_date) or pd.isnull(planned_delivery_date):
                return render_template("index.html", prediction="Invalid Date Input")

            # Calculate "Planned Shipment Gap (days)"
            planned_shipment_gap = (planned_delivery_date - shipment_date).days

            # Prepare a DataFrame with the input data
            input_data = pd.DataFrame({
                'Weather Conditions': [weather_conditions],
                'Traffic Conditions': [traffic_conditions],
                'Vehicle Type': [vehicle_type],
                'Distance (km)': [distance],
                'Planned Shipment Gap (days)': [planned_shipment_gap],
            })

            # Apply the same transformations used during training
            X_transformed = clmn.transform(input_data)

            # Use the trained model to make a prediction
            prediction = model_log.predict(X_transformed)

            # Convert the prediction back to a human-readable format
            prediction_result = "On Time" if prediction[0] == 0 else "Delayed"

            return render_template("index.html", prediction=prediction_result)

        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
