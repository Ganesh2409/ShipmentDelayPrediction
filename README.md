
# Shipment Delay Prediction Web Application

This Flask web application predicts whether a shipment will be on time or delayed based on various factors such as weather, traffic conditions, shipment details, and the planned delivery date. The model has been trained using a dataset containing historical shipment data, and the predictions are based on the machine learning model `model_log.pkl`.

## Project Overview

The goal of this project is to provide a web-based interface where users can input shipment data, and the system will predict whether the shipment will arrive on time or be delayed. This is achieved using a trained machine learning model and a column transformer that processes the input data in the same way it was processed during model training.

### Key Features:
- Collects user input through a web form (e.g., origin, destination, shipment date, vehicle type, distance, etc.).
- Loads a pre-trained machine learning model (`model_log.pkl`) and a column transformer (`column_transformer.pkl`).
- Predicts whether the shipment will be **On Time** or **Delayed** based on the input data.
- Provides real-time prediction results directly on the webpage.

## Project Structure

```
/shipment-delay-prediction
├── app.py                # Main Flask app to handle prediction requests
├── model_files           # Folder containing the trained model and column transformer
│   ├── model_log.pkl     # Pickled machine learning model
│   └── column_transformer.pkl  # Pickled column transformer
├── static                # Folder for static files like CSS and JavaScript
│   └── /css              # CSS files for styling
├── templates             # HTML templates for rendering web pages
│   └── index.html        # HTML page for user input form and prediction result
├── data                  # Folder containing training data
│   ├── data # Dataset used for training
├── Delay_Prediction.ipynb  # Jupyter notebook used for model training and evaluation
├── Delay_Prediction.py
└── requirements.txt      # List of Python dependencies
└── README.md             # This README file
```

## How the App Works

### 1. **Input Form:**
   - Users fill out a form with the following details:
     - **Shipment ID**: Unique identifier for the shipment (not used for prediction).
     - **Origin**: Origin location of the shipment.
     - **Destination**: Destination location for the shipment.
     - **Shipment Date**: Date when the shipment was dispatched.
     - **Planned Delivery Date**: Date when the shipment is expected to arrive.
     - **Vehicle Type**: Type of vehicle used for the shipment (e.g., truck, van).
     - **Distance (km)**: Distance between origin and destination.
     - **Weather Conditions**: Weather conditions during shipment (e.g., sunny, rainy).
     - **Traffic Conditions**: Traffic conditions during shipment (e.g., heavy, moderate).

### 2. **Data Processing:**
   - When the form is submitted, the app converts the date strings to `datetime` format and calculates the "Planned Shipment Gap" (i.e., the difference in days between the planned delivery date and the shipment date).
   - The data is then transformed using the same column transformer that was applied during model training.

### 3. **Prediction:**
   - The transformed data is passed to the trained machine learning model (`model_log.pkl`), which predicts whether the shipment will be **On Time** (0) or **Delayed** (1).
   - The result is then displayed on the webpage as a human-readable prediction ("On Time" or "Delayed").

### 4. **Error Handling:**
   - If the input data is invalid (e.g., incorrect date format), an error message is displayed, informing the user to correct the input.

## Requirements

### Python Version:
- Python 3.8 or higher

### Dependencies:
Create a `requirements.txt` file using the following dependencies:

```
Flask==2.3.2
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
pickle-mixin==1.0.2
```

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application locally, follow these steps:

1. Clone the repository or download the project files.
2. Navigate to the project directory in your terminal.
3. Install the required dependencies using the command:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask app by running:

   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/` to access the application.

## Model Training

The model used in this application (`model_log.pkl`) was trained using historical shipment data. The dataset was preprocessed and features such as weather conditions, traffic conditions, vehicle type, and shipment gap were used to train a machine learning model that can predict whether a shipment will be on time or delayed.

The model training process can be found in the Jupyter notebook `Delay_Prediction.ipynb`.

### Key Steps in Model Training:
1. **Data Preprocessing**: Clean and preprocess the dataset.
2. **Feature Engineering**: Extract meaningful features such as "Planned Shipment Gap" and transform categorical variables.
3. **Model Selection**: Use a machine learning algorithm (e.g., Logistic Regression, Random Forest, etc.) to train the model.
4. **Model Evaluation**: Evaluate the model's performance using metrics such as accuracy, precision, recall, etc.
5. **Model Serialization**: Save the trained model (`model_log.pkl`) and column transformer (`column_transformer.pkl`) using `pickle`.

## Conclusion

This Flask-based web application provides an easy-to-use interface for predicting shipment delays. By inputting key shipment details, users can quickly determine whether their shipment is likely to be on time or delayed. The app utilizes a machine learning model trained on historical data to make these predictions.

Feel free to fork the repository, contribute, or modify it as per your requirements.


```
© 2024  Shipment Delay Prediction Web Application . Made with ❤️
```

