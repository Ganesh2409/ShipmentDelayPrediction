import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'data.csv')

# Basic exploration
print(df.head())
print(df.shape)
print(df.isnull().sum())

# Drop irrelevant columns
drop_columns = ["Shipment ID", "Origin", "Destination"]
df.drop(drop_columns, axis=1, inplace=True)

# Convert date columns to datetime format
df["Actual Delivery Date"] = pd.to_datetime(df["Actual Delivery Date"])
df["Planned Delivery Date"] = pd.to_datetime(df["Planned Delivery Date"])
df["Shipment Date"] = pd.to_datetime(df["Shipment Date"])

# Create new feature: Planned Shipment Gap
df["Planned Shipment Gap (days)"] = (df["Planned Delivery Date"] - df["Shipment Date"]).dt.days

# Drop date columns after feature engineering
drop_columns = ["Actual Delivery Date", "Planned Delivery Date", "Shipment Date"]
df.drop(drop_columns, axis=1, inplace=True)

# Handle missing values
df.fillna({'Vehicle Type': df['Vehicle Type'].mode()[0]}, inplace=True)

print(df.info())

# Separate features and target
X = df.drop(columns=['Delayed'])
y = df['Delayed']

# Define Column Transformer for preprocessing
clmn = ColumnTransformer(transformers=[
    ('encode_weather', OneHotEncoder(handle_unknown='ignore'), 
     ['Weather Conditions', 'Traffic Conditions', 'Vehicle Type']),
    ('scale_distance', StandardScaler(), ['Distance (km)']),
],
remainder='passthrough')


# Transform features
X_transformed = clmn.fit_transform(X)

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=24)

# Logistic Regression Model
model_log = LogisticRegression()
model_log.fit(X_train, y_train)
predictions_log = model_log.predict(X_test)

accuracy = accuracy_score(y_test, predictions_log)
print("Logistic Regression Accuracy:", accuracy)
print(classification_report(y_test, predictions_log))
print("Cross-validation scores:", cross_val_score(model_log, X_train, y_train, cv=5))

# SVM Model
model_svc = SVC()
model_svc.fit(X_train, y_train)
predictions_svc = model_svc.predict(X_test)

accuracy = accuracy_score(y_test, predictions_svc)
print("SVM Accuracy:", accuracy)
print(classification_report(y_test, predictions_svc))
print("Cross-validation scores:", cross_val_score(model_svc, X_train, y_train, cv=5))

# Random Forest Model
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions_rf)
print("Random Forest Accuracy:", accuracy)
print(classification_report(y_test, predictions_rf))
print("Cross-validation scores:", cross_val_score(model_rf, X_train, y_train, cv=5))

# XGBoost Model
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
predictions_xgb = model_xgb.predict(X_test)

accuracy = accuracy_score(y_test, predictions_xgb)
print("XGBoost Accuracy:", accuracy)
print(classification_report(y_test, predictions_xgb))
print("Cross-validation scores:", cross_val_score(model_xgb, X_train, y_train, cv=5))

# Save transformers and models
with open('column_transformer.pkl', 'wb') as f:
    pickle.dump(clmn, f)

with open('model_log.pkl', 'wb') as f:
    pickle.dump(model_log, f)

print("Models and transformer saved successfully!")
