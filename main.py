import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json



# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")



### 📚 Source:
- [WHO Cardiovascular Facts](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
                    
### 📊 Dataset Source:
- [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)
                    




        input_data = np.array([[age, gender_val, height, weight, systolic, diastolic,
                                chest_pain_val, diabetes_val, smoking_val, alcohol_val,
                                family_val, bmi, heart_rate]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

print(prediction)

# Extract accuracy
accuracy = float(metrics["accuracy"].strip('%'))



