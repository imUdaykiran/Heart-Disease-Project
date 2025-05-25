import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    page_icon="🫀"
)

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar
st.sidebar.title("🫀 Heart Disease Prediction")
st.sidebar.markdown("""
This AI-powered app predicts the **risk of heart disease** using key health indicators.

### 🔍 What You Can Do:
- Predict risk instantly
- Understand your risk factors
- Learn prevention tips

---

### 🧠 Did You Know?
- Heart disease is the **#1 cause of death** globally
- Lifestyle changes can prevent up to **80%** of cases
- High BP, obesity, and smoking are top risk factors

---

### 📚 Source:
- [WHO Cardiovascular Facts](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
                    
### 📊 Dataset Source:
- [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)
                    
""")

st.markdown("<h1 style='text-align: center; color : firebrick '> Heart Disease Prediction</h1>",
            unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(
    ["🔍 Prediction", "📈 Model Performance", "👤 Contact"])


with tab1:
    st.markdown(
        " :green[Use the form below to check your risk and get personalized health suggestions.]")

    # Default values
    default_values = {
        "age": 53,
        "gender": "Male",
        "height": 170,
        "weight": 75,
        "systolic": 130,
        "diastolic": 85,
        "chest_pain": "Yes",
        "diabetes": "Yes",
        "smoking": "No",
        "alcohol": "No",
        "family": "Yes",
        "heart_rate": 72,
    }

    # High-risk values
    high_risk_values = {
        "age": 67,
        "gender": "Male",
        "height": 165,
        "weight": 92,
        "systolic": 160,
        "diastolic": 100,
        "chest_pain": "Yes",
        "diabetes": "Yes",
        "smoking": "Yes",
        "alcohol": "Yes",
        "family": "Yes",
        "heart_rate": 105,
    }

    # Low-risk values
    low_risk_values = {
        "age": 34,
        "gender": "Female",
        "height": 168,
        "weight": 58,
        "systolic": 115,
        "diastolic": 75,
        "chest_pain": "No",
        "diabetes": "No",
        "smoking": "No",
        "alcohol": "No",
        "family": "No",
        "heart_rate": 68,
    }

    # Create session state for inputs
    for key in default_values:
        if key not in st.session_state:
            st.session_state[key] = default_values[key]

    # Buttons for autofill
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔥 Fill High-Risk Values"):
            for k, v in high_risk_values.items():
                st.session_state[k] = v
    with col2:
        if st.button("🧘‍♂️ Fill Low-Risk Values"):
            for k, v in low_risk_values.items():
                st.session_state[k] = v
    with col3:
        if st.button("🔁 Reset to Default"):
            for k, v in default_values.items():
                st.session_state[k] = v

    # Form
    with st.form("heart_form"):
        st.subheader("📝 Fill in your health information")

        col1, spacer, col2 = st.columns(
            [1, 0.05, 1])  # spacer for vertical line

        with col1:
            age = st.number_input("Age", 18, 120, st.session_state["age"])
            gender = st.selectbox("Gender", [
                                  "Male", "Female"], index=0 if st.session_state["gender"] == "Male" else 1)
            height = st.number_input(
                "Height (cm)", 100, 250, st.session_state["height"])
            weight = st.number_input(
                "Weight (kg)", 30, 200, st.session_state["weight"])
            systolic = st.number_input(
                "Systolic BP", 80, 200, st.session_state["systolic"])
            diastolic = st.number_input(
                "Diastolic BP", 60, 130, st.session_state["diastolic"])

        with spacer:
            st.markdown("""
                <div style='height: 100%; border-left: 2px solid #DDD; margin: auto;'></div>
            """, unsafe_allow_html=True)

        with col2:
            chest_pain = st.selectbox("Chest Pain?", [
                                      "Yes", "No"], index=0 if st.session_state["chest_pain"] == "Yes" else 1)
            diabetes = st.selectbox("Diabetes?", [
                                    "Yes", "No"], index=0 if st.session_state["diabetes"] == "Yes" else 1)
            smoking = st.selectbox(
                "Smoker?", ["Yes", "No"], index=0 if st.session_state["smoking"] == "Yes" else 1)
            alcohol = st.selectbox("Alcohol Use?", [
                                   "Yes", "No"], index=0 if st.session_state["alcohol"] == "Yes" else 1)
            family = st.selectbox("Family History?", [
                                  "Yes", "No"], index=0 if st.session_state["family"] == "Yes" else 1)
            heart_rate = st.number_input(
                "Heart Rate (bpm)", 50, 220, st.session_state["heart_rate"])

        submit = st.form_submit_button("🚀 Predict Risk")

    # Prediction logic
    if submit:
        gender_val = 1 if gender == "Male" else 0
        chest_pain_val = 1 if chest_pain == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0
        smoking_val = 1 if smoking == "Yes" else 0
        alcohol_val = 1 if alcohol == "Yes" else 0
        family_val = 1 if family == "Yes" else 0
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        input_data = np.array([[age, gender_val, height, weight, systolic, diastolic,
                                chest_pain_val, diabetes_val, smoking_val, alcohol_val,
                                family_val, bmi, heart_rate]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col_pred, col_graph = st.columns([1.5, 1])

        with col_pred:
            if prediction[0] == 1:
                st.error(
                    f"⚠️ **High Risk** of Heart Disease (Probability: `{prediction_proba[0][1]*100:.1f}%`)")
                st.markdown("""
                **🧑‍⚕️ Recommendations:**
                - Consult a cardiologist immediately  
                - Reduce salt, control BP  
                - Cardiac screening suggested  
                """)
            else:
                st.success(
                    f"✅ **Low Risk** of Heart Disease (Probability: `{prediction_proba[0][0]*100:.1f}%`)")
                st.markdown("""
                **💪 Stay Healthy Tips:**
                - Maintain physical activity  
                - Eat heart-friendly food  
                - Annual checkups recommended  
                """)

        with col_graph:
            st.markdown("#### 🧮 Risk Breakdown")
            st.plotly_chart(
                {
                    "data": [{
                        "type": "bar",
                        "x": ["Low Risk", "High Risk"],
                        "y": list(prediction_proba[0]),
                        "marker": {"color": ["green", "crimson"]}
                    }],
                    "layout": {
                        "height": 250,
                        "margin": {"l": 40, "r": 40, "t": 20, "b": 30}
                    }
                },
                use_container_width=True
            )

    # Tips Section
    st.markdown("---")
    st.header("💡 Heart Health Tips")
    tips = [
        "🥦 Eat a diet rich in vegetables, fruits, and whole grains",
        "🚶‍♀️ Aim for 30 mins of physical activity daily",
        "🛌 Get 7–8 hours of quality sleep",
        "🧘 Practice meditation and relaxation",
        "🧂 Cut back on sodium and red meat",
        "🩺 Check your blood pressure monthly",
        "📉 Maintain cholesterol and blood sugar in check"
    ]
    for tip in tips:
        st.markdown(f"- {tip}")


with open("model_metrics.json", "r") as f:
    metrics = json.load(f)

# Extract accuracy
accuracy = float(metrics["accuracy"].strip('%'))

# Extract classification report
report = metrics["classification_report"]

# Prepare table data
rows = []
for label in ['0', '1']:
    label_data = report[label]
    rows.append([
        label,
        round(label_data["precision"], 3),
        round(label_data["recall"], 3),
        round(label_data["f1-score"], 3),
        label_data["support"]
    ])

# Add average rows
rows.append([
    "Macro Avg",
    round(report["macro avg"]["precision"], 3),
    round(report["macro avg"]["recall"], 3),
    round(report["macro avg"]["f1-score"], 3),
    report["macro avg"]["support"]
])
rows.append([
    "Weighted Avg",
    round(report["weighted avg"]["precision"], 3),
    round(report["weighted avg"]["recall"], 3),
    round(report["weighted avg"]["f1-score"], 3),
    report["weighted avg"]["support"]
])

# Create DataFrame
df_report = pd.DataFrame(
    rows, columns=["Class", "Precision", "Recall", "F1-Score", "Support"])


with tab2:
    st.subheader(f"🎯 Model Accuracy: {accuracy:.2f}%")
    st.markdown("### 📋 Classification Report")
    st.dataframe(df_report, use_container_width=True)


with tab3:
    # Project Guide
    st.markdown("### 🧑‍🏫 Project Guide")
    st.write(
        "**Prof.Rekha**, Department of Computer Science and Engineering.")

    st.markdown("---")

    # Development Team
    st.markdown("### 👨‍💻 Development Team")
    st.write("""
    - **Asima Sadiya** — `1AJ21CS009`  
    - **Brunda M. C.** — `1AJ21CS017`  
    - **Dayyala Hyndavi** — `1AJ21CS026`  
    - **Lomada Nandini** — `1AJ21CS052`
    """)

    st.markdown("---")

    st.markdown("### 🎓 Academic Information")
    st.write("""
    - **Academic Year :**  `2024–2025`
    - **Department :**     `Computer Science and Engineering`
    - **College :**        `Cambridge Institute of Technology`
    """)

    st.markdown("---")

    # Contact & GitHub
    st.markdown("### 🔗 Contact & Resources")
    st.markdown(
        "**📍 Address:** KR Puram, Bengaluru - 560036 [(View on Maps)](https://maps.google.com/?q=Cambridge+Institute+of+Technology+KR+Puram)")
    st.write("**📧 Email**: admissions@cambridge.edu.in")

    st.markdown("---")

    # Disclaimer
    st.info("⚠️ This application is intended for educational purposes only and should not be used as a substitute for professional medical advice.")


# streamlit run main.py