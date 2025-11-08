import pandas as pd
import numpy as np
import joblib 
import streamlit as st
import matplotlib as plt
import seaborn as sns


# --------------------------
# Load trained model
model = joblib.load('rf_pipeline.pkl')
# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="NaN Masters | Water Potability Predictor",
    page_icon="💧",
    layout="centered"
)

# ---------------------------
# Header + About Section
# ---------------------------
st.title("💧 Water Potability Prediction App")

st.markdown("""
### 👥 Who We Are
**We are NaN Masters** — a curious crew of data explorers, problem solvers, and change-makers.  
Our mission? To turn raw data into real-world impact.

We dive deep into numbers, patterns, and possibilities to help communities access **clean, safe, and sustainable water**.  
For us, every dataset tells a story — and we use technology to make that story count.

With curiosity as our compass and innovation as our toolkit,  
we’re here to prove that even from **NaN (Not a Number)** beginnings, great solutions can flow.
""")

# ---------------------------
# Problem Overview
# ---------------------------
st.markdown("---")
st.subheader("🌍 Why This Matters")
st.markdown("""
Many communities worldwide lack access to **safe drinking water**, and testing is often **slow or unavailable**.  
Our model predicts whether water is safe or unsafe to drink using **chemical properties like pH, turbidity, hardness, and more**.

This helps communities and municipalities **take quick, data-driven action** to ensure water safety.
""")

# ---------------------------
# Prediction Mode
# ---------------------------
st.markdown("---")
st.subheader("🚀 Choose a Prediction Mode")
mode = st.radio(
    "Select how you'd like to predict water safety:",
    ("🔹 Manual Input", "📂 Batch CSV Upload"),
    horizontal=True
)
# ---------------------------
# Manual Input Mode
# ---------------------------
if mode == "🔹 Manual Input":
    st.sidebar.header("Enter Water Quality Features")

    def user_input_features():
        pH = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
        Hardness = st.sidebar.slider("Hardness (mg/L)", 0.0, 500.0, 150.0)
        Solids = st.sidebar.number_input("Solids (ppm)", 0.0, 50000.0, 20000.0)
        Chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 10.0, 3.0)
        Sulfate = st.sidebar.number_input("Sulfate (mg/L)", 0.0, 500.0, 200.0)
        Conductivity = st.sidebar.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 400.0)
        Organic_carbon = st.sidebar.slider("Organic Carbon (mg/L)", 0.0, 30.0, 10.0)
        Trihalomethanes = st.sidebar.number_input("Trihalomethanes (µg/L)", 0.0, 150.0, 50.0)
        Turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 4.0)

        data = {
            'ph': pH,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    st.subheader("🔍 Entered Water Quality Data:")
    st.write(input_df)

    processed_input = feature_engineering(input_df)

    if st.button("💧 Predict Water Safety"):
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        st.subheader("💡 Prediction Result")
        if prediction[0] == 1:
            st.success("✅ The water is predicted to be **SAFE** to drink.")
        else:
            st.error("⚠️ The water is predicted to be **UNSAFE** to drink.")

        st.write("Prediction Probability:")
        st.write(f"Unsafe: {round(prediction_proba[0][0]*100, 2)}% | Safe: {round(prediction_proba[0][1]*100, 2)}%")

# ---------------------------
# Batch CSV Upload Mode
# ---------------------------
elif mode == "📂 Batch CSV Upload":
    st.subheader("📁 Upload a CSV File for Batch Predictions")
    st.markdown("""
    Upload a CSV file with these columns:  
    `ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`
    """)

    uploaded_file = st.file_uploader("Upload your water quality dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("🚀 Predict for All Rows"):
            processed_df = feature_engineering(df)
            df['Potability_Prediction'] = model.predict(processed_df)

            st.success("✅ Predictions generated successfully!")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.info("Please upload a CSV file to continue.")

# ---------------------------
# Model Insights Section
# ---------------------------
st.markdown("---")
st.subheader("📊 Model Insights")

# Demo metrics (replace with real model metrics)
accuracy = 0.85
precision = 0.82
recall = 0.80
f1 = 0.81

st.markdown(f"""
**Model Performance (Demo Metrics):**  
- Accuracy: {accuracy}  
- Precision: {precision}  
- Recall: {recall}  
- F1 Score: {f1}
""")

# Feature importance placeholder
st.markdown("**Feature Importance:**")
feat_importances = pd.Series([0.15,0.12,0.10,0.08,0.08,0.07,0.06,0.05,0.04], 
                             index=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])
plt.figure(figsize=(6,4))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)

# ---------------------------
# Meet the Team Section
# ---------------------------
st.markdown("---")
st.subheader("👩‍💻 Meet the Team")
st.markdown("""
1. **Snenhlanhla Nsele** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/sinenhlanhla-nsele-126a6a18a)  
2. **Nonhlanhla Magagula** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/nonhlanhla-magagula-b741b3207)  
3. **Thandiwe Mkhabela** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/thandiwe-m)  
4. **Thabiso Seema** - Software Engineer - [LinkedIn](https://www.linkedin.com/in/thabisoseema)
""")

# ---------------------------
# Model Versioning Info
# ---------------------------
st.markdown("---")
st.subheader("🧩 Model Versioning Info")
st.markdown("""
- **Model Version:** 1.0  
- **Last Updated:** Nov 2025  
- **Training Data:** 3,000+ samples
""")

# ---------------------------
# Footer
# ---------------------------
st.caption("Created with ❤️ by NaN Masters | Powered by Streamlit")
