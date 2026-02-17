import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="GPA Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student GPA Predictor")
st.write("Enter student info and get a predicted GPA.")

MODEL_PATH = "gpa_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python train_model.py` first to create gpa_model.joblib.")
    st.stop()

model = joblib.load(MODEL_PATH)

study_hours = st.slider("Study hours per week", min_value=0, max_value=60, value=25)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)
sleep_hours = st.slider("Sleep hours per night", min_value=0, max_value=12, value=7)

input_df = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "sleep_hours": [sleep_hours],
})

if st.button("Predict GPA"):
    pred = model.predict(input_df)[0]
    pred = max(0.0, min(4.0, float(pred)))

    st.subheader("Result")
    st.metric("Predicted GPA", f"{pred:.2f} / 4.00")

    # Optional “risk” label for portfolio storytelling
    if pred < 2.0:
        st.warning("Risk: High (needs support)")
    elif pred < 3.0:
        st.info("Risk: Medium (could improve)")
    else:
        st.success("Risk: Low (doing well)")
