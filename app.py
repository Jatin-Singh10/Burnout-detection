"""
AI-Based Early Burnout Detection System for Students — Streamlit Dashboard
Loads the trained, calibrated model and serves live risk predictions with
risk tiering, an actionable recommendation engine, and a feature-importance
explanation view.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "burnout_model.joblib"
META_PATH = APP_DIR / "metadata.json"

st.set_page_config(
    page_title="Student Burnout Risk Detector",
    page_icon="🧠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load model + metadata (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text())
    return model, metadata

model, metadata = load_artifacts()
BEST_THRESHOLD = metadata["best_threshold"]
CAT_OPTIONS = metadata["categorical_options"]
NUM_RANGES = metadata["numeric_ranges"]


# ---------------------------------------------------------------------------
# Feature engineering — must mirror the training notebook exactly
# ---------------------------------------------------------------------------
def normalize_sleep_hours(value: str):
    mapping = {
        "less than 5 hours": 4.0, "5-6 hours": 5.5, "7-8 hours": 7.5,
        "more than 8 hours": 9.0, "5-6": 5.5, "7-8": 7.5, "8-9": 8.5,
        "10-11": 10.5, "2-3 hours": 2.5,
    }
    return mapping.get(str(value).strip().lower(), np.nan)


def add_domain_features(row: dict) -> pd.DataFrame:
    out = dict(row)
    ap = float(out.get("Academic Pressure", np.nan))
    ss = float(out.get("Study Satisfaction", np.nan))
    wsh = float(out.get("Work/Study Hours", np.nan))
    fs = float(out.get("Financial Stress", np.nan))

    out["Academic_Strain"] = ap - ss
    out["Low_Study_Satisfaction"] = float(ss <= 3)  # 3 ~ dataset median
    out["Workload_Burden"] = wsh * (1 + ap)
    out["Financial_Academic_Strain"] = fs * (1 + ap)
    sleep_hours = normalize_sleep_hours(out.get("Sleep Duration", ""))
    out["Sleep_Hours_Approx"] = sleep_hours
    out["Sleep_Work_Imbalance"] = (
        wsh / sleep_hours if sleep_hours and sleep_hours != 0 else np.nan
    )
    return pd.DataFrame([out])


def risk_tier(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def recommendation_engine(row: dict, probability: float):
    recs = []

    def num(col):
        try:
            return float(row.get(col, np.nan))
        except Exception:
            return np.nan

    academic_pressure = num("Academic Pressure")
    study_satisfaction = num("Study Satisfaction")
    work_hours = num("Work/Study Hours")
    financial_stress = num("Financial Stress")
    cgpa = num("CGPA")

    if not np.isnan(academic_pressure) and academic_pressure >= 4:
        recs.append("Review academic workload and create a realistic study schedule.")
    if not np.isnan(study_satisfaction) and study_satisfaction <= 2:
        recs.append("Discuss study difficulties with an academic mentor or counsellor.")
    if not np.isnan(work_hours) and work_hours >= 8:
        recs.append("Reduce prolonged study/work sessions and add structured breaks.")
    if not np.isnan(financial_stress) and financial_stress >= 4:
        recs.append("Consider connecting with available financial-aid or student-support services.")
    if not np.isnan(cgpa) and cgpa < 6:
        recs.append("Consider academic support, tutoring, or faculty guidance.")

    if probability >= 0.65:
        recs.append("Early-warning alert: consider timely support from a qualified counsellor.")
    elif probability >= 0.35:
        recs.append("Monitor the student's risk indicators and reassess after an appropriate interval.")

    if not recs:
        recs.append("Continue healthy study, sleep, social-support and workload-management habits.")
    return recs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧠 AI-Based Early Burnout Detection")
st.caption(
    "Explainable ML decision-support prototype — estimates a student's burnout / "
    "mental-health risk from academic, lifestyle and psychosocial indicators. "
    "This is a research prototype, **not a clinical diagnostic tool**."
)

with st.form("student_form"):
    st.subheader("Student Profile")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", CAT_OPTIONS["Gender"])
        age = st.slider("Age", 16, 40, 21)
        profession = st.selectbox("Profession", CAT_OPTIONS["Profession"], index=CAT_OPTIONS["Profession"].index("Student") if "Student" in CAT_OPTIONS["Profession"] else 0)
        degree = st.selectbox("Degree", CAT_OPTIONS["Degree"])
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Dietary Habits", [o for o in CAT_OPTIONS["Dietary Habits"] if o not in ("Others",)])

    with col2:
        academic_pressure = st.slider("Academic Pressure (0 = none, 5 = extreme)", 0.0, 5.0, 3.0, 1.0)
        study_satisfaction = st.slider("Study Satisfaction (0 = very low, 5 = very high)", 0.0, 5.0, 3.0, 1.0)
        work_study_hours = st.slider("Work/Study Hours per day", 0.0, 14.0, 6.0, 0.5)
        financial_stress = st.slider("Financial Stress (1 = low, 5 = high)", 1.0, 5.0, 3.0, 1.0)
        suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
        family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])
        work_pressure = st.slider("Work Pressure (usually 0 for full-time students)", 0.0, 5.0, 0.0, 1.0)
        job_satisfaction = st.slider("Job Satisfaction (usually 0 for full-time students)", 0.0, 5.0, 0.0, 1.0)

    city = st.text_input("City (optional, free text)", value="Delhi")

    submitted = st.form_submit_button("Assess Burnout Risk", use_container_width=True)

if submitted:
    raw_row = {
        "Gender": gender,
        "Age": age,
        "City": city,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
    }

    input_df = add_domain_features(raw_row)
    input_df = input_df[metadata["feature_columns"]]  # enforce column order

    prob = float(model.predict_proba(input_df)[:, 1][0])
    tier = risk_tier(prob)
    early_warning = prob >= BEST_THRESHOLD
    recs = recommendation_engine(raw_row, prob)

    st.divider()
    st.subheader("Result")

    tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]
    c1, c2 = st.columns(2)
    c1.metric("Risk Probability", f"{prob * 100:.1f}%")
    c2.markdown(f"### Risk Tier: :{tier_color}[{tier}]")

    if early_warning:
        st.error("⚠️ Early-warning threshold exceeded — recommend timely follow-up.")
    else:
        st.success("No early-warning alert triggered at the current threshold.")

    st.subheader("Personalised Recommendations")
    for r in recs:
        st.markdown(f"- {r}")

    with st.expander("Model details / transparency"):
        st.write(f"**Model:** {metadata['best_model_name']} (Platt-calibrated)")
        st.write(f"**Decision threshold:** {BEST_THRESHOLD:.2f}")
        st.write("**Held-out test performance:**")
        st.json(metadata["final_test_metrics"])

st.divider()
st.caption(
    "⚠️ This tool is a research/decision-support prototype trained on a public "
    "student-depression survey dataset. It is not a substitute for professional "
    "mental-health assessment. If you or someone you know is in crisis, please "
    "contact a counsellor or local mental-health helpline immediately."
)
