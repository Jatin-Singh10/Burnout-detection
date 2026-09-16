"""
AI-Based Early Burnout Detection System for Students — Enhanced Streamlit Dashboard
Loads the trained, calibrated model and serves live risk predictions with
categorized recommendations and an improved user experience.

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
    recs = {
        "Immediate Support": [],
        "Academic Strategies": [],
        "Wellness & Lifestyle": [],
        "Professional Guidance": []
    }

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
    suicidal_thoughts = row.get("Have you ever had suicidal thoughts ?", "No")
    sleep_duration = row.get("Sleep Duration", "")

    # --- 1. Immediate Support ---
    if suicidal_thoughts == "Yes":
        recs["Immediate Support"].append(
            "🚨 **CRITICAL:** Please contact a crisis hotline or professional therapist immediately. "
            "You are not alone, and help is available."
        )

    if probability >= 0.65:
        recs["Immediate Support"].append(
            "Early-warning alert: High risk of burnout detected. We strongly recommend reaching "
            "out to a qualified mental health counselor."
        )
    elif probability >= 0.35:
        recs["Immediate Support"].append(
            "Moderate risk: Consider scheduling a check-in with a counselor to prevent escalation."
        )

    # --- 2. Academic Strategies ---
    if not np.isnan(academic_pressure) and academic_pressure >= 4:
        recs["Academic Strategies"].append(
            "Review academic workload; try the Pomodoro technique to break large tasks into manageable chunks."
        )
    if not np.isnan(study_satisfaction) and study_satisfaction <= 2:
        recs["Academic Strategies"].append(
            "Connect with a peer study group or an academic mentor to find new ways to engage with your subjects."
        )
    if not np.isnan(cgpa) and cgpa < 6:
        recs["Academic Strategies"].append(
            "Explore campus tutoring services or faculty office hours for targeted academic support."
        )

    # --- 3. Wellness & Lifestyle ---
    if not np.isnan(work_hours) and work_hours >= 8:
        recs["Wellness & Lifestyle"].append(
            "Reduce prolonged study/work sessions. Schedule 'non-negotiable' downtime every day to recharge."
        )

    # Combination rule: High workload + Low sleep
    sleep_val = normalize_sleep_hours(sleep_duration)
    if not np.isnan(work_hours) and work_hours >= 8 and (not np.isnan(sleep_val) and sleep_val < 6):
        recs["Wellness & Lifestyle"].append(
            "⚠️ **Burnout Cycle Alert:** You are working long hours with very little sleep. "
            "This is a high-strain pattern; prioritize sleep immediately to avoid total exhaustion."
        )
    elif not np.isnan(sleep_val) and sleep_val < 6:
        recs["Wellness & Lifestyle"].append(
            "Prioritize a consistent sleep-wake cycle. Aim for 7-9 hours to improve cognitive function and mood."
        )

    # --- 4. Professional Guidance ---
    if not np.isnan(financial_stress) and financial_stress >= 4:
        recs["Professional Guidance"].append(
            "Consider connecting with student financial aid offices or scholarship advisors to reduce financial anxiety."
        )

    # Fallback
    if not any(recs.values()):
        recs["Wellness & Lifestyle"].append("Continue maintaining your healthy study, sleep, and social-support habits!")

    return recs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧠 AI-Based Early Burnout Detection")
st.markdown(
    """
    This interactive tool helps students estimate their risk of burnout based on academic,
    lifestyle, and psychosocial indicators. It provides an early warning to encourage
    proactive self-care and professional support.

    ***
    **Disclaimer:** This is a research prototype, **not a clinical diagnostic tool**.
    If you are in crisis, please contact a professional helper immediately.
    """
)

with st.form("student_form"):
    st.subheader("Student Profile Intake")

    # Use tabs for a cleaner, more interactive experience
    tab1, tab2, tab3 = st.tabs(["👤 Personal", "📚 Academic & Work", "🌿 Lifestyle & Health"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", CAT_OPTIONS["Gender"])
            age = st.slider("Age", 16, 40, 21)
            profession = st.selectbox("Profession", CAT_OPTIONS["Profession"], index=CAT_OPTIONS["Profession"].index("Student") if "Student" in CAT_OPTIONS["Profession"] else 0)
        with c2:
            degree = st.selectbox("Degree", CAT_OPTIONS["Degree"])
            city = st.text_input("City (optional)", value="Delhi")
            cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            academic_pressure = st.slider("Academic Pressure (0=None, 5=Extreme)", 0.0, 5.0, 3.0, 1.0)
            study_satisfaction = st.slider("Study Satisfaction (0=Low, 5=High)", 0.0, 5.0, 3.0, 1.0)
            work_study_hours = st.slider("Work/Study Hours per day", 0.0, 14.0, 6.0, 0.5)
        with c2:
            work_pressure = st.slider("Work Pressure (usually 0 for students)", 0.0, 5.0, 0.0, 1.0)
            job_satisfaction = st.slider("Job Satisfaction (usually 0 for students)", 0.0, 5.0, 0.0, 1.0)
            financial_stress = st.slider("Financial Stress (1=Low, 5=High)", 1.0, 5.0, 3.0, 1.0)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
            dietary_habits = st.selectbox("Dietary Habits", [o for o in CAT_OPTIONS["Dietary Habits"] if o not in ("Others",)])
            family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])
        with c2:
            suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])

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
    recs_dict = recommendation_engine(raw_row, prob)

    st.divider()
    st.subheader("Assessment Result")

    # --- Visual Result Card ---
    tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Risk Probability", f"{prob * 100:.1f}%")
    with res_col2:
        st.markdown(f"### Risk Tier: :{tier_color}[{tier}]")
        st.progress(prob)

    if early_warning:
        st.error("⚠️ **Early-warning threshold exceeded** — we recommend a timely follow-up with a professional.")
    else:
        st.success("No early-warning alert triggered at the current threshold.")

    # --- Categorized Recommendations ---
    st.subheader("Personalized Recommendations")

    # Map categories to icons and colors
    categories = {
        "Immediate Support": ("🚨", "error"),
        "Academic Strategies": ("📚", "info"),
        "Wellness & Lifestyle": ("🌿", "success"),
        "Professional Guidance": ("💼", "warning")
    }

    for cat, (icon, st_type) in categories.items():
        cat_recs = recs_dict[cat]
        if cat_recs:
            # Use a different streamlit container based on the category
            if st_type == "error":
                with st.expander(f"{icon} {cat}", expanded=True):
                    for r in cat_recs: st.markdown(f"- {r}")
            elif st_type == "info":
                with st.expander(f"{icon} {cat}"):
                    for r in cat_recs: st.markdown(f"- {r}")
            elif st_type == "success":
                with st.expander(f"{icon} {cat}"):
                    for r in cat_recs: st.markdown(f"- {r}")
            else:
                with st.expander(f"{icon} {cat}"):
                    for r in cat_recs: st.markdown(f"- {r}")

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
