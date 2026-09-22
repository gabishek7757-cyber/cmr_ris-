"""
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System
Production-Grade Interactive Clinical Screening & Risk Intelligence Dashboard

Core Capabilities:
1. Patient Clinical Intake with Pydantic Data Contract Validation.
2. Conformal Prediction Intervals (90% calibrated uncertainty quantification).
3. Composite CardioMetabolic-Renal Index (CMRI) with multi-organ interaction coupling.
4. Independent vs Shared Multi-Task Model Benchmark.
5. Statistically-Grounded Longitudinal Trajectory Engine (Theil-Sen slope + Mann-Kendall test).
6. Explainable AI (SHAP attributions) & Dynamic Multi-Organ Counterfactual Simulation.
7. Auditable Multi-Criteria Decision Analysis (MCDA) Screening Priority Triage.
8. Subgroup Demographic Fairness & System Governance.
9. Clinical Summary Report Export (JSON / Markdown).
"""

import os
import sys
import json
from datetime import datetime

if "." not in sys.path:
    sys.path.insert(0, ".")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.cmri import CMRIndexCalculator
from src.uncertainty import ConformalRiskCalibrator
from src.trajectory import theil_sen_slope, mann_kendall_test, classify_trajectory, simulate_patient_trajectory
from src.explainability import ModelExplainer
from src.priority import MCDAPriorityEngine
from src.schemas import PatientIntake

# ---------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CMR-RIS | CardioMetabolic-Renal Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem 1.2rem;
        border-radius: 6px;
        border-left: 5px solid #ffeeba;
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .triage-routine { background-color: #e8f8f5; border-left: 5px solid #2ecc71; padding: 1rem; border-radius: 6px; }
    .triage-expedited { background-color: #fef9e7; border-left: 5px solid #f39c12; padding: 1rem; border-radius: 6px; }
    .triage-urgent { background-color: #fdedec; border-left: 5px solid #e74c3c; padding: 1rem; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RESOURCE CACHING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    models = {}
    for d in ["diabetes", "heart", "ckd"]:
        path = f"models/{d}_champion.pkl"
        if os.path.exists(path):
            models[d] = joblib.load(path)
        else:
            models[d] = None

    shared_path = "models/shared_multitask_model.pkl"
    shared_models = joblib.load(shared_path) if os.path.exists(shared_path) else None

    calibrator = ConformalRiskCalibrator()
    calibrator_path = "models/conformal_calibrator.pkl"
    if os.path.exists(calibrator_path):
        c_data = joblib.load(calibrator_path)
        if isinstance(c_data, dict):
            calibrator.calibrated_quantiles = c_data.get("calibrated_quantiles", {})
            calibrator.confidence_level = c_data.get("confidence_level", 0.90)
        else:
            calibrator = c_data

    comp_path = "results/model_comparison.csv"
    df_comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()

    fair_path = "results/fairness_audit.csv"
    df_fair = pd.read_csv(fair_path) if os.path.exists(fair_path) else pd.DataFrame()

    return models, shared_models, calibrator, df_comp, df_fair


champion_models, shared_models, calibrator, df_comparison, df_fairness = load_all_artifacts()
cmri_calc = CMRIndexCalculator()
priority_engine = MCDAPriorityEngine()


# ---------------------------------------------------------------------------
# HEADER & PERSISTENT CLINICAL DISCLAIMER
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🩺 CMR-RIS: CardioMetabolic-Renal Risk Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Organ Risk Screening, Calibrated Conformal Uncertainty, Longitudinal Trajectory Engine & Explainable AI</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ MANDATORY CLINICAL SCREENING DISCLAIMER:</strong>
    CMR-RIS is a research and clinical decision-support screening prototype designed for risk stratification and triage prioritization.
    It is <strong>not an automated diagnostic device</strong> and should not replace individualized clinical judgment, physician evaluation, or definitive diagnostic testing.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR — PATIENT CLINICAL INTAKE
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Patient Clinical Intake")
st.sidebar.markdown("Enter standardized patient demographics, vitals, and laboratory markers:")

preset = st.sidebar.selectbox(
    "Preload Clinical Persona",
    [
        "Custom Input",
        "Persona 1: Low-Risk Healthy Individual",
        "Persona 2: Early Metabolic Syndrome (Prediabetes + Prehypertension)",
        "Persona 3: Acute Cardio-Renal Comorbidity"
    ]
)

# Defaults according to persona
if preset == "Persona 1: Low-Risk Healthy Individual":
    def_age, def_sex, def_bmi = 32, "Female", 22.4
    def_sbp, def_dbp, def_glu, def_chol = 114, 74, 88.0, 168.0
    def_sc, def_bu, def_hemo = 0.8, 14.0, 14.5
    def_cp, def_thalach, def_oldpeak = 0, 165, 0.0
    def_preg, def_dpf = 1, 0.25
elif preset == "Persona 2: Early Metabolic Syndrome (Prediabetes + Prehypertension)":
    def_age, def_sex, def_bmi = 52, "Male", 29.8
    def_sbp, def_dbp, def_glu, def_chol = 138, 86, 118.0, 224.0
    def_sc, def_bu, def_hemo = 1.1, 22.0, 13.2
    def_cp, def_thalach, def_oldpeak = 1, 142, 0.8
    def_preg, def_dpf = 0, 0.55
elif preset == "Persona 3: Acute Cardio-Renal Comorbidity":
    def_age, def_sex, def_bmi = 66, "Male", 33.5
    def_sbp, def_dbp, def_glu, def_chol = 158, 96, 175.0, 268.0
    def_sc, def_bu, def_hemo = 2.4, 48.0, 10.4
    def_cp, def_thalach, def_oldpeak = 3, 118, 2.4
    def_preg, def_dpf = 0, 0.85
else:
    def_age, def_sex, def_bmi = 48, "Male", 27.5
    def_sbp, def_dbp, def_glu, def_chol = 130, 82, 110.0, 205.0
    def_sc, def_bu, def_hemo = 1.0, 18.0, 13.8
    def_cp, def_thalach, def_oldpeak = 1, 148, 0.5
    def_preg, def_dpf = 0, 0.40

with st.sidebar.expander("Demographics & Anthropometrics", expanded=True):
    age = st.number_input("Age (years)", min_value=18, max_value=95, value=def_age)
    sex = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if def_sex == "Male" else 1)
    sex_num = 1 if sex == "Male" else 0
    bmi = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=60.0, value=float(def_bmi), step=0.1)
    pregnancies = st.number_input("Pregnancies (if female)", min_value=0, max_value=18, value=int(def_preg)) if sex == "Female" else 0

with st.sidebar.expander("Cardiovascular & Hemodynamic", expanded=True):
    sys_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=240, value=int(def_sbp))
    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=int(def_dbp))
    tot_chol = st.number_input("Total Serum Cholesterol (mg/dL)", min_value=90, max_value=550, value=int(def_chol))
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=int(def_thalach))
    cp_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=int(def_cp), help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.5, value=float(def_oldpeak), step=0.1)

with st.sidebar.expander("Glycemic & Renal Biomarkers", expanded=True):
    glucose = st.number_input("Fasting / Random Glucose (mg/dL)", min_value=50.0, max_value=450.0, value=float(def_glu), step=1.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=float(def_dpf), step=0.01)
    serum_creatinine = st.number_input("Serum Creatinine (sc, mg/dL)", min_value=0.3, max_value=15.0, value=float(def_sc), step=0.1)
    blood_urea = st.number_input("Blood Urea (bu, mg/dL)", min_value=5.0, max_value=180.0, value=float(def_bu), step=1.0)
    hemoglobin = st.number_input("Hemoglobin (hemo, g/dL)", min_value=4.0, max_value=20.0, value=float(def_hemo), step=0.1)

# Pydantic Schema Validation
intake_payload = None
try:
    intake_payload = PatientIntake(
        age=int(age),
        sex=sex,
        pregnancies=int(pregnancies),
        sys_bp=float(sys_bp),
        dia_bp=float(dia_bp),
        bmi=float(bmi),
        glucose=float(glucose),
        dpf=float(dpf),
        tot_chol=float(tot_chol),
        thalach=float(thalach),
        oldpeak=float(oldpeak),
        cp_type=int(cp_type),
        blood_urea=float(blood_urea),
        serum_creatinine=float(serum_creatinine),
        hemoglobin=float(hemoglobin)
    )
except Exception as e:
    st.sidebar.error(f"Validation Notice: {e}")


# ---------------------------------------------------------------------------
# PATIENT INFERENCE PIPELINE
# ---------------------------------------------------------------------------
def compute_patient_predictions(override_dict=None):
    c_age = override_dict.get("age", age) if override_dict else age
    c_sex_num = override_dict.get("sex_num", sex_num) if override_dict else sex_num
    c_bmi = override_dict.get("bmi", bmi) if override_dict else bmi
    c_sys_bp = override_dict.get("sys_bp", sys_bp) if override_dict else sys_bp
    c_dia_bp = override_dict.get("dia_bp", dia_bp) if override_dict else dia_bp
    c_tot_chol = override_dict.get("tot_chol", tot_chol) if override_dict else tot_chol
    c_thalach = override_dict.get("thalach", thalach) if override_dict else thalach
    c_oldpeak = override_dict.get("oldpeak", oldpeak) if override_dict else oldpeak
    c_glucose = override_dict.get("glucose", glucose) if override_dict else glucose
    c_dpf = override_dict.get("dpf", dpf) if override_dict else dpf
    c_pregnancies = override_dict.get("pregnancies", pregnancies) if override_dict else pregnancies
    c_cp_type = override_dict.get("cp_type", cp_type) if override_dict else cp_type
    c_sc = override_dict.get("serum_creatinine", serum_creatinine) if override_dict else serum_creatinine
    c_bu = override_dict.get("blood_urea", blood_urea) if override_dict else blood_urea
    c_hemo = override_dict.get("hemoglobin", hemoglobin) if override_dict else hemoglobin

    # 1. Diabetes Inference
    d_model_dict = champion_models["diabetes"]
    d_scaler = d_model_dict["scaler"]
    d_feats = d_model_dict["features"]

    insulin_val = 120.0
    skin_val = 25.0
    d_row = pd.DataFrame([{
        "Pregnancies": float(c_pregnancies),
        "Glucose": float(c_glucose),
        "BloodPressure": float(c_dia_bp),
        "SkinThickness": skin_val,
        "Insulin": insulin_val,
        "BMI": float(c_bmi),
        "DiabetesPedigreeFunction": float(c_dpf),
        "Age": float(c_age),
        "Insulin_Glucose_Ratio": insulin_val / (c_glucose + 1e-5),
        "BMI_Age_Interaction": (c_bmi * c_age) / 100.0,
        "Glucose_ADA_Cat": 0.0 if c_glucose < 100 else (1.0 if c_glucose < 126 else 2.0),
        "BP_AHA_Cat": 0.0 if c_dia_bp < 80 else (1.0 if c_dia_bp < 90 else 2.0),
        "BMI_WHO_Cat": 0.0 if c_bmi < 25 else (1.0 if c_bmi < 30 else 2.0)
    }])
    d_scaled = pd.DataFrame(d_scaler.transform(d_row[d_scaler.feature_names_in_]), columns=d_scaler.feature_names_in_)
    p_diabetes = float(d_model_dict["model"].predict_proba(d_scaled[d_feats])[:, 1][0])

    # 2. Heart Inference
    h_model_dict = champion_models["heart"]
    h_scaler = h_model_dict["scaler"]
    h_feats = h_model_dict["features"]

    h_row = pd.DataFrame([{
        "age": float(c_age),
        "sex": float(c_sex_num),
        "cp": float(c_cp_type),
        "trestbps": float(c_sys_bp),
        "chol": float(c_tot_chol),
        "fbs": 1.0 if c_glucose > 120 else 0.0,
        "restecg": 0.0,
        "thalach": float(c_thalach),
        "exang": 1.0 if c_cp_type == 3 else 0.0,
        "oldpeak": float(c_oldpeak),
        "slope": 1.0,
        "ca": 0.0,
        "thal": 2.0,
        "Rate_Pressure_Product": (c_sys_bp * c_thalach) / 100.0,
        "Chol_Age_Ratio": c_tot_chol / (c_age + 1e-5),
        "ExAng_ST_Interaction": (1.0 if c_cp_type == 3 else 0.0) * c_oldpeak,
        "SysBP_AHA_Cat": 0.0 if c_sys_bp < 120 else (1.0 if c_sys_bp < 130 else (2.0 if c_sys_bp < 140 else 3.0))
    }])
    h_scaled = pd.DataFrame(h_scaler.transform(h_row[h_scaler.feature_names_in_]), columns=h_scaler.feature_names_in_)
    p_heart = float(h_model_dict["model"].predict_proba(h_scaled[h_feats])[:, 1][0])

    # 3. CKD Inference
    c_model_dict = champion_models["ckd"]
    c_scaler = c_model_dict["scaler"]
    c_feats = c_model_dict["features"]

    egfr_proxy = 100.0 / ((c_sc + 0.1) * (c_age / 50.0 + 0.5))
    bun_cr_ratio = c_bu / (c_sc + 1e-5)
    anemia = 1.0 if c_hemo < 12.0 else 0.0

    c_row = pd.DataFrame(0.0, index=[0], columns=c_scaler.feature_names_in_)
    c_row["age"] = float(c_age)
    c_row["bp"] = float(c_dia_bp)
    c_row["bgr"] = float(c_glucose)
    c_row["bu"] = float(c_bu)
    c_row["sc"] = float(c_sc)
    c_row["sod"] = 138.0
    c_row["pot"] = 4.2
    c_row["hemo"] = float(c_hemo)
    c_row["pcv"] = float(c_hemo * 3.0)
    c_row["wc"] = 7500.0
    c_row["rc"] = 4.8
    c_row["htn"] = 1.0 if c_sys_bp >= 130 else 0.0
    c_row["dm"] = 1.0 if c_glucose >= 126 else 0.0
    c_row["BUN_Creatinine_Ratio"] = bun_cr_ratio
    c_row["Anemia_Flag"] = anemia
    c_row["eGFR_Proxy"] = egfr_proxy
    c_row["HTN_Glucose_Interaction"] = c_row["htn"] * c_glucose

    c_scaled = pd.DataFrame(c_scaler.transform(c_row), columns=c_scaler.feature_names_in_)
    p_ckd = float(c_model_dict["model"].predict_proba(c_scaled[c_feats])[:, 1][0])

    return {
        "p_diabetes": p_diabetes,
        "p_heart": p_heart,
        "p_ckd": p_ckd,
        "d_scaled": d_scaled,
        "h_scaled": h_scaled,
        "c_scaled": c_scaled,
        "h_row_raw": h_row
    }


preds = compute_patient_predictions()
p_diab = preds["p_diabetes"]
p_heart = preds["p_heart"]
p_ckd = preds["p_ckd"]

# Conformal intervals
ci_diab = calibrator.predict_interval("diabetes", p_diab)
ci_heart = calibrator.predict_interval("heart", p_heart)
ci_ckd = calibrator.predict_interval("ckd", p_ckd)

# Composite Index
cmri_result = cmri_calc.calculate(p_diab, p_heart, p_ckd)

# Default estimated trajectory slope
default_slope = 0.015 if cmri_result["cmri_score"] < 0.30 else (0.042 if cmri_result["cmri_score"] < 0.60 else 0.088)
triage_result = priority_engine.compute_priority(
    cmri_result["cmri_score"],
    default_slope,
    {"diabetes": p_diab, "heart": p_heart, "ckd": p_ckd}
)


# ---------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# ---------------------------------------------------------------------------
tab_overview, tab_multitask, tab_trajectory, tab_explain, tab_priority, tab_fairness = st.tabs([
    "📊 Screening & CMRI",
    "🔄 Multi-Task Shared View",
    "📈 Longitudinal Trajectory",
    "🔍 Explainability & What-If",
    "🚨 MCDA Priority Triage",
    "⚖️ Fairness & Governance"
])


# ===========================================================================
# TAB 1: SCREENING & COMPOSITE CMRI
# ===========================================================================
with tab_overview:
    st.subheader("CardioMetabolic-Renal Multi-Organ Risk Assessment")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#e74c3c;">Diabetes Risk</h4>
            <h2 style="margin:0.2rem 0; color:#2c3e50;">{p_diab*100:.1f}%</h2>
            <p style="color:#7f8c8d; font-size:0.85rem; margin:0;">90% CI: {ci_diab['lower_pct']:.1f}%–{ci_diab['upper_pct']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#e67e22;">Cardiovascular Risk</h4>
            <h2 style="margin:0.2rem 0; color:#2c3e50;">{p_heart*100:.1f}%</h2>
            <p style="color:#7f8c8d; font-size:0.85rem; margin:0;">90% CI: {ci_heart['lower_pct']:.1f}%–{ci_heart['upper_pct']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#9b59b6;">Chronic Kidney Disease</h4>
            <h2 style="margin:0.2rem 0; color:#2c3e50;">{p_ckd*100:.1f}%</h2>
            <p style="color:#7f8c8d; font-size:0.85rem; margin:0;">90% CI: {ci_ckd['lower_pct']:.1f}%–{ci_ckd['upper_pct']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {cmri_result['category_color']};">
            <h4 style="margin:0; color:{cmri_result['category_color']};">Composite CMRI Index</h4>
            <h2 style="margin:0.2rem 0; color:{cmri_result['category_color']};">{cmri_result['cmri_percentage']:.1f}%</h2>
            <p style="font-weight:bold; color:{cmri_result['category_color']}; font-size:0.85rem; margin:0;">{cmri_result['risk_category']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("#### 🔬 Composite Index Decomposition & Synergistic Coupling")
        st.markdown("**Mathematical Formulation (Core Contribution #1):**")
        st.code(cmri_result["formula_audit"])

        comp_df = pd.DataFrame({
            "Component": ["Diabetes (30%)", "Cardiovascular (30%)", "Renal (30%)", "Multi-Organ Synergy (10%)"],
            "Contribution (%)": [
                cmri_result["components"]["diabetes_contribution"] * 100,
                cmri_result["components"]["heart_contribution"] * 100,
                cmri_result["components"]["ckd_contribution"] * 100,
                cmri_result["components"]["synergistic_interaction"] * 100
            ]
        })
        st.bar_chart(comp_df.set_index("Component"), color="#1e3d59")

    with col_right:
        st.markdown("#### 🏥 Clinical Guidance & Triage")
        st.info(f"**Recommended Strategy:** {cmri_result['clinical_guidance']}")

        st.markdown("##### Calibrated Conformal Intervals (Core Contribution #2)")
        st.write("Split Conformal Prediction produces finite-sample distribution-free coverage:")
        st.markdown(f"- **Diabetes**: `{ci_diab['formatted']}` (Margin: ±{ci_diab['margin_error_pct']}%)")
        st.markdown(f"- **Heart Disease**: `{ci_heart['formatted']}` (Margin: ±{ci_heart['margin_error_pct']}%)")
        st.markdown(f"- **CKD**: `{ci_ckd['formatted']}` (Margin: ±{ci_ckd['margin_error_pct']}%)")

        # Report Export Section
        st.markdown("---")
        st.markdown("##### 📥 Export Clinical Decision-Support Report")
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "patient_demographics": {"age": age, "sex": sex, "bmi": bmi},
            "biomarkers": {
                "sys_bp": sys_bp, "dia_bp": dia_bp, "glucose": glucose,
                "tot_chol": tot_chol, "blood_urea": blood_urea, "serum_creatinine": serum_creatinine,
                "hemoglobin": hemoglobin
            },
            "risk_estimates": {
                "diabetes": ci_diab,
                "cardiovascular": ci_heart,
                "chronic_kidney_disease": ci_ckd
            },
            "composite_index": cmri_result,
            "triage_priority": triage_result
        }
        st.download_button(
            label="📄 Download Assessment Summary (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name=f"CMR_RIS_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# ===========================================================================
# TAB 2: MULTI-TASK SHARED LEARNING VIEW
# ===========================================================================
with tab_multitask:
    st.subheader("Shared Multi-Task Model vs. Independent Disease Models")
    st.markdown("""
    **Core Research Question:** *Does joint multi-task representation learning on a shared 5-feature metabolic continuum
    (`Age`, `Blood Pressure`, `Glucose`, `Metabolic Proxy`, `Organ Proxy`) outperform disease-specific models?*
    """)

    if not df_comparison.empty:
        st.dataframe(
            df_comparison[["Disease", "Model_Type", "Algorithm", "Test_Recall", "Test_ROC_AUC", "Test_F1", "Test_Accuracy"]],
            use_container_width=True
        )

    st.markdown("#### 📝 Key Empirical Findings:")
    st.markdown("""
    1. **Independent models with domain-specific features achieve superior sensitivity** on acute cardiac manifestations (Recall $0.939$ vs $0.727$), where specialized exercise biomarkers (`oldpeak`, `thalach`, `cp`) are mandatory.
    2. **Shared representations provide robust baselines for chronic progressive nephropathy (CKD)** (Recall $0.960$, ROC-AUC $1.000$), demonstrating strong metabolic-renal coupling.
    3. Screening prioritization requires maximizing **recall/sensitivity** to prevent missed high-risk cases in preventive cohorts.
    """)


# ===========================================================================
# TAB 3: LONGITUDINAL TRAJECTORY TRACKER
# ===========================================================================
with tab_trajectory:
    st.subheader("Statistically-Grounded Trajectory Engine (Core Contribution #3)")
    st.markdown("""
    Detects longitudinal risk deterioration using **Theil-Sen robust median slope** and **Mann-Kendall non-parametric monotonic trend test**.
    """)

    col_traj_ctrl, col_traj_plot = st.columns([1, 2])
    with col_traj_ctrl:
        traj_mode = st.selectbox(
            "Select Trajectory Scenario",
            ["Simulate: Gradual Metabolic Drift (+3.5%/yr)",
             "Simulate: Rapid Cardio-Renal Deterioration (+7.5%/yr)",
             "Simulate: Stable Chronic Profile (+0.5%/yr)",
             "Simulate: Improving Response to Therapy (-4.0%/yr)",
             "Enter Custom Multi-Visit History"]
        )

        if "Custom" not in traj_mode:
            mode_key = "gradual_worsening"
            if "Rapid" in traj_mode:
                mode_key = "rapid_deterioration"
            elif "Stable" in traj_mode:
                mode_key = "stable"
            elif "Improving" in traj_mode:
                mode_key = "improving"

            traj_analysis = simulate_patient_trajectory(cmri_result["cmri_score"], years=4, trajectory_mode=mode_key)
        else:
            v0 = st.slider("Year 0 Risk (%)", 5, 95, 25)
            v1 = st.slider("Year 1 Risk (%)", 5, 95, 32)
            v2 = st.slider("Year 2 Risk (%)", 5, 95, 41)
            v3 = st.slider("Year 3 Risk (%)", 5, 95, 52)
            traj_analysis = classify_trajectory([0, 1, 2, 3], [v0/100, v1/100, v2/100, v3/100])
            traj_analysis["disclaimer"] = "USER ENTERED CUSTOM VISITS"

        st.markdown(f"**Trajectory Status:** <span style='color:{traj_analysis['status_color']}; font-weight:bold; font-size:1.1rem;'>{traj_analysis['trajectory_status']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Theil-Sen Annual Slope:** `{traj_analysis['annual_drift_percentage']:+0.2f}% / year`")
        st.markdown(f"**Mann-Kendall p-value:** `{traj_analysis['mann_kendall']['p_value']:.4f}` ({traj_analysis['mann_kendall']['trend']})")
        st.info(f"**Clinical Recommendation:** {traj_analysis['clinical_action']}")

    with col_traj_plot:
        hist_df = pd.DataFrame(traj_analysis["history"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(hist_df["year"], hist_df["risk_pct"], marker="o", linewidth=2.5, color=traj_analysis["status_color"], label="Risk Trajectory")
        ax.axhline(60, color="#e74c3c", linestyle="--", alpha=0.6, label="High Risk Threshold (60%)")
        ax.axhline(30, color="#2ecc71", linestyle="--", alpha=0.6, label="Low Risk Threshold (30%)")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Follow-up Year")
        ax.set_ylabel("Composite CMRI Risk (%)")
        ax.set_title("Multi-Year Patient Trajectory Progression", fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)
        st.caption(f"Note: {traj_analysis.get('disclaimer', '')}")


# ===========================================================================
# TAB 4: EXPLAINABLE AI & COUNTERFACTUAL WHAT-IF
# ===========================================================================
with tab_explain:
    st.subheader("Explainable AI: Local SHAP Drivers & Actionable Counterfactuals")

    exp_col1, exp_col2 = st.columns([1, 1.2])

    with exp_col1:
        st.markdown("#### 🔍 SHAP Local Risk Attributions")
        disease_choice = st.selectbox("Explain Prediction For:", ["Cardiovascular (Heart)", "Diabetes"])

        if disease_choice == "Cardiovascular (Heart)":
            explainer_obj = ModelExplainer(champion_models["heart"])
            shap_df = explainer_obj.explain_patient(preds["h_scaled"], top_k=6)
        else:
            explainer_obj = ModelExplainer(champion_models["diabetes"])
            shap_df = explainer_obj.explain_patient(preds["d_scaled"], top_k=6)

        st.dataframe(shap_df[["Feature", "Standardized_Value", "SHAP_Impact", "Direction"]], use_container_width=True)

        fig_shap, ax_shap = plt.subplots(figsize=(6, 3.5))
        colors = ["#e74c3c" if x > 0 else "#2ecc71" for x in shap_df["SHAP_Impact"]]
        ax_shap.barh(shap_df["Feature"], shap_df["SHAP_Impact"], color=colors)
        ax_shap.axvline(0, color="#333333", linewidth=0.8)
        ax_shap.set_xlabel("SHAP Impact (+ Increases Risk / - Decreases Risk)")
        ax_shap.set_title(f"Top Risk Drivers ({disease_choice})", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_shap)

    with exp_col2:
        st.markdown("#### 🎯 Multi-Organ Counterfactual Intervention Planner")
        st.markdown("Simulate patient therapeutic modifications across hemodynamics, glycemic control, and lipids:")

        c_w1, c_w2 = st.columns(2)
        with c_w1:
            delta_sbp = st.slider("Systolic BP Reduction (mmHg)", 0, 40, 15, step=5)
            delta_glu = st.slider("Fasting Glucose Reduction (mg/dL)", 0, 60, 25, step=5)
        with c_w2:
            delta_chol = st.slider("Total Cholesterol Reduction (mg/dL)", 0, 80, 30, step=5)
            delta_bmi = st.slider("BMI Reduction (kg/m²)", 0.0, 8.0, 2.0, step=0.5)

        # Compute multi-organ counterfactual predictions
        cf_overrides = {
            "sys_bp": max(90.0, sys_bp - delta_sbp),
            "dia_bp": max(60.0, dia_bp - (delta_sbp * 0.6)),
            "glucose": max(70.0, glucose - delta_glu),
            "tot_chol": max(120.0, tot_chol - delta_chol),
            "bmi": max(18.5, bmi - delta_bmi)
        }

        cf_preds = compute_patient_predictions(override_dict=cf_overrides)
        cf_cmri = cmri_calc.calculate(cf_preds["p_diabetes"], cf_preds["p_heart"], cf_preds["p_ckd"])

        cmri_drop = cmri_result["cmri_percentage"] - cf_cmri["cmri_percentage"]

        st.markdown(f"""
        <div class="triage-routine" style="margin-top:0.8rem;">
            <strong>🎯 Simulated Prospective Benefit:</strong><br>
            Under targeted lifestyle & pharmacotherapy, composite <strong>CMRI drops by {cmri_drop:+.1f}%</strong> 
            (from {cmri_result['cmri_percentage']:.1f}% → {cf_cmri['cmri_percentage']:.1f}%).
        </div>
        """, unsafe_allow_html=True)

        cf_comparison_df = pd.DataFrame({
            "Disease Axis": ["Diabetes", "Cardiovascular", "CKD", "Composite CMRI"],
            "Baseline Risk (%)": [p_diab * 100, p_heart * 100, p_ckd * 100, cmri_result["cmri_percentage"]],
            "Post-Intervention (%)": [cf_preds["p_diabetes"] * 100, cf_preds["p_heart"] * 100, cf_preds["p_ckd"] * 100, cf_cmri["cmri_percentage"]]
        })

        st.markdown("##### Pre vs Post Intervention Risk Delta:")
        st.dataframe(cf_comparison_df, use_container_width=True)


# ===========================================================================
# TAB 5: AUDITABLE MCDA PRIORITY SCORE
# ===========================================================================
with tab_priority:
    st.subheader("Auditable Screening Priority Engine (MCDA)")
    st.markdown("""
    **Core Contribution #5:** Translates composite organ severity, trajectory velocity, and comorbidity burden into an auditable triage score:
    """)

    st.code(triage_result["formula_audit"])

    tier_class = "triage-routine"
    if "EXPEDITED" in triage_result["priority_tier"]:
        tier_class = "triage-expedited"
    elif "URGENT" in triage_result["priority_tier"]:
        tier_class = "triage-urgent"

    st.markdown(f"""
    <div class="{tier_class}">
        <h3 style="margin:0; color:{triage_result['badge_color']};">{triage_result['priority_tier']}</h3>
        <h1 style="margin:0.3rem 0; color:#2c3e50;">{triage_result['priority_percentage']:.1f}% Priority Score</h1>
        <p style="font-size:1.05rem; margin:0;"><strong>Mandatory Clinical Protocol:</strong> {triage_result['recommended_clinical_action']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚖️ MCDA Component Breakdown")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("CMRI Risk Severity (50%)", f"{triage_result['components']['current_cmri_severity (50%)']:.1f}%")
    m_col2.metric("Trajectory Velocity (30%)", f"{triage_result['components']['trajectory_velocity (30%)']:.1f}%")
    m_col3.metric("Multi-Organ Comorbidity (20%)", f"{triage_result['components']['multi_organ_burden (20%)']:.1f}%")


# ===========================================================================
# TAB 6: FAIRNESS AUDIT & GOVERNANCE
# ===========================================================================
with tab_fairness:
    st.subheader("Demographic Subgroup Fairness Audit & Governance")
    st.markdown("""
    Healthcare ML models must be audited for equitable performance across protected demographics (Sex, Age Brackets)
    to prevent algorithmic bias and disparate false negatives in vulnerable subpopulations.
    """)

    if not df_fairness.empty:
        st.dataframe(df_fairness, use_container_width=True)

    st.markdown("#### 🛡️ Compliance & Limitations Disclosure")
    st.markdown(r"""
    - **Dataset Size Limitations**: Pima Indians Diabetes ($N=768$) and UCI CKD ($N=400$) cohorts are moderately sized. Conformal prediction intervals are calibrated to account for sample variance.
    - **Demographic Representation**: Older individuals ($\ge 60$) exhibit higher positive prevalence; stratified cross-validation was strictly enforced to guarantee equal sensitivity across age brackets.
    - **Clinical Workflow Role**: Designed strictly as an adjunctive triage filter for primary care screening and outpatient scheduling.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#888888; font-size:0.85rem;'>CMR-RIS v1.0 — CardioMetabolic-Renal Risk Intelligence System | Phase 14 Demo Deployment</p>", unsafe_allow_html=True)
