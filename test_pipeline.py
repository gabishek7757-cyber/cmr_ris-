"""
Phase 15 — Edge Case Testing
Exercises the exact same feature-construction + inference path as app.py,
but headless (no Streamlit), so it can run and log results automatically.
"""
import sys, os, traceback
sys.path.insert(0, ".")
import joblib
import pandas as pd
import numpy as np
from src.cmri import CMRIndexCalculator
from src.uncertainty import ConformalRiskCalibrator
from src.priority import MCDAPriorityEngine

models = {d: joblib.load(f"models/{d}_champion.pkl") for d in ["diabetes", "heart", "ckd"]}
calibrator = ConformalRiskCalibrator()
c_data = joblib.load("models/conformal_calibrator.pkl")
if isinstance(c_data, dict):
    calibrator.calibrated_quantiles = c_data.get("calibrated_quantiles", {})
    calibrator.confidence_level = c_data.get("confidence_level", 0.90)
else:
    calibrator = c_data
cmri_calc = CMRIndexCalculator()
priority_engine = MCDAPriorityEngine()


def predict_patient(p):
    """p: dict of raw clinical inputs. Mirrors compute_patient_predictions() in app.py."""
    glucose, dia_bp, sys_bp = p["glucose"], p["dia_bp"], p["sys_bp"]
    bmi, age, dpf, pregnancies = p["bmi"], p["age"], p["dpf"], p["pregnancies"]
    sex_num, cp_type, tot_chol, thalach, oldpeak = p["sex_num"], p["cp_type"], p["tot_chol"], p["thalach"], p["oldpeak"]
    blood_urea, serum_creatinine, hemoglobin = p["blood_urea"], p["serum_creatinine"], p["hemoglobin"]

    # Diabetes
    d_model_dict = models["diabetes"]; d_scaler = d_model_dict["scaler"]; d_feats = d_model_dict["features"]
    insulin_val, skin_val = 120.0, 25.0
    d_row = pd.DataFrame([{
        "Pregnancies": float(pregnancies), "Glucose": float(glucose), "BloodPressure": float(dia_bp),
        "SkinThickness": skin_val, "Insulin": insulin_val, "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf), "Age": float(age),
        "Insulin_Glucose_Ratio": insulin_val / (glucose + 1e-5),
        "BMI_Age_Interaction": (bmi * age) / 100.0,
        "Glucose_ADA_Cat": 0.0 if glucose < 100 else (1.0 if glucose < 126 else 2.0),
        "BP_AHA_Cat": 0.0 if dia_bp < 80 else (1.0 if dia_bp < 90 else 2.0),
        "BMI_WHO_Cat": 0.0 if bmi < 25 else (1.0 if bmi < 30 else 2.0)
    }])
    d_scaled = pd.DataFrame(d_scaler.transform(d_row[d_scaler.feature_names_in_]), columns=d_scaler.feature_names_in_)
    p_diabetes = float(d_model_dict["model"].predict_proba(d_scaled[d_feats])[:, 1][0])

    # Heart
    h_model_dict = models["heart"]; h_scaler = h_model_dict["scaler"]; h_feats = h_model_dict["features"]
    h_row = pd.DataFrame([{
        "age": float(age), "sex": float(sex_num), "cp": float(cp_type), "trestbps": float(sys_bp),
        "chol": float(tot_chol), "fbs": 1.0 if glucose > 120 else 0.0, "restecg": 0.0,
        "thalach": float(thalach), "exang": 1.0 if cp_type == 3 else 0.0, "oldpeak": float(oldpeak),
        "slope": 1.0, "ca": 0.0, "thal": 2.0,
        "Rate_Pressure_Product": (sys_bp * thalach) / 100.0,
        "Chol_Age_Ratio": tot_chol / (age + 1e-5),
        "ExAng_ST_Interaction": (1.0 if cp_type == 3 else 0.0) * oldpeak,
        "SysBP_AHA_Cat": 0.0 if sys_bp < 120 else (1.0 if sys_bp < 130 else (2.0 if sys_bp < 140 else 3.0))
    }])
    h_scaled = pd.DataFrame(h_scaler.transform(h_row[h_scaler.feature_names_in_]), columns=h_scaler.feature_names_in_)
    p_heart = float(h_model_dict["model"].predict_proba(h_scaled[h_feats])[:, 1][0])

    # CKD
    c_model_dict = models["ckd"]; c_scaler = c_model_dict["scaler"]; c_feats = c_model_dict["features"]
    egfr_proxy = 100.0 / ((serum_creatinine + 0.1) * (age / 50.0 + 0.5))
    bun_cr_ratio = blood_urea / (serum_creatinine + 1e-5)
    anemia = 1.0 if hemoglobin < 12.0 else 0.0
    c_row = pd.DataFrame(0.0, index=[0], columns=c_scaler.feature_names_in_)
    c_row["age"] = float(age); c_row["bp"] = float(dia_bp); c_row["bgr"] = float(glucose)
    c_row["bu"] = float(blood_urea); c_row["sc"] = float(serum_creatinine)
    c_row["sod"] = 138.0; c_row["pot"] = 4.2; c_row["hemo"] = float(hemoglobin)
    c_row["pcv"] = float(hemoglobin * 3.0); c_row["wc"] = 7500.0; c_row["rc"] = 4.8
    c_row["htn"] = 1.0 if sys_bp >= 130 else 0.0; c_row["dm"] = 1.0 if glucose >= 126 else 0.0
    c_row["BUN_Creatinine_Ratio"] = bun_cr_ratio; c_row["Anemia_Flag"] = anemia
    c_row["eGFR_Proxy"] = egfr_proxy; c_row["HTN_Glucose_Interaction"] = c_row["htn"] * glucose
    c_scaled = pd.DataFrame(c_scaler.transform(c_row), columns=c_scaler.feature_names_in_)
    p_ckd = float(c_model_dict["model"].predict_proba(c_scaled[c_feats])[:, 1][0])

    ci_d = calibrator.predict_interval("diabetes", p_diabetes)
    ci_h = calibrator.predict_interval("heart", p_heart)
    ci_c = calibrator.predict_interval("ckd", p_ckd)
    cmri = cmri_calc.calculate(p_diabetes, p_heart, p_ckd)
    slope = 0.015 if cmri["cmri_score"] < 0.30 else (0.042 if cmri["cmri_score"] < 0.60 else 0.088)
    triage = priority_engine.compute_priority(cmri["cmri_score"], slope, {"diabetes": p_diabetes, "heart": p_heart, "ckd": p_ckd})

    return {"p_diabetes": p_diabetes, "p_heart": p_heart, "p_ckd": p_ckd,
            "ci_diabetes": ci_d, "ci_heart": ci_h, "ci_ckd": ci_c,
            "cmri": cmri, "triage": triage}


# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------
cases = {
    "normal_patient": dict(glucose=110, dia_bp=75, sys_bp=120, bmi=24, age=35, dpf=0.4,
                            pregnancies=1, sex_num=0, cp_type=0, tot_chol=180, thalach=150,
                            oldpeak=0.5, blood_urea=15, serum_creatinine=0.9, hemoglobin=13.5),

    "all_low_risk": dict(glucose=85, dia_bp=65, sys_bp=105, bmi=20, age=22, dpf=0.1,
                          pregnancies=0, sex_num=0, cp_type=0, tot_chol=150, thalach=170,
                          oldpeak=0.0, blood_urea=10, serum_creatinine=0.6, hemoglobin=14.5),

    "all_high_risk": dict(glucose=230, dia_bp=105, sys_bp=170, bmi=41, age=68, dpf=1.8,
                           pregnancies=6, sex_num=1, cp_type=3, tot_chol=310, thalach=95,
                           oldpeak=4.5, blood_urea=90, serum_creatinine=4.2, hemoglobin=8.5),

    "extreme_glucose_900": dict(glucose=900, dia_bp=75, sys_bp=120, bmi=24, age=35, dpf=0.4,
                                 pregnancies=1, sex_num=0, cp_type=0, tot_chol=180, thalach=150,
                                 oldpeak=0.5, blood_urea=15, serum_creatinine=0.9, hemoglobin=13.5),

    "extreme_zero_values": dict(glucose=1, dia_bp=1, sys_bp=1, bmi=1, age=1, dpf=0.01,
                                 pregnancies=0, sex_num=0, cp_type=0, tot_chol=1, thalach=1,
                                 oldpeak=0.0, blood_urea=1, serum_creatinine=0.1, hemoglobin=1),

    "elderly_borderline": dict(glucose=126, dia_bp=90, sys_bp=140, bmi=30, age=75, dpf=0.9,
                                pregnancies=3, sex_num=1, cp_type=1, tot_chol=240, thalach=120,
                                oldpeak=1.5, blood_urea=40, serum_creatinine=1.4, hemoglobin=11.8),
}

log_lines = ["# Phase 15 — Edge Case Test Results\n", "Auto-generated by `test_pipeline.py`. Run against the real trained models, CMRI, conformal calibrator, and priority engine.\n"]

print(f"{'Test Case':<24} {'P(Diab)':>8} {'P(Heart)':>9} {'P(CKD)':>8} {'CMRI%':>7} {'Category':<16} {'Priority':<10} {'Status'}")
print("-" * 110)

all_passed = True
for name, patient in cases.items():
    try:
        r = predict_patient(patient)
        probs = [r["p_diabetes"], r["p_heart"], r["p_ckd"], r["cmri"]["cmri_score"]]
        sane = all(0.0 <= v <= 1.0 and not np.isnan(v) for v in probs)
        status = "PASS" if sane else "FAIL (out of range/NaN)"
        if not sane:
            all_passed = False
        print(f"{name:<24} {r['p_diabetes']*100:>7.1f}% {r['p_heart']*100:>8.1f}% {r['p_ckd']*100:>7.1f}% "
              f"{r['cmri']['cmri_percentage']:>6.1f}% {r['cmri']['risk_category']:<16} "
              f"{r['triage'].get('priority_category', r['triage'].get('priority_level','?')):<10} {status}")

        log_lines.append(f"### {name}")
        log_lines.append(f"- Input: {patient}")
        log_lines.append(f"- P(Diabetes): {r['p_diabetes']:.4f} | 90% CI: {r['ci_diabetes']}")
        log_lines.append(f"- P(Heart): {r['p_heart']:.4f} | 90% CI: {r['ci_heart']}")
        log_lines.append(f"- P(CKD): {r['p_ckd']:.4f} | 90% CI: {r['ci_ckd']}")
        log_lines.append(f"- CMRI: {r['cmri']['cmri_percentage']}% -> {r['cmri']['risk_category']}")
        log_lines.append(f"- Priority: {r['triage']}")
        log_lines.append(f"- **Result: {status}**\n")
    except Exception as e:
        all_passed = False
        print(f"{name:<24} CRASHED: {e}")
        log_lines.append(f"### {name}\n- **Result: CRASH** — {e}\n```\n{traceback.format_exc()}\n```\n")

print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED — see above"))

with open("TESTING_RESULTS.md", "w") as f:
    f.write("\n".join(log_lines))
print("\nFull log written to TESTING_RESULTS.md")
