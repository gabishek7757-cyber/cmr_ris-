# CMR-RIS: CardioMetabolic-Renal Risk Intelligence System
### A Unified, Trajectory-Aware, Explainable Early-Warning System for Diabetes, Cardiovascular Disease, and Chronic Kidney Disease

---

## 1. Why This Version Is More Publishable

Your original plan (3 independent models + SHAP + a trend chart) is a **correct** end-to-end pipeline, but it reads as three small projects bolted together. Reviewers/journals for this kind of work look for a *single conceptual contribution*, not three trained classifiers. The upgrade below gives you that contribution:

| Weakness in original plan | Upgrade |
|---|---|
| Diabetes, heart disease, CKD treated as unrelated problems | Modeled as one **interconnected metabolic-vascular-renal risk continuum** (clinically real: T2D, CVD and CKD share overlapping pathophysiology — often called cardiovascular-kidney-metabolic (CKM) syndrome in current literature) |
| Trajectory states ("stable", "increasing") defined informally | Trajectory computed with a **documented statistical trend test** (Theil-Sen slope + Mann-Kendall significance), not eyeballed thresholds |
| Risk categories (LOW/MODERATE/HIGH) are just probability cutoffs | Add **conformal prediction intervals** — a HIGH risk of 78% ± tight interval is a different claim than 78% ± wide interval, and reviewers will ask for this |
| Screening Priority is described but not mathematically defined | Defined as an explicit **Multi-Criteria Decision Analysis (MCDA)** weighted formula, fully auditable |
| No fairness/bias discussion | Add a subgroup fairness audit (age, sex) — standard expectation in any healthcare-ML paper today |
| No real longitudinal data addressed | Explicitly separate **real longitudinal validation** (Framingham Heart Study, which has genuine repeated-visit data) from a clearly-labeled **synthetic trajectory simulator** used only for the CKD/diabetes demo, where public data is cross-sectional |

This turns the project from "three sklearn models in a Streamlit app" into a defensible **system-level contribution**: a shared risk-continuum framework + statistically grounded monitoring + auditable prioritization — which is exactly the kind of framing PBL/paper reviewers reward.

---

## 2. Revised Objective Statement

> To design and implement CMR-RIS, a machine-learning decision-support system that (a) estimates individualized risk across three physiologically interconnected chronic conditions — diabetes, cardiovascular disease, and chronic kidney disease — using both disease-specific and shared-representation models; (b) quantifies estimation uncertainty; (c) statistically tracks risk trajectories over time; (d) explains predictions at both the global and individual level; and (e) produces a transparent, formula-based screening-priority score to support (not replace) clinical triage.

---

## 3. System Architecture

```
                    ┌───────────────────────────┐
                    │   Patient Health Record    │
                    └──────────────┬────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │  Preprocessing (per disease)│
                    │  imputation · encoding ·    │
                    │  scaling · leakage checks   │
                    └──────────────┬────────────┘
                                   ▼
        ┌──────────────────────────────────────────────┐
        │              Modeling Layer (dual)            │
        │                                                │
        │  (A) Disease-specific models                   │
        │      LogReg / RF / XGB / SVM  ×3 diseases       │
        │                                                │
        │  (B) Shared-representation multi-task model     │
        │      (captures cross-disease correlation via    │
        │       shared metabolic features: BMI, glucose,  │
        │       BP, lipids, age, eGFR proxy)               │
        └──────────────┬─────────────────┬────────────────┘
                        ▼                 ▼
              Per-disease probability   Composite CMR Index
                        │                 │
                        ▼                 ▼
        ┌───────────────────────────────────────────┐
        │  Conformal Prediction → uncertainty band    │
        └──────────────────┬──────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │  Risk Stratification (LOW/MODERATE/HIGH)    │
        └──────────────────┬──────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │  Explainable AI: SHAP + counterfactual        │
        │  ("risk would drop to X% if glucose − Y")     │
        └──────────────────┬──────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │  Trajectory Engine (Theil-Sen + Mann-Kendall) │
        └──────────────────┬──────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │  MCDA Screening-Priority Score (documented)   │
        └──────────────────┬──────────────────────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │        Streamlit Dashboard (clinician +       │
        │        patient view) + fairness report tab    │
        └───────────────────────────────────────────┘
```

---

## 4. Datasets (real, publicly available)

| Disease | Dataset | Notes |
|---|---|---|
| Diabetes | Pima Indians Diabetes (UCI/Kaggle) | Cross-sectional; use for classification only |
| Heart Disease | UCI Heart Disease (Cleveland/Statlog combined) | Cross-sectional classification |
| Heart Disease (trajectory validation) | **Framingham Heart Study** (Kaggle mirror, teaching subset) | Has genuine repeated-exam structure — use this for a *real* trajectory demonstration, not synthetic |
| CKD | UCI Chronic Kidney Disease dataset | Cross-sectional classification |

Because Pima/CKD datasets are cross-sectional (one row per patient, no repeat visits), be explicit in the report: **trajectory analysis is demonstrated on Framingham (real longitudinal data) and additionally on a clearly labeled synthetic simulator for diabetes/CKD**, generated using clinically plausible progression rates (e.g., glucose drift per year) sourced from published literature and cited as such. Never present the synthetic trajectories as real patient outcomes.

---

## 5. The Five Technical Contributions (this is your "paper")

### 5.1 Composite CardioMetabolic-Renal Risk Index (CMRI)
A single interpretable score combining all three disease probabilities, not just three separate percentages:

```
CMRI = w1·P(diabetes) + w2·P(heart) + w3·P(ckd) + w4·P(diabetes)·P(heart)·P(ckd)
```
The interaction term captures compounding risk when multiple systems are elevated simultaneously — document weight selection (start with equal weights, then justify via literature or a simple logistic meta-model trained on the outcome "≥2 elevated risks").

### 5.2 Shared-Representation Multi-Task Model
Train one multi-output model (e.g., a shared-trunk neural net or multi-output gradient boosting) on the union of common features (age, BMI, blood pressure, glucose, lipids) predicting all three disease probabilities jointly, and **compare its performance against the three independent models**. This comparison — does shared learning help or hurt? — is itself a genuine research question and gives you a results section that isn't just "our model got 0.85 AUC."

### 5.3 Statistically Grounded Trajectory Classification
Replace ad hoc "increasing/decreasing" labels with:
- **Theil-Sen slope estimator** on the time series of risk scores (robust to outliers)
- **Mann-Kendall trend test** for statistical significance of the trend
- Classify: Stable (non-significant trend), Improving (significant negative slope), Gradually Increasing (significant positive slope, low magnitude), Rapidly Increasing (significant positive slope, high magnitude — threshold documented from the data's own slope distribution, not invented)

### 5.4 Uncertainty via Conformal Prediction
Wrap each disease model with split conformal prediction to output a calibrated interval (e.g., "78% risk, 90% confidence interval 71–84%"). This directly addresses the most common reviewer criticism of healthcare-ML projects: presenting a bare point probability as if it were exact.

### 5.5 Auditable MCDA Screening Priority
```
Priority Score = α·(normalized current CMRI) + β·(normalized trajectory slope) + γ·(count of HIGH-risk diseases / 3)
```
Document α, β, γ explicitly (e.g., 0.5/0.3/0.2), publish the formula in the dashboard itself, and show priority bucket boundaries. This makes the "HIGH/MEDIUM/LOW priority" output fully traceable — a core requirement for any clinical decision-support paper.

**Add-on for rigor (if time permits):** a subgroup fairness audit — recompute recall/precision of each disease model split by sex and by age band, and report disparities. Even a simple table here substantially strengthens a PBL/paper submission.

---

## 6. Same-Day Execution Timeline

Realistic for a focused single day (~9–10 hrs). Core = must finish. Stretch = do if ahead of schedule.

| Time block | Task | Priority |
|---|---|---|
| Hr 1 | Download all 4 datasets, initial inspection (shape, missingness, target balance, leakage check) | Core |
| Hr 2 | Preprocessing pipeline (shared `preprocessing.py`): imputation, encoding, scaling, stratified split for all 3 diseases | Core |
| Hr 3 | EDA notebook per disease (distributions, correlation heatmaps, target vs feature plots) — keep concise, 6–8 plots each | Core |
| Hr 4 | Train disease-specific models (LogReg/RF/XGB/SVM) × 3 diseases, build comparison table (Acc/Prec/Recall/F1/ROC-AUC), select final model per disease prioritizing recall | Core |
| Hr 5 | Train shared multi-task model on common features; compare vs disease-specific models — write 3–4 sentence finding | Core |
| Hr 6 | Implement CMRI composite score + conformal prediction wrapper for uncertainty bands | Core |
| Hr 7 | Trajectory engine: Theil-Sen + Mann-Kendall on Framingham subset (real) + synthetic simulator for diabetes/CKD (clearly labeled) | Core |
| Hr 8 | SHAP explainability (global + per-patient) + one counterfactual example ("if glucose dropped by X, risk falls to Y%") | Core |
| Hr 9 | MCDA priority score module + Streamlit dashboard (patient input → risk table → trend chart → SHAP bars → priority) | Core |
| Hr 10 | Fairness audit table + limitations section + disclaimer text + README + slide outline | Stretch (do fairness table first if time is short; write-up can be finished after) |

---

## 7. Deliverables Checklist

- [ ] `data/` — 4 raw datasets + data dictionary
- [ ] `notebooks/` — EDA per disease
- [ ] `src/preprocessing.py`, `modeling.py`, `cmri.py`, `trajectory.py`, `explainability.py`, `priority.py`
- [ ] `models/` — saved per-disease models + shared multi-task model
- [ ] `app.py` — Streamlit dashboard (clinician view + patient view)
- [ ] Model comparison tables (disease-specific vs shared model)
- [ ] Trajectory validation plots (Framingham real + synthetic demo, clearly labeled)
- [ ] Fairness audit table
- [ ] `README.md` with disclaimer, limitations, and methodology summary
- [ ] Report/PPT following the paper skeleton below

---

## 8. Suggested Paper / Report Skeleton

1. **Title:** CMR-RIS: A Multi-Task, Trajectory-Aware, Explainable AI System for Cardiometabolic-Renal Risk Screening
2. **Abstract** — problem, the CKM-continuum framing, 5 contributions, headline result (e.g., recall achieved, whether multi-task helped)
3. **Introduction** — why treating these 3 diseases jointly matters clinically
4. **Related Work** — single-disease prediction papers vs your multi-disease framing
5. **Data & Preprocessing**
6. **Methodology** — the 5 contributions (Section 5 above)
7. **Results** — model comparison tables, multi-task vs independent comparison, trajectory validation, fairness table
8. **Discussion** — what the composite index and priority score add over single-disease tools
9. **Limitations** — dataset size/bias, cross-sectional vs longitudinal, non-clinical thresholds, no external validation
10. **Conclusion & Future Work** — external validation on hospital data, more diseases, calibration studies

---

## 9. Mandatory Disclaimer (keep in every report/app screen)

> This system is an experimental machine-learning risk-screening and decision-support tool. It does not diagnose disease, is not clinically validated, and must not replace evaluation by a qualified healthcare professional. Risk thresholds and priority scores are project-defined, not clinical standards.
