# CMR-RIS — Complete Project Roadmap

This is the full phase-by-phase roadmap for the entire project, from environment setup to final paper/demo. Each phase lists its goal, concrete tasks, inputs/outputs, tools, dependencies, and a time estimate. Phases are numbered in execution order — don't start a phase until its dependencies are done.

---

## Roadmap at a Glance

| # | Phase | Depends on | Est. time |
|---|---|---|---|
| 0 | Environment & Repo Setup | — | 20 min |
| 1 | Dataset Acquisition & Validation | 0 | 40 min |
| 2 | Preprocessing Pipeline | 1 | 45 min |
| 3 | Exploratory Data Analysis | 2 | 50 min |
| 4 | Feature Engineering | 3 | 30 min |
| 5 | Disease-Specific Modeling | 4 | 60 min |
| 6 | Shared Multi-Task Modeling | 4 | 45 min |
| 7 | Model Evaluation & Selection | 5, 6 | 30 min |
| 8 | Composite CMR Index (CMRI) | 7 | 25 min |
| 9 | Uncertainty Quantification (Conformal) | 7 | 30 min |
| 10 | Trajectory Engine | 1 (Framingham), 8 | 45 min |
| 11 | Explainable AI (SHAP + Counterfactuals) | 7 | 40 min |
| 12 | Screening Priority (MCDA) | 8, 10 | 25 min |
| 13 | Fairness Audit | 7 | 30 min |
| 14 | Streamlit Dashboard | 8–13 | 70 min |
| 15 | Testing | 14 | 40 min |
| 16 | Documentation, Paper Skeleton, PPT | all | 60 min |
| 17 | Final Packaging & Demo Run | 15, 16 | 20 min |

**Total: ~9–10 focused hours.** Core path (0→9→10→11→12→14→15) is mandatory; Phase 13 (fairness) and parts of 16 (full paper writing) can be trimmed if time runs short — but keep at least a stub for both, since their absence is the most common reviewer complaint.

---

## Phase 0 — Environment & Repo Setup
**Goal:** A working, reproducible project skeleton.

**Tasks**
- Create project folder using the structure below.
- Set up virtual environment; `pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib seaborn scipy joblib`.
- Initialize git repo, `.gitignore` for data/models if needed.

**Output:** Empty but structured repo, working Python environment.

```
cmr_ris/
├── data/{diabetes.csv, heart.csv, ckd.csv, framingham.csv}
├── notebooks/
├── models/
├── src/{preprocessing.py, modeling.py, cmri.py, uncertainty.py,
│        trajectory.py, explainability.py, priority.py, fairness.py}
├── app.py
├── requirements.txt
└── README.md
```

---

## Phase 1 — Dataset Acquisition & Validation
**Goal:** Four validated datasets, each understood before any code touches them.

**Tasks**
- Download: Pima Diabetes, UCI Heart Disease (Cleveland/Statlog), UCI CKD, Framingham (teaching subset).
- For each: log shape, dtypes, missing-value %, duplicate count, target balance, and any leakage-risk columns (e.g., columns only available after diagnosis).
- Write a one-paragraph note per dataset: what it measures, population, known limitations.

**Output:** `data/` populated + a short `data/data_notes.md` documenting the above for all four datasets.

**Watch for:** CKD and Pima datasets are small (a few hundred rows) — flag this explicitly as a limitation now, not later.

---

## Phase 2 — Preprocessing Pipeline
**Goal:** Clean, model-ready tables for all three diseases, built through one shared, reusable module.

**Tasks**
- `src/preprocessing.py`: functions for median/mode imputation, categorical encoding, scaling, stratified train/test split (per disease).
- Apply SMOTE/class-weighting only after the split, never before.
- Save cleaned train/test splits (or regenerate on demand from raw + fixed random seed — prefer this for reproducibility).

**Output:** Reusable `preprocessing.py`, verified clean data for diabetes/heart/CKD.

---

## Phase 3 — Exploratory Data Analysis
**Goal:** Understand each dataset before modeling; produce the plots your report will actually use.

**Tasks (per disease, in its own notebook)**
- Target distribution, missingness heatmap, feature histograms/box plots.
- Correlation heatmap; feature-vs-target plots for top candidates (glucose, BMI, BP, cholesterol, eGFR-related features).
- One paragraph per notebook: what looks predictive, what looks noisy, any surprising outliers and whether they look like real physiology vs data errors.

**Output:** 3 EDA notebooks, ~6–8 plots each, no claims of causation.

---

## Phase 4 — Feature Engineering
**Goal:** A small set of justified derived features, not arbitrary rules.

**Tasks**
- BMI categories, age groups, BP categories — using standard, citable medical reference ranges (cite the source in code comments/README).
- Identify the **shared feature set** across all three diseases (age, BMI, systolic/diastolic BP, glucose, lipids) — this shared set is what Phase 6's multi-task model will use.
- Confirm no engineered feature leaks post-outcome information.

**Output:** Updated preprocessing pipeline producing both disease-specific and shared feature matrices.

---

## Phase 5 — Disease-Specific Modeling
**Goal:** Independent baseline for each disease.

**Tasks**
- Train Logistic Regression, Random Forest, XGBoost, SVM for each of the 3 diseases.
- 5-fold stratified CV; record Accuracy, Precision, Recall, F1, ROC-AUC for each model/disease.

**Output:** `models/{disease}_{algo}.pkl` for all combinations + a raw results table (12 rows: 3 diseases × 4 algorithms).

---

## Phase 6 — Shared Multi-Task Modeling
**Goal:** Test whether joint learning across diseases beats independent models — this is your core research question.

**Tasks**
- Build one multi-output model (multi-output gradient boosting, or a small shared-trunk neural net with 3 output heads) trained on the shared feature set from Phase 4, predicting all 3 disease probabilities jointly.
- Same CV protocol and metrics as Phase 5, so the comparison is apples-to-apples.

**Output:** `models/shared_multitask_model.pkl` + comparison numbers ready for Phase 7.

---

## Phase 7 — Model Evaluation & Selection
**Goal:** Pick the final model per disease, and write the multi-task vs independent finding.

**Tasks**
- Build the full comparison table (independent models × multi-task model, all 3 diseases).
- Select final per-disease model prioritizing recall (screening context), not raw accuracy — document why.
- Write 3–5 sentences: did the shared model help, hurt, or tie? For which disease? Hypothesize why (e.g., shared metabolic features help CKD more than heart disease, or vice versa).

**Output:** Final model table + `results/model_comparison.csv` + written finding (goes straight into the paper's Results section).

---

## Phase 8 — Composite CardioMetabolic-Renal Index (CMRI)
**Goal:** One interconnected score, not three isolated percentages.

**Tasks**
- Implement `src/cmri.py`: `CMRI = w1·P(diabetes) + w2·P(heart) + w3·P(ckd) + w4·P(diabetes)·P(heart)·P(ckd)`.
- Start with w1=w2=w3=0.3, w4=0.1 (documented as a starting choice); optionally fit w's via a simple logistic meta-model predicting "≥2 elevated risks" as a sanity check.
- Unit test on a few synthetic patient profiles (all-low, all-high, one-high-two-low) to confirm sane behavior.

**Output:** Working `cmri.py`, documented weight rationale.

---

## Phase 9 — Uncertainty Quantification (Conformal Prediction)
**Goal:** Every risk score ships with a calibrated interval, not a bare point estimate.

**Tasks**
- Implement split conformal prediction per disease model using a held-out calibration set.
- Output format: `"Diabetes Risk: 78% (90% CI: 71–84%)"`.
- Sanity-check calibration: for a 90% interval, roughly 90% of held-out true outcomes should fall inside it — report this check number.

**Output:** `src/uncertainty.py`, calibration check result (goes in Results/Limitations).

---

## Phase 10 — Trajectory Engine
**Goal:** Statistically defensible trend classification, validated on real repeated-visit data.

**Tasks**
- Implement Theil-Sen slope + Mann-Kendall test in `src/trajectory.py`.
- Real validation: run on Framingham's repeated-exam subset for heart-disease risk trajectories.
- Synthetic demo: for diabetes/CKD (cross-sectional datasets only), build a clearly-labeled synthetic longitudinal simulator using literature-sourced yearly drift rates — label every synthetic output visibly as "SIMULATED, not real patient data" in both code output and dashboard.
- Classify trend into Stable / Improving / Gradually Increasing / Rapidly Increasing using slope magnitude thresholds derived from the data's own slope distribution (e.g., quartiles), not invented numbers.

**Output:** `trajectory.py`, one real trajectory example (Framingham) + one synthetic example (diabetes), trend plots for both.

---

## Phase 11 — Explainable AI
**Goal:** Per-patient and global explanations, plus one actionable counterfactual.

**Tasks**
- SHAP summary plots (global) per disease model.
- SHAP force/waterfall plot for 2–3 example patients (per-individual explanation).
- One counterfactual example: "if glucose decreased by X, risk falls from 78% to Y%" — computed by perturbing the input and re-scoring, not by SHAP itself.
- Confirm wording throughout: "contributed to the model's prediction," never "caused."

**Output:** `explainability.py`, SHAP plots, one worked counterfactual example.

---

## Phase 12 — Screening Priority (MCDA)
**Goal:** A fully auditable priority score.

**Tasks**
- Implement `src/priority.py`: `Priority = α·norm(CMRI) + β·norm(trajectory slope) + γ·(count of HIGH-risk diseases / 3)`, with α=0.5, β=0.3, γ=0.2 documented.
- Bucket into LOW/MEDIUM/HIGH with boundaries shown in the README and the dashboard itself.
- Test on the same synthetic patient profiles from Phase 8 to confirm sane, monotonic behavior.

**Output:** `priority.py`, documented formula, test cases.

---

## Phase 13 — Fairness Audit
**Goal:** Subgroup performance check — a near-mandatory section for any healthcare-ML submission.

**Tasks**
- Recompute recall/precision/F1 per disease model, split by sex and by age band (e.g., <40, 40–60, 60+).
- Build one summary table; flag any subgroup with notably lower recall.
- Write 2–3 sentences on likely cause (usually: dataset subgroup imbalance) — do not overclaim a fix, just document the disparity honestly.

**Output:** `fairness.py`, fairness table, written note for Limitations.

---

## Phase 14 — Streamlit Dashboard
**Goal:** Bring every module together into one interactive app.

**Tasks**
- Patient input form → calls preprocessing → 3 disease models + multi-task model → CMRI → conformal intervals → risk table.
- Trend tab: historical risk line chart (real Framingham example + synthetic demo, clearly labeled) using `trajectory.py`.
- Explainability tab: SHAP bar chart + counterfactual text.
- Priority tab: MCDA score + formula shown to the user.
- Fairness tab (or footer note) summarizing the audit.
- Persistent disclaimer banner on every page.
- Two view modes if time allows: "Clinician view" (full detail) vs "Patient view" (simplified).

**Output:** Working `app.py`.

---

## Phase 15 — Testing
**Goal:** Confirm the system doesn't break or mislead on edge cases.

**Tasks**
- Valid input, missing-field input, extreme/out-of-range input (e.g., glucose=0 or 900).
- All-low-risk and all-high-risk synthetic patients — confirm CMRI, trajectory, and priority all behave sanely.
- Model load/predict smoke test for all saved `.pkl` files.
- Confirm conformal intervals still render sensibly on edge-case inputs.

**Output:** A short `TESTING.md` log of cases tried and results.

---

## Phase 16 — Documentation, Paper Skeleton, PPT
**Goal:** Turn the working system into a reviewable write-up.

**Tasks**
- `README.md`: architecture diagram, setup instructions, disclaimer, limitations.
- Fill the paper skeleton (Title → Abstract → Intro → Related Work → Data → Methodology [5 contributions] → Results [model comparison, multi-task finding, calibration check, fairness table] → Discussion → Limitations → Conclusion).
- Slide deck: 1 slide per major phase (problem, architecture, 5 contributions, results table, dashboard screenshot, limitations).

**Output:** `README.md`, report draft, PPT outline.

---

## Phase 17 — Final Packaging & Demo Run
**Goal:** Everything runs cleanly end-to-end for a live demo.

**Tasks**
- Clean run from a fresh environment using `requirements.txt` only.
- Walk through the dashboard once as if presenting: input → risk table → trend → explanation → priority.
- Freeze final `models/` and confirm `app.py` loads them without retraining.

**Output:** Demo-ready repo.

---

## Critical Path Reminder

If you're short on time, the sequence you cannot skip is:

**Phase 1 → 2 → 5 → 7 → 8 → 10 → 11 → 12 → 14 → 15**

Phases 6 (multi-task), 9 (conformal), 13 (fairness), and the full write-up in 16 add the most *research* value — do them if the core path finishes early, and at minimum leave a documented stub for each ("we identified this as important but time-limited it to X") rather than omitting them silently.
