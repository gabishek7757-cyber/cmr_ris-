I'm continuing a Data Science PBL course project called CMR-RIS (CardioMetabolic-Renal Risk
Intelligence System) — a chronic disease risk-screening system for diabetes, cardiovascular
disease, and chronic kidney disease. I have the project folder already built and running in
Google Colab (mounted via Google Drive at /content/drive/MyDrive/cmr_ris). I'm attaching/have
access to the project zip. Please read this context and continue from where it left off —
don't restart planning from scratch.

## Project concept
Not just 3 separate disease classifiers — the core idea is treating diabetes, heart disease,
and CKD as one interconnected "cardiometabolic-renal risk continuum" (a real, clinically
recognized overlap), with 5 specific research-grade contributions:
1. Composite CardioMetabolic-Renal Risk Index (CMRI) combining all 3 disease probabilities
   with an interaction term for compounding risk
2. A shared-representation multi-task model compared against independent per-disease models
3. Statistically-grounded trajectory classification (Theil-Sen slope + Mann-Kendall test),
   not ad hoc "increasing/decreasing" labels
4. Conformal prediction for calibrated uncertainty intervals on every risk score
5. An auditable MCDA (multi-criteria decision analysis) formula for screening priority,
   not a black-box "priority: HIGH" label

Full detail is in the project's docs/01_PROJECT_PLAN.md and docs/02_ROADMAP.md files —
read those first if available.

## What's already done (don't redo these)

**Phase 0 — Setup:** Project structure created, requirements.txt has pandas, numpy,
scikit-learn, xgboost, shap, streamlit, matplotlib, seaborn, scipy, joblib.

**Phase 1 — Dataset acquisition & validation:** 4 real datasets downloaded and validated,
sitting in data/:
- data/diabetes.csv — Pima Indians Diabetes, 768 rows x 9 cols, target "Outcome"
- data/heart.csv — UCI Heart Disease (Cleveland), 303 rows x 14 cols, target "target"
- data/ckd.csv — UCI Chronic Kidney Disease, 400 rows x 26 cols, target "classification"
- data/framingham.csv — Framingham Heart Study, 4240 rows x 16 cols, target "TenYearCHD"

Known data-quality issues already documented in data/data_notes.md:
- diabetes: Glucose/BloodPressure/SkinThickness/Insulin/BMI use 0 to encode missing values
- heart: 1 exact duplicate row
- ckd: target column had a "ckd\t" (trailing tab) typo in 2 rows; heavy missingness in
  several columns (rbc 38% missing, rc/wc/sod/pot also high)
- framingham: ~9% missing in glucose, moderate missingness elsewhere

IMPORTANT CORRECTION already made: the public Framingham file is cross-sectional (one row
per participant), NOT true multi-visit longitudinal data. Real longitudinal FHS data needs
restricted BioLINCC access. So Phase 10 (trajectory) will use a clearly-labeled SYNTHETIC
simulator for all 3 diseases rather than claiming real longitudinal validation. This is
documented honestly in data/data_notes.md — keep that honesty in any future writeup.

**Phase 2 — Preprocessing (COMPLETE, tested, working):** src/preprocessing.py has 4 working
functions: load_diabetes(), load_heart(), load_ckd(), load_framingham() — each fixes the
issues above and returns (X_train, X_test, y_train, y_test, scaler, extra) with stratified
splits, median/mode imputation, and StandardScaler applied. Verified output when running
`python3 src/preprocessing.py`:
```
diabetes     -> X_train (614, 8), X_test (154, 8), train balance: {0: 400, 1: 214}
heart        -> X_train (241, 13), X_test (61, 13), train balance: {1: 131, 0: 110}
ckd          -> X_train (320, 24), X_test (80, 24), train balance: {1: 200, 0: 120}
framingham   -> X_train (3392, 15), X_test (848, 15), train balance: {0: 2877, 1: 515}
```

## What's NOT done yet — continue here

Follow docs/02_ROADMAP.md starting at **Phase 3 — Exploratory Data Analysis**:
- Phase 3: EDA notebooks per disease (distributions, correlation heatmaps, target vs feature)
- Phase 4: Feature engineering (BMI categories, age groups, shared feature set for multi-task model)
- Phase 5: Disease-specific models (LogReg, Random Forest, XGBoost, SVM x 3 diseases)
- Phase 6: Shared multi-task model, compared against Phase 5's independent models
- Phase 7: Model evaluation & final selection (prioritize recall, not accuracy)
- Phase 8: CMRI composite index
- Phase 9: Conformal prediction for uncertainty
- Phase 10: Trajectory engine (Theil-Sen + Mann-Kendall, synthetic simulator, clearly labeled)
- Phase 11: SHAP explainability + one counterfactual example
- Phase 12: MCDA screening priority score
- Phase 13: Fairness audit (subgroup recall/precision by sex, age band)
- Phase 14: Streamlit dashboard tying everything together
- Phase 15: Testing (edge cases, TESTING.md log)
- Phase 16: Documentation / paper skeleton / PPT
- Phase 17: Final packaging

## Working environment
Google Colab, project mounted from Google Drive at /content/drive/MyDrive/cmr_ris so files
persist across sessions. Please write code as Colab-ready cells I can paste in directly,
test any code before giving it to me (verify shapes, no errors, sane output), and keep
building files that get saved into the correct existing folders (src/, notebooks/, models/,
results/) rather than restructuring the project.

Please start by asking me to share the docs/01_PROJECT_PLAN.md and docs/02_ROADMAP.md
content (or confirm you can read them from the uploaded project), then proceed to Phase 3.
