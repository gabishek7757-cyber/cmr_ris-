# Dataset Validation Notes (Phase 1)

## 1. Diabetes — `diabetes.csv` (Pima Indians Diabetes)
- **Source:** UCI / National Institute of Diabetes and Digestive and Kidney Diseases, via GitHub mirror (npradaschnor/Pima-Indians-Diabetes-Dataset)
- **Shape:** 768 rows × 9 columns
- **Target:** `Outcome` (0/1) — 500 negative / 268 positive → moderate class imbalance (~35% positive)
- **Duplicates:** 0
- **Missing values:** No NaNs, BUT missingness is encoded as **0** in physiologically impossible fields:
  - Glucose: 5 zeros, BloodPressure: 35 zeros, SkinThickness: 227 zeros, Insulin: 374 zeros, BMI: 11 zeros
  - **Action for Phase 2:** convert these zeros to NaN before imputing (a real patient cannot have 0 BMI or 0 blood pressure). This is a classic data-quality trap in this dataset and must be handled explicitly.
- **Population:** women of Pima Indian heritage, age 21+ — a narrow population; note as a generalizability limitation later.
- **Leakage risk:** none obvious — all features are pre-outcome measurements.

## 2. Heart Disease — `heart.csv` (UCI Cleveland)
- **Source:** UCI Heart Disease (Cleveland subset), via GitHub mirror (mrdbourke/zero-to-mastery-ml), pre-binarized target
- **Shape:** 303 rows × 14 columns
- **Target:** `target` (0/1, already binarized from the original 0–4 scale) — 138 negative / 165 positive → fairly balanced
- **Duplicates:** 1 exact duplicate row found → drop in Phase 2
- **Missing values:** none (already cleaned in this mirror)
- **Leakage risk:** none obvious — standard clinical/exercise-test features

## 3. Chronic Kidney Disease — `ckd.csv` (UCI CKD)
- **Source:** UCI Chronic Kidney Disease dataset, via GitHub mirror (aiplanethub/Datasets)
- **Shape:** 400 rows × 26 columns (id + 24 features + target)
- **Target:** `classification` — 248 "ckd" / 150 "notckd" / **2 rows with "ckd\t" (trailing tab typo)** → must be stripped/normalized in Phase 2 before encoding
- **Duplicates:** 0
- **Missing values:** substantial and uneven across columns — most affected: `rbc` (152 missing, 38%), `rc` (130), `wc` (105), `pot`/`sod` (~88 each), `pcv` (70), `pc` (65). This is the dataset most in need of a careful imputation strategy (median for numeric, mode for categorical, and consider flagging heavy-missingness columns like `rbc` rather than naively imputing).
- **Population:** 400 patients from a single hospital (Tamil Nadu, India) over ~2 months — small, single-site; flag as a strong generalizability limitation.
- **Leakage risk:** none obvious, but small N (400) means the final model should be evaluated with caution (wide confidence intervals expected).

## 4. Framingham Heart Study — `framingham.csv`
- **Source:** Framingham Heart Study teaching dataset (Kaggle-derived), via GitHub mirror (matackett/sta210)
- **Shape:** 4,240 rows × 16 columns
- **Target:** `TenYearCHD` (0/1) — 3,596 negative / 644 positive → imbalanced (~15% positive)
- **Duplicates:** 0
- **Missing values:** `glucose` (388 missing, ~9%), `education` (105), `BPMeds` (53), `totChol` (50), `cigsPerDay` (29), `BMI` (19), `heartRate` (1)
- **IMPORTANT CORRECTION to the original project roadmap:** this publicly available Framingham file is a **single-exam, cross-sectional extract** (one row per participant, one baseline exam predicting a 10-year outcome) — **not** a true multi-visit longitudinal file with repeated risk-score measurements per patient. The genuine multi-exam longitudinal FHS data requires restricted access via BioLINCC and is not freely redistributable.
  - **Revised plan for Phase 10 (Trajectory Engine):** since no freely available dataset in this project has genuine repeated per-patient visits, the trajectory engine will be demonstrated entirely on the **clearly-labeled synthetic simulator** (literature-informed yearly drift rates) for all three diseases, rather than validated against real repeated-measures data. This will be stated explicitly in the Limitations section — a more honest framing than the original roadmap assumed.

## Cross-Dataset Notes
- No two datasets should be merged — they come from different populations, time periods, and collection protocols (Pima women only, Cleveland cardiac patients, Tamil Nadu CKD patients, Framingham MA residents). Each disease model is trained independently, exactly as planned.
- Class imbalance is present in diabetes (35% positive) and strongly present in Framingham (15% positive) — stratified splitting is mandatory for both; class-weighting or SMOTE (applied only after splitting) should be evaluated in Phase 5/6.
- All four datasets are small-to-moderate by modern ML standards (300–4,240 rows) — this will be restated in the paper's Limitations section regarding generalizability and confidence-interval width.
