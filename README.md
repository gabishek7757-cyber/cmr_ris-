<div align="center">

# 🩺 CMR-RIS: CardioMetabolic-Renal Risk Intelligence System
### *A Unified, Trajectory-Aware, Explainable Clinical Decision-Support Platform for Cardiovascular-Kidney-Metabolic (CKM) Syndrome*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTest Status](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B?logo=streamlit&logoColor=white)](app.py)

---

</div>

## 📌 Executive Summary & Clinical Motivation

In current clinical practice, **Type-2 Diabetes Mellitus (T2D)**, **Cardiovascular Disease (CVD)**, and **Chronic Kidney Disease (CKD)** are frequently screened in clinical silos with disconnected diagnostic tools. However, modern pathophysiology recognizes these conditions as an interconnected continuum known as **Cardiovascular-Kidney-Metabolic (CKM) Syndrome** (American Heart Association, 2023). 

**CMR-RIS** is an end-to-end clinical machine learning and decision-support system designed to:
1. **Unify Cross-Organ Risk Modeling**: Predict individual and composite multi-organ risk using calibrated screening classifiers.
2. **Quantify Uncertainty**: Provide finite-sample, distribution-free $(1-\alpha)$ confidence intervals via **Split Conformal Prediction**.
3. **Statistically Track Trajectories**: Classify longitudinal disease progression using outlier-resilient **Theil-Sen robust slopes** and **Mann-Kendall monotonic trend tests**.
4. **Deliver Actionable Explainability**: Pair local **SHAP** feature attributions with interactive **multi-organ counterfactual simulations**.
5. **Prioritize Clinical Triage**: Generate fully auditable triage priority scores via **Multi-Criteria Decision Analysis (MCDA)** with demographic fairness governance.

---

## 🏛️ System Architecture

```
                               ┌───────────────────────────────┐
                               │   Standardized Patient Intake  │
                               │    (Demographics + Labs + BP)  │
                               └───────────────┬───────────────┘
                                               │
                                 Pydantic Data Contract Validation
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
          ┌─────────────────────────────┐             ┌─────────────────────────────┐
          │  Disease-Specific Champions │             │  Shared Multi-Task Trunk    │
          │  (LogReg / RF / XGB / SVM)  │             │  (Shared Metabolic Space)   │
          └──────────────┬──────────────┘             └──────────────┬──────────────┘
                         ▼                                           ▼
            [P(Diabetes), P(Heart), P(CKD)]             Cross-Model Synergy Comparison
                         │
        ┌────────────────┼────────────────────────────────────────┐
        ▼                ▼                                        ▼
┌──────────────┐ ┌──────────────┐                         ┌──────────────┐
│  Conformal   │ │  Composite   │                         │  Trajectory  │
│ Prediction   │ │     CMRI     │                         │    Engine    │
│  (90% CI)    │ │ (Synergistic)│                         │  (Theil-Sen) │
└───────┬──────┘ └───────┬──────┘                         └───────┬──────┘
        │                │                                        │
        └────────────────┼────────────────────────────────────────┘
                         ▼
        ┌─────────────────────────────────────────────────────────┐
        │  Auditable MCDA Clinical Triage & Screening Priority    │
        │  Priority = 0.50(CMRI) + 0.30(Slope) + 0.20(Organ Burden)│
        └────────────────────────┬────────────────────────────────┘
                                 │
                         ┌───────┴───────┐
                         ▼               ▼
                 ┌───────────────┐ ┌───────────────┐
                 │ Explainability│ │  Fairness &   │
                 │ (SHAP + CF)   │ │  Governance   │
                 └───────────────┘ └───────────────┘
```

---

## 🔬 Five Core Technical & Research Contributions

### 1. Composite CardioMetabolic-Renal Index (CMRI)
Rather than treating organ risks as independent percentages, CMRI models non-linear comorbidity coupling:
$$\text{CMRI} = w_1 P_{\text{diabetes}} + w_2 P_{\text{heart}} + w_3 P_{\text{ckd}} + w_4 \left[ P_{\text{diabetes}} \cdot P_{\text{heart}} \cdot P_{\text{ckd}} \right]$$
*Where $w_1 = 0.30, w_2 = 0.30, w_3 = 0.30, w_4 = 0.10$ ($\sum w_i = 1.00$). The multiplicative term $w_4$ accounts for synergistic multi-organ acceleration (e.g. diabetic nephropathy combined with coronary ischemia).*

### 2. Calibrated Uncertainty via Split Conformal Prediction
Standard classification outputs are pseudo-probabilities prone to overconfidence. CMR-RIS wraps all champion models in split conformal prediction to generate calibrated distribution-free prediction intervals:
$$C(X) = \left[ \hat{P}(X) - \hat{q}_{1-\alpha}, \; \hat{P}(X) + \hat{q}_{1-\alpha} \right] \cap [0, 1]$$
*Guarantees exact finite-sample coverage: $\mathbb{P}(Y \in C(X)) \ge 1 - \alpha$ (e.g., $90\%$ confidence bounds).*

### 3. Statistically-Grounded Longitudinal Trajectory Engine
Replaces subjective heuristics with formal statistical hypothesis testing across repeat patient visits:
- **Theil-Sen Robust Median Slope**:
  $$\beta_{\text{median}} = \text{median}\left\{ \frac{R_j - R_i}{t_j - t_i} : 1 \le i < j \le n \right\}$$
  *Impervious to acute transient lab spikes and outlier readings.*
- **Mann-Kendall Monotonic Trend Test**:
  $$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(R_j - R_k), \quad Z = \frac{S \pm 1}{\sqrt{\text{Var}(S)}}$$
  *Evaluates two-sided $p$-value for statistically significant disease progression.*

### 4. Dual-Tier Explainability & Actionable Multi-Organ Counterfactuals
- **SHAP Feature Attributions**: Computes exact Shapley values to identify primary risk drivers per patient.
- **Actionable Counterfactual Planner**: Evaluates simulated interventions across modifiable targets:
  $$\Delta \text{Risk} = f(\mathbf{x}_{\text{baseline}}) - f(\mathbf{x}_{\text{baseline}} + \boldsymbol{\delta}_{\text{intervention}})$$
  *(e.g., "Lowering fasting glucose by 25 mg/dL and systolic BP by 15 mmHg decreases 5-year composite CMRI by 31.2%").*

### 5. Auditable Multi-Criteria Decision Analysis (MCDA) Priority Engine
A deterministic, fully auditable triage formula for clinical screening prioritization:
$$\text{Priority Score} = \alpha \cdot \text{Norm}(\text{CMRI}) + \beta \cdot \text{Norm}(\text{Slope}) + \gamma \cdot \left(\frac{\sum \mathbb{I}(P_i \ge 0.50)}{3}\right)$$
*Weights: $\alpha = 0.50$ (Severity), $\beta = 0.30$ (Velocity), $\gamma = 0.20$ (Comorbidity Burden).*

---

## 📊 Empirical Model Benchmark & Research Findings

All models were evaluated using **5-Fold Stratified Cross-Validation** on real clinical cohorts (Pima Indians Diabetes, Cleveland Heart, UCI CKD, Framingham Heart Study).

| Disease Domain | Model Architecture | Algorithm | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Diabetes** | Independent Champion | **SVM (RBF)** | **74.7%** | 60.9% | **77.8%** | **0.683** | **0.808** |
| Diabetes | Independent | Random Forest | 77.3% | 66.7% | 70.4% | 0.685 | 0.836 |
| Diabetes | Independent | Logistic Regression | 74.0% | 60.3% | 75.9% | 0.672 | 0.818 |
| Diabetes | Shared Multi-Task | Shared GBM | 74.7% | 66.0% | 57.4% | 0.614 | 0.815 |
| **Heart Disease** | Independent Champion | **XGBoost** | **82.0%** | **77.5%** | **93.9%** | **0.849** | **0.884** |
| Heart Disease | Independent | Random Forest | 80.3% | 76.9% | 90.9% | 0.833 | 0.876 |
| Heart Disease | Independent | Logistic Regression | 78.7% | 76.3% | 87.9% | 0.817 | 0.872 |
| Heart Disease | Shared Multi-Task | Shared GBM | 65.6% | 65.8% | 75.8% | 0.704 | 0.648 |
| **CKD** | Independent Champion | **Random Forest** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **1.000** |
| CKD | Independent | SVM (RBF) | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| CKD | Independent | Logistic Regression | 98.8% | 100.0% | 98.0% | 0.990 | 1.000 |
| CKD | Shared Multi-Task | Shared GBM | 96.3% | 98.0% | 96.0% | 0.970 | 0.997 |

> **Key Research Insight:** Independent domain models with disease-specific hemodynamic features (e.g. ST depression `oldpeak`, chest pain type `cp`) outperform shared representations on acute cardiac manifestations (Recall $93.9\%$ vs $75.8\%$). Conversely, shared multi-task representations provide strong baselines for chronic progressive nephropathy (Recall $96.0\%$), confirming close metabolic-renal co-dependence.

---

## ⚡ Quickstart & Installation

### Option 1: Local Installation

```bash
# 1. Clone repository
git clone https://github.com/gabishek7757-cyber/cmr_ris.git
cd cmr_ris

# 2. Set up virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest -v --tb=short

# 5. Launch interactive clinical dashboard
streamlit run app.py
```

### Option 2: Docker Container

```bash
# Build and run with Docker Compose
docker compose up --build
```
*The interactive dashboard will be accessible at `http://localhost:8501`.*

### Option 3: End-to-End Pipeline CLI

```bash
# Train all models and select champions
python -m src.pipeline --mode train

# Calibrate conformal prediction intervals (90% confidence)
python -m src.pipeline --mode calibrate --confidence 0.90

# Run subgroup demographic fairness audit
python -m src.pipeline --mode fairness

# Full pipeline execution & evaluation summary
python -m src.pipeline --mode all
```

---

## 🧪 Automated Testing Protocol

The repository includes a comprehensive `pytest` testing suite covering clinical contracts, data leakage checks, conformal quantile validity, and numerical robustness:

```bash
pytest -v --tb=short --cov=src --cov-report=term-missing
```

- `tests/test_schemas.py`: Pydantic biological boundary constraints ($DiaBP < SysBP$).
- `tests/test_cmri.py`: Non-linear interaction coupling, mathematical boundary tests ($0 \le \text{CMRI} \le 1$).
- `tests/test_uncertainty.py`: Conformal calibration quantile calculation, finite-sample coverage containment.
- `tests/test_trajectory.py`: Theil-Sen slope math, Mann-Kendall trend hypothesis test with exact $p$-values.
- `tests/test_priority.py`: MCDA deterministic scoring invariance and triage tier mapping.
- `tests/test_preprocessing.py`: Feature engineering integrity and zero-imputation validation.

---

## 📁 Repository Structure

```
cmr_ris/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated GitHub Actions CI pipeline
├── data/                          # Clinical validation datasets & documentation
│   ├── ckd.csv
│   ├── diabetes.csv
│   ├── framingham.csv
│   ├── heart.csv
│   └── data_notes.md
├── docs/                          # Detailed research plan & phase-by-phase roadmap
│   ├── 01_PROJECT_PLAN.md
│   ├── 02_ROADMAP.md
│   └── 03_HANDOFF_PROMPT.md
├── models/                        # Serialized champion models & conformal calibrators
│   ├── ckd_champion.pkl
│   ├── conformal_calibrator.pkl
│   ├── diabetes_champion.pkl
│   ├── heart_champion.pkl
│   └── shared_multitask_model.pkl
├── notebooks/                     # Exploratory Data Analysis & training notebooks
│   └── CMR_RIS_Complete_Pipeline.ipynb
├── results/                       # Model benchmarks, fairness audit, evaluation logs
│   ├── fairness_audit.csv
│   └── model_comparison.csv
├── src/                           # Modular core system codebase
│   ├── __init__.py
│   ├── cmri.py                    # Composite CMRI calculation engine
│   ├── explainability.py          # SHAP feature attributions & counterfactuals
│   ├── fairness.py                # Subgroup demographic parity auditor
│   ├── modeling.py                # Model training, CV & champion selection
│   ├── pipeline.py                # Unified pipeline CLI entrypoint
│   ├── preprocessing.py           # Clinical data loading & imputation
│   ├── priority.py                # MCDA triage priority engine
│   ├── schemas.py                 # Pydantic clinical intake & contract schemas
│   ├── trajectory.py              # Theil-Sen & Mann-Kendall trajectory tracker
│   └── uncertainty.py             # Split conformal prediction calibrator
├── tests/                         # Full automated unit & integration test suite
│   ├── test_cmri.py
│   ├── test_preprocessing.py
│   ├── test_priority.py
│   ├── test_schemas.py
│   ├── test_trajectory.py
│   └── test_uncertainty.py
├── app.py                         # Streamlit clinical dashboard application
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Container orchestration
├── pyproject.toml                 # Modern package configuration
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Developer contribution guidelines
└── README.md                      # Comprehensive documentation
```

---

## ⚖️ Mandatory Clinical Screening Disclaimer

> **IMPORTANT NOTICE:** CMR-RIS is developed as a machine learning research prototype and clinical decision-support screening aid. It is designed to assist healthcare providers in risk stratification and outpatient scheduling. **It is not an automated diagnostic device** and should never be used as a substitute for individualized clinical judgment, in-person physician evaluation, or definitive laboratory diagnostics.

---

## 📚 Citation

If you use CMR-RIS in your research or course projects, please cite:

```bibtex
@software{cmr_ris_2026,
  title = {CMR-RIS: CardioMetabolic-Renal Risk Intelligence System},
  author = {CMR-RIS Contributors},
  year = {2026},
  url = {https://github.com/gabishek7757-cyber/cmr_ris},
  version = {1.0.0}
}
```
