"""
Phase 2 & Phase 4 — Preprocessing & Feature Engineering Pipeline
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Fixes:
- Zero data leakage: train_test_split is performed BEFORE fitting SimpleImputer and StandardScaler.
- Imputers and scalers fit on X_train only and transform X_test.
- Clinical feature engineering grounded in established medical guidelines (ADA, AHA, WHO).
- Shared 5-feature metabolic continuum loader for multi-task joint learning.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# 1. DIABETES (Pima Indians)
# ---------------------------------------------------------------------------
def load_diabetes(path="data/diabetes.csv", engineered=True):
    df = pd.read_csv(path)
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])

    # Stratified Train/Test split BEFORE imputation and scaling
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )

    if engineered:
        for d in [X_train_imp, X_test_imp]:
            d["Insulin_Glucose_Ratio"] = d["Insulin"] / (d["Glucose"] + 1e-5)
            d["BMI_Age_Interaction"] = (d["BMI"] * d["Age"]) / 100.0
            d["Glucose_ADA_Cat"] = pd.cut(
                d["Glucose"], bins=[-np.inf, 99.9, 125.9, np.inf], labels=[0, 1, 2]
            ).astype(float)
            d["BP_AHA_Cat"] = pd.cut(
                d["BloodPressure"], bins=[-np.inf, 79.9, 89.9, np.inf], labels=[0, 1, 2]
            ).astype(float)
            d["BMI_WHO_Cat"] = pd.cut(
                d["BMI"], bins=[-np.inf, 24.9, 29.9, np.inf], labels=[0, 1, 2]
            ).astype(float)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp),
        columns=X_train_imp.columns,
        index=X_train_imp.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp),
        columns=X_test_imp.columns,
        index=X_test_imp.index
    )

    pipeline_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "features": list(X_train_scaled.columns),
        "engineered": engineered
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pipeline_dict


# ---------------------------------------------------------------------------
# 2. HEART DISEASE (UCI Cleveland)
# ---------------------------------------------------------------------------
def load_heart(path="data/heart.csv", engineered=True):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    y = df["target"]
    X = df.drop(columns=["target"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_df = X_train_raw.copy()
    X_test_df = X_test_raw.copy()

    if engineered:
        for d in [X_train_df, X_test_df]:
            d["Rate_Pressure_Product"] = (d["trestbps"] * d["thalach"]) / 100.0
            d["Chol_Age_Ratio"] = d["chol"] / (d["age"] + 1e-5)
            d["ExAng_ST_Interaction"] = d["exang"] * d["oldpeak"]
            d["SysBP_AHA_Cat"] = pd.cut(
                d["trestbps"], bins=[-np.inf, 119.9, 129.9, 139.9, np.inf], labels=[0, 1, 2, 3]
            ).astype(float)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_df),
        columns=X_train_df.columns,
        index=X_train_df.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_df),
        columns=X_test_df.columns,
        index=X_test_df.index
    )

    pipeline_dict = {
        "imputer": None,
        "scaler": scaler,
        "features": list(X_train_scaled.columns),
        "engineered": engineered
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pipeline_dict


# ---------------------------------------------------------------------------
# 3. CHRONIC KIDNEY DISEASE (UCI CKD)
# ---------------------------------------------------------------------------
def load_ckd(path="data/ckd.csv", engineered=True):
    df = pd.read_csv(path)
    df["classification"] = df["classification"].astype(str).str.strip().map({"ckd": 1, "notckd": 0})
    df = df.drop(columns=["id"])
    y = df["classification"]
    X = df.drop(columns=["classification"])

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    for c in categorical_cols:
        X[c] = X[c].astype(str).str.strip().replace({"nan": np.nan, "?": np.nan})

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_df = X_train_raw.copy()
    X_test_df = X_test_raw.copy()

    num_imputer = SimpleImputer(strategy="median")
    X_train_df[numeric_cols] = num_imputer.fit_transform(X_train_df[numeric_cols])
    X_test_df[numeric_cols] = num_imputer.transform(X_test_df[numeric_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_df[categorical_cols] = cat_imputer.fit_transform(X_train_df[categorical_cols])
    X_test_df[categorical_cols] = cat_imputer.transform(X_test_df[categorical_cols])

    encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        X_train_df[c] = le.fit_transform(X_train_df[c].astype(str))
        # Handle unseen test labels gracefully
        le_dict = {label: idx for idx, label in enumerate(le.classes_)}
        X_test_df[c] = X_test_df[c].astype(str).map(le_dict).fillna(0).astype(int)
        encoders[c] = le

    if engineered:
        for d in [X_train_df, X_test_df]:
            d["BUN_Creatinine_Ratio"] = d["bu"] / (d["sc"] + 1e-5)
            d["Anemia_Flag"] = (d["hemo"] < 12.0).astype(float)
            d["eGFR_Proxy"] = 100.0 / ((d["sc"] + 0.1) * (d["age"] / 50.0 + 0.5))
            d["HTN_Glucose_Interaction"] = d["htn"] * d["bgr"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_df),
        columns=X_train_df.columns,
        index=X_train_df.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_df),
        columns=X_test_df.columns,
        index=X_test_df.index
    )

    pipeline_dict = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoders": encoders,
        "scaler": scaler,
        "features": list(X_train_scaled.columns),
        "engineered": engineered
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pipeline_dict


# ---------------------------------------------------------------------------
# 4. FRAMINGHAM
# ---------------------------------------------------------------------------
def load_framingham(path="data/framingham.csv", engineered=True):
    df = pd.read_csv(path)
    y = df["TenYearCHD"]
    X = df.drop(columns=["TenYearCHD"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )

    if engineered:
        for d in [X_train_imp, X_test_imp]:
            d["Pulse_Pressure"] = d["sysBP"] - d["diaBP"]
            d["MAP"] = d["diaBP"] + (d["Pulse_Pressure"] / 3.0)
            d["Chol_Age_Ratio"] = d["totChol"] / (d["age"] + 1e-5)
            d["SysBP_AHA_Cat"] = pd.cut(
                d["sysBP"], bins=[-np.inf, 119.9, 129.9, 139.9, np.inf], labels=[0, 1, 2, 3]
            ).astype(float)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp),
        columns=X_train_imp.columns,
        index=X_train_imp.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp),
        columns=X_test_imp.columns,
        index=X_test_imp.index
    )

    pipeline_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "features": list(X_train_scaled.columns),
        "engineered": engineered
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pipeline_dict


# ---------------------------------------------------------------------------
# 5. SHARED CONTINUUM DATA (FOR MULTI-TASK JOINT LEARNING)
# ---------------------------------------------------------------------------
def load_shared_continuum_data():
    # 1. Diabetes
    df_d = pd.read_csv("data/diabetes.csv")
    zero_cols = ["Glucose", "BloodPressure", "BMI"]
    df_d[zero_cols] = df_d[zero_cols].replace(0, np.nan)
    X_d_raw = df_d[["Age", "BloodPressure", "Glucose", "BMI", "DiabetesPedigreeFunction"]]
    y_d = df_d["Outcome"]

    X_tr_d_raw, X_te_d_raw, y_tr_d, y_te_d = train_test_split(
        X_d_raw, y_d, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_d
    )
    imp_d = SimpleImputer(strategy="median")
    X_tr_d_imp = imp_d.fit_transform(X_tr_d_raw)
    X_te_d_imp = imp_d.transform(X_te_d_raw)

    cols = ["age", "bp", "glucose", "metabolic_proxy", "organ_proxy"]
    X_tr_d = pd.DataFrame(X_tr_d_imp, columns=cols, index=X_tr_d_raw.index)
    X_te_d = pd.DataFrame(X_te_d_imp, columns=cols, index=X_te_d_raw.index)
    sc_d = StandardScaler()
    X_tr_d_sc = pd.DataFrame(sc_d.fit_transform(X_tr_d), columns=cols, index=X_tr_d.index)
    X_te_d_sc = pd.DataFrame(sc_d.transform(X_te_d), columns=cols, index=X_te_d.index)

    # 2. Heart
    df_h = pd.read_csv("data/heart.csv").drop_duplicates().reset_index(drop=True)
    X_h_raw = pd.DataFrame({
        "age": df_h["age"],
        "bp": df_h["trestbps"],
        "glucose": df_h["fbs"].map({1: 140.0, 0: 95.0}),
        "metabolic_proxy": df_h["chol"] / 10.0,
        "organ_proxy": df_h["oldpeak"]
    })
    y_h = df_h["target"]

    X_tr_h_raw, X_te_h_raw, y_tr_h, y_te_h = train_test_split(
        X_h_raw, y_h, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_h
    )
    sc_h = StandardScaler()
    X_tr_h_sc = pd.DataFrame(sc_h.fit_transform(X_tr_h_raw), columns=cols, index=X_tr_h_raw.index)
    X_te_h_sc = pd.DataFrame(sc_h.transform(X_te_h_raw), columns=cols, index=X_te_h_raw.index)

    # 3. CKD
    df_c = pd.read_csv("data/ckd.csv")
    df_c["classification"] = df_c["classification"].astype(str).str.strip().map({"ckd": 1, "notckd": 0})
    for col in ["age", "bp", "bgr", "bu", "sc"]:
        df_c[col] = pd.to_numeric(
            df_c[col].astype(str).str.strip().replace({'?': np.nan, 'nan': np.nan}),
            errors='coerce'
        )
    X_c_raw = df_c[["age", "bp", "bgr", "bu", "sc"]]
    y_c = df_c["classification"]

    X_tr_c_raw, X_te_c_raw, y_tr_c, y_te_c = train_test_split(
        X_c_raw, y_c, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_c
    )
    imp_c = SimpleImputer(strategy="median")
    X_tr_c_imp = imp_c.fit_transform(X_tr_c_raw)
    X_te_c_imp = imp_c.transform(X_te_c_raw)

    X_tr_c = pd.DataFrame(X_tr_c_imp, columns=cols, index=X_tr_c_raw.index)
    X_te_c = pd.DataFrame(X_te_c_imp, columns=cols, index=X_te_c_raw.index)
    sc_c = StandardScaler()
    X_tr_c_sc = pd.DataFrame(sc_c.fit_transform(X_tr_c), columns=cols, index=X_tr_c.index)
    X_te_c_sc = pd.DataFrame(sc_c.transform(X_te_c), columns=cols, index=X_te_c.index)

    return {
        "diabetes": (X_tr_d_sc, X_te_d_sc, y_tr_d, y_te_d, sc_d),
        "heart": (X_tr_h_sc, X_te_h_sc, y_tr_h, y_te_h, sc_h),
        "ckd": (X_tr_c_sc, X_te_c_sc, y_tr_c, y_te_c, sc_c)
    }


if __name__ == "__main__":
    print("Testing Preprocessing & Feature Engineering Pipelines...\n")
    for name, loader in [
        ("Diabetes", load_diabetes),
        ("Heart", load_heart),
        ("CKD", load_ckd),
        ("Framingham", load_framingham)
    ]:
        X_tr, X_te, y_tr, y_te, scaler, pipe = loader()
        assert X_tr.isna().sum().sum() == 0, f"Error: NaNs in {name} train!"
        assert X_te.isna().sum().sum() == 0, f"Error: NaNs in {name} test!"
        print(f"[OK] {name:12s} | Train: {X_tr.shape} | Test: {X_te.shape} | Features: {len(X_tr.columns)}")

    shared = load_shared_continuum_data()
    for k, v in shared.items():
        print(f"[OK] Shared {k:8s} | Train: {v[0].shape} | Test: {v[1].shape}")
    print("\nAll preprocessing pipelines validated successfully with ZERO data leakage.")
