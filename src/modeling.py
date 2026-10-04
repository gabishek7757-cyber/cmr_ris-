"""
Phase 5, 6 & 7 — Disease-Specific & Shared Multi-Task Modeling
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Features:
- 4 core screening algorithms: Logistic Regression, Random Forest, XGBoost, SVM (RBF).
- 5-Fold Stratified Cross-Validation on training splits.
- Shared 5-feature continuum multi-task learning.
- Screening champion selection prioritizing Recall (Sensitivity) and ROC-AUC.
- Full model serialization and benchmark export to results/model_comparison.csv.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.preprocessing import (
    load_diabetes, load_heart, load_ckd, load_shared_continuum_data
)

RANDOM_STATE = 42


def get_algorithms(random_state=RANDOM_STATE):
    """Returns the suite of 4 core screening algorithms with balanced weighting."""
    return {
        "logreg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "rf": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=random_state
        ),
        "xgb": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=1.5,
            eval_metric="logloss",
            random_state=random_state
        ),
        "svm": SVC(
            probability=True, class_weight="balanced", kernel="rbf", random_state=random_state
        )
    }


def train_disease_models(disease_name, loader_func, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler, pipe = loader_func(engineered=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    algos = get_algorithms()
    results = []

    for name, model in algos.items():
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        save_path = os.path.join(output_dir, f"{disease_name}_{name}.pkl")
        payload = {
            "model": model,
            "scaler": scaler,
            "pipeline": pipe,
            "features": list(X_train.columns),
            "disease": disease_name,
            "algorithm": name.upper(),
            "metrics": {"recall": rec, "roc_auc": roc, "f1": f1, "accuracy": acc}
        }
        joblib.dump(payload, save_path)

        results.append({
            "Disease": disease_name.capitalize(),
            "Model_Type": "Independent",
            "Algorithm": name.upper(),
            "CV_Recall_Mean": float(np.mean(cv_res['test_recall'])),
            "Test_Accuracy": float(acc),
            "Test_Precision": float(prec),
            "Test_Recall": float(rec),
            "Test_F1": float(f1),
            "Test_ROC_AUC": float(roc),
            "Model_Path": save_path
        })

    return pd.DataFrame(results)


def train_shared_multitask_model(output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    shared_data = load_shared_continuum_data()

    results = []
    mt_models = {}

    for dname in ["diabetes", "heart", "ckd"]:
        X_train, X_test, y_train, y_test, scaler = shared_data[dname]

        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        save_path = os.path.join(output_dir, f"{dname}_shared_gbm.pkl")
        payload = {
            "model": model,
            "scaler": scaler,
            "features": list(X_train.columns),
            "disease": dname,
            "algorithm": "SHARED_GBM",
            "metrics": {"recall": rec, "roc_auc": roc, "f1": f1, "accuracy": acc}
        }
        joblib.dump(payload, save_path)
        mt_models[dname] = payload

        results.append({
            "Disease": dname.capitalize(),
            "Model_Type": "Shared Multi-Task",
            "Algorithm": "SHARED_GBM",
            "CV_Recall_Mean": np.nan,
            "Test_Accuracy": float(acc),
            "Test_Precision": float(prec),
            "Test_Recall": float(rec),
            "Test_F1": float(f1),
            "Test_ROC_AUC": float(roc),
            "Model_Path": save_path
        })

    joblib.dump(mt_models, os.path.join(output_dir, "shared_multitask_model.pkl"))
    return pd.DataFrame(results)


def run_full_modeling_suite(results_dir="results", models_dir="models"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Training Independent Disease Models (Phase 5)...")
    df_d = train_disease_models("diabetes", load_diabetes, models_dir)
    df_h = train_disease_models("heart", load_heart, models_dir)
    df_c = train_disease_models("ckd", load_ckd, models_dir)

    print("Training Shared Multi-Task Models (Phase 6)...")
    df_mt = train_shared_multitask_model(models_dir)

    df_all = pd.concat([df_d, df_h, df_c, df_mt], ignore_index=True)

    csv_path = os.path.join(results_dir, "model_comparison.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Exported model comparison to {csv_path}")

    # Select Champions Prioritizing Recall & Sensitivity (Phase 7)
    champions = {}
    for d in ["Diabetes", "Heart", "Ckd"]:
        sub = df_all[df_all["Disease"] == d].copy()
        best_row = sub.sort_values(by=["Test_Recall", "Test_ROC_AUC"], ascending=False).iloc[0]
        champions[d] = best_row.to_dict()

        source_path = best_row["Model_Path"]
        champ_path = os.path.join(models_dir, f"{d.lower()}_champion.pkl")
        data = joblib.load(source_path)
        joblib.dump(data, champ_path)
        print(f"[CHAMPION] {d.upper()}: {best_row['Algorithm']} (Recall: {best_row['Test_Recall']:.3f}, AUC: {best_row['Test_ROC_AUC']:.3f}) -> {champ_path}")

    return df_all, champions


if __name__ == "__main__":
    df_all, champions = run_full_modeling_suite()
    print("\nBenchmark completed successfully.")
