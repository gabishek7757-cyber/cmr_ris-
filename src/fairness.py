"""
Phase 13 — Subgroup Fairness Audit
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Evaluates model recall, precision, F1-score, and accuracy across:
- Biological Sex (Male vs Female)
- Age Brackets (<45 Younger, 45–60 Middle-Aged, >=60 Older)
Generates results/fairness_audit.csv for compliance, clinical audit, and research paper reporting.
"""

import os
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def evaluate_subgroup_metrics(df_eval, group_col, group_label_map, dimension_name):
    records = []
    for val, label in group_label_map.items():
        sub = df_eval[df_eval[group_col] == val]
        if len(sub) >= 5 and sub["y_true"].nunique() > 1:
            records.append({
                "Subgroup_Dimension": dimension_name,
                "Subgroup": label,
                "Sample_N": len(sub),
                "Positive_Prevalence": float(np.round(sub["y_true"].mean(), 3)),
                "Recall": float(np.round(recall_score(sub["y_true"], sub["y_pred"], zero_division=0), 3)),
                "Precision": float(np.round(precision_score(sub["y_true"], sub["y_pred"], zero_division=0), 3)),
                "F1_Score": float(np.round(f1_score(sub["y_true"], sub["y_pred"], zero_division=0), 3)),
                "Accuracy": float(np.round(accuracy_score(sub["y_true"], sub["y_pred"]), 3))
            })
    return records


def run_full_fairness_audit(models_dir="models", results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    from src.preprocessing import load_diabetes, load_heart, load_ckd

    cohorts = [
        ("Diabetes", "diabetes_champion.pkl", load_diabetes, "Age"),
        ("Heart", "heart_champion.pkl", load_heart, "age"),
        ("CKD", "ckd_champion.pkl", load_ckd, "age")
    ]

    all_records = []

    for disease_name, pkl_name, loader, age_field in cohorts:
        champ_path = os.path.join(models_dir, pkl_name)
        data = joblib.load(champ_path)
        model = data["model"]
        features = data["features"]

        _, X_test, _, y_test, _, _ = loader(engineered=True)
        y_pred = model.predict(X_test[features])

        df_eval = X_test.copy()
        df_eval["y_true"] = y_test.values
        df_eval["y_pred"] = y_pred

        # Overall Baseline
        all_records.append({
            "Disease": disease_name,
            "Subgroup_Dimension": "All Cohort",
            "Subgroup": "Overall Test Baseline",
            "Sample_N": len(df_eval),
            "Positive_Prevalence": float(np.round(df_eval["y_true"].mean(), 3)),
            "Recall": float(np.round(recall_score(df_eval["y_true"], df_eval["y_pred"], zero_division=0), 3)),
            "Precision": float(np.round(precision_score(df_eval["y_true"], df_eval["y_pred"], zero_division=0), 3)),
            "F1_Score": float(np.round(f1_score(df_eval["y_true"], df_eval["y_pred"], zero_division=0), 3)),
            "Accuracy": float(np.round(accuracy_score(df_eval["y_true"], df_eval["y_pred"]), 3))
        })

        # Sex Disparity (if present)
        for sex_col in ["sex", "Sex"]:
            if sex_col in df_eval.columns:
                recs = evaluate_subgroup_metrics(
                    df_eval, sex_col, {1: "Male", 0: "Female"}, "Biological Sex"
                )
                for r in recs:
                    r["Disease"] = disease_name
                    all_records.append(r)

        # Age Strata (using standardized tertiles / cutoffs)
        if age_field in df_eval.columns:
            age_s = df_eval[age_field]
            bins = [-np.inf, -0.4, 0.4, np.inf]
            labels = ["Younger (<45 / Low Tertile)", "Middle-Aged (45-60 / Mid)", "Older (>=60 / High)"]
            df_eval["Age_Strata"] = pd.cut(age_s, bins=bins, labels=labels)
            recs = evaluate_subgroup_metrics(
                df_eval, "Age_Strata", {l: l for l in labels}, "Age Bracket"
            )
            for r in recs:
                r["Disease"] = disease_name
                all_records.append(r)

    df_fairness = pd.DataFrame(all_records)
    csv_path = os.path.join(results_dir, "fairness_audit.csv")
    df_fairness.to_csv(csv_path, index=False)
    return df_fairness


if __name__ == "__main__":
    print("Running Full Subgroup Fairness Audit across all disease models...\n")
    df_fair = run_full_fairness_audit()
    print(df_fair[["Disease", "Subgroup_Dimension", "Subgroup", "Sample_N", "Recall", "F1_Score", "Accuracy"]].to_string(index=False))
    print("\nFairness audit exported to results/fairness_audit.csv successfully.")
