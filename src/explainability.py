"""
Phase 11 — Explainable AI & Actionable Counterfactuals
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Core Contribution #4:
1. SHAP local feature attributions (TreeExplainer / KernelExplainer / Sensitivity).
2. Actionable clinical counterfactual what-if simulation:
   "If fasting glucose is reduced from 150 mg/dL to 105 mg/dL, model-estimated risk falls by 21.4%."
"""

import os
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import shap
import warnings


class ModelExplainer:
    def __init__(self, model_dict, background_data=None):
        self.model = model_dict["model"]
        self.scaler = model_dict.get("scaler", None)
        self.pipeline = model_dict.get("pipeline", None)
        self.features = list(model_dict["features"])
        self.background_data = background_data
        self.explainer = None
        self.explainer_mode = None

        self._setup_explainer()

    def _setup_explainer(self):
        # 1. Tree Explainer for XGBoost, Random Forest, Gradient Boosting
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_mode = "tree"
            return
        except Exception:
            pass

        # 2. Kernel / Linear Explainer for SVM or Logistic Regression
        try:
            if self.background_data is not None:
                bg = shap.sample(self.background_data[self.features], min(30, len(self.background_data)))
            else:
                bg = pd.DataFrame(np.zeros((1, len(self.features))), columns=self.features)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)
                self.explainer_mode = "kernel"
                return
        except Exception:
            self.explainer_mode = "perturbation_sensitivity"

    def explain_patient(self, patient_df, top_k=6):
        """Computes local feature impact on model risk for a single patient."""
        X_df = patient_df[self.features].copy()
        n_feats = len(self.features)
        sv = None

        try:
            if self.explainer_mode == "tree":
                shap_obj = self.explainer(X_df)
                val = shap_obj.values
                if isinstance(val, list):
                    sv = np.asarray(val[1][0]).ravel()
                elif val.ndim == 3:
                    sv = np.asarray(val[0, :, 1]).ravel()
                elif val.ndim == 2:
                    sv = np.asarray(val[0]).ravel()
                else:
                    sv = np.asarray(val).ravel()
            elif self.explainer_mode == "kernel":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shap_vals = self.explainer.shap_values(X_df, nsamples=40)
                    if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                        sv = np.asarray(shap_vals[1]).ravel()
                    elif isinstance(shap_vals, list):
                        sv = np.asarray(shap_vals[0]).ravel()
                    else:
                        sv = np.asarray(shap_vals).ravel()
        except Exception:
            sv = None

        # Fallback to finite-difference marginal sensitivity if SHAP encounters edge dimension
        if sv is None or len(sv) != n_feats:
            base_p = float(self.model.predict_proba(X_df)[:, 1][0])
            sv_list = []
            for col in self.features:
                mod_df = X_df.copy()
                mod_df[col] = 0.0
                new_p = float(self.model.predict_proba(mod_df)[:, 1][0])
                sv_list.append(base_p - new_p)
            sv = np.array(sv_list, dtype=float)

        sv = np.asarray(sv, dtype=float).ravel()[:n_feats]
        patient_vals = X_df.iloc[0].values.ravel()[:n_feats]

        impacts = pd.DataFrame({
            "Feature": self.features[:n_feats],
            "Standardized_Value": np.round(patient_vals, 3),
            "SHAP_Impact": np.round(sv, 4),
            "Absolute_Impact": np.round(np.abs(sv), 4)
        }).sort_values(by="Absolute_Impact", ascending=False)

        impacts["Direction"] = impacts["SHAP_Impact"].apply(
            lambda x: "Increases Risk (+)" if x > 0 else "Decreases Risk (-)"
        )
        return impacts.head(top_k)

    def compute_counterfactual(self, patient_df, modifiable_interventions):
        """Simulates clinical intervention impact on patient risk."""
        baseline_prob = float(self.model.predict_proba(patient_df[self.features])[:, 1][0])
        counterfactual_df = patient_df.copy()
        applied_actions = []

        for feat, delta_or_val in modifiable_interventions.items():
            if feat in counterfactual_df.columns:
                old_val = float(counterfactual_df.loc[counterfactual_df.index[0], feat])
                new_val = old_val + float(delta_or_val)
                counterfactual_df.loc[counterfactual_df.index[0], feat] = new_val
                applied_actions.append(f"{feat}: {old_val:+.2f} SD -> {new_val:+.2f} SD")

        new_prob = float(self.model.predict_proba(counterfactual_df[self.features])[:, 1][0])
        risk_reduction = baseline_prob - new_prob

        return {
            "baseline_risk_pct": float(np.round(baseline_prob * 100, 1)),
            "counterfactual_risk_pct": float(np.round(new_prob * 100, 1)),
            "absolute_risk_drop_pct": float(np.round(risk_reduction * 100, 1)),
            "relative_risk_reduction_pct": float(np.round((risk_reduction / (baseline_prob + 1e-5)) * 100, 1)),
            "interventions": applied_actions,
            "clinical_narrative": (
                f"With targeted modifiable intervention ({', '.join(applied_actions)}), "
                f"estimated model-predicted risk decreases from {baseline_prob*100:.1f}% to {new_prob*100:.1f}% "
                f"(an absolute risk reduction of {risk_reduction*100:.1f}%)."
            )
        }


if __name__ == "__main__":
    import joblib
    from src.preprocessing import load_heart

    print("Testing Explainability Module (SHAP + Counterfactuals)...\n")
    heart_champ = joblib.load("models/heart_champion.pkl")
    X_tr_h, X_te_h, y_tr_h, y_te_h, _, _ = load_heart(engineered=True)

    explainer = ModelExplainer(heart_champ, background_data=X_tr_h)
    sample_pt = X_te_h.iloc[0:1]
    top_drivers = explainer.explain_patient(sample_pt, top_k=5)

    print("Top Local Risk Drivers:")
    print(top_drivers[["Feature", "Standardized_Value", "SHAP_Impact", "Direction"]].to_string(index=False))

    cf = explainer.compute_counterfactual(sample_pt, {"trestbps": -0.8, "chol": -0.5})
    print(f"\nActionable Counterfactual Simulation:\n{cf['clinical_narrative']}")
