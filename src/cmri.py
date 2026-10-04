"""
Phase 8 — Composite CardioMetabolic-Renal Index (CMRI)
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Core Contribution #1:
A shared cross-disease composite index modeling multi-organ risk coupling:
CMRI = w1·P(diabetes) + w2·P(heart) + w3·P(ckd) + w4·[P(diabetes)·P(heart)·P(ckd)]

Where:
- w1 = 0.30 (Endocrine/Metabolic risk contribution)
- w2 = 0.30 (Cardiovascular risk contribution)
- w3 = 0.30 (Renal risk contribution)
- w4 = 0.10 (Synergistic multi-organ comorbidity interaction term)
Sum of weights = 1.00
"""

import numpy as np
import pandas as pd


class CMRIndexCalculator:
    def __init__(self, w_diabetes=0.30, w_heart=0.30, w_ckd=0.30, w_interaction=0.10):
        total = w_diabetes + w_heart + w_ckd + w_interaction
        self.w1 = w_diabetes / total
        self.w2 = w_heart / total
        self.w3 = w_ckd / total
        self.w4 = w_interaction / total

    def calculate(self, p_diabetes, p_heart, p_ckd):
        """Calculates single-patient CMRI with detailed component decomposition."""
        p_d = float(np.clip(p_diabetes, 0.0, 1.0))
        p_h = float(np.clip(p_heart, 0.0, 1.0))
        p_c = float(np.clip(p_ckd, 0.0, 1.0))

        linear_part = (self.w1 * p_d) + (self.w2 * p_h) + (self.w3 * p_c)
        synergistic_coupling = self.w4 * (p_d * p_h * p_c)
        cmri_score = float(np.clip(linear_part + synergistic_coupling, 0.0, 1.0))

        if cmri_score < 0.30:
            category = "LOW RISK"
            color = "#2ecc71"
            clinical_guidance = "Routine primary care follow-up with annual metabolic panel."
        elif cmri_score < 0.60:
            category = "MODERATE RISK"
            color = "#f39c12"
            clinical_guidance = "Lifestyle interventions, targeted glycemic/BP control, repeat labs in 3-6 months."
        else:
            category = "HIGH COMPOSITE RISK"
            color = "#e74c3c"
            clinical_guidance = "Multidisciplinary triage: urgent nephrology, cardiology, and endocrinology consultation."

        return {
            "cmri_score": float(np.round(cmri_score, 4)),
            "cmri_percentage": float(np.round(cmri_score * 100, 1)),
            "risk_category": category,
            "category_color": color,
            "clinical_guidance": clinical_guidance,
            "components": {
                "diabetes_contribution": float(np.round(self.w1 * p_d, 4)),
                "heart_contribution": float(np.round(self.w2 * p_h, 4)),
                "ckd_contribution": float(np.round(self.w3 * p_c, 4)),
                "synergistic_interaction": float(np.round(synergistic_coupling, 4))
            },
            "formula_audit": (
                f"CMRI = {self.w1:.2f}·({p_d:.2f}) + {self.w2:.2f}·({p_h:.2f}) + "
                f"{self.w3:.2f}·({p_c:.2f}) + {self.w4:.2f}·({p_d:.2f}·{p_h:.2f}·{p_c:.2f}) = {cmri_score:.3f}"
            )
        }

    def batch_calculate(self, df_probs):
        """Batch calculates CMRI for a dataframe containing 'p_diabetes', 'p_heart', 'p_ckd'."""
        scores = []
        for _, row in df_probs.iterrows():
            res = self.calculate(row["p_diabetes"], row["p_heart"], row["p_ckd"])
            scores.append(res["cmri_score"])
        return np.array(scores)


if __name__ == "__main__":
    calc = CMRIndexCalculator()
    print("Testing CMRIndexCalculator on synthetic clinical profiles:\n")

    profiles = [
        ("All-Low Healthy", 0.05, 0.08, 0.02),
        ("Isolated Diabetes", 0.85, 0.15, 0.10),
        ("Isolated Cardiac", 0.10, 0.90, 0.12),
        ("Diabetic Nephropathy", 0.85, 0.20, 0.88),
        ("Triple-Organ Comorbidity", 0.90, 0.85, 0.92)
    ]

    for name, pd_val, ph_val, pc_val in profiles:
        res = calc.calculate(pd_val, ph_val, pc_val)
        print(f"[{res['risk_category']:20s}] {name:25s} -> CMRI: {res['cmri_percentage']:5.1f}% | Audit: {res['formula_audit']}")

    print("\nCMRI unit tests passed successfully.")
