"""
Phase 12 — Auditable Screening Priority Engine (MCDA)
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Core Contribution #5:
A Multi-Criteria Decision Analysis (MCDA) triage score:
Priority = α·norm(CMRI) + β·norm(trajectory slope) + γ·(count of HIGH-risk diseases / 3)

Where:
- α = 0.50 (Current Cross-Organ Risk Severity)
- β = 0.30 (Trajectory Velocity / Deterioration Rate)
- γ = 0.20 (Multi-Organ High-Risk Disease Burden)
"""

import os
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd


class MCDAPriorityEngine:
    def __init__(self, alpha=0.50, beta=0.30, gamma=0.20, slope_min=-0.05, slope_max=0.15):
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.slope_min = slope_min
        self.slope_max = slope_max

    def compute_priority(self, cmri_score, theil_sen_annual_slope, disease_probs_dict):
        """Computes triage priority score and auditable mathematical breakdown."""
        norm_cmri = float(np.clip(cmri_score, 0.0, 1.0))
        raw_slope = float(theil_sen_annual_slope)
        norm_slope = float(np.clip(
            (raw_slope - self.slope_min) / (self.slope_max - self.slope_min), 0.0, 1.0
        ))

        high_risk_count = sum(1 for p in disease_probs_dict.values() if p >= 0.50)
        high_risk_ratio = high_risk_count / 3.0

        component_cmri = self.alpha * norm_cmri
        component_slope = self.beta * norm_slope
        component_multidisease = self.gamma * high_risk_ratio

        priority_score = float(np.clip(component_cmri + component_slope + component_multidisease, 0.0, 1.0))

        if priority_score < 0.35:
            tier = "ROUTINE (LOW PRIORITY)"
            badge_color = "#2ecc71"
            action = "Schedule standard annual preventive follow-up and lifestyle coaching."
        elif priority_score < 0.65:
            tier = "EXPEDITED (MEDIUM PRIORITY)"
            badge_color = "#f39c12"
            action = "Recommend clinical consultation and repeat multi-organ screening within 3–6 months."
        else:
            tier = "URGENT (HIGH PRIORITY)"
            badge_color = "#e74c3c"
            action = "Flag for immediate clinician triage, specialist referral (Cardiology/Nephrology/Endo), and aggressive risk management."

        return {
            "priority_score": float(np.round(priority_score, 4)),
            "priority_percentage": float(np.round(priority_score * 100, 1)),
            "priority_tier": tier,
            "badge_color": badge_color,
            "recommended_clinical_action": action,
            "components": {
                "current_cmri_severity (50%)": float(np.round(component_cmri * 100, 2)),
                "trajectory_velocity (30%)": float(np.round(component_slope * 100, 2)),
                "multi_organ_burden (20%)": float(np.round(component_multidisease * 100, 2))
            },
            "formula_audit": (
                f"Priority = {self.alpha:.2f}*({norm_cmri:.2f}) + {self.beta:.2f}*({norm_slope:.2f}) + "
                f"{self.gamma:.2f}*({high_risk_ratio:.2f}) = {priority_score:.3f}"
            )
        }


if __name__ == "__main__":
    print("Testing MCDA Screening Priority Engine...\n")
    engine = MCDAPriorityEngine()

    test_scenarios = [
        {"name": "Stable Low-Risk Patient", "cmri": 0.15, "slope": 0.005, "probs": {"diab": 0.12, "heart": 0.18, "ckd": 0.10}},
        {"name": "Rapid Deterioration (High Velocity)", "cmri": 0.45, "slope": 0.085, "probs": {"diab": 0.55, "heart": 0.40, "ckd": 0.35}},
        {"name": "Multi-System Chronic Comorbidity", "cmri": 0.84, "slope": 0.050, "probs": {"diab": 0.88, "heart": 0.82, "ckd": 0.79}},
    ]

    for sc in test_scenarios:
        res = engine.compute_priority(sc["cmri"], sc["slope"], sc["probs"])
        print(f"[{res['priority_tier']:28s}] {sc['name']:35s} -> Score: {res['priority_percentage']:5.1f}% | Audit: {res['formula_audit']}")
