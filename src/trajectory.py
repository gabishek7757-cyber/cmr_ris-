"""
Phase 10 — Statistically-Grounded Trajectory Engine
CMR-RIS: CardioMetabolic-Renal Risk Intelligence System

Core Contribution #3:
Statistically robust trend detection across longitudinal visits using:
1. Theil-Sen Robust Slope Estimator (resilient against outlier lab spikes).
2. Mann-Kendall Non-Parametric Monotonic Trend Test (p-value and Kendall tau).
3. Longitudinal risk trajectory simulation for cross-sectional profiles.
"""

import os
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats


def theil_sen_slope(timestamps, values):
    """Computes the Theil-Sen robust median slope."""
    t = np.asarray(timestamps, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(t)
    if n < 2:
        return 0.0

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if dt != 0:
                slopes.append((v[j] - v[i]) / dt)

    if not slopes:
        return 0.0
    return float(np.median(slopes))


def mann_kendall_test(values):
    """Performs the Mann-Kendall non-parametric monotonic trend test."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3:
        return {"S": 0, "tau": 0.0, "p_value": 1.0, "trend": "Indeterminate (N < 3)"}

    # S Statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += int(np.sign(v[j] - v[i]))

    # Variance of S
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0

    # Z Statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    tau = float(s / (0.5 * n * (n - 1)))

    if p_value < 0.05:
        trend = "Statistically Significant Increase" if s > 0 else "Statistically Significant Decrease"
    else:
        trend = "No Statistically Significant Trend (Stable/Drifting)"

    return {
        "S": int(s),
        "Z": float(np.round(z, 3)),
        "tau": float(np.round(tau, 3)),
        "p_value": float(np.round(p_value, 4)),
        "trend": trend
    }


def classify_trajectory(timestamps, risk_scores):
    """Classifies risk trajectory into clinical triage categories."""
    t = np.asarray(timestamps, dtype=float)
    r = np.asarray(risk_scores, dtype=float)
    slope = theil_sen_slope(t, r)
    mk = mann_kendall_test(r)

    if slope < -0.02 and mk["p_value"] < 0.15:
        status = "IMPROVING"
        color = "#2ecc71"
        action = "Positive response to ongoing therapeutic intervention."
    elif abs(slope) <= 0.02 or mk["p_value"] >= 0.15:
        status = "STABLE"
        color = "#3498db"
        action = "Maintain current care protocol; re-evaluate at regular interval."
    elif slope <= 0.06:
        status = "GRADUALLY INCREASING"
        color = "#f39c12"
        action = "Early progressive risk detected; intensify lifestyle and medication review."
    else:
        status = "RAPIDLY DETERIORATING"
        color = "#e74c3c"
        action = "High-velocity clinical deterioration; urgent triage recommended."

    return {
        "theil_sen_annual_slope": float(np.round(slope, 4)),
        "annual_drift_percentage": float(np.round(slope * 100, 2)),
        "trajectory_status": status,
        "status_color": color,
        "clinical_action": action,
        "mann_kendall": mk,
        "history": [{"year": float(t[i]), "risk_pct": float(np.round(r[i] * 100, 1))} for i in range(len(t))]
    }


def simulate_patient_trajectory(baseline_cmri, years=4, trajectory_mode="gradual_worsening"):
    """Simulates a multi-year longitudinal sequence with physiological drift for demo/testing."""
    drift_rates = {
        "rapid_deterioration": 0.075,
        "gradual_worsening": 0.035,
        "stable": 0.005,
        "improving": -0.040
    }
    drift = drift_rates.get(trajectory_mode, 0.03)
    np.random.seed(42)

    years_arr = np.arange(0, years + 1, 1.0)
    scores = []
    current_val = baseline_cmri

    for y in years_arr:
        noise = np.random.normal(0, 0.015)
        val = np.clip(current_val + (drift * y) + noise, 0.02, 0.98)
        scores.append(float(val))

    analysis = classify_trajectory(years_arr, scores)
    analysis["is_simulated"] = True
    analysis["disclaimer"] = "SIMULATED LONGITUDINAL TRAJECTORY (Not real patient history)"
    return analysis


if __name__ == "__main__":
    print("Testing Trajectory Engine (Theil-Sen + Mann-Kendall):\n")

    scenarios = [
        ("Stable Patient", [0, 1, 2, 3], [0.22, 0.23, 0.22, 0.24]),
        ("Improving Lifestyle", [0, 1, 2, 3], [0.65, 0.58, 0.49, 0.42]),
        ("Gradual Metabolic Drift", [0, 1, 2, 3], [0.30, 0.35, 0.38, 0.44]),
        ("Acute Cardio-Renal Deterioration", [0, 1, 2, 3], [0.25, 0.38, 0.55, 0.72])
    ]

    for name, ts, rs in scenarios:
        res = classify_trajectory(ts, rs)
        print(f"[{res['trajectory_status']:22s}] {name:35s} | Slope: {res['annual_drift_percentage']:+5.1f}%/yr | MK p: {res['mann_kendall']['p_value']:.4f}")

    print("\nSimulated Demo Trajectory:")
    sim = simulate_patient_trajectory(0.40, years=4, trajectory_mode="gradual_worsening")
    print(f"Status: {sim['trajectory_status']} | Annual Drift: {sim['annual_drift_percentage']:+5.1f}%/yr | Label: {sim['disclaimer']}")
