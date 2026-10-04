"""
Unit tests for Phase 10 Statistically-Grounded Longitudinal Trajectory Engine.
"""

import pytest
import numpy as np
from src.trajectory import theil_sen_slope, mann_kendall_test, classify_trajectory


def test_theil_sen_monotonic_increase():
    """Test Theil-Sen slope calculation on clearly increasing risk data."""
    timestamps = [0, 1, 2, 3, 4]
    values = [0.20, 0.25, 0.30, 0.35, 0.40]
    slope = theil_sen_slope(timestamps, values)
    assert np.isclose(slope, 0.05)


def test_theil_sen_outlier_resilience():
    """Test that a single acute spike outlier does not distort the median slope."""
    timestamps = [0, 1, 2, 3, 4]
    values = [0.20, 0.22, 0.85, 0.26, 0.28]  # Spike at index 2
    slope = theil_sen_slope(timestamps, values)
    assert 0.01 <= slope <= 0.04


def test_mann_kendall_significance():
    """Test Mann-Kendall test detects statistically significant upward trend."""
    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    res = mann_kendall_test(values)
    assert res["S"] > 0
    assert res["tau"] > 0.8
    assert res["p_value"] < 0.05
    assert "Significant Increase" in res["trend"]


def test_mann_kendall_short_series():
    """Test Mann-Kendall handles small samples (N < 3) gracefully."""
    values = [0.20, 0.30]
    res = mann_kendall_test(values)
    assert res["p_value"] == 1.0
    assert "Indeterminate" in res["trend"]


def test_classify_trajectory_categories():
    """Test trajectory classification mapping with timestamps and scores."""
    # Stable series
    t_stable = [0, 1, 2, 3]
    r_stable = [0.30, 0.31, 0.30, 0.31]
    res_stable = classify_trajectory(t_stable, r_stable)
    assert res_stable["trajectory_status"] == "STABLE"

    # Rapidly deteriorating series
    t_rapid = [0, 1, 2, 3, 4]
    r_rapid = [0.20, 0.30, 0.45, 0.60, 0.80]
    res_rapid = classify_trajectory(t_rapid, r_rapid)
    assert res_rapid["trajectory_status"] == "RAPIDLY DETERIORATING"
