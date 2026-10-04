"""
Unit tests for Phase 9 Conformal Uncertainty Quantification.
"""

import pytest
import numpy as np
from src.uncertainty import ConformalRiskCalibrator


def test_conformal_interval_bounds():
    """Test that conformal prediction intervals stay clamped in [0.0, 1.0]."""
    calibrator = ConformalRiskCalibrator(confidence_level=0.90)
    calibrator.calibrated_quantiles["diabetes"] = {"q_val": 0.15}

    # Extreme low point probability
    ci_low = calibrator.predict_interval("diabetes", 0.05)
    assert ci_low["lower_bound"] == 0.0
    assert ci_low["upper_bound"] == 0.20
    assert ci_low["point_percentage"] == 5.0

    # Extreme high point probability
    ci_high = calibrator.predict_interval("diabetes", 0.95)
    assert ci_high["lower_bound"] == 0.80
    assert ci_high["upper_bound"] == 1.0
    assert ci_high["point_percentage"] == 95.0


def test_conformal_margin_of_error():
    """Test that margin of error is correctly reported."""
    calibrator = ConformalRiskCalibrator(confidence_level=0.90)
    calibrator.calibrated_quantiles["heart"] = {"q_val": 0.085}

    ci = calibrator.predict_interval("heart", 0.50)
    assert ci["margin_error_pct"] == 8.5
    assert ci["lower_pct"] == 41.5
    assert ci["upper_pct"] == 58.5
