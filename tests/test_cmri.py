"""
Unit tests for Phase 8 Composite CardioMetabolic-Renal Index (CMRI).
"""

import pytest
import numpy as np
from src.cmri import CMRIndexCalculator


@pytest.fixture
def calculator():
    return CMRIndexCalculator(
        w_diabetes=0.30,
        w_heart=0.30,
        w_ckd=0.30,
        w_interaction=0.10
    )


def test_cmri_bounds(calculator):
    """Test that CMRI is bounded strictly in [0.0, 1.0]."""
    # All zeros
    res_zero = calculator.calculate(0.0, 0.0, 0.0)
    assert res_zero["cmri_score"] == 0.0
    assert res_zero["risk_category"] == "LOW RISK"

    # All ones
    res_one = calculator.calculate(1.0, 1.0, 1.0)
    assert np.isclose(res_one["cmri_score"], 1.0)
    assert res_one["risk_category"] == "HIGH COMPOSITE RISK"


def test_synergistic_interaction_term(calculator):
    """Test that the synergistic interaction term is strictly positive when all 3 risks are present."""
    res_single = calculator.calculate(0.8, 0.0, 0.0)
    assert res_single["components"]["synergistic_interaction"] == 0.0

    res_triple = calculator.calculate(0.8, 0.8, 0.8)
    assert res_triple["components"]["synergistic_interaction"] > 0.0
    # 0.10 * (0.8 * 0.8 * 0.8) = 0.0512
    assert np.isclose(res_triple["components"]["synergistic_interaction"], 0.10 * (0.8**3), atol=1e-4)


def test_risk_category_thresholds(calculator):
    """Test that risk stratification thresholds map correctly."""
    # Low: < 0.30
    res_low = calculator.calculate(0.1, 0.1, 0.1)
    assert res_low["risk_category"] == "LOW RISK"
    assert res_low["category_color"] == "#2ecc71"

    # Moderate: 0.30 - 0.60
    res_mod = calculator.calculate(0.4, 0.5, 0.4)
    assert res_mod["risk_category"] == "MODERATE RISK"
    assert res_mod["category_color"] == "#f39c12"

    # High: >= 0.60
    res_high = calculator.calculate(0.8, 0.7, 0.8)
    assert res_high["risk_category"] == "HIGH COMPOSITE RISK"
    assert res_high["category_color"] == "#e74c3c"
