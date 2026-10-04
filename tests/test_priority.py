"""
Unit tests for Phase 12 Auditable MCDA Triage Priority Engine.
"""

import pytest
import numpy as np
from src.priority import MCDAPriorityEngine


@pytest.fixture
def engine():
    return MCDAPriorityEngine(alpha=0.50, beta=0.30, gamma=0.20)


def test_mcda_priority_bounds(engine):
    """Test priority score bounds in [0.0, 1.0]."""
    # Low severity, zero slope, no high risk disease
    res_low = engine.compute_priority(
        cmri_score=0.10,
        theil_sen_annual_slope=0.0,
        disease_probs_dict={"diabetes": 0.1, "heart": 0.1, "ckd": 0.1}
    )
    assert 0.0 <= res_low["priority_score"] < 0.35
    assert res_low["priority_tier"] == "ROUTINE (LOW PRIORITY)"

    # High severity, positive slope, triple high risk
    res_high = engine.compute_priority(
        cmri_score=0.85,
        theil_sen_annual_slope=0.10,
        disease_probs_dict={"diabetes": 0.8, "heart": 0.9, "ckd": 0.7}
    )
    assert res_high["priority_score"] >= 0.65
    assert res_high["priority_tier"] == "URGENT (HIGH PRIORITY)"


def test_mcda_formula_components(engine):
    """Test that individual component percentages sum to the total priority percentage."""
    res = engine.compute_priority(
        cmri_score=0.50,
        theil_sen_annual_slope=0.05,
        disease_probs_dict={"diabetes": 0.6, "heart": 0.2, "ckd": 0.1}
    )
    comps = res["components"]
    total_from_comps = sum(comps.values())
    assert np.isclose(total_from_comps, res["priority_percentage"], atol=0.1)
