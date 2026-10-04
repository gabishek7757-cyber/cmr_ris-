"""
Unit tests for CMR-RIS Pydantic clinical intake & contract schemas.
"""

import pytest
from pydantic import ValidationError
from src.schemas import (
    PatientIntake,
    ConformalInterval,
    CMRIOutput,
    TriagePriorityResult
)


def test_valid_patient_intake():
    """Test valid patient intake payload instantiation and clinical conversion."""
    data = {
        "age": 45,
        "sex": "Male",
        "pregnancies": 0,
        "sys_bp": 130.0,
        "dia_bp": 85.0,
        "bmi": 28.5,
        "glucose": 115.0,
        "dpf": 0.45,
        "tot_chol": 210.0,
        "thalach": 145.0,
        "oldpeak": 0.8,
        "cp_type": 1,
        "blood_urea": 22.0,
        "serum_creatinine": 1.1,
        "hemoglobin": 14.2
    }
    intake = PatientIntake(**data)
    assert intake.age == 45
    assert intake.sex_numeric == 1
    assert intake.sys_bp == 130.0
    assert intake.dia_bp == 85.0

    c_dict = intake.to_clinical_dict()
    assert isinstance(c_dict, dict)
    assert c_dict["glucose"] == 115.0
    assert c_dict["sex_num"] == 1


def test_invalid_blood_pressure_constraint():
    """Test that diastolic BP >= systolic BP raises a biological validation error."""
    invalid_data = {
        "age": 50,
        "sex": "Female",
        "sys_bp": 110.0,
        "dia_bp": 120.0,  # Invalid: Dia >= Sys
        "bmi": 25.0,
        "glucose": 100.0,
        "tot_chol": 190.0,
        "thalach": 140.0,
        "blood_urea": 20.0,
        "serum_creatinine": 0.9,
        "hemoglobin": 13.0
    }
    with pytest.raises(ValidationError) as excinfo:
        PatientIntake(**invalid_data)
    assert "Diastolic BP" in str(excinfo.value)


def test_conformal_interval_schema():
    """Test ConformalInterval schema validation."""
    ci = ConformalInterval(
        point_probability=0.75,
        point_percentage=75.0,
        lower_bound=0.68,
        upper_bound=0.82,
        lower_pct=68.0,
        upper_pct=82.0,
        margin_error_pct=7.0,
        confidence_level=0.90,
        formatted="75.0% (90% CI: 68.0%–82.0%)"
    )
    assert ci.point_probability == 0.75
    assert ci.lower_pct <= ci.upper_pct


def test_cmri_output_schema():
    """Test CMRIOutput schema validation."""
    cmri = CMRIOutput(
        cmri_score=0.42,
        cmri_percentage=42.0,
        risk_category="MODERATE RISK",
        category_color="#f39c12",
        clinical_guidance="Lifestyle interventions.",
        components={"diabetes": 0.1, "heart": 0.2, "ckd": 0.1, "interaction": 0.02},
        formula_audit="CMRI = 0.30*(0.40) + ..."
    )
    assert cmri.risk_category == "MODERATE RISK"
    assert cmri.cmri_score == 0.42


def test_triage_priority_result_schema():
    """Test TriagePriorityResult schema validation."""
    res = TriagePriorityResult(
        priority_score=0.72,
        priority_percentage=72.0,
        priority_tier="URGENT (HIGH PRIORITY)",
        badge_color="#e74c3c",
        recommended_clinical_action="Immediate clinician triage.",
        components={"severity": 36.0, "velocity": 21.0, "burden": 15.0},
        formula_audit="Priority = 0.50*(0.72) + ..."
    )
    assert res.priority_tier == "URGENT (HIGH PRIORITY)"
    assert res.priority_score == 0.72
