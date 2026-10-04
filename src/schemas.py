"""
CMR-RIS Data Contracts & Clinical Schemas (Pydantic v2)
CardioMetabolic-Renal Risk Intelligence System

Provides strict validation, biological constraint checking, and structured payloads
for patient intake, model predictions, conformal intervals, CMRI indices, and triage priority.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PatientIntake(BaseModel):
    """Validated standardized patient clinical intake payload."""

    # Demographics & Lifestyle
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: Literal["Female", "Male", 0, 1] = Field(
        ..., description="Biological sex (0=Female, 1=Male or string representation)"
    )
    pregnancies: int = Field(
        default=0, ge=0, le=25, description="Number of pregnancies (for female metabolic assessment)"
    )

    # Hemodynamics & Vitals
    sys_bp: float = Field(..., ge=50.0, le=260.0, description="Systolic Blood Pressure (mmHg)")
    dia_bp: float = Field(..., ge=30.0, le=160.0, description="Diastolic Blood Pressure (mmHg)")
    bmi: float = Field(..., ge=10.0, le=75.0, description="Body Mass Index (kg/m²)")

    # Metabolic & Glycemic
    glucose: float = Field(..., ge=40.0, le=600.0, description="Fasting Blood Glucose (mg/dL)")
    dpf: float = Field(
        default=0.47, ge=0.0, le=3.0, description="Diabetes Pedigree Function (genetic score)"
    )

    # Cardiovascular & Lipid Panel
    tot_chol: float = Field(..., ge=80.0, le=600.0, description="Total Serum Cholesterol (mg/dL)")
    thalach: float = Field(..., ge=40.0, le=240.0, description="Maximum Heart Rate Achieved (bpm)")
    oldpeak: float = Field(
        default=0.0, ge=0.0, le=8.0, description="ST depression induced by exercise relative to rest"
    )
    cp_type: int = Field(
        default=0, ge=0, le=3, description="Chest Pain Type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)"
    )

    # Renal & Hematologic
    blood_urea: float = Field(..., ge=1.0, le=300.0, description="Blood Urea Nitrogen (mg/dL)")
    serum_creatinine: float = Field(..., ge=0.1, le=20.0, description="Serum Creatinine (mg/dL)")
    hemoglobin: float = Field(..., ge=2.0, le=22.0, description="Hemoglobin (g/dL)")

    @property
    def sex_numeric(self) -> int:
        if isinstance(self.sex, str):
            return 1 if self.sex.lower() == "male" else 0
        return int(self.sex)

    @field_validator("dia_bp")
    @classmethod
    def validate_bp_relationship(cls, v: float, info) -> float:
        # Cross-validation handled at model validator for full context
        return v

    @model_validator(mode="after")
    def check_blood_pressures(self):
        if self.dia_bp >= self.sys_bp:
            raise ValueError(
                f"Diastolic BP ({self.dia_bp} mmHg) must be strictly less than Systolic BP ({self.sys_bp} mmHg)."
            )
        return self

    def to_clinical_dict(self) -> Dict[str, float]:
        """Converts intake model to feature mapping dictionary for model inference."""
        return {
            "glucose": float(self.glucose),
            "dia_bp": float(self.dia_bp),
            "sys_bp": float(self.sys_bp),
            "bmi": float(self.bmi),
            "age": int(self.age),
            "dpf": float(self.dpf),
            "pregnancies": int(self.pregnancies),
            "sex_num": self.sex_numeric,
            "cp_type": int(self.cp_type),
            "tot_chol": float(self.tot_chol),
            "thalach": float(self.thalach),
            "oldpeak": float(self.oldpeak),
            "blood_urea": float(self.blood_urea),
            "serum_creatinine": float(self.serum_creatinine),
            "hemoglobin": float(self.hemoglobin)
        }


class ConformalInterval(BaseModel):
    """Conformal prediction interval result."""
    point_probability: float = Field(..., ge=0.0, le=1.0)
    point_percentage: float = Field(..., ge=0.0, le=100.0)
    lower_bound: float = Field(..., ge=0.0, le=1.0)
    upper_bound: float = Field(..., ge=0.0, le=1.0)
    lower_pct: float = Field(..., ge=0.0, le=100.0)
    upper_pct: float = Field(..., ge=0.0, le=100.0)
    margin_error_pct: float = Field(..., ge=0.0, le=100.0)
    confidence_level: float = Field(default=0.90, ge=0.50, le=0.99)
    formatted: str


class CMRIOutput(BaseModel):
    """Composite CardioMetabolic-Renal Index output schema."""
    cmri_score: float = Field(..., ge=0.0, le=1.0)
    cmri_percentage: float = Field(..., ge=0.0, le=100.0)
    risk_category: Literal["LOW RISK", "MODERATE RISK", "HIGH COMPOSITE RISK"]
    category_color: str
    clinical_guidance: str
    components: Dict[str, float]
    formula_audit: str


class TrajectoryOutput(BaseModel):
    """Longitudinal trajectory test and trend output."""
    theil_sen_slope: float
    mann_kendall_p_value: float
    mann_kendall_tau: float
    trend_description: str
    classification_label: str
    risk_velocity_tier: str


class TriagePriorityResult(BaseModel):
    """MCDA Triage Screening Priority output schema."""
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_percentage: float = Field(..., ge=0.0, le=100.0)
    priority_tier: Literal[
        "ROUTINE (LOW PRIORITY)",
        "EXPEDITED (MEDIUM PRIORITY)",
        "URGENT (HIGH PRIORITY)"
    ]
    badge_color: str
    recommended_clinical_action: str
    components: Dict[str, float]
    formula_audit: str


class FullPatientAssessment(BaseModel):
    """Comprehensive multi-organ clinical assessment output."""
    patient_id: Optional[str] = "CMR-PATIENT-AUTO"
    timestamp: str
    intake: PatientIntake
    diabetes_risk: ConformalInterval
    heart_risk: ConformalInterval
    ckd_risk: ConformalInterval
    cmri: CMRIOutput
    priority: TriagePriorityResult
