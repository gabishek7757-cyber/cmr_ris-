"""
Unit tests for Phase 2 & 4 Preprocessing and Feature Engineering Pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import load_diabetes, load_heart, load_ckd


def test_load_diabetes():
    """Test diabetes data loader and engineered features."""
    X_train, X_test, y_train, y_test, scaler, pipe = load_diabetes(engineered=True)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert "Insulin_Glucose_Ratio" in X_train.columns
    assert "Glucose_ADA_Cat" in X_train.columns


def test_load_heart():
    """Test heart disease data loader and engineered features."""
    X_train, X_test, y_train, y_test, scaler, pipe = load_heart(engineered=True)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert "Rate_Pressure_Product" in X_train.columns
    assert "Chol_Age_Ratio" in X_train.columns


def test_load_ckd():
    """Test CKD data loader and engineered features."""
    X_train, X_test, y_train, y_test, scaler, pipe = load_ckd(engineered=True)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert "BUN_Creatinine_Ratio" in X_train.columns
    assert "eGFR_Proxy" in X_train.columns
