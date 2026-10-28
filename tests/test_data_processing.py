"""
Unit tests for data preprocessing and serverless ETL logic.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add lambda directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lambda')))
from lambda_function import process_telemetry_dataframe

def test_deduplication():
    df_with_dupes = pd.DataFrame({
        'patient_id': ['PT-1', 'PT-2', 'PT-2'],
        'age': [65, 72, 72],
        'heart_rate': [80, 95, 95],
        'blood_pressure': [120, 140, 140],
        'glucose': [100, 130, 130],
        'underlying_conditions': [1, 2, 2],
        'readmitted_30days': [0, 1, 1]
    })
    cleaned = process_telemetry_dataframe(df_with_dupes)
    assert len(cleaned) == 2, "Duplicate row was not removed by deduplication."

def test_median_imputation():
    df_with_nans = pd.DataFrame({
        'patient_id': ['PT-1', 'PT-2', 'PT-3'],
        'age': [50.0, None, 70.0],
        'heart_rate': [80.0, 90.0, None],
        'blood_pressure': [120.0, None, 140.0],
        'glucose': [None, 110.0, 130.0],
        'underlying_conditions': [0, 1, 2],
        'readmitted_30days': [0, 1, 0]
    })
    cleaned = process_telemetry_dataframe(df_with_nans)
    assert not cleaned[['age', 'heart_rate', 'blood_pressure', 'glucose']].isnull().any().any(), \
        "Missing clinical values were not imputed."
    # Age median of [50, 70] is 60
    assert cleaned.loc[cleaned['patient_id'] == 'PT-2', 'age'].values[0] == 60.0

def test_min_max_normalization():
    df_raw = pd.DataFrame({
        'patient_id': ['PT-1', 'PT-2'],
        'age': [20.0, 80.0],
        'heart_rate': [50.0, 150.0],
        'blood_pressure': [100.0, 200.0],
        'glucose': [80.0, 180.0],
        'underlying_conditions': [0, 2],
        'readmitted_30days': [0, 1]
    })
    cleaned = process_telemetry_dataframe(df_raw)
    
    for col in ['age_norm', 'heart_rate_norm', 'blood_pressure_norm', 'glucose_norm']:
        assert col in cleaned.columns, f"Normalized column {col} missing."
        assert cleaned[col].min() >= 0.0, f"{col} has values below 0.0"
        assert cleaned[col].max() <= 1.0, f"{col} has values above 1.0"

