"""
Unit tests for model inference calculations and risk scoring boundaries.
"""
import pytest
import os
import sys

def calculate_heuristic_risk(age, heart_rate, blood_pressure, conditions):
    return min(100, int((age * 0.3) + (heart_rate * 0.2) + (blood_pressure * 0.2) + (conditions * 10)))

def test_heuristic_risk_boundaries():
    # Test low risk baseline (young, healthy vitals, no conditions)
    low_risk = calculate_heuristic_risk(age=20, heart_rate=60, blood_pressure=100, conditions=0)
    # (20*0.3) + (60*0.2) + (100*0.2) + 0 = 6 + 12 + 20 = 38
    assert 0 <= low_risk <= 100

    # Test high risk senior (elderly, high vitals, multi-comorbid)
    high_risk = calculate_heuristic_risk(age=85, heart_rate=140, blood_pressure=180, conditions=4)
    # (85*0.3) + (140*0.2) + (180*0.2) + 40 = 25.5 + 28 + 36 + 40 = 129.5 -> capped at 100
    assert high_risk == 100

def test_risk_categorization():
    def get_category(score):
        if score > 60:
            return "High"
        elif score > 30:
            return "Moderate"
        return "Low"

    assert get_category(85) == "High"
    assert get_category(45) == "Moderate"
    assert get_category(25) == "Low"
