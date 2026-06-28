# tests/unit/test_data_pipeline.py
"""
Unit tests for the data pipeline.
Run with: pytest tests/unit/test_data_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np


# ─── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Minimal valid HR DataFrame for testing."""
    return pd.DataFrame({
        "Age": [35, 28, 45, 52, 30],
        "Attrition": ["Yes", "No", "No", "Yes", "No"],
        "BusinessTravel": ["Travel_Rarely"] * 5,
        "DailyRate": [1000, 800, 1200, 900, 1100],
        "Department": ["Sales", "Research & Development",
                       "Sales", "Human Resources",
                       "Research & Development"],
        "DistanceFromHome": [5, 10, 2, 20, 8],
        "Education": [3, 4, 2, 3, 4],
        "EducationField": ["Marketing"] * 5,
        "EmployeeCount": [1, 1, 1, 1, 1],
        "EmployeeNumber": [1, 2, 3, 4, 5],
        "EnvironmentSatisfaction": [3, 4, 2, 1, 3],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "HourlyRate": [50, 60, 45, 55, 65],
        "JobInvolvement": [3, 4, 2, 3, 4],
        "JobLevel": [2, 3, 1, 2, 3],
        "JobRole": ["Sales Executive"] * 5,
        "JobSatisfaction": [3, 4, 2, 1, 4],
        "MaritalStatus": ["Single", "Married", "Single",
                          "Divorced", "Married"],
        "MonthlyIncome": [5000, 8000, 3000, 6500, 9000],
        "MonthlyRate": [10000] * 5,
        "NumCompaniesWorked": [2, 4, 1, 3, 2],
        "Over18": ["Y", "Y", "Y", "Y", "Y"],
        "OverTime": ["Yes", "No", "Yes", "No", "Yes"],
        "PercentSalaryHike": [11, 13, 15, 12, 14],
        "PerformanceRating": [3, 4, 3, 3, 4],
        "RelationshipSatisfaction": [4, 3, 2, 3, 4],
        "StandardHours": [80, 80, 80, 80, 80],
        "StockOptionLevel": [0, 1, 2, 0, 1],
        "TotalWorkingYears": [10, 8, 15, 12, 7],
        "TrainingTimesLastYear": [2, 3, 1, 2, 3],
        "WorkLifeBalance": [3, 4, 3, 2, 3],
        "YearsAtCompany": [5, 6, 8, 4, 3],
        "YearsInCurrentRole": [3, 4, 6, 2, 2],
        "YearsSinceLastPromotion": [1, 2, 3, 1, 0],
        "YearsWithCurrManager": [3, 4, 5, 2, 2],
    })


# ─── Validation Tests ─────────────────────────────────────────
class TestValidation:
    """Tests for src/data/validate.py"""

    def test_valid_data_key_checks_pass(self, sample_raw_df):
        """
        Key quality checks pass on valid data.
        Note: row_count check fails on small sample (needs >1000)
        — this is expected and correct behaviour.
        """
        from src.data.validate import validate_hr_data
        results = validate_hr_data(sample_raw_df)
        # These checks should always pass on valid data
        assert results["checks"]["nulls"]["passed"] == True
        assert results["checks"]["attrition_values"]["passed"] == True
        assert results["checks"]["age_range"]["passed"] == True

    def test_row_count_check(self, sample_raw_df):
        from src.data.validate import validate_hr_data
        results = validate_hr_data(sample_raw_df)
        assert results["checks"]["row_count"]["value"] == 5

    def test_null_check_passes(self, sample_raw_df):
        from src.data.validate import validate_hr_data
        results = validate_hr_data(sample_raw_df)
        assert results["checks"]["nulls"]["null_count"] == 0

    def test_attrition_values_valid(self, sample_raw_df):
        from src.data.validate import validate_hr_data
        results = validate_hr_data(sample_raw_df)
        assert results["checks"]["attrition_values"]["passed"] is True

    def test_invalid_attrition_fails(self, sample_raw_df):
        from src.data.validate import validate_hr_data
        bad_df = sample_raw_df.copy()
        bad_df.loc[0, "Attrition"] = "Maybe"
        results = validate_hr_data(bad_df)
        assert results["checks"]["attrition_values"]["passed"] is False

    def test_age_range_valid(self, sample_raw_df):
        from src.data.validate import validate_hr_data
        results = validate_hr_data(sample_raw_df)
        assert results["checks"]["age_range"]["passed"] is True

    def test_null_introduced_detected(self, sample_raw_df):
        from src.data.validate import validate_hr_data
        bad_df = sample_raw_df.copy()
        bad_df.loc[0, "Age"] = None
        results = validate_hr_data(bad_df)
        assert results["checks"]["nulls"]["null_count"] > 0


# ─── Cleaning Tests ───────────────────────────────────────────
class TestCleaning:
    """Tests for src/data/clean.py"""

    def test_attrition_encoded_binary(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert result["Attrition"].dtype in ["int64", "int32"]
        assert set(result["Attrition"].unique()).issubset({0, 1})

    def test_overtime_encoded_binary(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert set(result["OverTime"].unique()).issubset({0, 1})

    def test_constant_columns_removed(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert "EmployeeCount" not in result.columns
        assert "Over18" not in result.columns
        assert "StandardHours" not in result.columns
        assert "EmployeeNumber" not in result.columns

    def test_derived_columns_added(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert "TenureRatio" in result.columns
        assert "SatisfactionScore" in result.columns

    def test_no_nulls_after_cleaning(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert result.isnull().sum().sum() == 0

    def test_row_count_preserved(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert len(result) == len(sample_raw_df)

    def test_tenure_ratio_between_0_and_1(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        assert (result["TenureRatio"] >= 0).all()
        assert (result["TenureRatio"] <= 1).all()

    def test_column_count_reduced(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        # Started with 35, removed 4, added 2 = 33
        assert result.shape[1] == 33

    def test_attrition_yes_becomes_1(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        # Row 0 had Attrition=Yes — should be 1
        assert result.iloc[0]["Attrition"] == 1

    def test_attrition_no_becomes_0(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        result = clean_hr_data(sample_raw_df)
        # Row 1 had Attrition=No — should be 0
        assert result.iloc[1]["Attrition"] == 0


# ─── Feature Engineering Tests ────────────────────────────────
class TestFeatureEngineering:
    """Tests for src/features/feature_engineering.py"""

    @pytest.fixture
    def clean_df(self, sample_raw_df):
        from src.data.clean import clean_hr_data
        return clean_hr_data(sample_raw_df)

    def test_encoding_removes_text_columns(self, clean_df):
        from src.features.feature_engineering import encode_categorical
        result = encode_categorical(clean_df)
        obj_cols = result.select_dtypes(include=["object"]).columns
        assert len(obj_cols) == 0

    def test_select_features_separates_target(self, clean_df):
        from src.features.feature_engineering import (
            encode_categorical, select_features
        )
        encoded = encode_categorical(clean_df)
        X, y = select_features(encoded)
        assert "Attrition" not in X.columns
        assert len(y) == len(X)

    def test_select_features_target_is_binary(self, clean_df):
        from src.features.feature_engineering import (
            encode_categorical, select_features
        )
        encoded = encode_categorical(clean_df)
        X, y = select_features(encoded)
        assert set(y.unique()).issubset({0, 1})

    def test_split_preserves_total_rows(self, clean_df):
        from src.features.feature_engineering import (
            encode_categorical, select_features, split_data
        )
        encoded = encode_categorical(clean_df)
        X, y = select_features(encoded)
        # Use larger test_size for small 5-row sample
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.4
        )
        assert len(X_train) + len(X_test) == len(X)

    def test_encoded_columns_are_numeric(self, clean_df):
        from src.features.feature_engineering import encode_categorical
        result = encode_categorical(clean_df)
        non_numeric = result.select_dtypes(
            exclude=["number", "bool"]
        ).columns
        assert len(non_numeric) == 0


# ─── Agent Tools Tests ────────────────────────────────────────
class TestAgentTools:
    """Tests for src/agents/tools/hr_tools.py"""

    def test_get_employee_details_found(self):
        import json
        from src.agents.tools.hr_tools import get_employee_details
        result = json.loads(get_employee_details(1))
        assert result["status"] == "found"
        assert result["employee_id"] == 1
        assert "department" in result
        assert "monthly_income" in result

    def test_get_employee_details_not_found(self):
        import json
        from src.agents.tools.hr_tools import get_employee_details
        result = json.loads(get_employee_details(9999))
        assert result["status"] == "not_found"

    def test_get_employee_details_has_all_fields(self):
        import json
        from src.agents.tools.hr_tools import get_employee_details
        result = json.loads(get_employee_details(1))
        required_fields = [
            "employee_id", "age", "department", "job_role",
            "monthly_income", "overtime", "attrition"
        ]
        for field in required_fields:
            assert field in result

    def test_get_attrition_risk_returns_level(self):
        import json
        from src.agents.tools.hr_tools import get_attrition_risk
        result = json.loads(get_attrition_risk(1))
        assert result["status"] == "success"
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(result["risk_score"], int)

    def test_get_attrition_risk_has_recommendation(self):
        import json
        from src.agents.tools.hr_tools import get_attrition_risk
        result = json.loads(get_attrition_risk(1))
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0

    def test_get_department_stats_sales(self):
        import json
        from src.agents.tools.hr_tools import get_department_stats
        result = json.loads(get_department_stats("Sales"))
        assert result["status"] == "found"
        assert result["employee_count"] > 0
        assert "median_salary" in result

    def test_get_department_stats_invalid(self):
        import json
        from src.agents.tools.hr_tools import get_department_stats
        result = json.loads(get_department_stats("InvalidDept"))
        assert result["status"] == "not_found"

    def test_high_risk_employee_score(self):
        """Employee 7 is known HIGH risk from our EDA."""
        import json
        from src.agents.tools.hr_tools import get_attrition_risk
        result = json.loads(get_attrition_risk(7))
        assert result["risk_level"] == "HIGH"
        assert result["risk_score"] >= 7