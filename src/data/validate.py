# src/data/validate.py
"""
Data validation module.
Checks the HR dataset meets expected schema and quality rules.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns in the dataset
EXPECTED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate",
    "Department", "DistanceFromHome", "Education",
    "EducationField", "EmployeeCount", "EmployeeNumber",
    "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "Over18", "OverTime",
    "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StandardHours",
    "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]


def validate_hr_data(df: pd.DataFrame) -> dict:
    """
    Validate the HR DataFrame against quality rules.

    Args:
        df: Raw HR DataFrame to validate

    Returns:
        dict with validation results
    """
    logger.info("Starting data validation...")

    results = {
        "passed": True,
        "checks": {}
    }

    # Check 1: Row count
    row_check = len(df) >= 1000
    results["checks"]["row_count"] = {
        "passed": row_check,
        "value": len(df),
        "expected": ">= 1000"
    }
    logger.info(f"Row count check: {len(df)} rows - {'PASS' if row_check else 'FAIL'}")

    # Check 2: All expected columns present
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    col_check = len(missing_cols) == 0
    results["checks"]["columns"] = {
        "passed": col_check,
        "missing": missing_cols
    }
    logger.info(f"Column check: {'PASS' if col_check else f'FAIL - missing {missing_cols}'}")

    # Check 3: No null values
    null_count = df.isnull().sum().sum()
    null_check = null_count == 0
    results["checks"]["nulls"] = {
        "passed": null_check,
        "null_count": int(null_count)
    }
    logger.info(f"Null check: {null_count} nulls - {'PASS' if null_check else 'FAIL'}")

    # Check 4: Attrition column has only Yes/No
    valid_attrition = set(df["Attrition"].unique()) == {"Yes", "No"}
    results["checks"]["attrition_values"] = {
        "passed": valid_attrition,
        "values_found": df["Attrition"].unique().tolist()
    }
    logger.info(f"Attrition values check: {'PASS' if valid_attrition else 'FAIL'}")

    # Check 5: Age is reasonable
    age_check = df["Age"].between(18, 70).all()
    results["checks"]["age_range"] = {
        "passed": bool(age_check),
        "min": int(df["Age"].min()),
        "max": int(df["Age"].max())
    }
    logger.info(f"Age range check: {df['Age'].min()}-{df['Age'].max()} - {'PASS' if age_check else 'FAIL'}")

    # Overall result
    results["passed"] = all(
        check["passed"] for check in results["checks"].values()
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.ingest import load_hr_data
    df = load_hr_data()

    results = validate_hr_data(df)

    print(f"\n{'='*50}")
    print("VALIDATION RESULTS")
    print(f"{'='*50}")
    for check_name, details in results["checks"].items():
        status = "✅ PASS" if details["passed"] else "❌ FAIL"
        print(f"{status} {check_name}")
    print(f"{'='*50}")
    print(f"Overall: {'✅ ALL PASSED' if results['passed'] else '❌ FAILED'}")