# src/data/clean.py
"""
Data cleaning module.
Transforms raw HR data (Bronze) into clean data (Silver).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_hr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the IBM HR Analytics dataset.
    Bronze (raw) -> Silver (clean) transformation.

    Changes:
    1. Convert Yes/No columns to 1/0
    2. Drop constant columns (add no information)
    3. Drop identifier column
    4. Add derived features

    Args:
        df: Raw HR DataFrame

    Returns:
        Cleaned DataFrame ready for analysis
    """
    logger.info("Starting data cleaning...")
    df = df.copy()

    # Step 1: Convert Yes/No to 1/0
    binary_columns = ["Attrition", "OverTime"]
    for col in binary_columns:
        df[col] = (df[col] == "Yes").astype(int)
    logger.info(f"Converted binary columns: {binary_columns}")

    # Step 2: Drop constant columns
    # These columns have the same value for every row
    constant_columns = ["EmployeeCount", "Over18", "StandardHours"]
    df = df.drop(columns=constant_columns)
    logger.info(f"Dropped constant columns: {constant_columns}")

    # Step 3: Drop identifier column
    # EmployeeNumber is just an ID — not useful for ML
    df = df.drop(columns=["EmployeeNumber"])
    logger.info("Dropped EmployeeNumber")

    # Step 4: Add derived features
    # Tenure ratio: what fraction of career spent here
    df["TenureRatio"] = (
        df["YearsAtCompany"] /
        (df["TotalWorkingYears"] + 1)
    ).round(3)

    # Satisfaction composite: average of all satisfaction scores
    satisfaction_cols = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance"
    ]
    df["SatisfactionScore"] = df[satisfaction_cols].mean(axis=1).round(3)

    logger.info("Added derived features: TenureRatio, SatisfactionScore")

    # Final check
    assert df.isnull().sum().sum() == 0, "Nulls found after cleaning!"
    logger.info(f"Cleaning complete: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.ingest import load_hr_data

    # Load raw data
    df_raw = load_hr_data()
    print(f"Raw data: {df_raw.shape}")

    # Clean it
    df_clean = clean_hr_data(df_raw)
    print(f"Clean data: {df_clean.shape}")

    print(f"\n{'='*50}")
    print("NEW COLUMNS ADDED:")
    new_cols = ["TenureRatio", "SatisfactionScore"]
    print(df_clean[new_cols].head())

    print(f"\n{'='*50}")
    print("ATTRITION (now 0/1):")
    print(df_clean["Attrition"].value_counts())

    print(f"\n{'='*50}")
    print("COLUMNS REMOVED:")
    removed = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    for col in removed:
        status = "✅ Removed" if col not in df_clean.columns else "❌ Still there"
        print(f"  {status}: {col}")