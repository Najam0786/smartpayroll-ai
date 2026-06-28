# src/features/feature_engineering.py
"""
Feature engineering module.
Creates ML-ready features from the Silver layer data.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns to numbers.
    ML models need numbers — not text.

    Args:
        df: Clean HR DataFrame

    Returns:
        DataFrame with encoded categorical columns
    """
    df = df.copy()

    # Binary encoding — only two values
    binary_cols = ["Gender", "Over18"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
            logger.info(f"Binary encoded: {col}")

    # One-hot encoding — multiple categories
    # drop_first=True avoids multicollinearity
    ohe_cols = ["BusinessTravel", "Department",
                "EducationField", "JobRole", "MaritalStatus"]

    ohe_cols_present = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols_present, drop_first=True)
    logger.info(f"One-hot encoded: {ohe_cols_present}")

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    Separate features (X) from target (y).

    Args:
        df: Fully encoded DataFrame

    Returns:
        X: feature matrix
        y: target series (Attrition)
    """
    target = "Attrition"
    X = df.drop(columns=[target])
    y = df[target]

    logger.info(f"Features: {X.shape[1]} columns")
    logger.info(f"Target: {y.value_counts().to_dict()}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets.
    Stratify keeps the 16.1% attrition rate in both splits.

    Args:
        X: Feature matrix
        y: Target series
        test_size: Fraction for test (default 20%)
        random_state: For reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y    # Keep same attrition % in both sets
    )

    logger.info(f"Train: {len(X_train):,} rows")
    logger.info(f"Test:  {len(X_test):,} rows")
    logger.info(f"Train attrition: {y_train.mean():.1%}")
    logger.info(f"Test attrition:  {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.ingest import load_hr_data
    from src.data.clean import clean_hr_data

    # Load and clean data
    df = clean_hr_data(load_hr_data())
    print(f"Input shape: {df.shape}")

    # Encode
    df_encoded = encode_categorical(df)
    print(f"After encoding: {df_encoded.shape}")

    # Split features and target
    X, y = select_features(df_encoded)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"\n{'='*50}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'='*50}")
    print(f"Total features: {X.shape[1]}")
    print(f"Train samples:  {len(X_train):,}")
    print(f"Test samples:   {len(X_test):,}")
    print(f"Attrition rate: {y.mean():.1%}")