# src/data/ingest.py
"""
Data ingestion module.
Loads the IBM HR Analytics dataset from local storage.
In production: loads from Azure Blob Storage.
"""

import logging
from pathlib import Path

import pandas as pd

# Set up logging for this module
logger = logging.getLogger(__name__)


def load_hr_data(file_path: str = None) -> pd.DataFrame:
    """
    Load the IBM HR Analytics dataset.

    Args:
        file_path: Path to the CSV file.
                   Defaults to data/raw/ folder.

    Returns:
        pd.DataFrame with 1470 rows and 35 columns
    """
    # Default path if none provided
    if file_path is None:
        file_path = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    else:
        file_path = Path(file_path)

    # Check file exists before trying to read
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please download from Kaggle and place in data/raw/"
        )

    logger.info(f"Loading HR data from: {file_path}")

    # Read CSV into DataFrame
    df = pd.read_csv(file_path)

    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    # Quick test — run this file directly to verify it works
    logging.basicConfig(level=logging.INFO)

    df = load_hr_data()

    print(f"\n{'='*50}")
    print("IBM HR Analytics Dataset")
    print(f"{'='*50}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"{'='*50}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nAttrition breakdown:")
    print(df['Attrition'].value_counts())