# src/data/pipeline.py
"""
Data pipeline orchestrator.
Runs the complete Bronze -> Silver transformation.
Ingest -> Validate -> Clean -> Save
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline() -> pd.DataFrame:
    """
    Run the complete data pipeline.

    Steps:
    1. Ingest raw CSV
    2. Validate data quality
    3. Clean and transform
    4. Save processed data

    Returns:
        Cleaned DataFrame (Silver layer)
    """
    from src.data.ingest import load_hr_data
    from src.data.validate import validate_hr_data
    from src.data.clean import clean_hr_data

    logger.info("=" * 55)
    logger.info("SMARTPAYROLL DATA PIPELINE STARTING")
    logger.info("=" * 55)

    # Step 1: Ingest
    logger.info("\n[STEP 1] Ingesting raw data...")
    df_raw = load_hr_data()

    # Step 2: Validate
    logger.info("\n[STEP 2] Validating data quality...")
    results = validate_hr_data(df_raw)

    if not results["passed"]:
        raise ValueError("Data validation failed! Check logs.")
    logger.info("Validation passed!")

    # Step 3: Clean
    logger.info("\n[STEP 3] Cleaning data...")
    df_clean = clean_hr_data(df_raw)

    # Step 4: Save
    logger.info("\n[STEP 4] Saving processed data...")
    output_path = Path("data/processed/hr_silver.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    logger.info(f"Saved to: {output_path}")

    logger.info("\n" + "=" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Input:  {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")
    logger.info(f"Output: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
    logger.info(f"Saved:  {output_path}")
    logger.info("=" * 55)

    return df_clean


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s"
    )
    run_pipeline()