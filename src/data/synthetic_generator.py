# src/data/synthetic_generator.py
"""
Synthetic payroll data generator.
Creates realistic monthly payroll records for all 1,470 employees.
Injects known anomalies for testing the anomaly detection system.

Why synthetic data?
Real payroll data is confidential.
Synthetic data lets us build and test with known ground truth.
We KNOW which records are anomalies — so we can measure accuracy.

Anomaly types injected (~2% of records):
1. ZERO_NET_PAY: net pay is zero (processing error)
2. NET_EXCEEDS_GROSS: net > gross (calculation error)
3. HIGH_TAX_RATE: tax rate > statutory maximum
4. MISSING_PENSION: pension deduction not applied
5. LARGE_DEVIATION: >25% change from previous month
"""

import random
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)
fake = Faker("es_ES")  # Spanish locale for realistic names

# Country tax rates (simplified)
COUNTRY_TAX_RATES = {
    "ES": {
        "income_tax": 0.24,
        "social_security": 0.0635,
        "pension": 0.04
    },
    "BE": {
        "income_tax": 0.32,
        "social_security": 0.1307,
        "pension": 0.035
    },
    "DE": {
        "income_tax": 0.28,
        "social_security": 0.0945,
        "pension": 0.0935
    },
}

# Department to country mapping
DEPT_COUNTRIES = {
    "Sales": ["ES", "BE"],
    "Research & Development": ["DE", "BE"],
    "Human Resources": ["ES", "DE"],
}

# Anomaly types
ANOMALY_TYPES = [
    "ZERO_NET_PAY",
    "NET_EXCEEDS_GROSS",
    "HIGH_TAX_RATE",
    "MISSING_PENSION",
    "LARGE_DEVIATION",
]


def generate_clean_record(
    employee_id: int,
    employee_name: str,
    department: str,
    country: str,
    monthly_income: float,
    pay_period: str,
    noise: float = 0.02,
) -> dict:
    """
    Generate a clean (non-anomaly) payroll record.

    Args:
        employee_id: Employee identifier
        employee_name: Full name
        department: Department name
        country: Country code (ES/BE/DE)
        monthly_income: Base monthly salary
        pay_period: YYYY-MM format
        noise: Random variation (2% by default)

    Returns:
        dict representing one payroll record
    """
    rates = COUNTRY_TAX_RATES.get(country, COUNTRY_TAX_RATES["ES"])

    # Add small random variation to gross
    gross = monthly_income * (1 + np.random.normal(0, noise))
    gross = round(max(gross, 1000), 2)

    income_tax = round(gross * rates["income_tax"], 2)
    social_security = round(gross * rates["social_security"], 2)
    pension = round(gross * rates["pension"], 2)
    other = round(gross * np.random.uniform(0, 0.01), 2)
    net = round(gross - income_tax - social_security - pension - other, 2)

    return {
        "record_id": f"PR-{employee_id:04d}-{pay_period}",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "department": department,
        "country": country,
        "pay_period": pay_period,
        "gross_pay": gross,
        "income_tax": income_tax,
        "social_security": social_security,
        "pension": pension,
        "other_deductions": other,
        "net_pay": net,
        "is_anomaly": False,
        "anomaly_type": "",
    }


def inject_anomaly(record: dict, anomaly_type: str) -> dict:
    """
    Inject a specific anomaly into a clean record.
    Returns the modified record with anomaly flags.

    Args:
        record: Clean payroll record
        anomaly_type: Type of anomaly to inject

    Returns:
        Modified record with is_anomaly=True
    """
    record = record.copy()
    record["is_anomaly"] = True
    record["anomaly_type"] = anomaly_type

    if anomaly_type == "ZERO_NET_PAY":
        record["net_pay"] = 0.0

    elif anomaly_type == "NET_EXCEEDS_GROSS":
        record["net_pay"] = round(
            record["gross_pay"] * random.uniform(1.05, 1.20), 2
        )

    elif anomaly_type == "HIGH_TAX_RATE":
        record["income_tax"] = round(
            record["gross_pay"] * random.uniform(0.55, 0.70), 2
        )
        record["net_pay"] = round(
            record["gross_pay"]
            - record["income_tax"]
            - record["social_security"]
            - record["pension"]
            - record["other_deductions"],
            2
        )

    elif anomaly_type == "MISSING_PENSION":
        record["net_pay"] = round(
            record["net_pay"] + record["pension"], 2
        )
        record["pension"] = 0.0

    elif anomaly_type == "LARGE_DEVIATION":
        direction = random.choice(["up", "down"])
        factor = random.uniform(1.30, 1.60) if direction == "up" \
            else random.uniform(0.40, 0.70)
        record["gross_pay"] = round(record["gross_pay"] * factor, 2)
        deduction_rate = 0.40  # Approximate total deduction rate
        record["net_pay"] = round(
            record["gross_pay"] * (1 - deduction_rate), 2
        )

    return record


def generate_payroll_dataset(
    hr_df: pd.DataFrame,
    months: int = 6,
    anomaly_rate: float = 0.02,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate complete payroll dataset for all employees.

    Args:
        hr_df: IBM HR Silver layer DataFrame
        months: Number of months to generate
        anomaly_rate: Fraction of records to make anomalous
        random_seed: For reproducibility

    Returns:
        DataFrame with all payroll records + anomaly labels
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Generate pay periods (last N months)
    today = date.today()
    pay_periods = []
    for i in range(months - 1, -1, -1):
        d = today.replace(day=1) - timedelta(days=i * 30)
        pay_periods.append(d.strftime("%Y-%m"))

    # Determine which records get anomalies
    total_records = len(hr_df) * months
    anomaly_count = int(total_records * anomaly_rate)

    all_pairs = [
        (i, p)
        for i in range(len(hr_df))
        for p in pay_periods
    ]
    anomaly_pairs = set(
        map(
            tuple,
            random.sample(
                all_pairs,
                min(anomaly_count, len(all_pairs))
            )
        )
    )

    logger.info(
        f"Generating {total_records:,} records "
        f"({anomaly_count} anomalies = {anomaly_rate:.0%})"
    )

    all_records = []

    for idx, row in hr_df.iterrows():
        employee_id = idx + 1
        employee_name = fake.name()
        department = str(row.get("Department", "Sales"))
        country = random.choice(
            DEPT_COUNTRIES.get(department, ["ES"])
        )
        monthly_income = float(row.get("MonthlyIncome", 5000))

        for pay_period in pay_periods:
            record = generate_clean_record(
                employee_id=employee_id,
                employee_name=employee_name,
                department=department,
                country=country,
                monthly_income=monthly_income,
                pay_period=pay_period,
            )

            # Inject anomaly if selected
            if (idx, pay_period) in anomaly_pairs:
                anomaly_type = random.choice(ANOMALY_TYPES)
                record = inject_anomaly(record, anomaly_type)

            all_records.append(record)

    df = pd.DataFrame(all_records)

    # Summary
    anomaly_summary = df[df["is_anomaly"]]["anomaly_type"].value_counts()
    logger.info(f"Generation complete: {len(df):,} records")
    logger.info(f"Anomaly breakdown:\n{anomaly_summary.to_string()}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Loading HR Silver data...")
    hr_df = pd.read_parquet("data/processed/hr_silver.parquet")
    print(f"Employees: {len(hr_df):,}")

    print("\nGenerating synthetic payroll data...")
    payroll_df = generate_payroll_dataset(
        hr_df,
        months=6,
        anomaly_rate=0.02
    )

    # Save
    output_path = Path("data/synthetic/payroll.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payroll_df.to_parquet(output_path, index=False)

    print(f"\n{'='*55}")
    print("SYNTHETIC PAYROLL DATA GENERATED")
    print("=" * 55)
    print(f"Total records:  {len(payroll_df):,}")
    print(f"Employees:      {payroll_df['employee_id'].nunique():,}")
    print(f"Pay periods:    {payroll_df['pay_period'].nunique()}")
    print(f"Anomalies:      {payroll_df['is_anomaly'].sum():,} "
          f"({payroll_df['is_anomaly'].mean():.1%})")
    print(f"\nAnomaly types:")
    print(
        payroll_df[payroll_df["is_anomaly"]]["anomaly_type"]
        .value_counts()
        .to_string()
    )
    print(f"\nSample record:")
    print(payroll_df.head(1).to_string())
    print(f"\nSaved to: {output_path}")