# src/models/anomaly/detect.py
"""
Payroll anomaly detection — two-layer approach.

LAYER 1: Deterministic rules (fast, no ML cost)
  - Zero or negative net pay
  - Net pay exceeds gross pay
  - Tax rate outside statutory band
  - Missing pension deduction
  - Excessive total deductions

LAYER 2: Isolation Forest (catches subtle patterns)
  - Statistical outlier detection
  - Trained on clean records
  - Flags records that look unusual vs the population

WHY TWO LAYERS?
  Rules catch 60-70% of anomalies at zero cost.
  Isolation Forest catches the remaining subtle ones.
  Combined: better precision and recall than either alone.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Statutory tax bands per country (min, max)
TAX_BANDS = {
    "ES": (0.10, 0.47),
    "BE": (0.25, 0.50),
    "DE": (0.14, 0.45),
    "NL": (0.19, 0.495),
    "FR": (0.11, 0.45),
}


@dataclass
class AnomalyFlag:
    """A single anomaly flag from any detection layer."""
    rule: str
    severity: Literal["CRITICAL", "WARNING"]
    detail: str
    detected_by: Literal["RULE", "MODEL"]


def apply_rules(record: pd.Series) -> list[AnomalyFlag]:
    """
    Apply deterministic business rules to one payroll record.
    These rules are always applied — zero ML inference cost.

    Args:
        record: Single payroll record as pandas Series

    Returns:
        List of AnomalyFlag objects (empty if clean)
    """
    flags = []

    gross = float(record.get("gross_pay", 0))
    net = float(record.get("net_pay", 0))
    tax = float(record.get("income_tax", 0))
    pension = float(record.get("pension", 0))
    country = str(record.get("country", "ES"))

    # CRITICAL: Zero or negative net pay
    if net <= 0:
        flags.append(AnomalyFlag(
            rule="ZERO_OR_NEGATIVE_NET_PAY",
            severity="CRITICAL",
            detail=f"Net pay is {net:.2f} — must be positive",
            detected_by="RULE",
        ))

    # CRITICAL: Net pay exceeds gross
    if gross > 0 and net > gross:
        flags.append(AnomalyFlag(
            rule="NET_EXCEEDS_GROSS",
            severity="CRITICAL",
            detail=f"Net ({net:.2f}) exceeds gross ({gross:.2f})",
            detected_by="RULE",
        ))

    # CRITICAL: Missing pension on large salary
    if pension == 0 and gross > 1500:
        flags.append(AnomalyFlag(
            rule="MISSING_PENSION",
            severity="CRITICAL",
            detail=f"No pension deduction on gross of {gross:.2f}",
            detected_by="RULE",
        ))

    # WARNING/CRITICAL: Tax rate outside statutory band
    if gross > 0 and tax > 0:
        tax_rate = tax / gross
        band = TAX_BANDS.get(country, (0.10, 0.55))

        if tax_rate > band[1]:
            flags.append(AnomalyFlag(
                rule="TAX_RATE_TOO_HIGH",
                severity="CRITICAL",
                detail=(
                    f"Tax rate {tax_rate:.1%} exceeds "
                    f"statutory max {band[1]:.0%} for {country}"
                ),
                detected_by="RULE",
            ))
        elif tax_rate < band[0]:
            flags.append(AnomalyFlag(
                rule="TAX_RATE_TOO_LOW",
                severity="WARNING",
                detail=(
                    f"Tax rate {tax_rate:.1%} below "
                    f"statutory min {band[0]:.0%} for {country}"
                ),
                detected_by="RULE",
            ))

    # WARNING: Excessive total deductions
    if gross > 0:
        deduction_rate = 1 - (net / gross)
        if deduction_rate > 0.80:
            flags.append(AnomalyFlag(
                rule="EXCESSIVE_DEDUCTIONS",
                severity="WARNING",
                detail=f"Total deductions are {deduction_rate:.0%} of gross",
                detected_by="RULE",
            ))

    return flags


def train_isolation_forest(
    payroll_df: pd.DataFrame,
    contamination: float = 0.02,
    model_path: str = "data/processed/isolation_forest.joblib",
) -> IsolationForest:
    """
    Train Isolation Forest on payroll features.

    Isolation Forest principle:
    Anomalies are easier to isolate than normal points.
    The model builds random trees and measures how many
    splits are needed to isolate each point.
    Fewer splits = more anomalous.

    Args:
        payroll_df: Full payroll DataFrame
        contamination: Expected fraction of anomalies
        model_path: Where to save the trained model

    Returns:
        Fitted IsolationForest model
    """
    feature_cols = [
        "gross_pay", "income_tax", "social_security",
        "pension", "net_pay", "other_deductions",
    ]

    # Add derived features
    df = payroll_df.copy()
    df["net_to_gross"] = df["net_pay"] / (df["gross_pay"] + 1e-6)
    df["tax_rate"] = df["income_tax"] / (df["gross_pay"] + 1e-6)
    df["deduction_rate"] = (
        df["income_tax"] + df["social_security"] + df["pension"]
    ) / (df["gross_pay"] + 1e-6)

    extended_cols = feature_cols + [
        "net_to_gross", "tax_rate", "deduction_rate"
    ]
    available = [c for c in extended_cols if c in df.columns]

    X = df[available].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Save model and scaler
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path.replace(".joblib", "_scaler.joblib"))
    joblib.dump(available, model_path.replace(".joblib", "_features.joblib"))

    logger.info(f"Isolation Forest trained and saved to {model_path}")

    # Evaluate if labels available
    if "is_anomaly" in df.columns:
        predictions = model.predict(X_scaled)
        y_true = df["is_anomaly"].astype(int).values
        y_pred = (predictions == -1).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        logger.info("Isolation Forest evaluation:")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall:    {recall:.3f}")
        logger.info(f"  True positives: {tp}/{y_true.sum()}")

    return model


def detect_anomalies(
    payroll_df: pd.DataFrame,
    model_path: str = "data/processed/isolation_forest.joblib",
) -> pd.DataFrame:
    """
    Run full two-layer anomaly detection on a payroll batch.

    Args:
        payroll_df: Payroll DataFrame to scan
        model_path: Path to fitted Isolation Forest model

    Returns:
        DataFrame with anomaly flags and severity
    """
    # Load model
    model = joblib.load(model_path)
    scaler = joblib.load(model_path.replace(".joblib", "_scaler.joblib"))
    feature_cols = joblib.load(
        model_path.replace(".joblib", "_features.joblib")
    )

    results = []

    for _, record in payroll_df.iterrows():
        # Layer 1: Rules
        flags = apply_rules(record)

        # Layer 2: Isolation Forest
        df_record = record.to_frame().T.copy()
        df_record["net_to_gross"] = (
            df_record["net_pay"] / (df_record["gross_pay"] + 1e-6)
        )
        df_record["tax_rate"] = (
            df_record["income_tax"] / (df_record["gross_pay"] + 1e-6)
        )
        df_record["deduction_rate"] = (
            df_record["income_tax"]
            + df_record["social_security"]
            + df_record["pension"]
        ) / (df_record["gross_pay"] + 1e-6)

        available = [c for c in feature_cols if c in df_record.columns]
        X = df_record[available].fillna(0).values
        X_scaled = scaler.transform(X)

        iso_score = float(model.decision_function(X_scaled)[0])
        iso_pred = model.predict(X_scaled)[0]

        # Add model flag if no rule caught it
        if iso_pred == -1 and not flags:
            flags.append(AnomalyFlag(
                rule="STATISTICAL_OUTLIER",
                severity="WARNING",
                detail=f"Isolation Forest score: {iso_score:.3f}",
                detected_by="MODEL",
            ))

        # Overall severity
        if any(f.severity == "CRITICAL" for f in flags):
            severity = "CRITICAL"
        elif flags:
            severity = "WARNING"
        else:
            severity = "CLEAN"

        results.append({
            "record_id": record.get("record_id", ""),
            "employee_id": record.get("employee_id", 0),
            "pay_period": record.get("pay_period", ""),
            "predicted_severity": severity,
            "flag_count": len(flags),
            "flags": " | ".join(f.rule for f in flags),
            "isolation_score": iso_score,
            "requires_review": severity != "CLEAN",
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Loading synthetic payroll data...")
    payroll_df = pd.read_parquet("data/synthetic/payroll.parquet")
    print(f"Total records: {len(payroll_df):,}")

    print("\nTraining Isolation Forest...")
    model = train_isolation_forest(payroll_df, contamination=0.02)

    print("\nRunning anomaly detection on latest month...")
    latest = payroll_df["pay_period"].max()
    latest_df = payroll_df[payroll_df["pay_period"] == latest].copy()
    print(f"Records in {latest}: {len(latest_df)}")

    results = detect_anomalies(latest_df)

    print(f"\n{'='*55}")
    print("ANOMALY DETECTION RESULTS")
    print("=" * 55)
    print(results["predicted_severity"].value_counts().to_string())

    # Evaluate against ground truth
    merged = latest_df.merge(results, on="record_id")
    actual = merged["is_anomaly"].sum()
    detected = merged["requires_review"].sum()
    tp = (
        merged["is_anomaly"] &
        merged["requires_review"]
    ).sum()

    print(f"\nActual anomalies:   {actual}")
    print(f"Flagged for review: {detected}")
    print(f"True positives:     {tp}")
    if actual > 0:
        print(f"Recall:             {tp/actual:.1%}")
    if detected > 0:
        print(f"Precision:          {tp/detected:.1%}")