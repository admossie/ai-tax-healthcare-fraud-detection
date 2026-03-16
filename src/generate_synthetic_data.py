"""
generate_synthetic_data.py
==========================
Generates two realistic synthetic datasets for training the AI fraud
detection model:

  • 50,000 tax filing records  → data/raw/tax_records.csv
  • 50,000 healthcare claims   → data/raw/healthcare_claims.csv

Fraud labels are derived from composite risk signals that mirror known
real-world fraud patterns in each domain (income under-reporting,
excessive deductions, claim upcoding, duplicate billing, etc.).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

TAX_FRAUD_RATE = 0.07          # ~7 % of tax filings are fraudulent
HEALTHCARE_FRAUD_RATE = 0.09   # ~9 % of healthcare claims are fraudulent

FILING_STATUSES = ["single", "married_jointly", "married_separately", "head_of_household"]
STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
SPECIALTIES = [
    "general_practice", "cardiology", "orthopedics", "neurology",
    "oncology", "dermatology", "psychiatry", "radiology", "emergency",
    "physical_therapy",
]
# Specialty fraud-risk weights (higher = riskier based on CMS data)
SPECIALTY_RISK = {
    "general_practice": 0.20, "cardiology": 0.55, "orthopedics": 0.60,
    "neurology": 0.50, "oncology": 0.65, "dermatology": 0.45,
    "psychiatry": 0.70, "radiology": 0.40, "emergency": 0.35,
    "physical_therapy": 0.75,
}


# ──────────────────────────────────────────────────────────────────────────────
# Tax records generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_tax_records(
    n_samples: int = 50_000,
    fraud_rate: float = TAX_FRAUD_RATE,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of ``n_samples`` synthetic tax filing records."""
    rng = np.random.default_rng(seed=random_state)

    # ── Demographics & filing metadata ──────────────────────────────────────
    taxpayer_age = rng.integers(18, 85, size=n_samples)
    filing_year = rng.integers(2020, 2026, size=n_samples)
    filing_status = rng.choice(FILING_STATUSES, size=n_samples,
                               p=[0.38, 0.44, 0.06, 0.12])
    state_code = rng.choice(STATES, size=n_samples)
    days_to_file = rng.integers(1, 290, size=n_samples)   # 1 Jan → mid-Oct

    # ── Income streams ───────────────────────────────────────────────────────
    w2_income = rng.lognormal(mean=10.8, sigma=0.65, size=n_samples).round(2)
    self_employment_income = (
        rng.lognormal(mean=9.2, sigma=1.10, size=n_samples)
        * rng.binomial(1, 0.28, size=n_samples)          # only 28 % have SE income
    ).round(2)
    investment_income = (
        rng.lognormal(mean=8.5, sigma=1.20, size=n_samples)
        * rng.binomial(1, 0.18, size=n_samples)
    ).round(2)

    # Reported income: occasionally under-reported (fraud signal)
    underreport_factor = np.where(
        rng.uniform(size=n_samples) < 0.12,
        rng.uniform(0.40, 0.85, size=n_samples),   # fraudsters under-report
        rng.uniform(0.95, 1.05, size=n_samples),   # normal measurement noise
    )
    reported_income = (
        (w2_income + self_employment_income + investment_income) * underreport_factor
    ).round(2)

    # ── Deductions ───────────────────────────────────────────────────────────
    income_base = np.maximum(reported_income, 1.0)
    business_deductions = (
        income_base * rng.uniform(0.01, 0.25, size=n_samples)
        * (1 + rng.binomial(1, 0.10, size=n_samples) * rng.uniform(0.5, 3.0, size=n_samples))
    ).round(2)
    charitable_deductions = (
        income_base * rng.uniform(0.0, 0.15, size=n_samples)
        * (1 + rng.binomial(1, 0.08, size=n_samples) * rng.uniform(1.0, 4.0, size=n_samples))
    ).round(2)
    medical_deductions = (
        income_base * rng.uniform(0.0, 0.10, size=n_samples)
    ).round(2)
    total_deductions = (business_deductions + charitable_deductions + medical_deductions).round(2)

    # ── Dependents & refund ──────────────────────────────────────────────────
    claimed_dependents = rng.integers(0, 7, size=n_samples)
    # Fraudulent returns tend to claim inflated refunds
    base_refund = np.maximum(
        (total_deductions * 0.22 + claimed_dependents * 1_500
         - reported_income * 0.05),
        0.0,
    )
    refund_inflate = np.where(
        rng.uniform(size=n_samples) < 0.09,
        rng.uniform(1.5, 4.0, size=n_samples),
        1.0,
    )
    refund_amount = (base_refund * refund_inflate).round(2)

    num_prior_audits = rng.integers(0, 4, size=n_samples)
    num_amendments = rng.integers(0, 3, size=n_samples)
    round_number_refund = (refund_amount % 500 < 10).astype(int)

    # ── Occupation risk score (proxy for cash-heavy / SE industries) ─────────
    occupation_risk_score = rng.beta(a=2, b=5, size=n_samples).round(4)

    # ── Fraud signal ─────────────────────────────────────────────────────────
    income_gap = np.maximum(w2_income + self_employment_income - reported_income, 0)

    raw_signal = (
        0.30 * (income_gap / (w2_income + 1))
        + 0.20 * (total_deductions / (reported_income + 1)).clip(0, 5) / 5
        + 0.15 * occupation_risk_score
        + 0.10 * (refund_amount / (reported_income + 1)).clip(0, 2) / 2
        + 0.08 * (num_prior_audits / 3.0)
        + 0.07 * (num_amendments / 2.0)
        + 0.05 * round_number_refund
        + 0.05 * (claimed_dependents / 6.0)
        + rng.normal(0, 0.04, size=n_samples)
    )
    threshold = np.quantile(raw_signal, 1 - fraud_rate)
    is_fraud = (raw_signal >= threshold).astype(int)

    return pd.DataFrame({
        "tax_id":                  np.arange(1, n_samples + 1),
        "filing_year":             filing_year,
        "taxpayer_age":            taxpayer_age,
        "filing_status":           filing_status,
        "state_code":              state_code,
        "days_to_file":            days_to_file,
        "w2_income":               w2_income,
        "self_employment_income":  self_employment_income,
        "investment_income":       investment_income,
        "reported_income":         reported_income,
        "business_deductions":     business_deductions,
        "charitable_deductions":   charitable_deductions,
        "medical_deductions":      medical_deductions,
        "total_deductions":        total_deductions,
        "claimed_dependents":      claimed_dependents,
        "refund_amount":           refund_amount,
        "round_number_refund":     round_number_refund,
        "num_prior_audits":        num_prior_audits,
        "num_amendments":          num_amendments,
        "occupation_risk_score":   occupation_risk_score,
        "is_fraud":                is_fraud,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Healthcare claims generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_healthcare_claims(
    n_samples: int = 50_000,
    fraud_rate: float = HEALTHCARE_FRAUD_RATE,
    random_state: int = 99,
) -> pd.DataFrame:
    """Return a DataFrame of ``n_samples`` synthetic healthcare claim records."""
    rng = np.random.default_rng(seed=random_state)

    # ── Patient demographics ─────────────────────────────────────────────────
    patient_age = rng.integers(0, 95, size=n_samples)
    patient_gender = rng.choice(["M", "F"], size=n_samples, p=[0.49, 0.51])

    # ── Provider attributes ──────────────────────────────────────────────────
    n_providers = 2_000
    provider_ids = rng.integers(1000, 1000 + n_providers, size=n_samples)
    specialty_arr = rng.choice(SPECIALTIES, size=n_samples)
    provider_specialty_risk = np.array([SPECIALTY_RISK[s] for s in specialty_arr]).round(4)
    provider_tenure_years = rng.integers(0, 40, size=n_samples)
    # High-velocity providers submit many claims per week
    provider_claim_velocity = rng.poisson(lam=12.0, size=n_samples)

    # ── Claim financials ─────────────────────────────────────────────────────
    claim_amount = rng.gamma(shape=3.0, scale=900.0, size=n_samples).round(2)
    # Allowed amount is generally lower; upcoding inflates claim_amount
    upcode_factor = np.where(
        rng.uniform(size=n_samples) < 0.11,
        rng.uniform(1.4, 3.5, size=n_samples),   # upcoded
        rng.uniform(0.85, 1.05, size=n_samples),  # normal
    )
    claim_amount = (claim_amount * upcode_factor).round(2)
    allowed_amount = (claim_amount * rng.uniform(0.55, 0.95, size=n_samples)).round(2)

    # ── Diagnosis & procedure complexity ────────────────────────────────────
    diagnosis_code_count = rng.integers(1, 12, size=n_samples)
    procedure_code_count = rng.integers(1, 8, size=n_samples)
    procedure_risk_score = rng.beta(a=2, b=4, size=n_samples).round(4)
    diagnosis_complexity = rng.beta(a=2, b=3, size=n_samples).round(4)

    # ── Temporal features ────────────────────────────────────────────────────
    claim_hour = rng.integers(0, 24, size=n_samples)
    is_weekend = rng.binomial(n=1, p=0.26, size=n_samples)
    days_since_last_claim = rng.integers(0, 180, size=n_samples)
    num_prior_claims_ytd = rng.poisson(lam=5.0, size=n_samples)

    # ── Fraud-indicator features ─────────────────────────────────────────────
    duplicate_claim_flag = rng.binomial(n=1, p=0.05, size=n_samples)
    unbundling_score = rng.beta(a=1.5, b=5, size=n_samples).round(4)
    upcoding_score = ((upcode_factor - 1.0) / 2.5).clip(0, 1).round(4)

    # ── Fraud signal ─────────────────────────────────────────────────────────
    raw_signal = (
        0.25 * upcoding_score
        + 0.20 * provider_specialty_risk
        + 0.15 * (claim_amount / np.percentile(claim_amount, 90)).clip(0, 4) / 4
        + 0.10 * duplicate_claim_flag
        + 0.10 * unbundling_score
        + 0.08 * (provider_claim_velocity / np.percentile(provider_claim_velocity, 90)).clip(0, 3) / 3
        + 0.05 * (claim_hour <= 5).astype(float)
        + 0.04 * (days_since_last_claim <= 3).astype(float)
        + 0.03 * (num_prior_claims_ytd >= 10).astype(float)
        + rng.normal(0, 0.04, size=n_samples)
    )
    threshold = np.quantile(raw_signal, 1 - fraud_rate)
    is_fraud = (raw_signal >= threshold).astype(int)

    return pd.DataFrame({
        "claim_id":                np.arange(1, n_samples + 1),
        "patient_age":             patient_age,
        "patient_gender":          patient_gender,
        "provider_id":             provider_ids,
        "provider_specialty":      specialty_arr,
        "provider_specialty_risk": provider_specialty_risk,
        "provider_tenure_years":   provider_tenure_years,
        "provider_claim_velocity": provider_claim_velocity,
        "claim_amount":            claim_amount,
        "allowed_amount":          allowed_amount,
        "diagnosis_code_count":    diagnosis_code_count,
        "procedure_code_count":    procedure_code_count,
        "procedure_risk_score":    procedure_risk_score,
        "diagnosis_complexity":    diagnosis_complexity,
        "claim_hour":              claim_hour,
        "is_weekend":              is_weekend,
        "days_since_last_claim":   days_since_last_claim,
        "num_prior_claims_ytd":    num_prior_claims_ytd,
        "duplicate_claim_flag":    duplicate_claim_flag,
        "unbundling_score":        unbundling_score,
        "upcoding_score":          upcoding_score,
        "is_fraud":                is_fraud,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_tax_records(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    destination = output_path or RAW_DIR / "tax_records.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def save_healthcare_claims(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    destination = output_path or RAW_DIR / "healthcare_claims.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating 50,000 tax records …")
    tax_df = generate_tax_records(n_samples=50_000)
    tax_path = save_tax_records(tax_df)
    fraud_count = tax_df["is_fraud"].sum()
    print(f"  ✓ Saved {len(tax_df):,} records ({fraud_count:,} fraud, "
          f"{fraud_count / len(tax_df):.1%}) → {tax_path}")

    print("Generating 50,000 healthcare claims …")
    hc_df = generate_healthcare_claims(n_samples=50_000)
    hc_path = save_healthcare_claims(hc_df)
    fraud_count = hc_df["is_fraud"].sum()
    print(f"  ✓ Saved {len(hc_df):,} records ({fraud_count:,} fraud, "
          f"{fraud_count / len(hc_df):.1%}) → {hc_path}")

    print("\nDone. Raw datasets ready for feature engineering.")


if __name__ == "__main__":
    main()
