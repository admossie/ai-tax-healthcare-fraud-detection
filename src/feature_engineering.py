"""
feature_engineering.py
======================
Builds domain-specific features for:
  • Tax records        (data/raw/tax_records.csv)
  • Healthcare claims  (data/raw/healthcare_claims.csv)

Outputs four CSVs into data/processed/:
  tax_train.csv, tax_test.csv,
  healthcare_train.csv, healthcare_test.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


# ── Tax features ──────────────────────────────────────────────────────────────

def load_tax_records(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RAW_DIR / "tax_records.csv")


def build_tax_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inc = out["reported_income"].clip(lower=1)
    out["deduction_to_income_ratio"]  = (out["total_deductions"] / inc).round(4)
    out["refund_to_income_ratio"]     = (out["refund_amount"] / inc).round(4)
    out["se_income_share"]            = (out["self_employment_income"] / inc).round(4)
    out["income_gap"]                 = (
        (out["w2_income"] + out["self_employment_income"] - out["reported_income"]).clip(lower=0)
    ).round(2)
    out["income_gap_ratio"]           = (out["income_gap"] / inc).round(4)
    out["high_charitable_deduction"]  = (out["charitable_deductions"] / inc > 0.10).astype(int)
    out["late_filer"]                 = (out["days_to_file"] > 240).astype(int)
    out["multiple_amendments"]        = (out["num_amendments"] >= 2).astype(int)
    out["repeat_audit_history"]       = (out["num_prior_audits"] >= 2).astype(int)
    # Encode filing_status ordinally
    status_map = {"single": 0, "married_jointly": 1,
                  "married_separately": 2, "head_of_household": 3}
    out["filing_status_code"] = out["filing_status"].map(status_map).fillna(0).astype(int)
    return out.drop(columns=["filing_status", "state_code"])


def split_tax(df: pd.DataFrame, test_size: float = 0.20,
              random_state: int = 42) -> tuple[Path, Path]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["is_fraud"]
    )
    train_path = PROC_DIR / "tax_train.csv"
    test_path  = PROC_DIR / "tax_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    return train_path, test_path


# ── Healthcare features ───────────────────────────────────────────────────────

def load_healthcare_claims(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RAW_DIR / "healthcare_claims.csv")


def build_healthcare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_to_allowed_ratio"]   = (out["claim_amount"] / out["allowed_amount"].clip(lower=1)).round(4)
    out["amount_per_procedure"]      = (out["claim_amount"] / out["procedure_code_count"].clip(lower=1)).round(2)
    out["diagnosis_procedure_ratio"] = (out["diagnosis_code_count"] / out["procedure_code_count"].clip(lower=1)).round(4)
    out["night_claim"]               = ((out["claim_hour"] <= 5) | (out["claim_hour"] >= 22)).astype(int)
    out["rapid_reclaim"]             = (out["days_since_last_claim"] <= 3).astype(int)
    out["high_velocity_provider"]    = (out["provider_claim_velocity"] >= 20).astype(int)
    out["velocity_x_amount"]         = (out["provider_claim_velocity"] * out["claim_amount"]).round(2)
    out["elderly_patient"]           = (out["patient_age"] >= 70).astype(int)
    # Encode gender
    out["gender_code"] = (out["patient_gender"] == "M").astype(int)
    return out.drop(columns=["patient_gender", "provider_specialty"])


def split_healthcare(df: pd.DataFrame, test_size: float = 0.20,
                     random_state: int = 42) -> tuple[Path, Path]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["is_fraud"]
    )
    train_path = PROC_DIR / "healthcare_train.csv"
    test_path  = PROC_DIR / "healthcare_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    return train_path, test_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Engineering tax features …")
    tax_df = build_tax_features(load_tax_records())
    tr, te = split_tax(tax_df)
    print(f"  ✓ {tr.name}  ({len(pd.read_csv(tr)):,} rows)")
    print(f"  ✓ {te.name}  ({len(pd.read_csv(te)):,} rows)")

    print("Engineering healthcare features …")
    hc_df = build_healthcare_features(load_healthcare_claims())
    tr, te = split_healthcare(hc_df)
    print(f"  ✓ {tr.name}  ({len(pd.read_csv(tr)):,} rows)")
    print(f"  ✓ {te.name}  ({len(pd.read_csv(te)):,} rows)")

    print("\nDone. Processed datasets ready for training.")


if __name__ == "__main__":
	main()
