"""
tests/test_feature_engineering.py
==================================
Unit tests for src/feature_engineering.py.

Run with:  python -m pytest tests/ -v
"""
import pandas as pd
import pytest

from src.feature_engineering import build_tax_features, build_healthcare_features


# ── Minimal fixtures ──────────────────────────────────────────────────────────

def _make_tax_row(**overrides) -> dict:
    base = {
        "tax_id": 1,
        "filing_year": 2024,
        "taxpayer_age": 40,
        "filing_status": "married_jointly",
        "state_code": "CA",
        "days_to_file": 90,
        "w2_income": 80_000,
        "self_employment_income": 0,
        "investment_income": 500,
        "reported_income": 80_500,
        "business_deductions": 2_000,
        "charitable_deductions": 500,
        "medical_deductions": 1_000,
        "total_deductions": 12_000,
        "claimed_dependents": 2,
        "refund_amount": 1_500,
        "round_number_refund": 0,
        "num_prior_audits": 0,
        "num_amendments": 0,
        "occupation_risk_score": 0.2,
        "is_fraud": 0,
    }
    base.update(overrides)
    return base


def _make_hc_row(**overrides) -> dict:
    base = {
        "claim_id": 1,
        "provider_id": "P001",
        "patient_age": 55,
        "patient_gender": "F",
        "provider_specialty": "General Practice",
        "provider_specialty_risk": 0.3,
        "provider_tenure_years": 10,
        "provider_claim_velocity": 12,
        "claim_amount": 2_500,
        "allowed_amount": 1_800,
        "diagnosis_code_count": 3,
        "procedure_code_count": 2,
        "procedure_risk_score": 0.3,
        "diagnosis_complexity": 0.4,
        "claim_hour": 10,
        "is_weekend": 0,
        "days_since_last_claim": 30,
        "num_prior_claims_ytd": 4,
        "duplicate_claim_flag": 0,
        "unbundling_score": 0.05,
        "upcoding_score": 0.1,
        "is_fraud": 0,
    }
    base.update(overrides)
    return base


# ── Tax feature tests ─────────────────────────────────────────────────────────

class TestBuildTaxFeatures:
    def _build(self, **overrides) -> pd.Series:
        df = pd.DataFrame([_make_tax_row(**overrides)])
        return build_tax_features(df).iloc[0]

    def test_deduction_to_income_ratio(self):
        row = self._build(total_deductions=10_000, reported_income=100_000)
        assert row["deduction_to_income_ratio"] == pytest.approx(0.1, abs=1e-4)

    def test_refund_to_income_ratio(self):
        row = self._build(refund_amount=5_000, reported_income=100_000)
        assert row["refund_to_income_ratio"] == pytest.approx(0.05, abs=1e-4)

    def test_income_gap_positive(self):
        # W-2 + SE > reported → gap > 0
        row = self._build(w2_income=100_000, self_employment_income=10_000, reported_income=80_000)
        assert row["income_gap"] == pytest.approx(30_000, abs=1)

    def test_income_gap_clipped_at_zero(self):
        # reported > W-2 + SE → no gap
        row = self._build(w2_income=50_000, self_employment_income=0, reported_income=80_000)
        assert row["income_gap"] == 0.0

    def test_high_charitable_deduction_flag_on(self):
        row = self._build(charitable_deductions=15_000, reported_income=100_000)
        assert row["high_charitable_deduction"] == 1

    def test_high_charitable_deduction_flag_off(self):
        row = self._build(charitable_deductions=5_000, reported_income=100_000)
        assert row["high_charitable_deduction"] == 0

    def test_late_filer_flag(self):
        assert self._build(days_to_file=250)["late_filer"] == 1
        assert self._build(days_to_file=200)["late_filer"] == 0

    def test_multiple_amendments_flag(self):
        assert self._build(num_amendments=2)["multiple_amendments"] == 1
        assert self._build(num_amendments=1)["multiple_amendments"] == 0

    def test_repeat_audit_history_flag(self):
        assert self._build(num_prior_audits=2)["repeat_audit_history"] == 1
        assert self._build(num_prior_audits=1)["repeat_audit_history"] == 0

    def test_filing_status_code_values(self):
        codes = {
            "single": 0,
            "married_jointly": 1,
            "married_separately": 2,
            "head_of_household": 3,
        }
        for status, expected in codes.items():
            assert self._build(filing_status=status)["filing_status_code"] == expected

    def test_filing_status_and_state_code_dropped(self):
        df = pd.DataFrame([_make_tax_row()])
        result = build_tax_features(df)
        assert "filing_status" not in result.columns
        assert "state_code" not in result.columns

    def test_zero_income_no_division_error(self):
        row = self._build(reported_income=0, total_deductions=1_000, refund_amount=500)
        # Clipped to 1, so ratios are large but finite
        assert pd.notna(row["deduction_to_income_ratio"])

    def test_output_row_count_preserved(self):
        df = pd.DataFrame([_make_tax_row() for _ in range(50)])
        assert len(build_tax_features(df)) == 50


# ── Healthcare feature tests ──────────────────────────────────────────────────

class TestBuildHealthcareFeatures:
    def _build(self, **overrides) -> pd.Series:
        df = pd.DataFrame([_make_hc_row(**overrides)])
        return build_healthcare_features(df).iloc[0]

    def test_amount_to_allowed_ratio(self):
        row = self._build(claim_amount=2_000, allowed_amount=1_000)
        assert row["amount_to_allowed_ratio"] == pytest.approx(2.0, abs=1e-4)

    def test_amount_per_procedure(self):
        row = self._build(claim_amount=3_000, procedure_code_count=3)
        assert row["amount_per_procedure"] == pytest.approx(1_000.0, abs=1e-2)

    def test_diagnosis_procedure_ratio(self):
        row = self._build(diagnosis_code_count=6, procedure_code_count=3)
        assert row["diagnosis_procedure_ratio"] == pytest.approx(2.0, abs=1e-4)

    def test_night_claim_flag_early_morning(self):
        assert self._build(claim_hour=3)["night_claim"] == 1

    def test_night_claim_flag_late_night(self):
        assert self._build(claim_hour=23)["night_claim"] == 1

    def test_night_claim_flag_daytime(self):
        assert self._build(claim_hour=14)["night_claim"] == 0

    def test_rapid_reclaim_flag(self):
        assert self._build(days_since_last_claim=2)["rapid_reclaim"] == 1
        assert self._build(days_since_last_claim=7)["rapid_reclaim"] == 0

    def test_high_velocity_provider_flag(self):
        assert self._build(provider_claim_velocity=20)["high_velocity_provider"] == 1
        assert self._build(provider_claim_velocity=15)["high_velocity_provider"] == 0

    def test_velocity_x_amount(self):
        row = self._build(provider_claim_velocity=10, claim_amount=500)
        assert row["velocity_x_amount"] == pytest.approx(5_000.0, abs=1e-2)

    def test_elderly_patient_flag(self):
        assert self._build(patient_age=70)["elderly_patient"] == 1
        assert self._build(patient_age=69)["elderly_patient"] == 0

    def test_gender_code_male(self):
        assert self._build(patient_gender="M")["gender_code"] == 1

    def test_gender_code_female(self):
        assert self._build(patient_gender="F")["gender_code"] == 0

    def test_patient_gender_and_specialty_dropped(self):
        df = pd.DataFrame([_make_hc_row()])
        result = build_healthcare_features(df)
        assert "patient_gender" not in result.columns
        assert "provider_specialty" not in result.columns

    def test_zero_allowed_amount_no_division_error(self):
        row = self._build(allowed_amount=0, claim_amount=1_000)
        assert pd.notna(row["amount_to_allowed_ratio"])

    def test_output_row_count_preserved(self):
        df = pd.DataFrame([_make_hc_row() for _ in range(50)])
        assert len(build_healthcare_features(df)) == 50
