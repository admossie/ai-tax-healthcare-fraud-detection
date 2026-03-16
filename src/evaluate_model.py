"""
evaluate_model.py
=================
Evaluates both trained models against their respective held-out test sets
and writes full metric reports to reports/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT       = Path(__file__).resolve().parents[1]
PROC_DIR   = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET     = "is_fraud"
ID_COLS    = {"tax_id", "claim_id", "provider_id", TARGET}


def evaluate_model(model_name: str, test_csv: Path) -> Path:
    model_path  = MODEL_DIR  / f"{model_name}.joblib"
    report_path = REPORT_DIR / f"{model_name}_eval_metrics.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X_test, y_test = df[feature_cols], df[TARGET]

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "model":        model_name,
        "test_samples": int(len(df)),
        "fraud_rate":   round(float(y_test.mean()), 4),
        "roc_auc":      round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc":       round(float(average_precision_score(y_test, y_proba)), 4),
        "precision":    round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":       round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":           round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    pairs = [
        ("tax_fraud_model",         PROC_DIR / "tax_test.csv"),
        ("healthcare_fraud_model",   PROC_DIR / "healthcare_test.csv"),
    ]
    for model_name, test_csv in pairs:
        print(f"Evaluating {model_name} …")
        rp = evaluate_model(model_name, test_csv)
        print(f"  ✓ Report → {rp.name}")
    print("\nDone.")


if __name__ == "__main__":
	main()
