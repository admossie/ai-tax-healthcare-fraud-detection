"""
train_model.py
==============
Trains two independent XGBoost fraud detection models:
  • Tax fraud model        → models/tax_fraud_model.joblib
  • Healthcare fraud model → models/healthcare_fraud_model.joblib

Training metrics are written to reports/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR   = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET     = "is_fraud"
ID_COLS    = {"tax_id", "claim_id", "provider_id", TARGET}


def _build_pipeline(scale_pos_weight: float = 10.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def _quick_metrics(y_true, y_proba) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "train_samples": int(len(y_true)),
        "fraud_rate":    round(float(y_true.mean()), 4),
        "roc_auc":       round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc":        round(float(average_precision_score(y_true, y_proba)), 4),
        "precision":     round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":        round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":            round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_model(train_path: Path, model_name: str) -> tuple[Path, Path]:
    """Train one model from *train_path* and persist it as *model_name*."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X, y = df[feature_cols], df[TARGET]

    fraud_rate = float(y.mean())
    spw = round((1 - fraud_rate) / max(fraud_rate, 1e-6), 1)
    pipeline = _build_pipeline(scale_pos_weight=spw)
    pipeline.fit(X, y)

    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = _quick_metrics(y, y_proba)

    model_path  = MODEL_DIR  / f"{model_name}.joblib"
    report_path = REPORT_DIR / f"{model_name}_train_metrics.json"
    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model_path, report_path


def main() -> None:
    for label, name in [("tax", "tax_fraud_model"), ("healthcare", "healthcare_fraud_model")]:
        train_csv = PROC_DIR / f"{label}_train.csv"
        print(f"Training {label} fraud model …")
        mp, rp = train_model(train_csv, name)
        print(f"  ✓ Model  → {mp.name}")
        print(f"  ✓ Report → {rp.name}")
    print("\nDone.")


if __name__ == "__main__":
	main()
