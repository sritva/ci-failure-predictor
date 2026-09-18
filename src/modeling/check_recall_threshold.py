"""Diagnostic script: evaluate test-set performance at F1 (0.73) vs F2 (0.37) thresholds."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from src.modeling.split import per_repo_time_split


DATA_PATH = Path("data/processed/combined_features.parquet")
MODEL_PKL_PATH = Path("src/modeling/model.pkl")
METRICS_JSON_PATH = Path("reports/metrics.json")


def evaluate_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute precision, recall, F1, F2, and confusion matrix at a specific threshold."""
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

    return {
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "raw_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    """Load model artifact, compute test metrics at 0.73 and 0.37, and report side-by-side."""
    print("=" * 80)
    print("XGBOOST THRESHOLD SENSITIVITY CHECK: F1 (0.73) vs F2 (0.37)")
    print("=" * 80)

    # 1. Load model and features without retraining
    print(f"Loading model artifact from {MODEL_PKL_PATH}...")
    artifact = joblib.load(MODEL_PKL_PATH)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    prod_threshold = float(artifact["operating_threshold"])  # 0.73
    f2_threshold = 0.37

    # 2. Load dataset and reproduce primary test split
    print(f"Loading data from {DATA_PATH} and reproducing primary test split...")
    df = pd.read_parquet(DATA_PATH)
    _, test_df = per_repo_time_split(df, test_frac=0.2)

    y_test = test_df["label"].values
    test_scores = model.predict_proba(test_df[feature_names])[:, 1]
    n_test = len(y_test)
    n_failures = int(np.sum(y_test == 1))

    print(f"Test set: {n_test} rows, {n_failures} failures ({n_failures / n_test * 100:.1f}%)")

    # 3. Compute metrics for both thresholds
    res_073 = evaluate_threshold_metrics(y_test, test_scores, prod_threshold)
    res_037 = evaluate_threshold_metrics(y_test, test_scores, f2_threshold)

    # 4. Print side-by-side comparison table
    print("\n" + "=" * 80)
    print(f"{'Metric':<25} | {'Threshold 0.73 (F1-optimal)':<24} | {'Threshold 0.37 (F2-optimal)':<24}")
    print("=" * 80)
    print(f"{'Precision':<25} | {res_073['precision']*100:<23.1f}% | {res_037['precision']*100:<23.1f}%")
    print(f"{'Recall':<25} | {res_073['recall']*100:<23.1f}% | {res_037['recall']*100:<23.1f}%")
    print(f"{'F1 Score':<25} | {res_073['f1']:<24.4f} | {res_037['f1']:<24.4f}")
    print(f"{'F2 Score':<25} | {res_073['f2']:<24.4f} | {res_037['f2']:<24.4f}")
    print("-" * 80)
    print(f"{'True Positives (TP)':<25} | {res_073['confusion_matrix']['tp']:<24} | {res_037['confusion_matrix']['tp']:<24}")
    print(f"{'False Positives (FP)':<25} | {res_073['confusion_matrix']['fp']:<24} | {res_037['confusion_matrix']['fp']:<24}")
    print(f"{'True Negatives (TN)':<25} | {res_073['confusion_matrix']['tn']:<24} | {res_037['confusion_matrix']['tn']:<24}")
    print(f"{'False Negatives (FN)':<25} | {res_073['confusion_matrix']['fn']:<24} | {res_037['confusion_matrix']['fn']:<24}")
    print("-" * 80)
    print(f"{'Total Flagged for Review':<25} | {res_073['confusion_matrix']['tp'] + res_073['confusion_matrix']['fp']:<24} | {res_037['confusion_matrix']['tp'] + res_037['confusion_matrix']['fp']:<24}")
    print("=" * 80)

    # 5. Honest comparison vs Train OOF estimate
    oof_prec_037 = 0.352
    oof_rec_037 = 0.769
    delta_prec = res_037["precision"] - oof_prec_037
    delta_rec = res_037["recall"] - oof_rec_037

    print("\nHONEST COMPARISON (Train OOF Estimate vs Actual Test for Threshold 0.37):")
    print(
        f"Train OOF estimated 35.2% precision and 76.9% recall; on actual test it achieved "
        f"{res_037['precision']*100:.1f}% precision ({delta_prec*100:+.1f}%) and "
        f"{res_037['recall']*100:.1f}% recall ({delta_rec*100:+.1f}%)."
    )

    # 6. Append to reports/metrics.json without modifying or overwriting existing fields
    if METRICS_JSON_PATH.exists():
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        metrics_data["threshold_sensitivity_check"] = {
            "description": "Read-only evaluation of F1-optimal (0.73) vs F2-optimal (0.37) operating thresholds on primary test set.",
            "threshold_0_73_f1": res_073,
            "threshold_0_37_f2": res_037,
            "train_oof_vs_test_delta_at_0_37": {
                "train_oof_precision": oof_prec_037,
                "train_oof_recall": oof_rec_037,
                "test_precision": res_037["precision"],
                "test_recall": res_037["recall"],
                "precision_delta": delta_prec,
                "recall_delta": delta_rec,
            },
        }

        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nAppended 'threshold_sensitivity_check' section to {METRICS_JSON_PATH}.")


if __name__ == "__main__":
    main()
