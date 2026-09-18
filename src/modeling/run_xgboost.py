"""Runner script for Phase 6: XGBoost tuning, training, evaluation, and serialization."""

import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

from src.modeling.baselines import (
    FEATURE_COLS,
    RepoPriorClassifier,
    RecentRateHeuristic,
    build_logistic_regression_pipeline,
    find_optimal_threshold_oof,
    evaluate_model_performance,
    paired_bootstrap_difference,
)
from src.modeling.run_baselines import compute_step_0_diagnostics
from src.modeling.split import per_repo_time_split, time_aware_split
from src.modeling.xgboost_model import (
    find_optimal_threshold_xgb_oof,
    get_feature_importances,
    train_final_model,
    tune_xgboost,
)


DATA_PATH = Path("data/processed/combined_features.parquet")
REPORTS_DIR = Path("reports")
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
MODEL_PKL_PATH = Path("src/modeling/model.pkl")
MODEL_META_PATH = Path("src/modeling/model_metadata.json")


def main() -> None:
    """Execute hyperparameter tuning, model training, evaluation, and artifact saving."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}. Run feature pipeline first.")

    print("=" * 90)
    print("PHASE 6: XGBOOST TRAINING, PER-REPO CV TUNING & EVALUATION")
    print("=" * 90)

    print(f"\n1. Loading feature dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], utc=True)

    # -------------------------------------------------------------------------
    # Split Protocols
    # -------------------------------------------------------------------------
    print("\n2. Splitting data into train/test under PRIMARY (per-repo) & SECONDARY (global) protocols...")
    primary_train, primary_test = per_repo_time_split(df, test_frac=0.2)
    primary_diag = compute_step_0_diagnostics(df, primary_train, primary_test, protocol_name="per_repo")

    global_train, global_test = time_aware_split(df, test_frac=0.2)
    global_diag = compute_step_0_diagnostics(df, global_train, global_test, protocol_name="global")

    # -------------------------------------------------------------------------
    # Hyperparameter Tuning via per_repo_time_series_cv on Primary Train Set
    # -------------------------------------------------------------------------
    print("\n3. Tuning XGBoost hyperparameters via per_repo_time_series_cv (5 folds, 30 iterations)...")
    best_params, best_cv_score, tuning_history = tune_xgboost(
        primary_train,
        n_splits=5,
        n_iter=30,
        random_state=42,
    )

    print("\n" + "-" * 70)
    print("BEST HYPERPARAMETERS FOUND (Optimizing Mean Validation PR-AUC):")
    print("-" * 70)
    for param_name, param_val in best_params.items():
        print(f"  • {param_name:<20}: {param_val}")
    print(f"  • Mean CV PR-AUC      : {best_cv_score:.4f}")
    print("-" * 70)

    # -------------------------------------------------------------------------
    # Out-of-Fold Threshold Tuning on Primary Train Set
    # -------------------------------------------------------------------------
    print("\n4. Tuning operating threshold for XGBoost via OOF per-repo CV on TRAIN...")
    xgb_thresh, xgb_oof_f1, xgb_cv_diags = find_optimal_threshold_xgb_oof(
        train_df=primary_train,
        best_params=best_params,
        n_splits=5,
        random_state=42,
    )
    print(f"  • Chosen Operating Threshold: {xgb_thresh:.2f} (OOF F1: {xgb_oof_f1:.4f})")
    print("  • Validation fold breakdown:")
    for d in xgb_cv_diags:
        flag_str = " [FLAG: <15 positives!]" if d["flagged_low_positives"] else ""
        print(f"    Fold {d['fold']}: val_rows={d['val_rows']}, positives={d['val_positives']} ({d['val_positive_pct']:.1f}%){flag_str}")

    # -------------------------------------------------------------------------
    # Refit Final Model on Full Primary Train Set
    # -------------------------------------------------------------------------
    print("\n5. Refitting final XGBoost model on full primary training set...")
    final_xgb = train_final_model(primary_train, best_params, random_state=42)

    # -------------------------------------------------------------------------
    # Fit Logistic Regression Baseline on Full Primary Train Set
    # -------------------------------------------------------------------------
    print("6. Fitting Logistic Regression baseline on full primary training set for paired comparison...")
    lr_thresh, lr_oof_f1, lr_cv_diags = find_optimal_threshold_oof(
        model_factory=lambda: build_logistic_regression_pipeline(random_state=42),
        train_df=primary_train,
        n_splits=5,
    )
    lr_pipeline = build_logistic_regression_pipeline(random_state=42)
    lr_pipeline.fit(primary_train[FEATURE_COLS], primary_train["label"])

    # Also fit Repo-prior baseline on primary train
    repo_prior_model = RepoPriorClassifier()
    repo_prior_model.fit(primary_train["repo"], primary_train["label"])
    recent_rate_model = RecentRateHeuristic()

    # -------------------------------------------------------------------------
    # Evaluate on Primary Protocol Test Set
    # -------------------------------------------------------------------------
    print("\n7. Evaluating models on PRIMARY test set (per-repo split)...")
    y_test_primary = primary_test["label"].values
    repos_primary = primary_test["repo"]

    xgb_scores_primary = final_xgb.predict_proba(primary_test[FEATURE_COLS])[:, 1]
    lr_scores_primary = lr_pipeline.predict_proba(primary_test[FEATURE_COLS])[:, 1]
    repo_scores_primary = repo_prior_model.predict_proba(repos_primary)[:, 1]
    recent_scores_primary = recent_rate_model.predict_proba(primary_test)[:, 1]

    xgb_metrics_primary = evaluate_model_performance(y_test_primary, xgb_scores_primary, threshold=xgb_thresh, repos=repos_primary)
    lr_metrics_primary = evaluate_model_performance(y_test_primary, lr_scores_primary, threshold=lr_thresh, repos=repos_primary)

    # Paired bootstrap: XGBoost vs LR, Repo-prior, Recent-rate
    print("   Running paired bootstrap comparisons (1000 resamples, seed 42)...")
    paired_xgb_vs_lr_primary = paired_bootstrap_difference(y_test_primary, xgb_scores_primary, lr_scores_primary, n_bootstraps=1000, random_state=42)
    paired_xgb_vs_repo_primary = paired_bootstrap_difference(y_test_primary, xgb_scores_primary, repo_scores_primary, n_bootstraps=1000, random_state=42)
    paired_xgb_vs_recent_primary = paired_bootstrap_difference(y_test_primary, xgb_scores_primary, recent_scores_primary, n_bootstraps=1000, random_state=42)

    # Feature importances
    feature_importances = get_feature_importances(final_xgb)

    # -------------------------------------------------------------------------
    # Evaluate on Secondary Protocol Test Set (Global Split)
    # -------------------------------------------------------------------------
    print("\n8. Evaluating the SAME models on SECONDARY test set (global split)...")
    y_test_global = global_test["label"].values
    repos_global = global_test["repo"]

    xgb_scores_global = final_xgb.predict_proba(global_test[FEATURE_COLS])[:, 1]
    lr_scores_global = lr_pipeline.predict_proba(global_test[FEATURE_COLS])[:, 1]
    repo_scores_global = repo_prior_model.predict_proba(repos_global)[:, 1]
    recent_scores_global = recent_rate_model.predict_proba(global_test)[:, 1]

    xgb_metrics_global = evaluate_model_performance(y_test_global, xgb_scores_global, threshold=xgb_thresh, repos=repos_global)
    lr_metrics_global = evaluate_model_performance(y_test_global, lr_scores_global, threshold=lr_thresh, repos=repos_global)

    paired_xgb_vs_lr_global = paired_bootstrap_difference(y_test_global, xgb_scores_global, lr_scores_global, n_bootstraps=1000, random_state=42)

    # -------------------------------------------------------------------------
    # Print Comparison Table (Primary Protocol)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("PRIMARY PROTOCOL (PER-REPO) TEST EVALUATION TABLE")
    print("=" * 95)
    tbl_hdr = f"{'Model':<24} | {'PR-AUC':<8} | {'95% CI (PR-AUC)':<18} | {'ROC-AUC':<8} | {'Thresh':<6} | {'F1':<6} | {'P@10%':<6} | {'R@10%':<6} | {'P@20%':<6} | {'R@20%':<6}"
    print(tbl_hdr)
    print("-" * len(tbl_hdr))

    for name, res in [("XGBoost (tuned)", xgb_metrics_primary), ("Logistic regression", lr_metrics_primary)]:
        prauc = f"{res['pr_auc']:.4f}"
        ci = f"[{res['pr_auc_ci_95'][0]:.4f}, {res['pr_auc_ci_95'][1]:.4f}]"
        roc = f"{res['roc_auc']:.4f}"
        th = f"{res.get('threshold', 0.0):.2f}"
        f1 = f"{res.get('f1', 0.0):.4f}"
        p10 = f"{res['top_k_percent']['top_10%']['precision']:.3f}"
        r10 = f"{res['top_k_percent']['top_10%']['recall']:.3f}"
        p20 = f"{res['top_k_percent']['top_20%']['precision']:.3f}"
        r20 = f"{res['top_k_percent']['top_20%']['recall']:.3f}"
        print(f"{name:<24} | {prauc:<8} | {ci:<18} | {roc:<8} | {th:<6} | {f1:<6} | {p10:<6} | {r10:<6} | {p20:<6} | {r20:<6}")
    print("-" * len(tbl_hdr))

    print("\nPAIRED BOOTSTRAP RESULT (XGBoost vs Logistic Regression on Primary Test Set):")
    print(f"  • Mean Difference (XGB - LR): {paired_xgb_vs_lr_primary['mean_difference']:+.4f}")
    print(f"  • 95% Bootstrap CI          : [{paired_xgb_vs_lr_primary['ci_lower']:+.4f}, {paired_xgb_vs_lr_primary['ci_upper']:+.4f}]")
    print(f"  • XGBoost Win Fraction       : {paired_xgb_vs_lr_primary['win_fraction']*100:.1f}% (across 1000 resamples)")

    print("\nPAIRED BOOTSTRAP RESULT (XGBoost vs Other Baselines on Primary Test Set):")
    print(f"  • XGBoost minus Repo-prior  : mean diff = {paired_xgb_vs_repo_primary['mean_difference']:+.4f} | 95% CI = [{paired_xgb_vs_repo_primary['ci_lower']:+.4f}, {paired_xgb_vs_repo_primary['ci_upper']:+.4f}] | win rate = {paired_xgb_vs_repo_primary['win_fraction']*100:.1f}%")
    print(f"  • XGBoost minus Recent-rate : mean diff = {paired_xgb_vs_recent_primary['mean_difference']:+.4f} | 95% CI = [{paired_xgb_vs_recent_primary['ci_lower']:+.4f}, {paired_xgb_vs_recent_primary['ci_upper']:+.4f}] | win rate = {paired_xgb_vs_recent_primary['win_fraction']*100:.1f}%")

    print("\nTOP 5 FEATURES BY IMPORTANCE (Gain):")
    print("-" * 50)
    for rank, item in enumerate(feature_importances[:5], start=1):
        print(f"  {rank}. {item['feature']:<28}: gain = {item['gain']:.4f}")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # Secondary Protocol Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("SECONDARY PROTOCOL (GLOBAL) TEST EVALUATION TABLE")
    print("=" * 95)
    print(tbl_hdr)
    print("-" * len(tbl_hdr))
    for name, res in [("XGBoost (tuned)", xgb_metrics_global), ("Logistic regression", lr_metrics_global)]:
        prauc = f"{res['pr_auc']:.4f}"
        ci = f"[{res['pr_auc_ci_95'][0]:.4f}, {res['pr_auc_ci_95'][1]:.4f}]"
        roc = f"{res['roc_auc']:.4f}"
        th = f"{res.get('threshold', 0.0):.2f}"
        f1 = f"{res.get('f1', 0.0):.4f}"
        p10 = f"{res['top_k_percent']['top_10%']['precision']:.3f}"
        r10 = f"{res['top_k_percent']['top_10%']['recall']:.3f}"
        p20 = f"{res['top_k_percent']['top_20%']['precision']:.3f}"
        r20 = f"{res['top_k_percent']['top_20%']['recall']:.3f}"
        print(f"{name:<24} | {prauc:<8} | {ci:<18} | {roc:<8} | {th:<6} | {f1:<6} | {p10:<6} | {r10:<6} | {p20:<6} | {r20:<6}")
    print("-" * len(tbl_hdr))
    print(f"  • Global XGB vs LR Mean Diff: {paired_xgb_vs_lr_global['mean_difference']:+.4f} | 95% CI: [{paired_xgb_vs_lr_global['ci_lower']:+.4f}, {paired_xgb_vs_lr_global['ci_upper']:+.4f}] | Win Rate: {paired_xgb_vs_lr_global['win_fraction']*100:.1f}%\n")

    # -------------------------------------------------------------------------
    # 9. Save Model Artifact to src/modeling/model.pkl
    # -------------------------------------------------------------------------
    print(f"9. Serializing final fitted model to {MODEL_PKL_PATH}...")
    model_artifact = {
        "model": final_xgb,
        "feature_names": FEATURE_COLS,
        "operating_threshold": float(xgb_thresh),
        "best_hyperparameters": best_params,
        "cv_pr_auc": float(best_cv_score),
    }
    joblib.dump(model_artifact, MODEL_PKL_PATH)

    metadata_artifact = {
        "feature_names": FEATURE_COLS,
        "operating_threshold": float(xgb_thresh),
        "best_hyperparameters": best_params,
        "cv_pr_auc": float(best_cv_score),
        "test_pr_auc_per_repo": float(xgb_metrics_primary["pr_auc"]),
        "test_roc_auc_per_repo": float(xgb_metrics_primary["roc_auc"]),
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_artifact, f, indent=2)

    # -------------------------------------------------------------------------
    # 10. Save Complete Metrics to reports/metrics.json
    # -------------------------------------------------------------------------
    print(f"10. Saving all evaluation metrics to {METRICS_JSON_PATH}...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "per_repo": {
            "diagnostics": primary_diag,
            "xgboost": {
                "best_hyperparameters": best_params,
                "cv_pr_auc": float(best_cv_score),
                "chosen_threshold": float(xgb_thresh),
                "oof_f1": float(xgb_oof_f1),
                "metrics": xgb_metrics_primary,
                "feature_importances": feature_importances,
            },
            "logistic_regression": {
                "chosen_threshold": float(lr_thresh),
                "oof_f1": float(lr_oof_f1),
                "metrics": lr_metrics_primary,
            },
            "paired_bootstrap_vs_lr": paired_xgb_vs_lr_primary,
            "paired_bootstrap_vs_repo_prior": paired_xgb_vs_repo_primary,
            "paired_bootstrap_vs_recent_rate": paired_xgb_vs_recent_primary,
        },
        "global": {
            "diagnostics": global_diag,
            "xgboost": {
                "metrics": xgb_metrics_global,
            },
            "logistic_regression": {
                "metrics": lr_metrics_global,
            },
            "paired_bootstrap_vs_lr": paired_xgb_vs_lr_global,
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\n[DONE] Phase 6 evaluation complete. Metrics saved to {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
