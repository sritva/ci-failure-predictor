"""Runner script for Phase 5 baseline models under both per-repo and global split protocols."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.modeling.baselines import (
    FEATURE_COLS,
    RepoPriorClassifier,
    RecentRateHeuristic,
    build_decision_stump,
    fit_decision_stump,
    build_logistic_regression_pipeline,
    fit_dummy_baseline,
    find_optimal_threshold_oof,
    evaluate_model_performance,
    paired_bootstrap_difference,
)
from src.modeling.split import per_repo_time_split, time_aware_split


DATA_PATH = Path("data/processed/combined_features.parquet")
REPORTS_DIR = Path("reports")
OUTPUT_JSON_PATH = REPORTS_DIR / "baselines.json"


def compute_step_0_diagnostics(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol_name: str,
) -> Dict[str, Any]:
    """Compute and print comprehensive Step 0 diagnostics and flags for a split protocol."""
    total_n = len(df)
    train_n = len(train_df)
    test_n = len(test_df)

    train_failures = int((train_df["label"] == 1).sum()) if "label" in train_df.columns else 0
    test_failures = int((test_df["label"] == 1).sum()) if "label" in test_df.columns else 0

    train_fail_rate = float(train_failures / train_n) if train_n else 0.0
    test_fail_rate = float(test_failures / test_n) if test_n else 0.0

    print("=" * 80)
    print(f"STEP 0 DIAGNOSTICS - PROTOCOL: {protocol_name.upper()}")
    print("=" * 80)
    print(f"Total rows: {total_n}")
    print(f"  Train: {train_n} rows | Failures: {train_failures} ({train_fail_rate * 100:.2f}%)")
    print(f"  Test:  {test_n} rows | Failures: {test_failures} ({test_fail_rate * 100:.2f}%)")
    print("-" * 80)

    # Per-repo breakdown
    all_repos = sorted(df["repo"].unique().tolist())
    repo_stats: Dict[str, Dict[str, Any]] = {}
    flags: List[str] = []

    header = f"{'Repo':<28} | {'Train Rows':<10} {'Fails':<6} {'Rate%':<7} | {'Test Rows':<10} {'Fails':<6} {'Rate%':<7} | {'Train% Share':<12} {'Test% Share':<11} | {'Min Timestamp':<20} {'Max Timestamp':<20}"
    print(header)
    print("-" * len(header))

    for r in all_repos:
        r_df = df[df["repo"] == r]
        r_train = train_df[train_df["repo"] == r]
        r_test = test_df[test_df["repo"] == r]

        tr_count = len(r_train)
        tr_fails = int((r_train["label"] == 1).sum())
        tr_rate = float(tr_fails / tr_count) if tr_count else 0.0

        te_count = len(r_test)
        te_fails = int((r_test["label"] == 1).sum())
        te_rate = float(te_fails / te_count) if te_count else 0.0

        # Share of repo's own rows that landed in train
        repo_train_share_pct = float(tr_count / len(r_df) * 100) if len(r_df) else 0.0
        # Share of total test set made up by this repo
        test_set_share_pct = float(te_count / test_n * 100) if test_n else 0.0

        min_time = str(r_df["run_timestamp"].min())[:19]
        max_time = str(r_df["run_timestamp"].max())[:19]

        print(
            f"{r:<28} | {tr_count:<10} {tr_fails:<6} {tr_rate*100:<7.1f} | "
            f"{te_count:<10} {te_fails:<6} {te_rate*100:<7.1f} | "
            f"{repo_train_share_pct:<12.1f} {test_set_share_pct:<11.1f} | {min_time:<20} {max_time:<20}"
        )

        repo_stats[r] = {
            "total_rows": len(r_df),
            "train_rows": tr_count,
            "train_failures": tr_fails,
            "train_failure_rate": tr_rate,
            "test_rows": te_count,
            "test_failures": te_fails,
            "test_failure_rate": te_rate,
            "repo_train_share_pct": repo_train_share_pct,
            "test_set_share_pct": test_set_share_pct,
            "min_timestamp": min_time,
            "max_timestamp": max_time,
        }

        # Check flags
        if te_count == 0:
            flags.append(f"FLAG: Repo '{r}' has 0 test rows in {protocol_name} split.")
        if test_set_share_pct > 50.0:
            flags.append(f"FLAG: Repo '{r}' makes up >50% ({test_set_share_pct:.1f}%) of {protocol_name} test set.")
        if protocol_name == "global" and repo_train_share_pct < 20.0:
            flags.append(f"FLAG: Repo '{r}' has <20% ({repo_train_share_pct:.1f}%) of its rows in train set in global split.")

    print("-" * 80)
    if flags:
        print("FLAGS TRIPPED:")
        for flg in flags:
            print(f"  >>> {flg}")
    else:
        print("No diagnostic flags tripped.")
    print("=" * 80 + "\n")

    return {
        "protocol": protocol_name,
        "total_rows": total_n,
        "train_rows": train_n,
        "test_rows": test_n,
        "train_failures": train_failures,
        "test_failures": test_failures,
        "train_failure_rate": train_fail_rate,
        "test_failure_rate": test_fail_rate,
        "repos": repo_stats,
        "flags": flags,
    }


def run_protocol_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol_name: str,
) -> Dict[str, Any]:
    """Execute training, OOF threshold tuning, testing, and paired bootstrap for one protocol."""
    print(f"\n{'#' * 80}")
    print(f"# RUNNING MODEL PIPELINE: PROTOCOL = {protocol_name.upper()}")
    print(f"{'#' * 80}\n")

    y_train = train_df["label"]
    y_test = test_df["label"].values
    test_repos = test_df["repo"]

    # -------------------------------------------------------------------------
    # 1. Fit Dummy Classifier
    # -------------------------------------------------------------------------
    print("Fitting Model 1: Dummy (prior)...")
    dummy_model = fit_dummy_baseline(train_df, y_train)
    dummy_scores = dummy_model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    dummy_results = evaluate_model_performance(y_test, dummy_scores, threshold=None, repos=test_repos)

    # -------------------------------------------------------------------------
    # 2. Fit Repo-Prior Classifier
    # -------------------------------------------------------------------------
    print("Fitting Model 2: Repo-prior...")
    repo_prior_model = RepoPriorClassifier()
    repo_prior_model.fit(train_df["repo"], y_train)
    repo_prior_scores = repo_prior_model.predict_proba(test_df["repo"])[:, 1]
    repo_prior_results = evaluate_model_performance(y_test, repo_prior_scores, threshold=None, repos=test_repos)

    # -------------------------------------------------------------------------
    # 3. Recent-Rate Heuristic (No fitting)
    # -------------------------------------------------------------------------
    print("Evaluating Model 3: Recent-rate heuristic...")
    recent_rate_model = RecentRateHeuristic()
    recent_rate_scores = recent_rate_model.predict_proba(test_df)[:, 1]
    recent_rate_results = evaluate_model_performance(y_test, recent_rate_scores, threshold=None, repos=test_repos)

    # -------------------------------------------------------------------------
    # 4. Decision Stump: OOF Threshold Tuning + Refit
    # -------------------------------------------------------------------------
    print("\nTuning operating threshold for Model 4: Decision Stump via 5-fold TimeSeriesSplit on TRAIN...")
    stump_th, stump_oof_f1, stump_cv_diags = find_optimal_threshold_oof(
        model_factory=lambda: build_decision_stump(random_state=42),
        train_df=train_df,
        n_splits=5,
    )
    print(f"  Decision Stump OOF Best Threshold: {stump_th:.2f} (OOF F1: {stump_oof_f1:.4f})")
    print("  Validation fold breakdown:")
    for d in stump_cv_diags:
        flag_str = " [FLAG: <15 positives!]" if d["flagged_low_positives"] else ""
        print(f"    Fold {d['fold']}: val_rows={d['val_rows']}, positives={d['val_positives']} ({d['val_positive_pct']:.1f}%){flag_str}")

    # Refit on full train
    stump_model, stump_split = fit_decision_stump(train_df, y_train, random_state=42)
    print(f"  Decision Stump Chosen Split on Full Train: Feature='{stump_split['feature_name']}' <= {stump_split['threshold']:.4f}")
    stump_scores = stump_model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    stump_results = evaluate_model_performance(y_test, stump_scores, threshold=stump_th, repos=test_repos)
    stump_results["oof_cv_diagnostics"] = stump_cv_diags
    stump_results["chosen_split"] = stump_split

    # -------------------------------------------------------------------------
    # 5. Logistic Regression: OOF Threshold Tuning + Refit
    # -------------------------------------------------------------------------
    print("\nTuning operating threshold for Model 5: Logistic Regression Pipeline via 5-fold TimeSeriesSplit on TRAIN...")
    lr_th, lr_oof_f1, lr_cv_diags = find_optimal_threshold_oof(
        model_factory=lambda: build_logistic_regression_pipeline(random_state=42),
        train_df=train_df,
        n_splits=5,
    )
    print(f"  Logistic Regression OOF Best Threshold: {lr_th:.2f} (OOF F1: {lr_oof_f1:.4f})")
    print("  Validation fold breakdown:")
    for d in lr_cv_diags:
        flag_str = " [FLAG: <15 positives!]" if d["flagged_low_positives"] else ""
        print(f"    Fold {d['fold']}: val_rows={d['val_rows']}, positives={d['val_positives']} ({d['val_positive_pct']:.1f}%){flag_str}")

    # Refit on full train
    lr_pipeline = build_logistic_regression_pipeline(random_state=42)
    lr_pipeline.fit(train_df[FEATURE_COLS], y_train)
    lr_scores = lr_pipeline.predict_proba(test_df[FEATURE_COLS])[:, 1]
    lr_results = evaluate_model_performance(y_test, lr_scores, threshold=lr_th, repos=test_repos)
    lr_results["oof_cv_diagnostics"] = lr_cv_diags

    # -------------------------------------------------------------------------
    # 6. Paired Bootstrap for PR-AUC Differences
    # -------------------------------------------------------------------------
    print("\nComputing paired bootstrap differences (1000 resamples, seed 42)...")
    paired_lr_vs_repo = paired_bootstrap_difference(y_test, lr_scores, repo_prior_scores, n_bootstraps=1000, random_state=42)
    paired_lr_vs_recent = paired_bootstrap_difference(y_test, lr_scores, recent_rate_scores, n_bootstraps=1000, random_state=42)
    paired_stump_vs_repo = paired_bootstrap_difference(y_test, stump_scores, repo_prior_scores, n_bootstraps=1000, random_state=42)

    paired_summary = {
        "lr_minus_repo_prior": paired_lr_vs_repo,
        "lr_minus_recent_rate": paired_lr_vs_recent,
        "stump_minus_repo_prior": paired_stump_vs_repo,
    }

    # -------------------------------------------------------------------------
    # 7. Print Console Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"BASELINE MODEL COMPARISON TABLE - PROTOCOL: {protocol_name.upper()}")
    print("=" * 95)
    tbl_hdr = f"{'Model':<24} | {'PR-AUC':<8} | {'95% CI (PR-AUC)':<18} | {'ROC-AUC':<8} | {'Thresh':<6} | {'F1':<6} | {'P@10%':<6} | {'R@10%':<6} | {'P@20%':<6} | {'R@20%':<6}"
    print(tbl_hdr)
    print("-" * len(tbl_hdr))

    models_map = [
        ("Dummy (prior)", dummy_results),
        ("Repo-prior", repo_prior_results),
        ("Recent-rate heuristic", recent_rate_results),
        ("Decision stump", stump_results),
        ("Logistic regression", lr_results),
    ]

    for name, res in models_map:
        prauc = f"{res['pr_auc']:.4f}"
        ci = f"[{res['pr_auc_ci_95'][0]:.4f}, {res['pr_auc_ci_95'][1]:.4f}]"
        roc = f"{res['roc_auc']:.4f}"
        th = f"{res.get('threshold', 0.0):.2f}" if "threshold" in res else "N/A"
        f1 = f"{res.get('f1', 0.0):.4f}" if "f1" in res else "N/A"
        p10 = f"{res['top_k_percent']['top_10%']['precision']:.3f}"
        r10 = f"{res['top_k_percent']['top_10%']['recall']:.3f}"
        p20 = f"{res['top_k_percent']['top_20%']['precision']:.3f}"
        r20 = f"{res['top_k_percent']['top_20%']['recall']:.3f}"

        print(f"{name:<24} | {prauc:<8} | {ci:<18} | {roc:<8} | {th:<6} | {f1:<6} | {p10:<6} | {r10:<6} | {p20:<6} | {r20:<6}")

    print("-" * len(tbl_hdr))
    print("\nPaired Bootstrap Differences (PR-AUC):")
    print(f"  • LR minus Repo-prior:      mean diff = {paired_lr_vs_repo['mean_difference']:+.4f} | 95% CI = [{paired_lr_vs_repo['ci_lower']:+.4f}, {paired_lr_vs_repo['ci_upper']:+.4f}] | LR win rate = {paired_lr_vs_repo['win_fraction']*100:.1f}%")
    print(f"  • LR minus Recent-rate:     mean diff = {paired_lr_vs_recent['mean_difference']:+.4f} | 95% CI = [{paired_lr_vs_recent['ci_lower']:+.4f}, {paired_lr_vs_recent['ci_upper']:+.4f}] | LR win rate = {paired_lr_vs_recent['win_fraction']*100:.1f}%")
    print(f"  • Stump minus Repo-prior:   mean diff = {paired_stump_vs_repo['mean_difference']:+.4f} | 95% CI = [{paired_stump_vs_repo['ci_lower']:+.4f}, {paired_stump_vs_repo['ci_upper']:+.4f}] | Stump win rate = {paired_stump_vs_repo['win_fraction']*100:.1f}%")

    print("\nPer-Repo PR-AUC (repos with >= 5 test failures):")
    for r_name, r_info in lr_results["per_repo_pr_auc"].items():
        if isinstance(r_info["pr_auc"], float):
            print(f"  • {r_name:<28}: failures = {r_info['test_failures']:<2} | LR PR-AUC = {r_info['pr_auc']:.4f}")
        else:
            print(f"  • {r_name:<28}: failures = {r_info['test_failures']:<2} | {r_info['pr_auc']}")
    print("=" * 95 + "\n")

    return {
        "models": {
            "dummy": dummy_results,
            "repo_prior": repo_prior_results,
            "recent_rate_heuristic": recent_rate_results,
            "decision_stump": stump_results,
            "logistic_regression": lr_results,
        },
        "paired_bootstrap": paired_summary,
    }


def main() -> None:
    """Load processed dataset, execute baselines under both protocols, and save json report."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}. Run feature pipeline first.")

    print(f"Loading feature dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], utc=True)

    # -------------------------------------------------------------------------
    # PROTOCOL 1: Per-Repo Chronological Split (PRIMARY)
    # -------------------------------------------------------------------------
    print("\nExecuting PRIMARY protocol: per_repo_time_split...")
    per_repo_train, per_repo_test = per_repo_time_split(df, test_frac=0.2)
    per_repo_diag = compute_step_0_diagnostics(df, per_repo_train, per_repo_test, protocol_name="per_repo")
    per_repo_results = run_protocol_pipeline(per_repo_train, per_repo_test, protocol_name="per_repo")

    # -------------------------------------------------------------------------
    # PROTOCOL 2: Global Chronological Split (SECONDARY)
    # -------------------------------------------------------------------------
    print("\nExecuting SECONDARY protocol: global time_aware_split...")
    global_train, global_test = time_aware_split(df, test_frac=0.2)
    global_diag = compute_step_0_diagnostics(df, global_train, global_test, protocol_name="global")
    global_results = run_protocol_pipeline(global_train, global_test, protocol_name="global")

    # -------------------------------------------------------------------------
    # Assemble and Write reports/baselines.json
    # -------------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "per_repo": {
            "diagnostics": per_repo_diag,
            **per_repo_results,
        },
        "global": {
            "diagnostics": global_diag,
            **global_results,
        },
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[DONE] Baseline report successfully written to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
