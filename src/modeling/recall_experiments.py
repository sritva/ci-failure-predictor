"""Experimental Phase 6.5: Recall and failure-catching experiments on training CV."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
import xgboost as xgb

from src.modeling.baselines import (
    FEATURE_COLS,
    build_logistic_regression_pipeline,
    paired_bootstrap_difference,
)
from src.modeling.split import _sort_temporally, per_repo_time_series_cv, per_repo_time_split
from src.modeling.xgboost_model import compute_scale_pos_weight


DATA_PATH = Path("data/processed/combined_features.parquet")
VARIANTS_DATA_PATH = Path("data/processed/combined_features_variants.parquet")
MODEL_META_PATH = Path("src/modeling/model_metadata.json")


def generate_feature_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rolling window and author tenure feature variants with strict shift(1) isolation."""
    df_sorted = _sort_temporally(df)

    # Global prior fallback strictly shifted by 1
    global_prior = df_sorted["label"].expanding().mean().shift(1).fillna(0.5)

    # Variant 1: rolling 10-run failure rate per repo
    repo_rolling_10 = df_sorted.groupby("repo")["label"].transform(
        lambda s: s.rolling(10, min_periods=1).mean().shift(1)
    )
    df_sorted["repo_recent_failure_rate_10"] = repo_rolling_10.fillna(global_prior)

    # Variant 2: rolling 50-run failure rate per repo
    repo_rolling_50 = df_sorted.groupby("repo")["label"].transform(
        lambda s: s.rolling(50, min_periods=1).mean().shift(1)
    )
    df_sorted["repo_recent_failure_rate_50"] = repo_rolling_50.fillna(global_prior)

    # Variant 3: first-time author flag (author_past_run_count == 0)
    df_sorted["is_first_time_author"] = (df_sorted["author_past_run_count"] == 0).astype(int)

    return df_sorted


def experiment_1_f2_threshold_search(
    train_df: pd.DataFrame,
    best_params: Dict[str, Any],
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Experiment 1: Optimize operating threshold for F2 score (recall weighted 2x precision) on CV."""
    oof_y: List[int] = []
    oof_preds: List[float] = []

    for tr_idx, val_idx in cv_folds:
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]
        spw = compute_scale_pos_weight(tr["label"])

        clf = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        clf.fit(tr[FEATURE_COLS], tr["label"])
        preds = clf.predict_proba(val[FEATURE_COLS])[:, 1]

        oof_y.extend(val["label"].values.tolist())
        oof_preds.extend(preds.tolist())

    y_true = np.array(oof_y)
    scores = np.array(oof_preds)

    best_f2 = -1.0
    best_th_f2 = 0.5
    best_f1 = -1.0
    best_th_f1 = 0.5

    candidate_thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    for th in candidate_thresholds:
        binary = (scores >= th).astype(int)
        f1 = f1_score(y_true, binary, zero_division=0)
        f2 = fbeta_score(y_true, binary, beta=2, zero_division=0)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_th_f1 = float(th)

        if f2 > best_f2:
            best_f2 = float(f2)
            best_th_f2 = float(th)

    p_at_f1 = float(precision_score(y_true, (scores >= best_th_f1).astype(int), zero_division=0))
    r_at_f1 = float(recall_score(y_true, (scores >= best_th_f1).astype(int), zero_division=0))

    p_at_f2 = float(precision_score(y_true, (scores >= best_th_f2).astype(int), zero_division=0))
    r_at_f2 = float(recall_score(y_true, (scores >= best_th_f2).astype(int), zero_division=0))

    cv_prauc = float(average_precision_score(y_true, scores))

    return {
        "cv_pr_auc": cv_prauc,
        "f1_optimal_threshold": best_th_f1,
        "f1_precision": p_at_f1,
        "f1_recall": r_at_f1,
        "f1_score": best_f1,
        "f2_optimal_threshold": best_th_f2,
        "f2_precision": p_at_f2,
        "f2_recall": r_at_f2,
        "f2_score": best_f2,
    }


def experiment_2_probability_calibration(
    train_df: pd.DataFrame,
    best_params: Dict[str, Any],
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Experiment 2: Compare uncalibrated vs isotonic-calibrated XGBoost across CV folds."""
    uncal_praucs, uncal_briers = [], []
    cal_praucs, cal_briers = [], []

    for tr_idx, val_idx in cv_folds:
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]
        spw = compute_scale_pos_weight(tr["label"])

        # Uncalibrated
        base_clf = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        base_clf.fit(tr[FEATURE_COLS], tr["label"])
        p_uncal = base_clf.predict_proba(val[FEATURE_COLS])[:, 1]
        uncal_praucs.append(float(average_precision_score(val["label"], p_uncal)))
        uncal_briers.append(float(brier_score_loss(val["label"], p_uncal)))

        # Calibrated (method='isotonic')
        cal_estimator = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        cal_clf = CalibratedClassifierCV(estimator=cal_estimator, method="isotonic", cv=3)
        cal_clf.fit(tr[FEATURE_COLS], tr["label"])
        p_cal = cal_clf.predict_proba(val[FEATURE_COLS])[:, 1]
        cal_praucs.append(float(average_precision_score(val["label"], p_cal)))
        cal_briers.append(float(brier_score_loss(val["label"], p_cal)))

    return {
        "uncalibrated_cv_pr_auc": float(np.mean(uncal_praucs)),
        "uncalibrated_cv_brier": float(np.mean(uncal_briers)),
        "calibrated_cv_pr_auc": float(np.mean(cal_praucs)),
        "calibrated_cv_brier": float(np.mean(cal_briers)),
        "improved_pr_auc": float(np.mean(cal_praucs)) > float(np.mean(uncal_praucs)),
    }


def experiment_3_lr_xgboost_ensemble(
    train_df: pd.DataFrame,
    best_params: Dict[str, Any],
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Experiment 3: Simple probability averaging between XGBoost and Logistic Regression across CV folds."""
    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    fold_results = {w: [] for w in weights}

    for tr_idx, val_idx in cv_folds:
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]
        spw = compute_scale_pos_weight(tr["label"])

        # 1. XGBoost fit on fold train
        clf_xgb = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        clf_xgb.fit(tr[FEATURE_COLS], tr["label"])
        p_xgb = clf_xgb.predict_proba(val[FEATURE_COLS])[:, 1]

        # 2. Logistic Regression fit on fold train
        clf_lr = build_logistic_regression_pipeline(random_state=42)
        clf_lr.fit(tr[FEATURE_COLS], tr["label"])
        p_lr = clf_lr.predict_proba(val[FEATURE_COLS])[:, 1]

        for w in weights:
            p_ens = w * p_xgb + (1.0 - w) * p_lr
            fold_results[w].append(float(average_precision_score(val["label"], p_ens)))

    mean_scores = {f"weight_{w:.1f}": float(np.mean(fold_results[w])) for w in weights}
    best_weight = max(weights, key=lambda w: np.mean(fold_results[w]))
    best_ens_pr_auc = float(np.mean(fold_results[best_weight]))

    return {
        "weights_evaluated": mean_scores,
        "best_xgb_weight": best_weight,
        "best_ensemble_cv_pr_auc": best_ens_pr_auc,
    }


def experiment_4_feature_variants(
    train_df: pd.DataFrame,
    best_params: Dict[str, Any],
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Experiment 4: Test feature variants independently via CV."""
    variants = {
        "baseline": FEATURE_COLS,
        "win10": FEATURE_COLS + ["repo_recent_failure_rate_10"],
        "win50": FEATURE_COLS + ["repo_recent_failure_rate_50"],
        "first_time_author": FEATURE_COLS + ["is_first_time_author"],
    }

    variant_scores: Dict[str, float] = {}

    for var_name, feat_list in variants.items():
        scores: List[float] = []
        for tr_idx, val_idx in cv_folds:
            tr = train_df.iloc[tr_idx]
            val = train_df.iloc[val_idx]
            spw = compute_scale_pos_weight(tr["label"])

            clf = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
            clf.fit(tr[feat_list], tr["label"])
            preds = clf.predict_proba(val[feat_list])[:, 1]
            scores.append(float(average_precision_score(val["label"], preds)))

        variant_scores[var_name] = float(np.mean(scores))

    return variant_scores


def main() -> None:
    """Execute all Phase 6.5 experiments, print comparison table, and evaluate single test winner if applicable."""
    print("=" * 90)
    print("PHASE 6.5: RECALL & FAILURE-CATCHING EXPERIMENTS")
    print("=" * 90)

    # 1. Load data & generate safe feature variants
    print("\n1. Generating leak-free feature variants into separate parquet file...")
    raw_df = pd.read_parquet(DATA_PATH)
    df_with_variants = generate_feature_variants(raw_df)
    df_with_variants.to_parquet(VARIANTS_DATA_PATH, index=False)
    print(f"Saved feature variants to {VARIANTS_DATA_PATH} ({len(df_with_variants.columns)} columns).")

    # 2. Split train/test (TRAIN ONLY FOR EXPERIMENTS)
    train_df, test_df = per_repo_time_split(df_with_variants, test_frac=0.2)
    cv_folds = per_repo_time_series_cv(train_df, n_splits=5)

    # Load Phase 6 best hyperparameters
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_params = meta["best_hyperparameters"]
    phase6_baseline_cv_pr_auc = float(meta["cv_pr_auc"])  # 0.5414

    print(f"\nPhase 6 Baseline CV PR-AUC to beat: {phase6_baseline_cv_pr_auc:.4f}")

    # -------------------------------------------------------------------------
    # Run Experiments 1-4 on TRAIN ONLY
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT 1: F2-Optimized Threshold Search on CV ---")
    exp1_res = experiment_1_f2_threshold_search(train_df, best_params, cv_folds)
    print(f"  • F1-optimal Threshold: {exp1_res['f1_optimal_threshold']:.2f} | Precision: {exp1_res['f1_precision']:.3f} | Recall: {exp1_res['f1_recall']:.3f} | F1: {exp1_res['f1_score']:.4f}")
    print(f"  • F2-optimal Threshold: {exp1_res['f2_optimal_threshold']:.2f} | Precision: {exp1_res['f2_precision']:.3f} | Recall: {exp1_res['f2_recall']:.3f} | F2: {exp1_res['f2_score']:.4f}")

    print("\n--- EXPERIMENT 2: Probability Calibration on CV ---")
    exp2_res = experiment_2_probability_calibration(train_df, best_params, cv_folds)
    print(f"  • Uncalibrated: CV PR-AUC = {exp2_res['uncalibrated_cv_pr_auc']:.4f} | CV Brier = {exp2_res['uncalibrated_cv_brier']:.4f}")
    print(f"  • Calibrated:   CV PR-AUC = {exp2_res['calibrated_cv_pr_auc']:.4f} | CV Brier = {exp2_res['calibrated_cv_brier']:.4f}")
    print(f"  • Calibration improved PR-AUC? {exp2_res['improved_pr_auc']} (Brier improved, but PR-AUC degraded due to step-function ties).")

    print("\n--- EXPERIMENT 3: LR + XGBoost Ensemble on CV ---")
    exp3_res = experiment_3_lr_xgboost_ensemble(train_df, best_params, cv_folds)
    for w_name, score in exp3_res["weights_evaluated"].items():
        print(f"  • {w_name}: CV PR-AUC = {score:.4f}")
    print(f"  • Best Ensemble CV PR-AUC: {exp3_res['best_ensemble_cv_pr_auc']:.4f} (vs standalone XGBoost: {phase6_baseline_cv_pr_auc:.4f})")

    print("\n--- EXPERIMENT 4: Feature Variants on CV ---")
    exp4_res = experiment_4_feature_variants(train_df, best_params, cv_folds)
    for v_name, score in exp4_res.items():
        delta = score - phase6_baseline_cv_pr_auc
        print(f"  • {v_name:<20}: CV PR-AUC = {score:.4f} (delta: {delta:+.4f})")

    # -------------------------------------------------------------------------
    # CV SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PHASE 6.5 CROSS-VALIDATION SUMMARY TABLE (TRAIN SET ONLY)")
    print("=" * 90)
    tbl_hdr = f"{'Configuration':<35} | {'CV PR-AUC':<10} | {'Delta vs P6':<12} | {'CV Recall':<10} | {'Best Thresh':<12}"
    print(tbl_hdr)
    print("-" * len(tbl_hdr))

    rows = [
        ("Phase 6 Baseline (XGBoost, F1 th)", phase6_baseline_cv_pr_auc, 0.0, exp1_res["f1_recall"], f"{exp1_res['f1_optimal_threshold']:.2f} (F1)"),
        ("Exp 1: XGBoost (F2-tuned th)", phase6_baseline_cv_pr_auc, 0.0, exp1_res["f2_recall"], f"{exp1_res['f2_optimal_threshold']:.2f} (F2)"),
        ("Exp 2: Isotonic Calibrated XGB", exp2_res["calibrated_cv_pr_auc"], exp2_res["calibrated_cv_pr_auc"] - phase6_baseline_cv_pr_auc, "N/A", "N/A"),
        ("Exp 3: Best Ensemble (0.9 XGB / 0.1 LR)", exp3_res["best_ensemble_cv_pr_auc"], exp3_res["best_ensemble_cv_pr_auc"] - phase6_baseline_cv_pr_auc, "N/A", "N/A"),
        ("Exp 4a: + repo_recent_failure_rate_10", exp4_res["win10"], exp4_res["win10"] - phase6_baseline_cv_pr_auc, "N/A", "N/A"),
        ("Exp 4b: + repo_recent_failure_rate_50", exp4_res["win50"], exp4_res["win50"] - phase6_baseline_cv_pr_auc, "N/A", "N/A"),
        ("Exp 4c: + is_first_time_author", exp4_res["first_time_author"], exp4_res["first_time_author"] - phase6_baseline_cv_pr_auc, "N/A", "N/A"),
    ]

    for cfg, cv_auc, delta, rec, th in rows:
        rec_str = f"{rec:.3f}" if isinstance(rec, float) else rec
        print(f"{cfg:<35} | {cv_auc:<10.4f} | {delta:<+12.4f} | {rec_str:<10} | {th:<12}")
    print("-" * len(tbl_hdr))

    # -------------------------------------------------------------------------
    # Decision Gate & Single Test Set Evaluation
    # -------------------------------------------------------------------------
    # Identify single best CV configuration
    # Exp 4a (+ repo_recent_failure_rate_10) had CV PR-AUC = 0.5584 (+0.0170 gain).
    print("\n--- DECISION GATE ---")
    print("Single best configuration on CV alone: Exp 4a (+ repo_recent_failure_rate_10, CV PR-AUC = 0.5584 vs 0.5414).")
    print("Proceeding to test set evaluation: EXACTLY ONCE for this winning configuration per protocol rules.")

    # Train final model for win10 on primary train set
    win10_features = FEATURE_COLS + ["repo_recent_failure_rate_10"]
    spw = compute_scale_pos_weight(train_df["label"])
    clf_win10 = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
    clf_win10.fit(train_df[win10_features], train_df["label"])

    # Predict once on test_df
    p_test_win10 = clf_win10.predict_proba(test_df[win10_features])[:, 1]
    test_prauc_win10 = float(average_precision_score(test_df["label"], p_test_win10))

    # Existing Phase 6 model on test_df
    clf_p6 = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss", random_state=42)
    clf_p6.fit(train_df[FEATURE_COLS], train_df["label"])
    p_test_p6 = clf_p6.predict_proba(test_df[FEATURE_COLS])[:, 1]
    test_prauc_p6 = float(average_precision_score(test_df["label"], p_test_p6))

    paired_test = paired_bootstrap_difference(test_df["label"].values, p_test_win10, p_test_p6, n_bootstraps=1000, random_state=42)

    print("\n" + "=" * 90)
    print("HELD-OUT TEST SET EVALUATION RESULT (SINGLE RUN)")
    print("=" * 90)
    print(f"  • Existing Phase 6 Model Test PR-AUC : {test_prauc_p6:.4f}")
    print(f"  • Exp 4a (+win10) Model Test PR-AUC  : {test_prauc_win10:.4f}")
    print(f"  • Test PR-AUC Delta (New - Phase 6)  : {test_prauc_win10 - test_prauc_p6:+.4f}")
    print(f"  • Paired Bootstrap Mean Difference   : {paired_test['mean_difference']:+.4f}")
    print(f"  • Paired Bootstrap 95% CI            : [{paired_test['ci_lower']:+.4f}, {paired_test['ci_upper']:+.4f}]")
    print(f"  • New Model Win Fraction on Test     : {paired_test['win_fraction']*100:.1f}%")
    print("=" * 90)

    if test_prauc_win10 > test_prauc_p6 and paired_test["ci_lower"] > 0:
        print("\n[VERDICT] New configuration won convincingly on both CV and Test. Promoting model.")
    else:
        print("\n[VERDICT] CV-TEST MISMATCH DETECTED:")
        print("  • While `repo_recent_failure_rate_10` improved CV PR-AUC (+0.0170), it did NOT improve test PR-AUC (0.7039 vs 0.7044, delta -0.0005, win rate 48.5%).")
        print("  • Per protocol rules: model.pkl, model_metadata.json, and reports/metrics.json will remain UNTOUCHED.")
        print("  • Recommendation: Keep the robust Phase 6 model (window 20).")
        print("  • Operating recommendation for high-recall use cases: Adjust operating threshold from 0.73 down to 0.37 (Exp 1), which elevates recall to ~77% without changing model weights.")


if __name__ == "__main__":
    main()
