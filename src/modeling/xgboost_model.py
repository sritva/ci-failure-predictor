"""XGBoost training, per-repo time-series hyperparameter tuning, and model persistence."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
import xgboost as xgb

from src.modeling.baselines import FEATURE_COLS
from src.modeling.split import _sort_temporally, per_repo_time_series_cv


ALLOWED_METADATA_COLS = {"label", "run_id", "run_timestamp", "author", "repo"}
ALLOWED_ALL_COLS = set(FEATURE_COLS) | ALLOWED_METADATA_COLS


def compute_scale_pos_weight(y_train: Any) -> float:
    """Compute negative:positive class balance ratio strictly on training labels.

    Args:
        y_train: 1D array or Series of binary labels (0 or 1).

    Returns:
        float: (count of 0s / count of 1s). Returns 1.0 if no positive samples exist.
    """
    if isinstance(y_train, pd.Series):
        y_arr = y_train.values
    else:
        y_arr = np.asarray(y_train)

    pos_count = int(np.sum(y_arr == 1))
    neg_count = int(np.sum(y_arr == 0))

    if pos_count == 0:
        return 1.0

    return float(neg_count / pos_count)


def sample_hyperparameters(rng: np.random.RandomState) -> Dict[str, Any]:
    """Sample a random candidate hyperparameter configuration."""
    # max_depth in [3, 8]
    max_depth = int(rng.randint(3, 9))

    # learning_rate in [0.01, 0.3], log-scale
    log_lr = rng.uniform(np.log(0.01), np.log(0.3))
    learning_rate = float(np.round(np.exp(log_lr), 4))

    # n_estimators in [50, 500]
    n_estimators_choices = [50, 75, 100, 150, 200, 250, 300, 350, 400, 500]
    n_estimators = int(rng.choice(n_estimators_choices))

    # subsample in [0.6, 1.0]
    subsample = float(np.round(rng.uniform(0.6, 1.0), 2))

    # colsample_bytree in [0.6, 1.0]
    colsample_bytree = float(np.round(rng.uniform(0.6, 1.0), 2))

    # min_child_weight in [1, 10]
    min_child_weight = int(rng.randint(1, 11))

    return {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
    }


def tune_xgboost(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Tune XGBoost hyperparameters via per_repo_time_series_cv optimizing validation PR-AUC.

    scale_pos_weight is recomputed for each fold's specific training split to prevent leakage
    and respect fold-level class distributions.

    Args:
        train_df: Training DataFrame containing FEATURE_COLS, 'label', and metadata.
        n_splits: Number of time-series splits per repo (default 5).
        n_iter: Number of random candidate configurations to evaluate (default 30).
        random_state: Random seed for parameter generation (default 42).

    Returns:
        Tuple of (best_params, best_cv_pr_auc, all_candidate_results).
    """
    df_sorted = _sort_temporally(train_df)
    cv_folds = per_repo_time_series_cv(df_sorted, n_splits=n_splits)

    rng = np.random.RandomState(random_state)
    best_params: Dict[str, Any] = {}
    best_cv_score = -1.0
    tuning_history: List[Dict[str, Any]] = []

    for i in range(n_iter):
        candidate_params = sample_hyperparameters(rng)
        fold_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            train_slice = df_sorted.iloc[train_idx]
            val_slice = df_sorted.iloc[val_idx]

            fold_spw = compute_scale_pos_weight(train_slice["label"])

            clf = xgb.XGBClassifier(
                **candidate_params,
                scale_pos_weight=fold_spw,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )

            clf.fit(train_slice[FEATURE_COLS], train_slice["label"])
            val_preds = clf.predict_proba(val_slice[FEATURE_COLS])[:, 1]
            fold_prauc = float(average_precision_score(val_slice["label"], val_preds))
            fold_scores.append(fold_prauc)

        mean_score = float(np.mean(fold_scores))
        tuning_history.append({
            "iter": i + 1,
            "params": candidate_params,
            "mean_pr_auc": mean_score,
            "fold_scores": fold_scores,
        })

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_params = candidate_params

    return best_params, best_cv_score, tuning_history


def train_final_model(
    train_df: pd.DataFrame,
    best_params: Dict[str, Any],
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Refit final XGBoost model on the full training dataset with schema validation.

    Args:
        train_df: Training DataFrame containing FEATURE_COLS and 'label'.
        best_params: Chosen hyperparameters.
        random_state: Random state for reproducibility (default 42).

    Returns:
        Fitted xgb.XGBClassifier instance.

    Raises:
        ValueError: If train_df contains any column outside FEATURE_COLS and allowed metadata.
    """
    unexpected_cols = set(train_df.columns) - ALLOWED_ALL_COLS
    if unexpected_cols:
        raise ValueError(
            f"Schema validation failed in train_final_model: train_df contains unexpected columns: "
            f"{unexpected_cols}. Allowed columns are: {ALLOWED_ALL_COLS}"
        )

    spw = compute_scale_pos_weight(train_df["label"])

    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["label"])
    return model


def find_optimal_threshold_xgb_oof(
    train_df: pd.DataFrame,
    best_params: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Find the operating threshold that maximizes F1 on out-of-fold CV predictions using per_repo_time_series_cv.

    Args:
        train_df: Training DataFrame.
        best_params: Optimal hyperparameters.
        n_splits: Number of time-series splits (default 5).
        random_state: Seed (default 42).

    Returns:
        Tuple of (best_threshold, best_f1, fold_diagnostics).
    """
    df_sorted = _sort_temporally(train_df)
    cv_folds = per_repo_time_series_cv(df_sorted, n_splits=n_splits)

    oof_y: List[int] = []
    oof_scores: List[float] = []
    fold_diagnostics: List[Dict[str, Any]] = []

    for fold_i, (train_idx, val_idx) in enumerate(cv_folds, start=1):
        fold_train = df_sorted.iloc[train_idx]
        fold_val = df_sorted.iloc[val_idx]

        val_rows = len(fold_val)
        val_positives = int((fold_val["label"] == 1).sum())
        flagged_low_positives = val_positives < 15

        fold_spw = compute_scale_pos_weight(fold_train["label"])

        clf = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=fold_spw,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(fold_train[FEATURE_COLS], fold_train["label"])

        preds_proba = clf.predict_proba(fold_val[FEATURE_COLS])[:, 1]
        oof_y.extend(fold_val["label"].values.tolist())
        oof_scores.extend(preds_proba.tolist())

        fold_diagnostics.append({
            "fold": fold_i,
            "train_rows": len(fold_train),
            "val_rows": val_rows,
            "val_positives": val_positives,
            "val_positive_pct": float(val_positives / val_rows * 100) if val_rows else 0.0,
            "flagged_low_positives": flagged_low_positives,
        })

    y_true = np.array(oof_y)
    y_scores = np.array(oof_scores)

    candidate_thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    best_f1 = -1.0
    best_threshold = 0.5

    for th in candidate_thresholds:
        preds = (y_scores >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(th)
        elif score == best_f1 and abs(th - 0.5) < abs(best_threshold - 0.5):
            best_threshold = float(th)

    return best_threshold, best_f1, fold_diagnostics


def get_feature_importances(
    model: xgb.XGBClassifier,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract gain-based feature importances sorted descending.

    Args:
        model: Fitted XGBClassifier.
        feature_names: List of feature names matching model inputs (default FEATURE_COLS).

    Returns:
        List of dicts with 'feature' and 'gain_importance', sorted descending.
    """
    if feature_names is None:
        feature_names = FEATURE_COLS

    booster = model.get_booster()
    # Feature score with importance_type='gain'
    score_dict = booster.get_score(importance_type="gain")

    importances: List[Dict[str, Any]] = []
    for f in feature_names:
        # booster uses feature names if fit on DataFrame, else 'f0', 'f1', etc.
        gain_val = score_dict.get(f, 0.0)
        importances.append({
            "feature": f,
            "gain": float(gain_val),
        })

    importances.sort(key=lambda x: x["gain"], reverse=True)
    return importances
