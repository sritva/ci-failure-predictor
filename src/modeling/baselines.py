"""Baseline models, feature specifications, threshold tuning, and evaluation metrics."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.modeling.split import time_series_cv_splits


# Exactly 10 model input features.
# INVARIANT: Never use as inputs: label, run_id, run_timestamp, author, repo.
FEATURE_COLS: List[str] = [
    "diff_size",
    "files_changed",
    "test_files_changed",
    "touches_test_files",
    "time_of_day",
    "day_of_week",
    "is_weekend",
    "author_past_run_count",
    "author_past_failure_rate",
    "repo_recent_failure_rate",
]

# Features with right-skewed counts to transform via log1p in Logistic Regression
LOG_COLS: List[str] = [
    "diff_size",
    "files_changed",
    "test_files_changed",
    "author_past_run_count",
]

PASSTHROUGH_COLS: List[str] = [c for c in FEATURE_COLS if c not in LOG_COLS]


# -----------------------------------------------------------------------------
# Model 1: Dummy Baseline
# -----------------------------------------------------------------------------
def fit_dummy_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """Fit a dummy classifier that predicts class prior probabilities."""
    clf = DummyClassifier(strategy="prior")
    clf.fit(X_train[FEATURE_COLS], y_train)
    return clf


# -----------------------------------------------------------------------------
# Model 2: Repo-Prior Baseline
# -----------------------------------------------------------------------------
class RepoPriorClassifier(BaseEstimator, ClassifierMixin):
    """Repo-prior classifier: score = each repo's failure rate in TRAIN.

    Repos unseen in train fall back to the global train rate.
    Evaluates how much signal is purely attributable to repository identity.
    """

    def __init__(self) -> None:
        self.repo_rates: Dict[str, float] = {}
        self.global_rate: float = 0.0

    def fit(self, repos: pd.Series, y: pd.Series) -> "RepoPriorClassifier":
        """Fit repo failure rates on training data only."""
        if len(repos) == 0 or len(y) == 0:
            self.global_rate = 0.0
            self.repo_rates = {}
            return self

        y_numeric = y.astype(float)
        self.global_rate = float(y_numeric.mean())
        df_temp = pd.DataFrame({"repo": repos.values, "label": y_numeric.values})
        self.repo_rates = df_temp.groupby("repo")["label"].mean().to_dict()
        return self

    def predict_proba(self, repos: pd.Series) -> np.ndarray:
        """Predict failure probability based on repo identity."""
        scores = np.array(
            [self.repo_rates.get(r, self.global_rate) for r in repos],
            dtype=float,
        )
        return np.column_stack([1.0 - scores, scores])

    def predict(self, repos: pd.Series, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels using threshold."""
        return (self.predict_proba(repos)[:, 1] >= threshold).astype(int)


# -----------------------------------------------------------------------------
# Model 3: Recent-Rate Heuristic
# -----------------------------------------------------------------------------
class RecentRateHeuristic(BaseEstimator, ClassifierMixin):
    """Recent-rate heuristic: score = repo_recent_failure_rate alone, no fitting."""

    def __init__(self) -> None:
        pass

    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> "RecentRateHeuristic":
        """No fitting required; returns self."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return repo_recent_failure_rate as failure probability."""
        if "repo_recent_failure_rate" not in X.columns:
            raise KeyError("DataFrame must contain 'repo_recent_failure_rate'.")
        scores = X["repo_recent_failure_rate"].values.astype(float)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels using threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# -----------------------------------------------------------------------------
# Model 4: Decision Stump
# -----------------------------------------------------------------------------
def build_decision_stump(random_state: int = 42) -> DecisionTreeClassifier:
    """Build an unfitted DecisionTreeClassifier with depth 1."""
    return DecisionTreeClassifier(
        max_depth=1,
        class_weight="balanced",
        random_state=random_state,
    )


def fit_decision_stump(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """Fit a single decision stump and extract the chosen split feature and threshold."""
    stump = build_decision_stump(random_state=random_state)
    stump.fit(X_train[FEATURE_COLS], y_train)

    tree = stump.tree_
    feat_idx = tree.feature[0]
    split_info = {
        "feature_index": int(feat_idx),
        "feature_name": FEATURE_COLS[feat_idx] if 0 <= feat_idx < len(FEATURE_COLS) else None,
        "threshold": float(tree.threshold[0]),
    }
    return stump, split_info


# -----------------------------------------------------------------------------
# Model 5: Logistic Regression Pipeline
# -----------------------------------------------------------------------------
def build_logistic_regression_pipeline(random_state: int = 42) -> Pipeline:
    """Construct Logistic Regression Pipeline.

    Applies log1p on right-skewed count columns, passes through remaining features,
    applies StandardScaler, and fits LogisticRegression with balanced class weights.
    The scaler lives inside the Pipeline so it is fit on train only.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("log1p", FunctionTransformer(np.log1p, validate=True), LOG_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ]
    )

    pipeline = Pipeline([
        ("transform", preprocessor),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )),
    ])
    return pipeline


# -----------------------------------------------------------------------------
# Out-of-Fold Threshold Selection (Train-Only)
# -----------------------------------------------------------------------------
def find_optimal_threshold_oof(
    model_factory: Callable[[], Any],
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Find the operating threshold that maximizes F1 on out-of-fold train predictions.

    INVARIANT:
    Uses time_series_cv_splits(train_df, n_splits=5) on train_df only.
    Test data is never seen or referenced.

    Args:
        model_factory: Callable that returns a fresh, unfitted model/pipeline instance.
        train_df: Training DataFrame containing FEATURE_COLS and 'label'.
        n_splits: Number of time-series CV splits (default 5).

    Returns:
        Tuple of (best_threshold, best_f1, fold_diagnostics).
    """
    splits = time_series_cv_splits(train_df, n_splits=n_splits)

    oof_y: List[int] = []
    oof_scores: List[float] = []
    fold_diagnostics: List[Dict[str, Any]] = []

    # Ensure train_df is sorted temporally consistent with time_series_cv_splits
    df_sorted = train_df.copy()
    df_sorted["run_timestamp"] = pd.to_datetime(df_sorted["run_timestamp"], utc=True)
    sort_cols = ["run_timestamp"]
    if "run_id" in df_sorted.columns:
        sort_cols.append("run_id")
    df_sorted = df_sorted.sort_values(by=sort_cols).reset_index(drop=True)

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_train = df_sorted.iloc[train_idx]
        fold_val = df_sorted.iloc[val_idx]

        val_rows = len(fold_val)
        val_positives = int((fold_val["label"] == 1).sum())
        flagged_low_positives = val_positives < 15

        fold_diagnostics.append({
            "fold": fold_i,
            "train_rows": len(fold_train),
            "val_rows": val_rows,
            "val_positives": val_positives,
            "val_positive_pct": float(val_positives / val_rows * 100) if val_rows else 0.0,
            "flagged_low_positives": flagged_low_positives,
        })

        model = model_factory()
        model.fit(fold_train[FEATURE_COLS], fold_train["label"])

        preds_proba = model.predict_proba(fold_val[FEATURE_COLS])[:, 1]
        oof_y.extend(fold_val["label"].values.tolist())
        oof_scores.extend(preds_proba.tolist())

    y_true = np.array(oof_y)
    y_scores = np.array(oof_scores)

    # Search thresholds in [0.01, 0.99]
    candidate_thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    best_f1 = -1.0
    best_threshold = 0.5

    for th in candidate_thresholds:
        preds = (y_scores >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        # Deterministically tie-break towards threshold closest to 0.5
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(th)
        elif score == best_f1 and abs(th - 0.5) < abs(best_threshold - 0.5):
            best_threshold = float(th)

    return best_threshold, best_f1, fold_diagnostics


# -----------------------------------------------------------------------------
# Ranking & Performance Metrics
# -----------------------------------------------------------------------------
def evaluate_top_k_percent(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fractions: List[float] = [0.1, 0.2, 0.3],
) -> Dict[str, Dict[str, float]]:
    """Evaluate precision and recall when flagging the top k% of PRs by predicted score.

    Args:
        y_true: Array of binary ground-truth labels (0 or 1).
        y_scores: Array of predicted failure probability scores.
        fractions: List of top proportions to evaluate (e.g. [0.1, 0.2, 0.3]).

    Returns:
        Dictionary mapping percentage string (e.g. 'top_10%') to {'precision', 'recall', 'flagged_count'}.
    """
    n = len(y_true)
    total_positives = int(np.sum(y_true))
    results: Dict[str, Dict[str, float]] = {}

    # Stable sort descending so ties are handled deterministically
    sorted_indices = np.argsort(-y_scores, kind="mergesort")

    for frac in fractions:
        pct_label = f"top_{int(round(frac * 100))}%"
        k = max(1, int(round(n * frac)))
        flagged_idx = sorted_indices[:k]

        tp = int(np.sum(y_true[flagged_idx]))
        prec = float(tp / k) if k > 0 else 0.0
        rec = float(tp / total_positives) if total_positives > 0 else 0.0

        results[pct_label] = {
            "k": k,
            "precision": prec,
            "recall": rec,
        }

    return results


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, int]:
    """Compute 95% bootstrap confidence interval for PR-AUC.

    # NOTE: This is an i.i.d. bootstrap on time-ordered data, so it's only a rough uncertainty guide.

    Returns:
        Tuple of (ci_lower, ci_upper, skipped_single_class_count).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    boot_scores: List[float] = []
    skipped = 0

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_y = y_true[idx]
        # Skip single-class resamples where PR-AUC cannot be calculated
        if len(np.unique(sample_y)) < 2:
            skipped += 1
            continue
        score = average_precision_score(sample_y, y_scores[idx])
        boot_scores.append(score)

    if not boot_scores:
        return 0.0, 0.0, skipped

    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_scores, low_pct))
    ci_upper = float(np.percentile(boot_scores, high_pct))

    return ci_lower, ci_upper, skipped


def paired_bootstrap_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Paired bootstrap for PR-AUC difference (scores_a minus scores_b).

    Uses identical resample indices for both models.
    # NOTE: This is an i.i.d. bootstrap on time-ordered data, so it's only a rough uncertainty guide.

    Returns:
        Dict with mean_difference, ci_lower, ci_upper, win_fraction, and skipped_count.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    diffs: List[float] = []
    skipped = 0

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            skipped += 1
            continue

        pr_a = average_precision_score(sample_y, scores_a[idx])
        pr_b = average_precision_score(sample_y, scores_b[idx])
        diffs.append(pr_a - pr_b)

    if not diffs:
        return {
            "mean_difference": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "win_fraction": 0.0,
            "skipped": skipped,
        }

    diffs_arr = np.array(diffs)
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    return {
        "mean_difference": float(np.mean(diffs_arr)),
        "ci_lower": float(np.percentile(diffs_arr, low_pct)),
        "ci_upper": float(np.percentile(diffs_arr, high_pct)),
        "win_fraction": float(np.mean(diffs_arr > 0)),
        "skipped": skipped,
    }


def evaluate_model_performance(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
    repos: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Calculate standard evaluation metrics on test data for a baseline model."""
    # Handle single-class or degenerate ground truth
    has_two_classes = len(np.unique(y_true)) > 1
    pr_auc = float(average_precision_score(y_true, y_scores)) if has_two_classes else 0.0
    roc_auc = float(roc_auc_score(y_true, y_scores)) if has_two_classes else 0.5

    ci_lower, ci_upper, skipped_boot = bootstrap_pr_auc_ci(y_true, y_scores)
    top_k = evaluate_top_k_percent(y_true, y_scores)

    results: Dict[str, Any] = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "pr_auc_ci_95": [ci_lower, ci_upper],
        "bootstrap_skipped": skipped_boot,
        "top_k_percent": top_k,
    }

    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        results.update({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": cm,
        })

    # Per-repo PR-AUC (for repos with at least 5 failures)
    if repos is not None:
        per_repo: Dict[str, Any] = {}
        df_eval = pd.DataFrame({"repo": repos.values, "y": y_true, "score": y_scores})
        for repo_name, grp in df_eval.groupby("repo"):
            repo_failures = int((grp["y"] == 1).sum())
            if repo_failures >= 5 and grp["y"].nunique() > 1:
                r_prauc = float(average_precision_score(grp["y"].values, grp["score"].values))
                per_repo[repo_name] = {
                    "test_failures": repo_failures,
                    "pr_auc": r_prauc,
                }
            else:
                per_repo[repo_name] = {
                    "test_failures": repo_failures,
                    "pr_auc": "insufficient failures",
                }
        results["per_repo_pr_auc"] = per_repo

    return results
