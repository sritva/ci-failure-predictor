"""Tests for baseline models, threshold tuning, and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.baselines import (
    FEATURE_COLS,
    RepoPriorClassifier,
    RecentRateHeuristic,
    build_decision_stump,
    fit_decision_stump,
    build_logistic_regression_pipeline,
    find_optimal_threshold_oof,
    evaluate_top_k_percent,
    bootstrap_pr_auc_ci,
    paired_bootstrap_difference,
    evaluate_model_performance,
)


class TestBaselineInvariants:
    """Test suite for project invariants and feature boundaries in baseline modeling."""

    def test_feature_list_excludes_restricted_columns(self):
        """Invariant: FEATURE_COLS must never contain label, run_id, run_timestamp, author, repo."""
        restricted = {"label", "run_id", "run_timestamp", "author", "repo"}
        intersection = set(FEATURE_COLS).intersection(restricted)
        assert len(intersection) == 0, f"Restricted columns found in FEATURE_COLS: {intersection}"

        # Must have exactly 10 model inputs
        assert len(FEATURE_COLS) == 10
        expected = [
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
        assert FEATURE_COLS == expected

    def test_repo_prior_train_isolation_and_fallback(self):
        """Repo-prior uses ONLY train statistics and falls back to global train rate for unseen repos."""
        train_repos = pd.Series(["repo_a"] * 8 + ["repo_b"] * 2)
        # repo_a: 2 fails / 8 = 0.25; repo_b: 2 fails / 2 = 1.0; global: 4 / 10 = 0.40
        train_labels = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

        clf = RepoPriorClassifier()
        clf.fit(train_repos, train_labels)

        assert clf.global_rate == pytest.approx(0.40)
        assert clf.repo_rates["repo_a"] == pytest.approx(0.25)
        assert clf.repo_rates["repo_b"] == pytest.approx(1.00)

        # Test set with known repos and an unseen repo
        test_repos = pd.Series(["repo_a", "repo_b", "unseen_repo"])
        test_probs = clf.predict_proba(test_repos)[:, 1]

        assert test_probs[0] == pytest.approx(0.25)
        assert test_probs[1] == pytest.approx(1.00)
        # unseen repo must fall back to global train failure rate
        assert test_probs[2] == pytest.approx(0.40)

    def test_scaler_fit_on_train_only(self):
        """Pipeline scaler must be fit on train data only; extreme test values must not taint scaler parameters."""
        # Train data with diff_size around 100
        n_train = 40
        train_df = pd.DataFrame({
            "diff_size": [100] * n_train,
            "files_changed": [2] * n_train,
            "test_files_changed": [1] * n_train,
            "touches_test_files": [True] * n_train,
            "time_of_day": [12] * n_train,
            "day_of_week": [2] * n_train,
            "is_weekend": [False] * n_train,
            "author_past_run_count": [5] * n_train,
            "author_past_failure_rate": [0.2] * n_train,
            "repo_recent_failure_rate": [0.1] * n_train,
            "label": [0, 1] * (n_train // 2),
        })

        pipe = build_logistic_regression_pipeline()
        pipe.fit(train_df[FEATURE_COLS], train_df["label"])

        scaler = pipe.named_steps["scaler"]
        train_scaler_mean = scaler.mean_.copy()

        # Extreme test data with diff_size = 10,000,000
        test_df = pd.DataFrame({
            "diff_size": [10_000_000] * 10,
            "files_changed": [500] * 10,
            "test_files_changed": [100] * 10,
            "touches_test_files": [True] * 10,
            "time_of_day": [12] * 10,
            "day_of_week": [2] * 10,
            "is_weekend": [False] * 10,
            "author_past_run_count": [1000] * 10,
            "author_past_failure_rate": [0.9] * 10,
            "repo_recent_failure_rate": [0.8] * 10,
            "label": [1] * 10,
        })

        # Calling predict_proba on test should NOT change scaler parameters
        _ = pipe.predict_proba(test_df[FEATURE_COLS])
        np.testing.assert_allclose(scaler.mean_, train_scaler_mean)

    def test_threshold_selection_never_receives_test_data(self):
        """OOF threshold selection takes only train_df and uses time_series_cv_splits without test data."""
        # Create chronological train dataframe
        n_rows = 50
        dates = pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC")
        train_df = pd.DataFrame({
            "run_id": list(range(1, n_rows + 1)),
            "run_timestamp": dates,
            "diff_size": np.random.randint(10, 500, size=n_rows),
            "files_changed": np.random.randint(1, 10, size=n_rows),
            "test_files_changed": np.random.randint(0, 3, size=n_rows),
            "touches_test_files": [True] * n_rows,
            "time_of_day": [12] * n_rows,
            "day_of_week": [1] * n_rows,
            "is_weekend": [False] * n_rows,
            "author_past_run_count": np.random.randint(0, 20, size=n_rows),
            "author_past_failure_rate": np.random.rand(n_rows),
            "repo_recent_failure_rate": np.random.rand(n_rows),
            "label": np.random.binomial(1, 0.3, size=n_rows),
        })

        best_th, best_f1, fold_diags = find_optimal_threshold_oof(
            model_factory=lambda: build_decision_stump(random_state=42),
            train_df=train_df,
            n_splits=3,
        )

        assert 0.01 <= best_th <= 0.99
        assert 0.0 <= best_f1 <= 1.0
        assert len(fold_diags) == 3


class TestEvaluationHelpers:
    """Test suite for top-k% flagging, bootstrap CIs, and paired differences."""

    def test_top_k_percent_hand_calculated(self):
        """Verify evaluate_top_k_percent on exact hand-calculated test vectors.

        10 samples, total positives = 2 (at index 0 and index 2).
        Scores sorted descending: indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
        - top 10% (k=1): flags index 0 (y=1) -> tp=1, prec=1.0, rec=1/2=0.5
        - top 20% (k=2): flags index 0 (y=1) and 1 (y=0) -> tp=1, prec=0.5, rec=0.5
        - top 30% (k=3): flags index 0, 1, 2 (y=1, 0, 1) -> tp=2, prec=2/3=0.6667, rec=1.0
        """
        y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        results = evaluate_top_k_percent(y_true, y_scores, fractions=[0.1, 0.2, 0.3])

        assert results["top_10%"]["k"] == 1
        assert results["top_10%"]["precision"] == pytest.approx(1.0)
        assert results["top_10%"]["recall"] == pytest.approx(0.5)

        assert results["top_20%"]["k"] == 2
        assert results["top_20%"]["precision"] == pytest.approx(0.5)
        assert results["top_20%"]["recall"] == pytest.approx(0.5)

        assert results["top_30%"]["k"] == 3
        assert results["top_30%"]["precision"] == pytest.approx(2.0 / 3.0)
        assert results["top_30%"]["recall"] == pytest.approx(1.0)

    def test_bootstrap_pr_auc_ci_validity(self):
        """Verify 95% bootstrap CI produces valid bounds where lower <= upper within [0, 1]."""
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.3, size=100)
        y_scores = rng.rand(100)

        ci_low, ci_high, skipped = bootstrap_pr_auc_ci(y_true, y_scores, n_bootstraps=200, random_state=42)

        assert 0.0 <= ci_low <= ci_high <= 1.0
        assert skipped == 0

    def test_paired_bootstrap_difference(self):
        """Verify paired bootstrap difference detects superior model with positive mean difference."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 10)
        # Model A is perfect predictor
        scores_a = y_true.astype(float)
        # Model B is inverse/poor predictor
        scores_b = 1.0 - scores_a

        paired = paired_bootstrap_difference(y_true, scores_a, scores_b, n_bootstraps=200, random_state=42)

        assert paired["mean_difference"] > 0.0
        assert paired["win_fraction"] == 1.0
        assert paired["ci_lower"] > 0.0
