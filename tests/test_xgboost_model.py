"""Tests for XGBoost model training, tuning, and invariants."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.baselines import FEATURE_COLS
from src.modeling.split import per_repo_time_series_cv
from src.modeling.xgboost_model import (
    compute_scale_pos_weight,
    train_final_model,
    tune_xgboost,
    get_feature_importances,
)


@pytest.fixture
def synthetic_multirepo_df():
    """Create a synthetic 2-repo DataFrame with 40 rows."""
    rows = []
    # repo_a: 20 rows
    for i in range(1, 21):
        rows.append({
            "run_id": i,
            "run_timestamp": f"2026-01-{i:02d} 10:00:00",
            "author": f"user_{i}",
            "repo": "repo_a",
            "label": 1 if i % 4 == 0 else 0,  # 5 positives, 15 negatives
            "diff_size": 100 + i,
            "files_changed": i,
            "test_files_changed": 1,
            "touches_test_files": True,
            "time_of_day": 12,
            "day_of_week": 2,
            "is_weekend": False,
            "author_past_run_count": i,
            "author_past_failure_rate": 0.2,
            "repo_recent_failure_rate": 0.25,
        })
    # repo_b: 20 rows
    for j in range(1, 21):
        rows.append({
            "run_id": 100 + j,
            "run_timestamp": f"2026-02-{j:02d} 10:00:00",
            "author": f"user_{j}",
            "repo": "repo_b",
            "label": 1 if j % 5 == 0 else 0,  # 4 positives, 16 negatives
            "diff_size": 200 + j,
            "files_changed": j,
            "test_files_changed": 0,
            "touches_test_files": False,
            "time_of_day": 14,
            "day_of_week": 3,
            "is_weekend": False,
            "author_past_run_count": j,
            "author_past_failure_rate": 0.1,
            "repo_recent_failure_rate": 0.2,
        })
    df = pd.DataFrame(rows)
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], utc=True)
    return df


class TestXGBoostInvariants:
    """Test suite for XGBoost scale_pos_weight, schema validation, and tuning isolation."""

    def test_compute_scale_pos_weight_hand_calculated(self):
        """Verify scale_pos_weight on exact hand-computed labels."""
        # 8 negatives, 2 positives -> 8 / 2 = 4.0
        y1 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        assert compute_scale_pos_weight(y1) == pytest.approx(4.0)

        # 9 negatives, 1 positive -> 9 / 1 = 9.0
        y2 = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert compute_scale_pos_weight(y2) == pytest.approx(9.0)

        # 0 positives -> returns defensive fallback 1.0
        y3 = np.array([0, 0, 0])
        assert compute_scale_pos_weight(y3) == pytest.approx(1.0)

    def test_tune_xgboost_temporal_leakage_protection(self, synthetic_multirepo_df):
        """Verify that per_repo_time_series_cv used by tune_xgboost never leaks future rows into training."""
        folds = per_repo_time_series_cv(synthetic_multirepo_df, n_splits=3)
        assert len(folds) == 3

        df_sorted = synthetic_multirepo_df.sort_values(by=["run_timestamp", "run_id"]).reset_index(drop=True)

        for tr_idx, val_idx in folds:
            tr_df = df_sorted.iloc[tr_idx]
            val_df = df_sorted.iloc[val_idx]

            for repo in ["repo_a", "repo_b"]:
                repo_tr = tr_df[tr_df["repo"] == repo]
                repo_val = val_df[val_df["repo"] == repo]

                assert len(repo_tr) > 0
                assert len(repo_val) > 0
                # Strict temporal ordering per repo: max train < min val
                assert repo_tr["run_timestamp"].max() < repo_val["run_timestamp"].min()

    def test_train_final_model_schema_validation(self, synthetic_multirepo_df):
        """Verify that extraneous unexpected columns in train_df raise a clear ValueError."""
        # Add an unexpected/leaky column
        leaky_df = synthetic_multirepo_df.copy()
        leaky_df["leaky_future_column"] = 123.45

        best_params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        }

        with pytest.raises(ValueError) as exc_info:
            train_final_model(leaky_df, best_params)

        assert "Schema validation failed" in str(exc_info.value)
        assert "leaky_future_column" in str(exc_info.value)

    def test_xgboost_integration_tiny_synthetic(self, synthetic_multirepo_df):
        """Integration test: train on synthetic train_df, assert predict_proba returns valid probabilities."""
        best_params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        }

        model = train_final_model(synthetic_multirepo_df, best_params)
        probs = model.predict_proba(synthetic_multirepo_df[FEATURE_COLS])

        assert probs.shape == (len(synthetic_multirepo_df), 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # Verify feature importances can be extracted
        importances = get_feature_importances(model)
        assert len(importances) == len(FEATURE_COLS)
        assert importances[0]["gain"] >= importances[-1]["gain"]
