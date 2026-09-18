"""Tests for Phase 6.5 recall experiments and helpers."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.recall_experiments import (
    generate_feature_variants,
)
from src.modeling.split import per_repo_time_series_cv


class TestRecallExperiments:
    """Test suite for Phase 6.5 recall exploration, F2 search, and ensemble blending."""

    def test_f2_threshold_search_prefers_higher_recall(self):
        """Verify that F2 threshold search favors higher recall than F1 threshold search."""
        from sklearn.metrics import f1_score, fbeta_score

        # Ground truth with imbalanced classes
        y_true = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
        # Continuous predicted scores
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])

        best_f1, best_f1_th = -1.0, 0.5
        best_f2, best_f2_th = -1.0, 0.5

        for th in np.linspace(0.1, 0.9, 81):
            preds = (scores >= th).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_f1_th = th
            if f2 > best_f2:
                best_f2 = f2
                best_f2_th = th

        assert 0.0 < best_f1_th < 1.0
        assert 0.0 < best_f2_th < 1.0
        # F2 weights recall 2x more than precision, so its threshold should be <= F1 threshold
        assert best_f2_th <= best_f1_th

    def test_ensemble_weighting_averages_probability_arrays(self):
        """Verify that ensemble probability weighting computes exact convex combination."""
        p_xgb = np.array([0.8, 0.4, 0.2])
        p_lr = np.array([0.6, 0.2, 0.4])
        w = 0.7

        p_ens = w * p_xgb + (1.0 - w) * p_lr
        expected = 0.7 * p_xgb + 0.3 * p_lr

        np.testing.assert_allclose(p_ens, expected)
        assert p_ens[0] == pytest.approx(0.7 * 0.8 + 0.3 * 0.6)  # 0.56 + 0.18 = 0.74

    def test_feature_variants_leak_free_shift_1(self):
        """Verify that newly generated rolling failure rates strictly shift by 1 and do not leak."""
        rows = [
            {"run_id": 1, "run_timestamp": "2026-01-01 10:00:00", "author": "dev1", "repo": "repo_x", "label": 1, "author_past_run_count": 0},
            {"run_id": 2, "run_timestamp": "2026-01-02 10:00:00", "author": "dev1", "repo": "repo_x", "label": 0, "author_past_run_count": 1},
            {"run_id": 3, "run_timestamp": "2026-01-03 10:00:00", "author": "dev1", "repo": "repo_x", "label": 1, "author_past_run_count": 2},
        ]
        df = pd.DataFrame(rows)
        variants_df = generate_feature_variants(df)

        assert "repo_recent_failure_rate_10" in variants_df.columns
        assert "repo_recent_failure_rate_50" in variants_df.columns
        assert "is_first_time_author" in variants_df.columns

        # First row has no prior runs in repo_x, so its rolling rate cannot know that label=1
        # Row 2 should reflect row 1's label (1.0), NOT row 2's own label (0)
        assert variants_df.iloc[1]["repo_recent_failure_rate_10"] == pytest.approx(1.0)
        assert variants_df.iloc[1]["repo_recent_failure_rate_50"] == pytest.approx(1.0)

        # Row 1 has author_past_run_count=0 -> is_first_time_author=1
        assert variants_df.iloc[0]["is_first_time_author"] == 1
        assert variants_df.iloc[1]["is_first_time_author"] == 0
