"""Tests for time-aware splitting, TimeSeriesSplit CV, and repo holdout strategies."""

import pandas as pd
import pytest

from src.modeling.split import (
    time_aware_split,
    time_series_cv_splits,
    repo_holdout_split,
    per_repo_time_split,
    per_repo_time_series_cv,
)


@pytest.fixture
def synthetic_df():
    """Create a 12-row synthetic DataFrame with explicit chronological timestamps and 2 repos."""
    timestamps = [
        "2026-01-01 10:00:00",
        "2026-01-02 10:00:00",
        "2026-01-03 10:00:00",
        "2026-01-04 10:00:00",
        "2026-01-05 10:00:00",
        "2026-01-06 10:00:00",
        "2026-01-07 10:00:00",
        "2026-01-08 10:00:00",
        "2026-01-09 10:00:00",
        "2026-01-10 10:00:00",
        "2026-01-11 10:00:00",
        "2026-01-12 10:00:00",
    ]
    repos = ["repo_a", "repo_b"] * 6
    labels = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]

    return pd.DataFrame({
        "run_id": list(range(1, 13)),
        "run_timestamp": timestamps,
        "author": [f"user_{i}" for i in range(12)],
        "repo": repos,
        "label": labels,
    })


class TestTimeAwareSplit:
    """Test suite for chronological time-aware train/test splitting."""

    def test_time_aware_split_boundary_and_counts(self, synthetic_df):
        """Test exact row cut point for test_frac=0.25 (9 train, 3 test)."""
        train_df, test_df = time_aware_split(synthetic_df, test_frac=0.25)

        assert len(train_df) == 9
        assert len(test_df) == 3
        # First 9 rows in chronological order must be train
        assert train_df["run_id"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Latest 3 rows must be test
        assert test_df["run_id"].tolist() == [10, 11, 12]

    def test_time_aware_split_temporal_boundary_assertion(self, synthetic_df):
        """Verify that train max timestamp <= test min timestamp strictly holds."""
        train_df, test_df = time_aware_split(synthetic_df, test_frac=0.2)

        max_train = train_df["run_timestamp"].max()
        min_test = test_df["run_timestamp"].min()
        assert max_train <= min_test
        assert max_train == pd.Timestamp("2026-01-09 10:00:00+00:00")
        assert min_test == pd.Timestamp("2026-01-10 10:00:00+00:00")

    def test_time_aware_split_invalid_fraction_raises_error(self, synthetic_df):
        """Verify invalid test_frac values raise ValueError."""
        with pytest.raises(ValueError):
            time_aware_split(synthetic_df, test_frac=1.5)
        with pytest.raises(ValueError):
            time_aware_split(synthetic_df, test_frac=0.0)


class TestTimeSeriesCVSplits:
    """Test suite for TimeSeriesSplit cross-validation wrapper."""

    def test_time_series_cv_splits_temporal_order(self, synthetic_df):
        """Verify every fold's test indices strictly succeed that fold's train indices."""
        splits = time_series_cv_splits(synthetic_df, n_splits=3)
        assert len(splits) == 3

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            assert len(train_idx) > 0
            assert len(test_idx) > 0

            # Every index in test_idx must be strictly greater than max index in train_idx
            assert min(test_idx) > max(train_idx)

            # Check timestamp ordering
            train_timestamps = pd.to_datetime(synthetic_df.iloc[train_idx]["run_timestamp"], utc=True)
            test_timestamps = pd.to_datetime(synthetic_df.iloc[test_idx]["run_timestamp"], utc=True)

            assert train_timestamps.max() <= test_timestamps.min()


class TestRepoHoldoutSplit:
    """Test suite for secondary repository-grouped holdout splitting."""

    def test_repo_holdout_split_multi_repo(self, synthetic_df):
        """Verify clean separation by repo on 2-repo data."""
        train_df, test_df = repo_holdout_split(synthetic_df, holdout_repo="repo_b")

        assert len(train_df) == 6
        assert len(test_df) == 6
        assert (train_df["repo"] == "repo_a").all()
        assert (test_df["repo"] == "repo_b").all()

    def test_repo_holdout_split_single_repo_raises_informative_error(self, synthetic_df):
        """Verify that single-repo datasets raise a clear, informative ValueError."""
        single_repo_df = synthetic_df[synthetic_df["repo"] == "repo_a"].copy()
        with pytest.raises(ValueError) as exc_info:
            repo_holdout_split(single_repo_df, holdout_repo="repo_a")

        assert "requires 2 or more repositories" in str(exc_info.value)
        assert "single-repo" in str(exc_info.value)

    def test_repo_holdout_split_missing_repo_raises_error(self, synthetic_df):
        """Verify querying a nonexistent repo raises ValueError with available repos."""
        with pytest.raises(ValueError) as exc_info:
            repo_holdout_split(synthetic_df, holdout_repo="nonexistent_repo")

        assert "Holdout repo 'nonexistent_repo' not found" in str(exc_info.value)


class TestPerRepoTimeSplit:
    """Test suite for per-repository chronological train/test splitting."""

    def test_per_repo_time_split_hand_calculated(self):
        """Verify exact per-repo counts and temporal boundaries on a multi-repo fixture.

        repo_a has 10 rows -> test_frac=0.2 -> n_test = round(2.0) = 2, n_train = 8
        repo_b has 5 rows  -> test_frac=0.2 -> n_test = round(1.0) = 1, n_train = 4
        Total train = 12, Total test = 3.
        """
        rows = []
        # repo_a: 10 rows (days 1 to 10)
        for i in range(1, 11):
            rows.append({
                "run_id": i,
                "run_timestamp": f"2026-01-{i:02d} 10:00:00",
                "repo": "repo_a",
                "label": i % 2,
            })
        # repo_b: 5 rows (days 1 to 5)
        for j in range(1, 6):
            rows.append({
                "run_id": 100 + j,
                "run_timestamp": f"2026-01-{j:02d} 12:00:00",
                "repo": "repo_b",
                "label": 0,
            })

        df = pd.DataFrame(rows)
        train_df, test_df = per_repo_time_split(df, test_frac=0.2)

        # Exact row count checks
        assert len(train_df) == 12
        assert len(test_df) == 3

        train_a = train_df[train_df["repo"] == "repo_a"]
        test_a = test_df[test_df["repo"] == "repo_a"]
        assert len(train_a) == 8
        assert len(test_a) == 2
        assert train_a["run_id"].tolist() == list(range(1, 9))
        assert test_a["run_id"].tolist() == [9, 10]

        train_b = train_df[train_df["repo"] == "repo_b"]
        test_b = test_df[test_df["repo"] == "repo_b"]
        assert len(train_b) == 4
        assert len(test_b) == 1
        assert train_b["run_id"].tolist() == [101, 102, 103, 104]
        assert test_b["run_id"].tolist() == [105]

        # Per-repo temporal boundary check
        assert train_a["run_timestamp"].max() <= test_a["run_timestamp"].min()
        assert train_b["run_timestamp"].max() <= test_b["run_timestamp"].min()

    def test_per_repo_time_split_tiny_repo_edge_cases(self):
        """Verify tiny-repo behavior: 1 row raises ValueError, 2 rows splits 1 train / 1 test."""
        # Case 1: Repo with 1 row cannot land in both train and test
        df_1_row = pd.DataFrame([{
            "run_id": 1,
            "run_timestamp": "2026-01-01 10:00:00",
            "repo": "tiny_repo",
            "label": 0,
        }])
        with pytest.raises(ValueError) as exc_info:
            per_repo_time_split(df_1_row, test_frac=0.2)
        assert "cannot be split into both train and test" in str(exc_info.value) or "land in both train and test" in str(exc_info.value)

        # Case 2: Repo with 2 rows: n_test = max(1, round(2*0.2)) = 1, n_train = 1
        df_2_rows = pd.DataFrame([
            {"run_id": 1, "run_timestamp": "2026-01-01 10:00:00", "repo": "tiny_repo", "label": 0},
            {"run_id": 2, "run_timestamp": "2026-01-02 10:00:00", "repo": "tiny_repo", "label": 1},
        ])
        train_df, test_df = per_repo_time_split(df_2_rows, test_frac=0.2)
        assert len(train_df) == 1
        assert len(test_df) == 1
        assert train_df.iloc[0]["run_id"] == 1
        assert test_df.iloc[0]["run_id"] == 2

    def test_per_repo_time_split_unsorted_input(self):
        """Verify that shuffled/unsorted input is correctly sorted chronologically per repo."""
        # Create records in reverse chronological order
        rows = [
            {"run_id": 3, "run_timestamp": "2026-01-03 10:00:00", "repo": "r1", "label": 1},
            {"run_id": 1, "run_timestamp": "2026-01-01 10:00:00", "repo": "r1", "label": 0},
            {"run_id": 2, "run_timestamp": "2026-01-02 10:00:00", "repo": "r1", "label": 0},
        ]
        df_unsorted = pd.DataFrame(rows)
        train_df, test_df = per_repo_time_split(df_unsorted, test_frac=0.33)

        assert len(train_df) == 2
        assert len(test_df) == 1
        assert train_df["run_id"].tolist() == [1, 2]
        assert test_df["run_id"].tolist() == [3]

    def test_per_repo_time_split_equal_timestamps_tie_broken_by_run_id(self):
        """Verify equal timestamps within a repo are deterministically tie-broken by run_id."""
        rows = [
            {"run_id": 50, "run_timestamp": "2026-01-01 12:00:00", "repo": "r1", "label": 0},
            {"run_id": 20, "run_timestamp": "2026-01-01 12:00:00", "repo": "r1", "label": 1},
        ]
        df = pd.DataFrame(rows)
        # 2 rows with test_frac=0.5 -> 1 train, 1 test
        train_df, test_df = per_repo_time_split(df, test_frac=0.5)

        # Lower run_id (20) should be sorted first and land in train; higher (50) in test
        assert train_df.iloc[0]["run_id"] == 20
        assert test_df.iloc[0]["run_id"] == 50


class TestPerRepoTimeSeriesCV:
    """Test suite for per-repository time-series cross-validation splits."""

    def test_per_repo_time_series_cv_no_temporal_leakage_per_repo(self):
        """Verify on a synthetic 2-repo example that no fold's validation indices for a
        given repo include rows earlier than that same repo's training indices in that fold."""
        rows = []
        for i in range(1, 15):
            rows.append({
                "run_id": i,
                "run_timestamp": f"2026-01-{i:02d} 10:00:00",
                "repo": "repo_a",
                "label": i % 2,
            })
        for j in range(1, 15):
            rows.append({
                "run_id": 100 + j,
                "run_timestamp": f"2026-02-{j:02d} 10:00:00",
                "repo": "repo_b",
                "label": (j + 1) % 2,
            })

        df = pd.DataFrame(rows)
        folds = per_repo_time_series_cv(df, n_splits=3)
        assert len(folds) == 3

        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], utc=True)
        df_sorted = df.sort_values(by=["run_timestamp", "run_id"]).reset_index(drop=True)

        for fold_k, (train_idx, val_idx) in enumerate(folds):
            train_sub = df_sorted.iloc[train_idx]
            val_sub = df_sorted.iloc[val_idx]

            for repo in ["repo_a", "repo_b"]:
                repo_train = train_sub[train_sub["repo"] == repo]
                repo_val = val_sub[val_sub["repo"] == repo]

                assert len(repo_train) > 0
                assert len(repo_val) > 0

                # Crucial check: for this repo, validation timestamps strictly succeed training timestamps
                max_train_t = repo_train["run_timestamp"].max()
                min_val_t = repo_val["run_timestamp"].min()
                assert max_train_t < min_val_t


