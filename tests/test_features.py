"""Tests for feature engineering, label construction, and leakage invariants."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_historical_features,
    is_test_file,
)


class TestLeakageInvariants:
    """Critical tests verifying that historical features never leak current or future outcomes."""

    def test_author_past_features_strictly_historical(self):
        """Construct synthetic runs for 1 author across 5 distinct timestamps and check hand-calculated values."""
        df = pd.DataFrame({
            "run_id": [1, 2, 3, 4, 5],
            "run_timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-03 10:00:00",
                "2026-01-04 10:00:00",
                "2026-01-05 10:00:00",
            ],
            "author": ["alice", "alice", "alice", "alice", "alice"],
            "repo": ["org/repo"] * 5,
            "label": [0, 1, 1, 0, 1],  # pass, fail, fail, pass, fail
        })

        res = add_historical_features(df)

        # Hand-calculated verification:
        # Row 0 (t0, label=0):
        #   - 0 prior runs
        #   - No author history -> fallback to global prior (0.5 for first row)
        assert res.loc[0, "author_past_run_count"] == 0
        assert res.loc[0, "author_past_failure_rate"] == pytest.approx(0.5)

        # Row 1 (t1, label=1):
        #   - 1 prior run (Row 0, label 0)
        #   - past failure rate = 0 / 1 = 0.0
        assert res.loc[1, "author_past_run_count"] == 1
        assert res.loc[1, "author_past_failure_rate"] == pytest.approx(0.0)

        # Row 2 (t2, label=1):
        #   - 2 prior runs (Row 0: 0, Row 1: 1)
        #   - past failure rate = 1 / 2 = 0.5
        assert res.loc[2, "author_past_run_count"] == 2
        assert res.loc[2, "author_past_failure_rate"] == pytest.approx(0.5)

        # Row 3 (t3, label=0):
        #   - 3 prior runs (Row 0: 0, Row 1: 1, Row 2: 1)
        #   - past failure rate = 2 / 3 ≈ 0.666667
        assert res.loc[3, "author_past_run_count"] == 3
        assert res.loc[3, "author_past_failure_rate"] == pytest.approx(2.0 / 3.0)

        # Row 4 (t4, label=1):
        #   - 4 prior runs (Row 0: 0, Row 1: 1, Row 2: 1, Row 3: 0)
        #   - past failure rate = 2 / 4 = 0.5
        assert res.loc[4, "author_past_run_count"] == 4
        assert res.loc[4, "author_past_failure_rate"] == pytest.approx(0.5)

    def test_author_isolation_and_global_prior_fallback(self):
        """Verify multiple authors have isolated histories and first runs use global expanding prior."""
        df = pd.DataFrame({
            "run_id": [1, 2, 3, 4],
            "run_timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-03 10:00:00",
                "2026-01-04 10:00:00",
            ],
            "author": ["alice", "alice", "bob", "alice"],
            "repo": ["org/repo"] * 4,
            "label": [1, 1, 0, 0],
        })

        res = add_historical_features(df)

        # Bob's first run is at index 2.
        # Bob has 0 prior runs.
        assert res.loc[2, "author"] == "bob"
        assert res.loc[2, "author_past_run_count"] == 0

        # Prior runs before Bob are rows 0 and 1 (both failures, labels [1, 1]).
        # Bob's first-time failure rate must default to the prior global expanding rate: (1 + 1)/2 = 1.0
        assert res.loc[2, "author_past_failure_rate"] == pytest.approx(1.0)

        # Alice's run at index 3:
        # Prior runs for Alice are index 0 (1) and index 1 (1). (Bob's label 0 must not affect Alice's past rate)
        assert res.loc[3, "author"] == "alice"
        assert res.loc[3, "author_past_run_count"] == 2
        assert res.loc[3, "author_past_failure_rate"] == pytest.approx(1.0)

    def test_repo_recent_failure_rate_rolling_20(self):
        """Verify repo_recent_failure_rate strictly computes rolling 20 window excluding current row."""
        # Create 25 runs for one repo with alternating outcomes
        n_runs = 25
        labels = [(i % 2) for i in range(n_runs)]  # 0, 1, 0, 1, ...
        timestamps = [f"2026-01-{i+1:02d} 12:00:00" for i in range(n_runs)]

        df = pd.DataFrame({
            "run_id": list(range(1, n_runs + 1)),
            "run_timestamp": timestamps,
            "author": [f"user_{i}" for i in range(n_runs)],
            "repo": ["org/repo"] * n_runs,
            "label": labels,
        })

        res = add_historical_features(df)

        # Row 0: no previous runs in repo, uses fallback
        assert res.loc[0, "repo_recent_failure_rate"] == pytest.approx(0.5)

        # Row 1: only Row 0 in window (label=0) -> 0.0
        assert res.loc[1, "repo_recent_failure_rate"] == pytest.approx(0.0)

        # Row 20 (21st run, index 20):
        # Window of 20 runs strictly preceding index 20 (indices 0 to 19):
        # In labels[0:20], exactly 10 are 1 and 10 are 0 -> mean is 0.5
        expected_rate_20 = float(np.mean(labels[0:20]))
        assert res.loc[20, "repo_recent_failure_rate"] == pytest.approx(expected_rate_20)

        # Row 24 (25th run, index 24):
        # Preceding 20 runs are indices 4 to 23 (strictly excluding index 24):
        expected_rate_24 = float(np.mean(labels[4:24]))
        assert res.loc[24, "repo_recent_failure_rate"] == pytest.approx(expected_rate_24)

    def test_adversarial_identical_timestamps_tie_breaking(self):
        """Adversarial test: two runs with the exact same timestamp.

        Asserts that deterministic tie-breaking on run_id prevents mutual leakage.
        """
        same_time = "2026-01-01 12:00:00"
        df = pd.DataFrame({
            "run_id": [200, 100],  # deliberately out of order
            "run_timestamp": [same_time, same_time],
            "author": ["alice", "alice"],
            "repo": ["org/repo", "org/repo"],
            "label": [0, 1],  # run 200 is pass (0), run 100 is fail (1)
        })

        res = add_historical_features(df)

        # Sorted order should put run 100 first, run 200 second
        assert res.loc[0, "run_id"] == 100
        assert res.loc[1, "run_id"] == 200

        # Run 100 was first:
        # past run count is 0, does NOT see run 200
        assert res.loc[0, "author_past_run_count"] == 0

        # Run 200 was second:
        # past run count is 1, only includes run 100 (which had label 1)
        assert res.loc[1, "author_past_run_count"] == 1
        assert res.loc[1, "author_past_failure_rate"] == pytest.approx(1.0)


class TestFileHeuristics:
    """Test suite for test-file heuristic function."""

    @pytest.mark.parametrize(
        "filepath",
        [
            "tests/test_auth.py",
            "tests/unit/test_models.py",
            "test_client.py",
            "src/models_test.py",
            "specs/models/user_spec.rb",
            "src/components/Button.test.js",
            "src/components/Modal.spec.ts",
            "tests/e2e/test_checkout.py",
            "pkg/server/server_test.go",
        ],
    )
    def test_is_test_file_positive(self, filepath):
        """Assert true for various common test file patterns."""
        assert is_test_file(filepath) is True

    @pytest.mark.parametrize(
        "filepath",
        [
            "src/main.py",
            "src/collect/github_client.py",
            "README.md",
            "package.json",
            "docker-compose.yml",
            "components/Button.tsx",
            "controllers/users.rb",
        ],
    )
    def test_is_test_file_negative(self, filepath):
        """Assert false for non-test application files."""
        assert is_test_file(filepath) is False
