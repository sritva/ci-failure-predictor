"""Tests for live feature computation and PR scoring."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from src.features.compute_live_features import compute_live_pr_features, get_global_prior
from src.service.schema import PRFeaturesRequest
from src.service.score_pr import find_a_live_open_pr, format_risk_comment


@pytest.fixture
def mock_pr_response():
    """Sample mock PR metadata response."""
    return {
        "number": 123,
        "title": "Add feature X",
        "user": {"login": "test_developer"},
        "created_at": "2026-09-10T10:00:00Z",
        "additions": 150,
        "deletions": 50,
        "changed_files": 4,
    }


@pytest.fixture
def mock_files_response():
    """Sample mock PR changed files response."""
    return [
        {"filename": "src/core.py"},
        {"filename": "src/utils.py"},
        {"filename": "tests/test_core.py"},
        {"filename": "docs/index.md"},
    ]


@pytest.fixture
def mock_repo_runs():
    """Sample mock completed workflow runs for a repository."""
    runs = []
    # 10 recent completed runs: 2 failures, 8 successes
    for i in range(10):
        runs.append({
            "id": 1000 + i,
            "created_at": f"2026-09-1{i % 5}T12:00:00Z",
            "status": "completed",
            "conclusion": "failure" if i in (1, 5) else "success",
            "actor": {"login": "test_developer" if i < 3 else f"other_user_{i}"},
        })
    return runs


def test_compute_live_pr_features_schema_match(mock_pr_response, mock_files_response, mock_repo_runs):
    """Verify returned dictionary keys exactly match PRFeaturesRequest fields."""
    with patch("src.features.compute_live_features.GitHubClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client

        client.get.side_effect = [
            MagicMock(json=lambda: mock_pr_response),
            MagicMock(json=lambda: {"workflow_runs": mock_repo_runs}),
            MagicMock(json=lambda: {"workflow_runs": [r for r in mock_repo_runs if r["actor"]["login"] == "test_developer"]}),
        ]
        client.get_paginated.return_value = iter(mock_files_response)

        features = compute_live_pr_features(
            owner="pallets",
            repo="flask",
            pr_number=123,
            now_dt=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        )

        expected_keys = set(PRFeaturesRequest.model_fields.keys())
        assert set(features.keys()) == expected_keys

        # Validate with Pydantic model directly
        validated_request = PRFeaturesRequest(**features)
        assert validated_request.diff_size == 200
        assert validated_request.files_changed == 4
        assert validated_request.test_files_changed == 1
        assert validated_request.touches_test_files is True


def test_first_time_author_global_prior_fallback(mock_pr_response, mock_files_response):
    """Verify a first-time author (0 past runs) falls back to the historical global prior."""
    with patch("src.features.compute_live_features.GitHubClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client

        # Repo has runs, but none by this author
        repo_runs = [
            {"id": 1, "created_at": "2026-09-15T12:00:00Z", "status": "completed", "conclusion": "success", "actor": {"login": "veteran"}},
            {"id": 2, "created_at": "2026-09-16T12:00:00Z", "status": "completed", "conclusion": "failure", "actor": {"login": "veteran"}},
        ]

        client.get.side_effect = [
            MagicMock(json=lambda: mock_pr_response),
            MagicMock(json=lambda: {"workflow_runs": repo_runs}),
            MagicMock(json=lambda: {"workflow_runs": []}),  # No author runs found
        ]
        client.get_paginated.return_value = iter(mock_files_response)

        features = compute_live_pr_features(
            owner="pallets",
            repo="flask",
            pr_number=123,
            now_dt=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        )

        assert features["author_past_run_count"] == 0
        expected_prior = get_global_prior()
        assert features["author_past_failure_rate"] == pytest.approx(expected_prior, abs=0.001)
        assert features["author_past_failure_rate"] not in (0.0, 1.0)


def test_now_vs_created_at_cutoff_distinction(mock_pr_response, mock_files_response):
    """Verify historical cutoff uses 'now' rather than PR 'created_at', capturing intermediate runs."""
    # PR created on 2026-09-01 (16 days prior)
    old_pr = dict(mock_pr_response)
    old_pr["created_at"] = "2026-09-01T10:00:00Z"
    old_pr_created_dt = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

    # Intermediate completed run happened on 2026-09-10 (AFTER PR created_at, but BEFORE evaluation 'now')
    intermediate_run = {
        "id": 555,
        "created_at": "2026-09-10T15:00:00Z",
        "status": "completed",
        "conclusion": "failure",
        "actor": {"login": "test_developer"},
    }

    # Evaluation moment 'now' is 2026-09-18
    evaluation_now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)

    with patch("src.features.compute_live_features.GitHubClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client

        # 1. Evaluate with correct cutoff: evaluation_now
        client.get.side_effect = [
            MagicMock(json=lambda: old_pr),
            MagicMock(json=lambda: {"workflow_runs": [intermediate_run]}),
            MagicMock(json=lambda: {"workflow_runs": [intermediate_run]}),
        ]
        client.get_paginated.return_value = iter(mock_files_response)

        features_with_now = compute_live_pr_features(
            owner="pallets",
            repo="flask",
            pr_number=123,
            now_dt=evaluation_now,
        )

        # 2. Evaluate with buggy cutoff: old_pr_created_dt
        client.get.side_effect = [
            MagicMock(json=lambda: old_pr),
            MagicMock(json=lambda: {"workflow_runs": [intermediate_run]}),
            MagicMock(json=lambda: {"workflow_runs": [intermediate_run]}),
        ]
        client.get_paginated.return_value = iter(mock_files_response)

        features_with_created_at = compute_live_pr_features(
            owner="pallets",
            repo="flask",
            pr_number=123,
            now_dt=old_pr_created_dt,
        )

        # Proves distinction matters:
        # With evaluation_now: intermediate run is counted (count=1, rate=1.0)
        assert features_with_now["author_past_run_count"] == 1
        assert features_with_now["author_past_failure_rate"] == 1.0
        assert features_with_now["repo_recent_failure_rate"] == 1.0

        # With created_at cutoff: intermediate run was dropped as 'future', resulting in 0 past runs
        assert features_with_created_at["author_past_run_count"] == 0
        assert features_with_created_at["author_past_failure_rate"] != 1.0


def test_adversarial_target_pr_contains_outcome_fields(mock_pr_response, mock_files_response, mock_repo_runs):
    """Verify function strictly ignores adversarial outcome/conclusion fields in target PR payload."""
    adversarial_pr = dict(mock_pr_response)
    adversarial_pr["conclusion"] = "failure"
    adversarial_pr["label"] = 1
    adversarial_pr["outcome"] = "failure"

    with patch("src.features.compute_live_features.GitHubClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client

        client.get.side_effect = [
            MagicMock(json=lambda: adversarial_pr),
            MagicMock(json=lambda: {"workflow_runs": mock_repo_runs}),
            MagicMock(json=lambda: {"workflow_runs": []}),
        ]
        client.get_paginated.return_value = iter(mock_files_response)

        features = compute_live_pr_features(
            owner="pallets",
            repo="flask",
            pr_number=123,
            now_dt=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        )

        # None of the forbidden outcome fields must ever be present
        for forbidden in ("conclusion", "label", "outcome", "status_conclusion"):
            assert forbidden not in features


def test_find_a_live_open_pr_success():
    """Verify find_a_live_open_pr returns the first open PR number."""
    with patch("src.service.score_pr.GitHubClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.return_value = MagicMock(json=lambda: [{"number": 5918, "title": "Sample PR"}])

        pr_num = find_a_live_open_pr("pallets", "flask")
        assert pr_num == 5918


def test_find_a_live_open_pr_no_open_prs():
    """Verify find_a_live_open_pr raises ValueError when repo has no open PRs."""
    with patch("src.service.score_pr.GitHubClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.return_value = MagicMock(json=lambda: [])

        with pytest.raises(ValueError, match="currently has zero open pull requests"):
            find_a_live_open_pr("empty", "repo")


def test_format_risk_comment_contains_sticky_marker():
    """Verify generated comment includes the sticky comment HTML tag and readable content."""
    pred = {
        "failure_risk_score": 0.825,
        "risk_label": "high",
        "top_risk_factors": [
            {"feature": "repo_recent_failure_rate", "contribution": 0.81},
            {"feature": "author_past_failure_rate", "contribution": 0.65},
            {"feature": "diff_size", "contribution": 0.32},
        ],
        "model_version": "1.0.0-xgb",
    }
    feats = {
        "diff_size": 450,
        "repo_recent_failure_rate": 0.45,
        "author_past_failure_rate": 0.50,
    }

    comment = format_risk_comment(pred, feats, "pallets", "flask", 5918)
    assert "<!-- ci-failure-predictor-comment -->" in comment
    assert "HIGH RISK" in comment
    assert "82.5%" in comment
    assert "Recent CI instability in this repository" in comment
    assert "1.0.0-xgb" in comment
