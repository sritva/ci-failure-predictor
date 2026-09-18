"""Unit tests for GitHubClient, verifying pagination, rate-limiting, and error handling."""

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch
import pytest
import requests

from src.collect.github_client import (
    GitHubClient,
    GitHubAPIError,
    GitHubAuthError,
    GitHubNotFoundError,
)


@pytest.fixture
def client():
    """Fixture providing a GitHubClient with a test token."""
    return GitHubClient(token="ghp_testtoken12345", timeout=5, backoff_delay=0.01)


def create_mock_response(
    status_code: int = 200,
    json_data: Any = None,
    headers: dict = None,
    links: dict = None,
    text: str = "",
) -> MagicMock:
    """Helper to construct a mock Response object."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.url = "https://api.github.com/test"
    resp.text = text

    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.return_value = {}

    resp.headers = requests.structures.CaseInsensitiveDict(headers or {})
    resp.links = links or {}
    return resp


class TestGitHubClient:
    """Test suite for GitHubClient methods."""

    def test_pagination_follows_link_header_across_pages(self, client):
        """Test that get_paginated follows Link header and yields all items."""
        page1_response = create_mock_response(
            status_code=200,
            json_data=[{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}],
            links={"next": {"url": "https://api.github.com/test?page=2", "rel": "next"}},
        )
        page2_response = create_mock_response(
            status_code=200,
            json_data=[{"id": 3, "name": "item3"}],
            links={},
        )

        with patch.object(client.session, "get", side_effect=[page1_response, page2_response]):
            items = list(client.get_paginated("https://api.github.com/test"))

        assert len(items) == 3
        assert [item["id"] for item in items] == [1, 2, 3]

    def test_pagination_with_nested_dict_response(self, client):
        """Test that get_paginated extracts items from dictionary response (e.g., workflow_runs)."""
        page1_response = create_mock_response(
            status_code=200,
            json_data={
                "total_count": 2,
                "workflow_runs": [{"id": 101, "name": "CI"}, {"id": 102, "name": "Lint"}],
            },
            links={},
        )

        with patch.object(client.session, "get", return_value=page1_response):
            items = list(client.get_paginated("/repos/owner/repo/actions/runs"))

        assert len(items) == 2
        assert items[0]["id"] == 101
        assert items[1]["id"] == 102

    @patch("time.sleep")
    def test_rate_limit_triggers_sleep_when_remaining_is_low(self, mock_sleep, client):
        """Test that client sleeps when X-RateLimit-Remaining drops below threshold (50)."""
        future_reset = int(time.time()) + 120
        mock_response = create_mock_response(
            status_code=200,
            json_data={"message": "ok"},
            headers={
                "X-RateLimit-Remaining": "35",
                "X-RateLimit-Reset": str(future_reset),
            },
        )

        with patch.object(client.session, "get", return_value=mock_response):
            client.get("/test")

        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        # Should be approximately (future_reset - now) + 5
        assert 115 <= sleep_arg <= 130

    @patch("time.sleep")
    def test_rate_limit_does_not_trigger_sleep_when_remaining_is_sufficient(self, mock_sleep, client):
        """Test that client does not sleep when quota is safely above threshold."""
        mock_response = create_mock_response(
            status_code=200,
            json_data={"message": "ok"},
            headers={
                "X-RateLimit-Remaining": "4900",
                "X-RateLimit-Reset": "1700000000",
            },
        )

        with patch.object(client.session, "get", return_value=mock_response):
            client.get("/test")

        mock_sleep.assert_not_called()

    def test_401_raises_specific_github_auth_error(self, client):
        """Test that HTTP 401 raises GitHubAuthError with informative details."""
        mock_response = create_mock_response(
            status_code=401,
            json_data={"message": "Bad credentials", "documentation_url": "https://docs.github.com"},
        )

        with patch.object(client.session, "get", return_value=mock_response):
            with pytest.raises(GitHubAuthError) as exc_info:
                client.get("/user")

            err_msg = str(exc_info.value)
            assert "401" in err_msg
            assert "Bad credentials" in err_msg
            assert "GITHUB_TOKEN" in err_msg

    def test_404_raises_specific_github_not_found_error(self, client):
        """Test that HTTP 404 raises GitHubNotFoundError."""
        mock_response = create_mock_response(
            status_code=404,
            json_data={"message": "Not Found"},
        )

        with patch.object(client.session, "get", return_value=mock_response):
            with pytest.raises(GitHubNotFoundError) as exc_info:
                client.get("/repos/owner/nonexistent")

            assert "404" in str(exc_info.value)
            assert "Not Found" in str(exc_info.value)

    @patch("time.sleep")
    def test_retry_once_on_5xx_server_error(self, mock_sleep, client):
        """Test that client retries once on 5xx server errors and succeeds if second attempt is 200."""
        fail_response = create_mock_response(status_code=502, text="Bad Gateway")
        ok_response = create_mock_response(status_code=200, json_data={"status": "recovered"})

        with patch.object(client.session, "get", side_effect=[fail_response, ok_response]) as mock_get:
            resp = client.get("/test")

        assert mock_get.call_count == 2
        assert resp.status_code == 200
        assert resp.json() == {"status": "recovered"}
        mock_sleep.assert_called_once_with(client.backoff_delay)


class TestFetchWorkflowRuns:
    """Test suite for fetch_workflow_runs caching and fetching logic."""

    def test_fetch_workflow_runs_caches_and_returns_runs(self, tmp_path):
        """Test fetching workflow runs writes to cache and returns run list."""
        from src.collect.fetch_workflow_runs import fetch_workflow_runs

        mock_client = MagicMock()
        mock_response = create_mock_response(
            status_code=200,
            json_data={
                "total_count": 2,
                "workflow_runs": [
                    {"id": 1, "name": "CI", "conclusion": "success", "pull_requests": [{"number": 10}]},
                    {"id": 2, "name": "CI", "conclusion": "failure", "pull_requests": [{"number": 10}]},
                ],
            },
            links={},
        )
        mock_client.get.return_value = mock_response

        runs = fetch_workflow_runs(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            max_runs=10,
            force=False,
            data_dir=str(tmp_path),
        )

        assert len(runs) == 2
        assert runs[0]["id"] == 1
        assert runs[1]["id"] == 2

        cache_file = tmp_path / "testowner__testrepo__runs.json"
        assert cache_file.exists()

        # Second call should hit cache without invoking mock_client.get
        mock_client.get.reset_mock()
        cached_runs = fetch_workflow_runs(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            max_runs=10,
            force=False,
            data_dir=str(tmp_path),
        )
        mock_client.get.assert_not_called()
        assert len(cached_runs) == 2

    def test_fetch_workflow_runs_force_bypasses_cache(self, tmp_path):
        """Test force=True bypasses existing cache and calls API."""
        from src.collect.fetch_workflow_runs import fetch_workflow_runs

        cache_file = tmp_path / "testowner__testrepo__runs.json"
        cache_file.write_text('{"runs": [{"id": 999}]}', encoding="utf-8")

        mock_client = MagicMock()
        mock_response = create_mock_response(
            status_code=200,
            json_data={"total_count": 1, "workflow_runs": [{"id": 1000}]},
            links={},
        )
        mock_client.get.return_value = mock_response

        runs = fetch_workflow_runs(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            max_runs=10,
            force=True,
            data_dir=str(tmp_path),
        )
        mock_client.get.assert_called_once()
        assert runs[0]["id"] == 1000


class TestFetchPRMetadata:
    """Test suite for fetch_pr_metadata deduplication and caching."""

    def test_fetch_pr_metadata_deduplicates_pr_numbers(self, tmp_path):
        """Test that duplicate PR numbers are only queried once."""
        from src.collect.fetch_pr_metadata import fetch_pr_metadata

        mock_client = MagicMock()
        pr_response = create_mock_response(
            status_code=200,
            json_data={
                "number": 42,
                "user": {"login": "octocat"},
                "created_at": "2026-01-01T00:00:00Z",
                "additions": 10,
                "deletions": 5,
                "changed_files": 2,
            },
        )
        mock_client.get.return_value = pr_response
        mock_client.get_paginated.return_value = iter([
            {"filename": "src/app.py"},
            {"filename": "tests/test_app.py"},
        ])

        # Pass duplicated PR numbers [42, 42, 42]
        result = fetch_pr_metadata(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            pr_numbers=[42, 42, 42],
            force=False,
            data_dir=str(tmp_path),
        )

        # Should only call GET /repos/testowner/testrepo/pulls/42 once
        assert mock_client.get.call_count == 1
        assert "42" in result
        pr_entry = result["42"]
        assert pr_entry["number"] == 42
        assert pr_entry["user"] == "octocat"
        assert pr_entry["filenames"] == ["src/app.py", "tests/test_app.py"]

        cache_file = tmp_path / "testowner__testrepo__prs.json"
        assert cache_file.exists()


class TestLinkRunsToPRs:
    """Test suite for linking workflow runs to PRs via head_sha."""

    def test_link_runs_to_prs_synthetic_match_and_non_match(self):
        """Test with 2 runs, 2 PRs: one match, one deliberate non-match."""
        from src.collect.link_runs_to_prs import link_runs_to_prs

        runs = [
            {"id": 101, "head_sha": "sha_matched_123", "conclusion": "failure"},
            {"id": 102, "head_sha": "sha_unmatched_456", "conclusion": "success"},
        ]

        prs = [
            {"number": 42, "head_sha": "sha_matched_123", "head": {"sha": "sha_matched_123"}},
            {"number": 43, "head_sha": "sha_other_789", "head": {"sha": "sha_other_789"}},
        ]

        matched_pairs, unmatched_count = link_runs_to_prs(runs, prs)

        # Assert matched pair is correct
        assert len(matched_pairs) == 1
        assert matched_pairs[0] == (runs[0], 42)

        # Assert unmatched count is 1
        assert unmatched_count == 1


class TestFetchAllPRs:
    """Test suite for fetch_all_prs caching and pagination."""

    def test_fetch_all_prs_caches_and_returns_retained_fields(self, tmp_path):
        """Test fetching all PRs retains required fields and writes to cache."""
        from src.collect.fetch_all_prs import fetch_all_prs

        mock_client = MagicMock()
        mock_response = create_mock_response(
            status_code=200,
            json_data=[
                {
                    "number": 1,
                    "user": {"login": "alice"},
                    "created_at": "2026-01-01T00:00:00Z",
                    "additions": 10,
                    "deletions": 2,
                    "changed_files": 1,
                    "head": {"sha": "sha_pr_1"},
                    "merged_at": "2026-01-02T00:00:00Z",
                    "state": "closed",
                },
                {
                    "number": 2,
                    "user": {"login": "bob"},
                    "created_at": "2026-01-03T00:00:00Z",
                    "additions": 5,
                    "deletions": 0,
                    "changed_files": 1,
                    "head": {"sha": "sha_pr_2"},
                    "merged_at": None,
                    "state": "open",
                },
            ],
            links={},
        )
        mock_client.get.return_value = mock_response

        prs = fetch_all_prs(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            max_prs=10,
            force=False,
            data_dir=str(tmp_path),
        )

        assert len(prs) == 2
        assert prs[0]["number"] == 1
        assert prs[0]["user"] == "alice"
        assert prs[0]["head_sha"] == "sha_pr_1"
        assert prs[0]["state"] == "closed"

        assert prs[1]["number"] == 2
        assert prs[1]["user"] == "bob"
        assert prs[1]["head_sha"] == "sha_pr_2"

        cache_file = tmp_path / "testowner__testrepo__all_prs.json"
        assert cache_file.exists()

        # Cache hit test
        mock_client.get.reset_mock()
        cached_prs = fetch_all_prs(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            max_prs=10,
            force=False,
            data_dir=str(tmp_path),
        )
        mock_client.get.assert_not_called()
        assert len(cached_prs) == 2


class TestRunCollection:
    """Test suite for run_collection top-level entry point."""

    @patch("src.collect.run_collection.fetch_pr_metadata")
    @patch("src.collect.run_collection.link_runs_to_prs")
    @patch("src.collect.run_collection.fetch_all_prs")
    @patch("src.collect.run_collection.fetch_workflow_runs")
    def test_run_collection_end_to_end_summary(
        self,
        mock_fetch_runs,
        mock_fetch_all_prs,
        mock_link_runs,
        mock_fetch_pr_metadata,
        monkeypatch,
        capsys,
    ):
        """Test run_collection processes repos, links by head_sha, and prints class balance summary."""
        from src.collect.run_collection import run_collection

        monkeypatch.setenv("TARGET_REPOS", "testowner/testrepo")

        run1 = {"id": 1, "head_sha": "sha1", "conclusion": "failure"}
        run2 = {"id": 2, "head_sha": "sha2", "conclusion": "success"}
        run3 = {"id": 3, "head_sha": "sha3_unmatched", "conclusion": "failure"}

        mock_fetch_runs.return_value = [run1, run2, run3]
        mock_fetch_all_prs.return_value = [
            {"number": 101, "head_sha": "sha1"},
            {"number": 102, "head_sha": "sha2"},
        ]
        mock_link_runs.return_value = ([(run1, 101), (run2, 102)], 1)
        mock_fetch_pr_metadata.return_value = {
            "101": {"number": 101, "filenames": ["test.py"]},
            "102": {"number": 102, "filenames": ["main.py"]},
        }

        run_collection(force=False, max_runs=10)

        captured = capsys.readouterr().out
        assert "COLLECTION SUMMARY REPORT" in captured
        assert "Repository: testowner/testrepo" in captured
        assert "Total Workflow Runs Fetched: 3" in captured
        assert "Runs Matched to a PR:        2" in captured
        assert "Unmatched Runs (e.g. force-push): 1" in captured
        assert "Total Unique Matched PRs:    2" in captured
        assert "failure: 1 (50.0%)" in captured
        assert "success: 1 (50.0%)" in captured



