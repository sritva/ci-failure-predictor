"""Data collection modules for CI Failure Predictor."""

from src.collect.github_client import (
    GitHubClient,
    GitHubAPIError,
    GitHubAuthError,
    GitHubNotFoundError,
)
from src.collect.fetch_workflow_runs import fetch_workflow_runs
from src.collect.fetch_all_prs import fetch_all_prs
from src.collect.link_runs_to_prs import link_runs_to_prs
from src.collect.fetch_pr_metadata import fetch_pr_metadata

__all__ = [
    "GitHubClient",
    "GitHubAPIError",
    "GitHubAuthError",
    "GitHubNotFoundError",
    "fetch_workflow_runs",
    "fetch_all_prs",
    "link_runs_to_prs",
    "fetch_pr_metadata",
]
