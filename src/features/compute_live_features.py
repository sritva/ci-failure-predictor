"""Module for computing live, leakage-free features for a GitHub pull request.

Fetches current PR metadata, changed files, and historical workflow run performance
at inference time, conforming to PRFeaturesRequest's schema.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.collect.github_client import GitHubClient
from src.features.build_features import is_test_file

logger = logging.getLogger("compute_live_features")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
COMBINED_FEATURES_PARQUET = BASE_DIR / "data" / "processed" / "combined_features.parquet"

# Last-resort fallback prior if parquet file is missing from local filesystem
DEFAULT_GLOBAL_PRIOR = 0.1540


def get_global_prior(parquet_path: Optional[Path] = None) -> float:
    """Compute the global failure prior from historical combined_features.parquet.

    NOTE ON FALLBACK PRIOR:
    This value is derived dynamically from `combined_features.parquet` whenever available.
    If the dataset is regenerated with more data or over a different time window, this
    prior will automatically reflect the updated historical distribution. The hardcoded
    constant `DEFAULT_GLOBAL_PRIOR` (0.1540) is strictly a fallback if the file is missing.
    """
    path = parquet_path or COMBINED_FEATURES_PARQUET
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if "label" in df.columns and len(df) > 0:
                return float(df["label"].mean())
        except Exception as exc:
            logger.warning("Could not read global prior from %s: %s", path, exc)
    return DEFAULT_GLOBAL_PRIOR


def compute_live_pr_features(
    owner: str,
    repo: str,
    pr_number: int,
    github_token: Optional[str] = None,
    now_dt: Optional[datetime] = None,
    parquet_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fetch live metadata for a pull request and compute leak-free features.

    Args:
        owner: Repository owner / organization.
        repo: Repository name.
        pr_number: Pull request number.
        github_token: Optional GitHub personal access token (uses GITHUB_TOKEN env if None).
        now_dt: Evaluation cutoff timestamp in UTC. Defaults to datetime.now(timezone.utc).
        parquet_path: Optional path to combined_features.parquet for global prior.

    Returns:
        Dict conforming to PRFeaturesRequest schema.
    """
    client = GitHubClient(token=github_token)

    pr_resp = client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    pr_data = pr_resp.json()

    forbidden_keys = {"conclusion", "label", "outcome", "status_conclusion"}
    target_pr_leaks = [k for k in forbidden_keys if k in pr_data]
    if target_pr_leaks:
        logger.warning(
            "Target PR payload contains outcome-like fields (%s); strictly ignoring them.",
            target_pr_leaks,
        )

    author = ""
    if isinstance(pr_data.get("user"), dict):
        author = pr_data["user"].get("login", "")
    elif pr_data.get("user"):
        author = str(pr_data.get("user"))

    additions = int(pr_data.get("additions", 0) or 0)
    deletions = int(pr_data.get("deletions", 0) or 0)
    diff_size = max(0, additions + deletions)
    files_changed = int(pr_data.get("changed_files", 0) or 0)

    created_at_raw = pr_data.get("created_at")
    if created_at_raw:
        pr_created_dt = pd.to_datetime(created_at_raw, utc=True)
    else:
        pr_created_dt = datetime.now(timezone.utc)

    time_of_day = int(pr_created_dt.hour)
    day_of_week = int(pr_created_dt.dayofweek)
    is_weekend = bool(day_of_week >= 5)

    files_url = f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
    raw_files = list(client.get_paginated(files_url))
    test_files_changed = 0
    for f in raw_files:
        filename = f.get("filename") if isinstance(f, dict) else ""
        if filename and is_test_file(filename):
            test_files_changed += 1

    touches_test_files = bool(test_files_changed > 0)

    # CRITICAL - HISTORICAL CUTOFF INVARIANT:
    # The historical cutoff for author_past_run_count, author_past_failure_rate,
    # and repo_recent_failure_rate MUST be the current evaluation moment ("now",
    # via datetime.now(timezone.utc)), NOT the PR's original created_at.
    # A PR can remain open for days or weeks while new commits land and intermediate
    # CI builds finish; using created_at would arbitrarily discard valid, safely-available
    # history that occurred before this inference call. Using "now" is completely
    # leakage-safe because "now" is the exact moment the prediction is requested.
    cutoff_dt = now_dt if now_dt is not None else datetime.now(timezone.utc)
    if not hasattr(cutoff_dt, "tzinfo") or cutoff_dt.tzinfo is None:
        cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)

    global_prior = get_global_prior(parquet_path)

    repo_runs_resp = client.get(
        f"/repos/{owner}/{repo}/actions/runs",
        params={"event": "pull_request", "status": "completed", "per_page": 100},
    )
    repo_runs_data = repo_runs_resp.json()
    raw_repo_runs = repo_runs_data.get("workflow_runs", [])

    valid_prior_repo_runs: List[Dict[str, Any]] = []
    for r in raw_repo_runs:
        if not isinstance(r, dict):
            continue
        created_str = r.get("created_at")
        conclusion = r.get("conclusion")
        if not created_str or conclusion not in ("success", "failure"):
            continue
        r_dt = pd.to_datetime(created_str, utc=True)
        if r_dt < cutoff_dt:
            valid_prior_repo_runs.append(r)

    valid_prior_repo_runs.sort(
        key=lambda r: pd.to_datetime(r.get("created_at"), utc=True),
        reverse=True,
    )

    recent_20_runs = valid_prior_repo_runs[:20]
    if recent_20_runs:
        repo_failures = sum(1 for r in recent_20_runs if r.get("conclusion") == "failure")
        repo_recent_failure_rate = round(float(repo_failures / len(recent_20_runs)), 4)
    else:
        repo_recent_failure_rate = round(float(global_prior), 4)

    author_runs: List[Dict[str, Any]] = []
    if author:
        try:
            author_runs_resp = client.get(
                f"/repos/{owner}/{repo}/actions/runs",
                params={
                    "event": "pull_request",
                    "actor": author,
                    "status": "completed",
                    "per_page": 100,
                },
            )
            raw_author_runs = author_runs_resp.json().get("workflow_runs", [])
            for r in raw_author_runs:
                if not isinstance(r, dict):
                    continue
                created_str = r.get("created_at")
                conclusion = r.get("conclusion")
                if not created_str or conclusion not in ("success", "failure"):
                    continue
                r_dt = pd.to_datetime(created_str, utc=True)
                if r_dt < cutoff_dt:
                    author_runs.append(r)
        except Exception as exc:
            logger.warning("Failed to fetch author-specific workflow runs: %s", exc)

    if not author_runs and author:
        for r in valid_prior_repo_runs:
            actor = (r.get("actor") or {}).get("login") or (r.get("triggering_actor") or {}).get("login")
            if actor == author:
                author_runs.append(r)

    author_past_run_count = len(author_runs)
    if author_past_run_count > 0:
        author_failures = sum(1 for r in author_runs if r.get("conclusion") == "failure")
        author_past_failure_rate = round(float(author_failures / author_past_run_count), 4)
    else:
        author_past_failure_rate = round(float(global_prior), 4)

    return {
        "diff_size": diff_size,
        "files_changed": files_changed,
        "test_files_changed": test_files_changed,
        "touches_test_files": touches_test_files,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "author_past_run_count": author_past_run_count,
        "author_past_failure_rate": author_past_failure_rate,
        "repo_recent_failure_rate": repo_recent_failure_rate,
    }
