"""Module for fetching and caching raw GitHub Actions workflow runs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.collect.github_client import GitHubClient


def fetch_workflow_runs(
    client: GitHubClient,
    owner: str,
    repo: str,
    max_runs: int = 2000,
    event: str = "pull_request",
    force: bool = False,
    data_dir: str = "data/raw",
) -> List[Dict[str, Any]]:
    """Fetch GitHub Actions workflow runs and cache the full raw API response.

    Args:
        client: Authenticated GitHubClient instance.
        owner: Repository owner/organization.
        repo: Repository name.
        max_runs: Maximum number of workflow runs to fetch.
        event: GitHub event to filter by (default: 'pull_request').
        force: If True, bypass cache and re-fetch from API.
        data_dir: Directory where raw JSON responses will be cached.

    Returns:
        List of workflow run dictionaries.
    """
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{owner}__{repo}__runs.json"

    if cache_path.exists() and not force:
        print(f"[fetch_workflow_runs] Cache hit: {cache_path}. Skipping API fetch.")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            if isinstance(cached_data, dict) and "runs" in cached_data:
                return cached_data["runs"][:max_runs]
            if isinstance(cached_data, list):
                return cached_data[:max_runs]
            return cached_data.get("workflow_runs", [])[:max_runs]

    print(f"[fetch_workflow_runs] Fetching up to {max_runs} (event={event}) workflow runs for {owner}/{repo}...")
    url = f"/repos/{owner}/{repo}/actions/runs"
    params = {"per_page": min(100, max_runs), "event": event}

    raw_pages: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    current_url: Optional[str] = url
    current_params: Optional[Dict[str, Any]] = params

    while current_url and len(runs) < max_runs:
        resp = client.get(current_url, params=current_params)
        page_data = resp.json()
        raw_pages.append(page_data)

        page_runs = page_data.get("workflow_runs", [])
        runs.extend(page_runs)

        if len(runs) >= max_runs or not page_runs:
            break

        # Follow next page link
        next_url = resp.links.get("next", {}).get("url")
        if next_url:
            current_url = next_url
            current_params = None  # Link header already encodes params
        else:
            break

    runs = runs[:max_runs]

    # Save full raw JSON pages and runs without trimming
    cache_payload = {
        "owner": owner,
        "repo": repo,
        "total_runs_cached": len(runs),
        "raw_pages": raw_pages,
        "runs": runs,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2)

    print(f"[fetch_workflow_runs] Successfully cached {len(runs)} raw runs to {cache_path}")
    return runs
