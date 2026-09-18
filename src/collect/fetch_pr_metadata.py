"""Module for fetching and caching pull request metadata and changed files."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm

from src.collect.github_client import GitHubClient, GitHubNotFoundError


def fetch_pr_metadata(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_numbers: Iterable[int],
    force: bool = False,
    data_dir: str = "data/raw",
) -> Dict[str, Dict[str, Any]]:
    """Fetch pull request metadata and changed files for a deduplicated list of PR numbers.

    Args:
        client: Authenticated GitHubClient instance.
        owner: Repository owner/organization.
        repo: Repository name.
        pr_numbers: Iterable of pull request numbers.
        force: If True, bypass cache and re-fetch from API.
        data_dir: Directory where raw JSON responses will be cached.

    Returns:
        Dict mapping string PR number to metadata including raw PR, raw files, and retained fields.
    """
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{owner}__{repo}__prs.json"

    # Deduplicate PR numbers upfront
    unique_prs = sorted(list({int(p) for p in pr_numbers if p is not None}))

    if cache_path.exists() and not force:
        print(f"[fetch_pr_metadata] Cache hit: {cache_path}. Skipping API fetch.")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return cached_data.get("prs", cached_data)

    print(f"[fetch_pr_metadata] Fetching metadata for {len(unique_prs)} unique PRs in {owner}/{repo}...")
    prs_result: Dict[str, Dict[str, Any]] = {}

    for pr_num in tqdm(unique_prs, desc=f"PR metadata ({owner}/{repo})"):
        pr_key = str(pr_num)
        try:
            # Fetch PR details
            pr_resp = client.get(f"/repos/{owner}/{repo}/pulls/{pr_num}")
            pr_data = pr_resp.json()

            # Fetch changed files
            files_url = f"/repos/{owner}/{repo}/pulls/{pr_num}/files"
            raw_files = list(client.get_paginated(files_url))

            filenames = [f.get("filename") for f in raw_files if isinstance(f, dict) and "filename" in f]

            # Retain required fields alongside raw responses
            prs_result[pr_key] = {
                "number": pr_data.get("number", pr_num),
                "user": pr_data.get("user", {}).get("login") if isinstance(pr_data.get("user"), dict) else None,
                "created_at": pr_data.get("created_at"),
                "additions": pr_data.get("additions"),
                "deletions": pr_data.get("deletions"),
                "changed_files": pr_data.get("changed_files"),
                "filenames": filenames,
                "raw_pr": pr_data,
                "raw_files": raw_files,
            }
        except GitHubNotFoundError:
            print(f"[fetch_pr_metadata] PR #{pr_num} not found (404). Skipping.")
            continue

    cache_payload = {
        "owner": owner,
        "repo": repo,
        "unique_prs_count": len(prs_result),
        "prs": prs_result,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2)

    print(f"[fetch_pr_metadata] Successfully cached {len(prs_result)} PRs to {cache_path}")
    return prs_result
