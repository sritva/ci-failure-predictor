"""Module for fetching and caching all pull requests for a repository."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.collect.github_client import GitHubClient


def fetch_all_prs(
    client: GitHubClient,
    owner: str,
    repo: str,
    max_prs: int = 2000,
    force: bool = False,
    data_dir: str = "data/raw",
) -> List[Dict[str, Any]]:
    """Fetch all pull requests (open, closed, merged) and cache the response.

    Args:
        client: Authenticated GitHubClient instance.
        owner: Repository owner/organization.
        repo: Repository name.
        max_prs: Maximum number of PRs to fetch.
        force: If True, bypass cache and re-fetch from API.
        data_dir: Directory where raw JSON responses will be cached.

    Returns:
        List of pull request dictionaries with retained fields.
    """
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{owner}__{repo}__all_prs.json"

    if cache_path.exists() and not force:
        print(f"[fetch_all_prs] Cache hit: {cache_path}. Skipping API fetch.")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            if isinstance(cached_data, dict) and "prs" in cached_data:
                return cached_data["prs"][:max_prs]
            if isinstance(cached_data, list):
                return cached_data[:max_prs]
            return []

    print(f"[fetch_all_prs] Fetching up to {max_prs} PRs for {owner}/{repo} (state=all, sort=created, desc)...")
    url = f"/repos/{owner}/{repo}/pulls"
    params = {
        "state": "all",
        "sort": "created",
        "direction": "desc",
        "per_page": min(100, max_prs),
    }

    raw_pages: List[List[Dict[str, Any]]] = []
    prs: List[Dict[str, Any]] = []
    current_url: Optional[str] = url
    current_params: Optional[Dict[str, Any]] = params

    while current_url and len(prs) < max_prs:
        resp = client.get(current_url, params=current_params)
        page_data = resp.json()

        if isinstance(page_data, list):
            raw_pages.append(page_data)
            for item in page_data:
                head = item.get("head") if isinstance(item.get("head"), dict) else {}
                head_sha = head.get("sha")
                user = item.get("user") if isinstance(item.get("user"), dict) else {}

                retained = {
                    "number": item.get("number"),
                    "user": user.get("login"),
                    "created_at": item.get("created_at"),
                    "additions": item.get("additions"),
                    "deletions": item.get("deletions"),
                    "changed_files": item.get("changed_files"),
                    "head_sha": head_sha,
                    "head": {"sha": head_sha},
                    "merged_at": item.get("merged_at"),
                    "state": item.get("state"),
                    "raw_pr": item,
                }
                prs.append(retained)
                if len(prs) >= max_prs:
                    break
        else:
            break

        if len(prs) >= max_prs or not page_data:
            break

        next_url = resp.links.get("next", {}).get("url")
        if next_url:
            current_url = next_url
            current_params = None
        else:
            break

    prs = prs[:max_prs]

    cache_payload = {
        "owner": owner,
        "repo": repo,
        "total_prs_cached": len(prs),
        "raw_pages": raw_pages,
        "prs": prs,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2)

    print(f"[fetch_all_prs] Successfully cached {len(prs)} PRs to {cache_path}")
    return prs
