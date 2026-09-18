"""Module for linking workflow runs to pull requests via commit head SHA."""

from typing import Any, Dict, List, Tuple


def link_runs_to_prs(
    runs: List[Dict[str, Any]],
    prs: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Dict[str, Any], int]], int]:
    """Link workflow runs to pull requests by matching run head_sha to PR head sha.

    Args:
        runs: List of workflow runs (filtered by event=pull_request).
        prs: List of all pull requests for the repository.

    Returns:
        Tuple containing:
            - List of (run_dict, matched_pr_number) tuples.
            - Count of runs that had no matching PR head_sha (e.g. force pushes).
    """
    # Build lookup map: PR head.sha -> PR number
    sha_to_pr: Dict[str, int] = {}
    for pr in prs:
        pr_number = pr.get("number")
        sha = pr.get("head_sha")
        if not sha and isinstance(pr.get("head"), dict):
            sha = pr["head"].get("sha")

        if sha and pr_number is not None:
            sha_to_pr[sha] = pr_number

    matched_pairs: List[Tuple[Dict[str, Any], int]] = []
    unmatched_count = 0

    for run in runs:
        run_sha = run.get("head_sha")
        if run_sha and run_sha in sha_to_pr:
            matched_pairs.append((run, sha_to_pr[run_sha]))
        else:
            unmatched_count += 1

    print(
        f"[link_runs_to_prs] Successfully matched {len(matched_pairs)} runs to PRs "
        f"({unmatched_count} runs had no matching PR head SHA)."
    )

    return matched_pairs, unmatched_count
