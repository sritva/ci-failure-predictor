"""Top-level collection runner for CI Failure Predictor."""

import argparse
import os
import sys
from collections import Counter
from typing import List, Set
from dotenv import load_dotenv

from src.collect.github_client import GitHubClient
from src.collect.fetch_workflow_runs import fetch_workflow_runs
from src.collect.fetch_all_prs import fetch_all_prs
from src.collect.link_runs_to_prs import link_runs_to_prs
from src.collect.fetch_pr_metadata import fetch_pr_metadata


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch and cache workflow runs and PR metadata for target repositories."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass cache and force re-fetching of all data from GitHub API.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=2000,
        help="Maximum number of workflow runs to fetch per repository (default: 2000).",
    )
    return parser.parse_args(args)


def run_collection(force: bool = False, max_runs: int = 2000) -> None:
    """Execute data collection for all configured repositories."""
    load_dotenv()
    target_repos_env = os.getenv("TARGET_REPOS", "")
    if not target_repos_env.strip():
        print(
            "ERROR: TARGET_REPOS not set or empty in environment / .env file.\n"
            "Please copy .env.example to .env and configure TARGET_REPOS (e.g., owner/repo1,owner/repo2)."
        )
        sys.exit(1)

    repo_list = [r.strip() for r in target_repos_env.split(",") if r.strip()]
    client = GitHubClient()

    overall_runs = 0
    overall_matched_runs = 0
    overall_unmatched_runs = 0
    overall_unique_prs = 0
    repo_summaries = []

    print(f"\n================ CI FAILURE PREDICTOR: DATA COLLECTION ================")
    print(f"Target Repositories: {', '.join(repo_list)}")
    print(f"Force re-fetch: {force}")
    print(f"Max runs per repo: {max_runs}\n")

    for repo_full in repo_list:
        if "/" not in repo_full:
            print(f"Skipping invalid repository identifier: '{repo_full}'. Expected 'owner/repo'.")
            continue

        owner, repo = repo_full.split("/", 1)
        print(f"\n--- Processing {owner}/{repo} ---")

        # 1. Fetch pull_request-event runs
        runs = fetch_workflow_runs(
            client=client,
            owner=owner,
            repo=repo,
            max_runs=max_runs,
            event="pull_request",
            force=force,
        )

        # 2. Fetch all PRs
        prs = fetch_all_prs(
            client=client,
            owner=owner,
            repo=repo,
            max_prs=max_runs,
            force=force,
        )

        # 3. Link runs to PRs via head_sha
        matched_pairs, unmatched_count = link_runs_to_prs(runs=runs, prs=prs)
        matched_runs = [run for run, _ in matched_pairs]
        matched_pr_numbers = sorted(list({pr_num for _, pr_num in matched_pairs}))

        # 4. Fetch file-change data only for matched PR numbers
        pr_metadata = fetch_pr_metadata(
            client=client,
            owner=owner,
            repo=repo,
            pr_numbers=matched_pr_numbers,
            force=force,
        )

        # 5. Class balance inspection among MATCHED runs only
        conclusions = [run.get("conclusion") for run in matched_runs]
        conclusion_counts = Counter(conclusions)

        total_repo_runs = len(runs)
        total_matched = len(matched_pairs)

        overall_runs += total_repo_runs
        overall_matched_runs += total_matched
        overall_unmatched_runs += unmatched_count
        overall_unique_prs += len(matched_pr_numbers)

        repo_summaries.append({
            "repo": repo_full,
            "total_runs": total_repo_runs,
            "matched_runs": total_matched,
            "unmatched_runs": unmatched_count,
            "matched_prs_count": len(matched_pr_numbers),
            "conclusions": conclusion_counts,
        })

    # Print summary report
    print("\n=================== COLLECTION SUMMARY REPORT ===================")
    for summary in repo_summaries:
        repo_name = summary["repo"]
        total_runs = summary["total_runs"]
        matched_runs = summary["matched_runs"]
        unmatched_runs = summary["unmatched_runs"]
        matched_prs = summary["matched_prs_count"]
        conclusions = summary["conclusions"]

        print(f"\nRepository: {repo_name}")
        print(f"  • Total Workflow Runs Fetched: {total_runs}")
        print(f"  • Runs Matched to a PR:        {matched_runs} ({(matched_runs / total_runs * 100) if total_runs else 0:.1f}%)")
        print(f"  • Unmatched Runs (e.g. force-push): {unmatched_runs}")
        print(f"  • Total Unique Matched PRs:    {matched_prs}")
        print(f"  • Run Conclusions (Matched Runs Only):")

        if matched_runs > 0:
            for conc, count in conclusions.most_common():
                conc_name = conc if conc is not None else "(in progress/queued)"
                pct = (count / matched_runs) * 100
                print(f"    - {conc_name:<16}: {count:>5} ({pct:>5.1f}%)")

            # Binary sanity check: failure vs success among matched runs
            failures = conclusions.get("failure", 0)
            successes = conclusions.get("success", 0)
            completed_total = failures + successes
            if completed_total > 0:
                fail_rate = (failures / completed_total) * 100
                succ_rate = (successes / completed_total) * 100
                print(f"  • Failure/Success Balance (Matched completed runs):")
                print(f"    - failure: {failures} ({fail_rate:.1f}%)")
                print(f"    - success: {successes} ({succ_rate:.1f}%)")
        else:
            print("    (No matched runs available to evaluate class balance)")

    print(f"\nTotals Across All Repositories:")
    print(f"  • Total Runs Fetched:      {overall_runs}")
    print(f"  • Total Matched Runs:      {overall_matched_runs}")
    print(f"  • Total Unmatched Runs:    {overall_unmatched_runs}")
    print(f"  • Total Unique Matched PRs:{overall_unique_prs}")
    print("=================================================================\n")


if __name__ == "__main__":
    args = parse_args()
    run_collection(force=args.force, max_runs=args.max_runs)
