"""Top-level feature generation script for all target repositories."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from src.features.build_features import build_feature_table


def run_build_features(processed_dir: str = "data/processed") -> None:
    """Build feature tables for all target repositories and combine them."""
    load_dotenv()
    target_repos_env = os.getenv("TARGET_REPOS", "")
    if not target_repos_env.strip():
        print(
            "ERROR: TARGET_REPOS not set in environment or .env file.\n"
            "Please configure TARGET_REPOS (e.g. owner/repo1,owner/repo2)."
        )
        sys.exit(1)

    repo_list = [r.strip() for r in target_repos_env.split(",") if r.strip()]
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    print("\n=============== CI FAILURE PREDICTOR: FEATURE ENGINEERING ===============")
    print(f"Target Repositories: {', '.join(repo_list)}\n")

    tables = []
    repo_breakdown = {}

    for repo_full in repo_list:
        if "/" not in repo_full:
            print(f"Skipping invalid repository identifier: '{repo_full}'. Expected 'owner/repo'.")
            continue

        owner, repo = repo_full.split("/", 1)
        print(f"--- Processing features for {owner}/{repo} ---")
        try:
            df_repo = build_feature_table(owner=owner, repo=repo, processed_dir=processed_dir)
            if not df_repo.empty:
                tables.append(df_repo)
                repo_breakdown[repo_full] = len(df_repo)
            else:
                repo_breakdown[repo_full] = 0
        except Exception as exc:
            print(f"Error building features for {repo_full}: {exc}")
            repo_breakdown[repo_full] = 0

    if not tables:
        print("\nERROR: No feature tables could be generated. Check raw data.")
        sys.exit(1)

    # Combine all tables
    combined_df = pd.concat(tables, ignore_index=True)
    combined_parquet = proc_path / "combined_features.parquet"
    combined_df.to_parquet(combined_parquet, index=False)

    total_rows = len(combined_df)
    failure_count = int((combined_df["label"] == 1).sum())
    success_count = int((combined_df["label"] == 0).sum())

    fail_pct = (failure_count / total_rows * 100) if total_rows else 0.0
    succ_pct = (success_count / total_rows * 100) if total_rows else 0.0

    print("\n=================== FEATURE ENGINEERING SUMMARY ===================")
    print(f"Total Rows Generated:     {total_rows}")
    print(f"Saved Combined Parquet:   {combined_parquet}")
    print(f"\nLabel Balance:")
    print(f"  • Failure (label=1):    {failure_count:>5} ({fail_pct:>5.1f}%)")
    print(f"  • Success (label=0):    {success_count:>5} ({succ_pct:>5.1f}%)")
    print(f"\nPer-Repository Breakdown:")
    for r_name, count in repo_breakdown.items():
        pct = (count / total_rows * 100) if total_rows else 0.0
        print(f"  • {r_name:<24}: {count:>5} rows ({pct:>5.1f}%)")
    print("===================================================================\n")


if __name__ == "__main__":
    run_build_features()
