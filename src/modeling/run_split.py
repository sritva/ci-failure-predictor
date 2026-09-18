"""Demonstration runner for Phase 4 train/test splits on combined feature table."""

from pathlib import Path
import pandas as pd

from src.modeling.split import (
    time_aware_split,
    repo_holdout_split,
)


def run_split(data_path: str = "data/processed/combined_features.parquet") -> None:
    """Load combined features and demonstrate time-aware and repo-holdout splits."""
    path = Path(data_path)
    if not path.exists():
        print(f"ERROR: Combined feature file not found at {path}. Run feature generation first.")
        return

    df = pd.read_parquet(path)
    total_rows = len(df)

    print("\n================= CI FAILURE PREDICTOR: PHASE 4 SPLITS =================")
    print(f"Loaded dataset: {path} ({total_rows} total rows)\n")

    # 1. Primary Time-Aware Split (80% train, 20% test)
    print("--- 1. Primary Time-Aware Split (80/20 Chronological) ---")
    train_df, test_df = time_aware_split(df, test_frac=0.2)

    # 2. Secondary Repo-Holdout Split
    print("\n--- 2. Secondary Repo-Holdout Split (Generalization Check) ---")
    unique_repos = df["repo"].unique().tolist() if "repo" in df.columns else []

    if len(unique_repos) >= 2:
        # Pick the largest repo as an example holdout
        largest_repo = df["repo"].value_counts().index[0]
        print(f"Found {len(unique_repos)} repositories. Demonstrating holdout with: '{largest_repo}'")
        train_holdout, test_holdout = repo_holdout_split(df, holdout_repo=largest_repo)

        h_train_fails = int((train_holdout["label"] == 1).sum())
        h_test_fails = int((test_holdout["label"] == 1).sum())
        print(f"  • Train label balance: {h_train_fails} failures / {len(train_holdout)} ({h_train_fails/len(train_holdout)*100:.1f}%)")
        print(f"  • Held-out label balance: {h_test_fails} failures / {len(test_holdout)} ({h_test_fails/len(test_holdout)*100:.1f}%)")
    else:
        print(
            f"Skipped repo-holdout: Dataset contains only {len(unique_repos)} repository "
            f"({unique_repos}). Repo holdout requires 2+ repositories."
        )

    print("\n(Note: Split datasets are held in-memory and not written to disk per project conventions.)")
    print("========================================================================\n")


if __name__ == "__main__":
    run_split()
