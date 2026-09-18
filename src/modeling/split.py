"""Time-aware train/test splitting and cross-validation strategies."""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def _sort_temporally(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe deterministically by run_timestamp and run_id."""
    df_sorted = df.copy()
    df_sorted["run_timestamp"] = pd.to_datetime(df_sorted["run_timestamp"], utc=True)
    sort_cols = ["run_timestamp"]
    if "run_id" in df_sorted.columns:
        sort_cols.append("run_id")
    return df_sorted.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)


def time_aware_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and test sets.

    CRITICAL SPLIT RULE:
    Splits must respect temporal order. Earliest (1 - test_frac) rows become train,
    latest test_frac rows become test. Never use random or stratified splits.

    Args:
        df: Input DataFrame containing features and labels.
        test_frac: Fraction of total rows to allocate to the test set (default 0.2).

    Returns:
        Tuple of (train_df, test_df) DataFrames.

    Raises:
        ValueError: If temporal boundary assertion fails (train max timestamp > test min timestamp).
    """
    if df.empty:
        return df.copy(), df.copy()

    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be strictly between 0 and 1, got {test_frac}")

    df_sorted = _sort_temporally(df)
    n_total = len(df_sorted)
    n_train = int(n_total * (1.0 - test_frac))

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train:].copy().reset_index(drop=True)

    # Defensive sanity check: verify no future data in train
    if not train_df.empty and not test_df.empty:
        max_train_time = train_df["run_timestamp"].max()
        min_test_time = test_df["run_timestamp"].min()
        if max_train_time > min_test_time:
            raise ValueError(
                f"Temporal leak detected: train max timestamp ({max_train_time}) "
                f"exceeds test min timestamp ({min_test_time})"
            )

    # Log label balance for train and test sets
    train_n = len(train_df)
    test_n = len(test_df)
    train_fails = int((train_df["label"] == 1).sum()) if "label" in train_df.columns else 0
    test_fails = int((test_df["label"] == 1).sum()) if "label" in test_df.columns else 0

    train_pct = (train_fails / train_n * 100) if train_n else 0.0
    test_pct = (test_fails / test_n * 100) if test_n else 0.0

    print(f"[time_aware_split] Total: {n_total} rows | Train (earliest {100*(1-test_frac):.0f}%): {train_n} rows | Test (latest {100*test_frac:.0f}%): {test_n} rows")
    print(f"  • Train label balance: {train_fails} failures / {train_n} ({train_pct:.1f}%)")
    print(f"  • Test label balance:  {test_fails} failures / {test_n} ({test_pct:.1f}%)")

    return train_df, test_df


def time_series_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate time-ordered train/validation fold indices using TimeSeriesSplit.

    Data is sorted by run_timestamp prior to folding to ensure that each fold's
    validation split strictly succeeds its training window.

    Args:
        df: Input DataFrame.
        n_splits: Number of time-series splits (default 5).

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    df_sorted = _sort_temporally(df)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df_sorted))


def repo_holdout_split(
    df: pd.DataFrame,
    holdout_repo: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a repository-grouped holdout split.

    Train on N-1 repos, test on 1 held-out repo.

    IMPORTANT NOTE:
    Per project memory conventions, this is a secondary / optional generalization check
    to test transferability to an unseen repository. It is NEVER a replacement for the
    primary time-aware split.

    Args:
        df: Input DataFrame containing a 'repo' column.
        holdout_repo: Repository identifier to hold out for testing.

    Returns:
        Tuple of (train_df, test_df) DataFrames.

    Raises:
        ValueError: If fewer than 2 repositories exist in df, or holdout_repo is not present.
    """
    if "repo" not in df.columns:
        raise ValueError("DataFrame must contain a 'repo' column to perform repo_holdout_split.")

    unique_repos = df["repo"].unique().tolist()
    if len(unique_repos) < 2:
        raise ValueError(
            f"repo_holdout_split requires 2 or more repositories, but only found {len(unique_repos)}: {unique_repos}. "
            "A repo-holdout cannot be performed on single-repo data."
        )

    if holdout_repo not in unique_repos:
        raise ValueError(
            f"Holdout repo '{holdout_repo}' not found in dataset. Available repositories: {unique_repos}"
        )

    train_df = df[df["repo"] != holdout_repo].copy().reset_index(drop=True)
    test_df = df[df["repo"] == holdout_repo].copy().reset_index(drop=True)

    print(f"[repo_holdout_split] Held out '{holdout_repo}':")
    print(f"  • Train (other repos): {len(train_df)} rows")
    print(f"  • Test (held out repo): {len(test_df)} rows")

    return train_df, test_df


def per_repo_time_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically within each repository into train and test sets.

    PRIMARY PROTOCOL FOR POOLED-MODEL EVALUATION:
    When multiple repositories have disparate data collection windows, a single
    global chronological split can inadvertently isolate entire repositories into
    train or test (e.g., older repos solely in train, newer repos solely in test).
    This function guarantees that every repository contributes its earliest
    (1 - test_frac) fraction of runs to train and its latest test_frac fraction
    to test.

    NOTE ON LIMITATION:
    Because splits are applied independently per repository, train rows from one
    repository may have a run_timestamp that post-dates test rows from another
    repository. As a result, shared cross-repo calendar/time effects (such as
    platform-wide GitHub Actions outages on a specific date) are not isolated
    across repositories.

    DETERMINISTIC ROUNDING RULE:
    For each repository with n rows, n_test = max(1, int(round(n * test_frac)))
    and n_train = n - n_test. Every repository must have at least 1 train row
    and 1 test row (i.e. n >= 2 and n_train >= 1); otherwise a ValueError is raised.

    Args:
        df: Input DataFrame containing features, 'repo', 'run_timestamp', and 'run_id'.
        test_frac: Fraction of rows per repo to allocate to test (default 0.2).

    Returns:
        Tuple of (train_df, test_df) DataFrames sorted chronologically.

    Raises:
        ValueError: If test_frac is invalid, 'repo' column is missing, any repo
            cannot produce both train and test rows, or any repo exhibits a
            temporal boundary violation (train max timestamp > test min timestamp).
    """
    if df.empty:
        return df.copy(), df.copy()

    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be strictly between 0 and 1, got {test_frac}")

    if "repo" not in df.columns:
        raise ValueError("DataFrame must contain a 'repo' column to perform per_repo_time_split.")

    train_pieces: List[pd.DataFrame] = []
    test_pieces: List[pd.DataFrame] = []

    for repo_name, repo_group in df.groupby("repo", sort=True):
        repo_sorted = _sort_temporally(repo_group)
        n = len(repo_sorted)

        # Deterministic rounding rule
        n_test = max(1, int(round(n * test_frac)))
        n_train = n - n_test

        if n_train < 1 or n_test < 1:
            raise ValueError(
                f"Repo '{repo_name}' has {n} rows, yielding {n_train} train rows and {n_test} "
                f"test rows with test_frac={test_frac}. Every repository must land in both "
                f"train and test (requires at least 1 row in each)."
            )

        repo_train = repo_sorted.iloc[:n_train].copy()
        repo_test = repo_sorted.iloc[n_train:].copy()

        # Per-repo temporal boundary guard
        max_train_time = repo_train["run_timestamp"].max()
        min_test_time = repo_test["run_timestamp"].min()
        if max_train_time > min_test_time:
            raise ValueError(
                f"Temporal leak detected in repo '{repo_name}': train max timestamp "
                f"({max_train_time}) exceeds test min timestamp ({min_test_time})."
            )

        train_pieces.append(repo_train)
        test_pieces.append(repo_test)

    train_df = pd.concat(train_pieces, ignore_index=True)
    test_df = pd.concat(test_pieces, ignore_index=True)

    # Sort final concatenated splits temporally for determinism
    train_df = _sort_temporally(train_df)
    test_df = _sort_temporally(test_df)

    n_total = len(df)
    train_n = len(train_df)
    test_n = len(test_df)
    train_fails = int((train_df["label"] == 1).sum()) if "label" in train_df.columns else 0
    test_fails = int((test_df["label"] == 1).sum()) if "label" in test_df.columns else 0

    train_pct = (train_fails / train_n * 100) if train_n else 0.0
    test_pct = (test_fails / test_n * 100) if test_n else 0.0

    print(f"[per_repo_time_split] Total: {n_total} rows across {df['repo'].nunique()} repos | Train: {train_n} rows | Test: {test_n} rows")
    print(f"  • Train label balance: {train_fails} failures / {train_n} ({train_pct:.1f}%)")
    print(f"  • Test label balance:  {test_fails} failures / {test_n} ({test_pct:.1f}%)")

    return train_df, test_df


def per_repo_time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate per-repository time-series cross-validation fold indices.

    For EACH repository independently:
      - Sorts that repository's rows by (run_timestamp, run_id).
      - Computes sklearn TimeSeriesSplit folds on just that repo's rows.
      - Combines fold k's training indices and validation indices across all repos
        into a single unified fold k.

    This ensures that fold k trains on "each repo's own earlier history" and
    validates on "each repo's own next chunk," rather than mixing absolute
    timestamps across repos with disparate collection windows.

    Args:
        df: Input DataFrame containing 'repo', 'run_timestamp', and 'run_id'.
        n_splits: Number of time-series splits per repo (default 5).

    Returns:
        List of (train_idx, val_idx) tuples containing integer positional indices
        into the temporally sorted DataFrame (_sort_temporally(df)).

    Raises:
        ValueError: If 'repo' column is missing or any repo has fewer rows than (n_splits + 1).
    """
    if "repo" not in df.columns:
        raise ValueError("DataFrame must contain a 'repo' column to perform per_repo_time_series_cv.")

    df_sorted = _sort_temporally(df)
    unique_repos = sorted(df_sorted["repo"].unique().tolist())

    repo_splits: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for r in unique_repos:
        repo_pos_indices = np.where(df_sorted["repo"] == r)[0]
        n_repo = len(repo_pos_indices)
        if n_repo < n_splits + 1:
            raise ValueError(
                f"Repo '{r}' has {n_repo} rows, but per_repo_time_series_cv requires at least "
                f"{n_splits + 1} rows to produce {n_splits} TimeSeriesSplit folds."
            )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        repo_splits[r] = []
        for rel_tr, rel_val in tscv.split(repo_pos_indices):
            tr_idx = repo_pos_indices[rel_tr]
            val_idx = repo_pos_indices[rel_val]
            repo_splits[r].append((tr_idx, val_idx))

    combined_folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        fold_train = np.sort(np.concatenate([repo_splits[r][k][0] for r in unique_repos]))
        fold_val = np.sort(np.concatenate([repo_splits[r][k][1] for r in unique_repos]))
        combined_folds.append((fold_train, fold_val))

    return combined_folds


