"""Feature engineering and labeling pipeline for CI failure prediction."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.collect.link_runs_to_prs import link_runs_to_prs


# Common patterns for identifying test files in software repositories
TEST_FILE_REGEX = re.compile(
    r"(^|[\\/._-])(tests?|specs?)([\\/._-]|$)"
    r"|test_[^\\/]+\.py$"
    r"|[^\\/]+_test\.[a-zA-Z0-9]+$"
    r"|[^\\/]+\.test\.[a-zA-Z0-9]+$"
    r"|[^\\/]+\.spec\.[a-zA-Z0-9]+$",
    re.IGNORECASE,
)


def is_test_file(path: str) -> bool:
    """Heuristic to determine if a file path is a test or specification file.

    Matches:
        - Directories containing 'test', 'tests', 'spec', 'specs'
        - Filenames like test_*.py, *_test.py, *_test.go
        - Filenames like *.test.js, *.test.ts, *.spec.ts, *.spec.js
        - Path components matching test/spec patterns

    Args:
        path: Relative or absolute file path.

    Returns:
        True if the file appears to be a test file, False otherwise.
    """
    if not path:
        return False
    normalized = path.replace("\\", "/").lower()
    return bool(TEST_FILE_REGEX.search(normalized))


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free historical author and repo failure features.

    CRITICAL INVARIANT:
    All features derived from history must be computed using ONLY runs with a
    timestamp strictly BEFORE the row being predicted. Dataframe is sorted by
    ['run_timestamp', 'run_id'] first, and all rolling/expanding operations are
    shifted by 1 within their respective groups before assignment.

    Args:
        df: DataFrame containing at least 'run_timestamp', 'run_id', 'author', 'repo', 'label'.

    Returns:
        DataFrame with added historical feature columns:
            - author_past_run_count
            - author_past_failure_rate
            - repo_recent_failure_rate
    """
    result = df.copy()

    # 1. Deterministic sorting: sort by timestamp, breaking ties by run_id
    result["run_timestamp"] = pd.to_datetime(result["run_timestamp"], utc=True)
    result = result.sort_values(by=["run_timestamp", "run_id"], ascending=[True, True]).reset_index(drop=True)

    # 2. Global prior expanding failure rate (strictly prior rows, shifted by 1)
    # Used as an unbiased fallback for an author's very first run
    global_prior = result["label"].expanding().mean().shift(1)
    # If the first row in the dataset has no prior rows, default fallback is neutral 0.5
    global_prior = global_prior.fillna(0.5)

    # -------------------------------------------------------------------------
    # HISTORICAL FEATURE 1: author_past_run_count
    # Count of prior runs for this author strictly before the current run
    # -------------------------------------------------------------------------
    result["author_past_run_count"] = result.groupby("author").cumcount()

    # -------------------------------------------------------------------------
    # HISTORICAL FEATURE 2: author_past_failure_rate
    # Expanding failure rate for this author shifted by 1 so current row is excluded.
    # For an author's first run (expanding mean is NaN), fallback to global_prior.
    # -------------------------------------------------------------------------
    author_expanding_rate = result.groupby("author")["label"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    result["author_past_failure_rate"] = author_expanding_rate.fillna(global_prior)

    # -------------------------------------------------------------------------
    # HISTORICAL FEATURE 3: repo_recent_failure_rate
    # Rolling failure rate over the preceding 20 runs in this repo, shifted by 1.
    # -------------------------------------------------------------------------
    repo_rolling_rate = result.groupby("repo")["label"].transform(
        lambda s: s.rolling(window=20, min_periods=1).mean().shift(1)
    )
    result["repo_recent_failure_rate"] = repo_rolling_rate.fillna(global_prior)

    return result


def build_feature_table(
    owner: str,
    repo: str,
    data_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    """Load raw cached data, extract static and historical features, and save to parquet.

    Args:
        owner: Repository owner.
        repo: Repository name.
        data_dir: Directory where raw JSON caches live.
        processed_dir: Directory where processed parquet files will be saved.

    Returns:
        Processed pandas DataFrame containing features and labels.
    """
    raw_path = Path(data_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    runs_file = raw_path / f"{owner}__{repo}__runs.json"
    all_prs_file = raw_path / f"{owner}__{repo}__all_prs.json"
    prs_meta_file = raw_path / f"{owner}__{repo}__prs.json"

    if not runs_file.exists():
        raise FileNotFoundError(f"Runs cache missing: {runs_file}. Run collection first.")
    if not all_prs_file.exists():
        raise FileNotFoundError(f"All PRs cache missing: {all_prs_file}. Run collection first.")

    # 1. Load raw data
    with open(runs_file, "r", encoding="utf-8") as f:
        runs_data = json.load(f)
        runs = runs_data.get("runs", runs_data) if isinstance(runs_data, dict) else runs_data

    with open(all_prs_file, "r", encoding="utf-8") as f:
        all_prs_data = json.load(f)
        all_prs = all_prs_data.get("prs", all_prs_data) if isinstance(all_prs_data, dict) else all_prs_data

    prs_meta: Dict[str, Any] = {}
    if prs_meta_file.exists():
        with open(prs_meta_file, "r", encoding="utf-8") as f:
            prs_meta_data = json.load(f)
            prs_meta = prs_meta_data.get("prs", prs_meta_data) if isinstance(prs_meta_data, dict) else prs_meta_data

    # Map all_prs by number for fallback details
    all_prs_by_number: Dict[int, Dict[str, Any]] = {}
    for p in all_prs:
        if isinstance(p, dict) and p.get("number") is not None:
            all_prs_by_number[p["number"]] = p

    # 2. Re-derive matched runs using link_runs_to_prs
    matched_pairs, unmatched_count = link_runs_to_prs(runs=runs, prs=all_prs)
    print(
        f"[build_feature_table] Matched {len(matched_pairs)} runs to PRs for {owner}/{repo} "
        f"({unmatched_count} unmatched runs excluded)."
    )

    rows: List[Dict[str, Any]] = []

    for run, pr_number in matched_pairs:
        # Phase 3 Labeling:
        # label: 1 if failure, 0 if success. Drop all other conclusions.
        conclusion = run.get("conclusion")
        if conclusion == "failure":
            label = 1
        elif conclusion == "success":
            label = 0
        else:
            # Drop rows with cancelled, skipped, neutral, timed_out, or None
            continue

        # Extract PR details from prs_meta (preferred) or all_prs
        pr_entry = prs_meta.get(str(pr_number)) or prs_meta.get(pr_number) or {}
        pr_fallback = all_prs_by_number.get(pr_number, {})

        additions = pr_entry.get("additions")
        if additions is None:
            additions = pr_fallback.get("additions", 0) or 0

        deletions = pr_entry.get("deletions")
        if deletions is None:
            deletions = pr_fallback.get("deletions", 0) or 0

        files_changed = pr_entry.get("changed_files")
        if files_changed is None:
            files_changed = pr_fallback.get("changed_files", 0) or 0

        filenames = pr_entry.get("filenames") or []

        test_files_changed = sum(1 for f in filenames if is_test_file(f))
        touches_test_files = test_files_changed > 0

        author = (
            pr_entry.get("user")
            or pr_fallback.get("user")
            or (run.get("actor", {}).get("login") if isinstance(run.get("actor"), dict) else None)
            or "unknown"
        )

        created_at_raw = run.get("created_at")
        run_timestamp = pd.to_datetime(created_at_raw, utc=True)
        time_of_day = run_timestamp.hour
        day_of_week = run_timestamp.dayofweek
        is_weekend = day_of_week >= 5

        rows.append({
            "run_id": run.get("id"),
            "run_timestamp": run_timestamp,
            "author": author,
            "repo": f"{owner}/{repo}",
            "label": label,
            "diff_size": additions + deletions,
            "files_changed": files_changed,
            "test_files_changed": test_files_changed,
            "touches_test_files": touches_test_files,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        })

    if not rows:
        print(f"[build_feature_table] Warning: No valid labeled rows produced for {owner}/{repo}.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 3. Add leak-free historical features
    df = add_historical_features(df)

    # 4. Save to parquet
    output_parquet = proc_path / f"{owner}__{repo}__features.parquet"
    df.to_parquet(output_parquet, index=False)
    print(f"[build_feature_table] Saved {len(df)} feature rows to {output_parquet}")

    return df
