"""CLI tool for scoring a pull request against the FastAPI inference service.

Fetches live features for a PR, requests failure risk prediction, formats a
plain-language GitHub comment with sticky HTML comment tags, and writes it to disk.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import urllib.request
import urllib.error
import sys

# Ensure UTF-8 printing in Windows console environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.collect.github_client import GitHubClient
from src.features.compute_live_features import compute_live_pr_features

logger = logging.getLogger("score_pr")
logging.basicConfig(level=logging.INFO)

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "repo_recent_failure_rate": "Recent CI instability in this repository",
    "author_past_failure_rate": "Author's historical CI failure rate",
    "author_past_run_count": "Author's prior CI run experience in this repository",
    "diff_size": "Large cumulative diff size (additions + deletions)",
    "files_changed": "High count of modified files in pull request",
    "test_files_changed": "Volume of changes across test files",
    "touches_test_files": "Pull request touches test files",
    "time_of_day": "Off-peak or late-night PR creation time",
    "day_of_week": "Day of the week PR was opened",
    "is_weekend": "Weekend pull request activity",
}


def find_a_live_open_pr(owner: str, repo: str, github_token: Optional[str] = None) -> int:
    """Query GitHub API to auto-discover the most recently opened PR for a repository.

    Args:
        owner: Repository owner.
        repo: Repository name.
        github_token: Optional GitHub API token.

    Returns:
        Integer pull request number.

    Raises:
        ValueError: If repository currently has 0 open pull requests.
    """
    client = GitHubClient(token=github_token)
    url = f"/repos/{owner}/{repo}/pulls"
    params = {"state": "open", "sort": "created", "direction": "desc", "per_page": 1}

    resp = client.get(url, params=params)
    prs = resp.json()

    if not isinstance(prs, list) or len(prs) == 0:
        raise ValueError(f"Repository {owner}/{repo} currently has zero open pull requests.")

    pr_num = prs[0].get("number")
    if pr_num is None:
        raise ValueError(f"Unable to extract pull request number from response: {prs[0]}")

    logger.info("Auto-discovered open PR #%d for %s/%s ('%s')", pr_num, owner, repo, prs[0].get("title", ""))
    return int(pr_num)


def format_risk_comment(
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Format an informative, plain-language PR comment with a sticky HTML marker.

    Args:
        prediction: Response from /predict endpoint.
        features: Computed feature dictionary.
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.

    Returns:
        Markdown-formatted string.
    """
    score = float(prediction.get("failure_risk_score", 0.0))
    risk_label = str(prediction.get("risk_label", "medium")).upper()
    model_version = prediction.get("model_version", "unknown")
    top_factors = prediction.get("top_risk_factors", [])

    # Visual indicators based on risk level
    if risk_label == "HIGH":
        badge = "🔴 **HIGH RISK**"
        recommendation = (
            "> ⚠️ **High failure risk detected**: Historical failure patterns or change characteristics "
            "indicate an elevated likelihood of CI failure. Consider running targeted unit/integration tests locally."
        )
    elif risk_label == "MEDIUM":
        badge = "🟡 **MEDIUM RISK**"
        recommendation = (
            "> ℹ️ **Moderate risk profile**: Failure likelihood is within typical operational bounds. "
            "Standard automated checks apply."
        )
    else:
        badge = "🟢 **LOW RISK**"
        recommendation = (
            "> ✅ **Clean risk profile**: Pull request exhibits low-risk markers across file scope and historical runs."
        )

    factor_lines = []
    for idx, factor in enumerate(top_factors[:3], start=1):
        feat_name = factor.get("feature", "")
        contrib = float(factor.get("contribution", 0.0))
        plain_desc = FEATURE_DESCRIPTIONS.get(feat_name, feat_name.replace("_", " ").title())
        feat_val = features.get(feat_name)
        val_str = f" (Value: `{feat_val}`)" if feat_val is not None else ""
        factor_lines.append(
            f"{idx}. **{plain_desc}** (`{feat_name}`){val_str} — Contribution: `{contrib:+.4f}`"
        )

    factors_md = "\n".join(factor_lines) if factor_lines else "*(No dominant risk factors)*"

    comment_body = f"""<!-- ci-failure-predictor-comment -->
## 🔍 CI Failure Risk Assessment for #{pr_number}

{badge} — Predicted Failure Risk: **{score * 100:.1f}%** *(Operating Threshold: 73.0%)*

{recommendation}

### Top Contributing Factors
{factors_md}

---
*Evaluated by [ci-failure-predictor](https://github.com/sritva/ci-failure-predictor) (Model Version: `{model_version}`) at PR-open time.*
"""
    return comment_body


def score_pr(
    owner: str,
    repo: str,
    pr_number: Optional[int] = None,
    api_url: str = "http://127.0.0.1:8000",
    output_file: Optional[str] = "risk_comment.md",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute live feature computation, model inference, and comment generation.

    Returns:
        Dict containing: 'pr_number', 'features', 'prediction', and 'comment_markdown'.
    """
    github_token = token or os.getenv("GITHUB_TOKEN")

    if pr_number is None:
        pr_number = find_a_live_open_pr(owner, repo, github_token)

    logger.info("Computing live features for %s/%s PR #%d...", owner, repo, pr_number)
    features = compute_live_pr_features(owner=owner, repo=repo, pr_number=pr_number, github_token=github_token)

    predict_url = f"{api_url.rstrip('/')}/predict"
    logger.info("Calling inference service at %s...", predict_url)

    req_data = json.dumps(features).encode("utf-8")
    req = urllib.request.Request(
        predict_url,
        data=req_data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            prediction = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to inference service at {predict_url}: {exc}") from exc

    logger.info(
        "Prediction received for PR #%d: score=%.4f label=%s",
        pr_number,
        prediction.get("failure_risk_score", 0.0),
        prediction.get("risk_label", "unknown"),
    )

    comment_md = format_risk_comment(prediction, features, owner, repo, pr_number)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(comment_md)
        logger.info("Saved formatted risk comment to %s", out_path)

    return {
        "pr_number": pr_number,
        "features": features,
        "prediction": prediction,
        "comment_markdown": comment_md,
    }


def main():
    """CLI entrypoint for scoring a PR."""
    parser = argparse.ArgumentParser(description="Score a pull request for CI failure risk.")
    parser.add_argument("--owner", required=True, help="Repository owner/org (e.g. pallets)")
    parser.add_argument("--repo", required=True, help="Repository name (e.g. flask)")
    parser.add_argument("--pr", type=int, default=None, help="Pull request number (auto-discovers if omitted)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI service base URL")
    parser.add_argument("--output-file", default="risk_comment.md", help="Path to write markdown comment")
    parser.add_argument("--token", default=None, help="GitHub API Token (optional, uses GITHUB_TOKEN if omitted)")

    args = parser.parse_args()

    result = score_pr(
        owner=args.owner,
        repo=args.repo,
        pr_number=args.pr,
        api_url=args.api_url,
        output_file=args.output_file,
        token=args.token,
    )

    print("\n" + "=" * 60)
    print(f"RESULT FOR {args.owner}/{args.repo} PR #{result['pr_number']}")
    print("=" * 60)
    print("\n[Computed Feature Dict]:")
    print(json.dumps(result["features"], indent=2))
    print("\n[Prediction Response]:")
    print(json.dumps(result["prediction"], indent=2))
    print("\n[Generated Comment Preview]:")
    print(result["comment_markdown"])
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
