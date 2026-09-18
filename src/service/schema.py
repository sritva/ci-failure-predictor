"""Request and response schemas for CI failure prediction inference service.

Includes strict range validation and detailed feature computation documentation
to prevent target leakage when callers (such as GitHub Actions) prepare feature payloads.
"""

from typing import List
from pydantic import BaseModel, ConfigDict, Field


class RiskFactor(BaseModel):
    """A single contributing feature to the failure risk prediction."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(
        ...,
        description="Name of the feature contributing to the prediction.",
    )
    contribution: float = Field(
        ...,
        description=(
            "Contribution score of this feature to failure risk (SHAP value log-odds "
            "impact on positive class, or normalized gain-weighted value)."
        ),
    )


class PRFeaturesRequest(BaseModel):
    """Feature payload for predicting CI pipeline failure risk on a pull request.

    IMPORTANT - LEAKAGE INVARIANT:
    All historical features (`author_past_run_count`, `author_past_failure_rate`,
    `repo_recent_failure_rate`) MUST be computed using only workflow runs that finished
    strictly BEFORE the current run or PR evaluation event. Including the current run
    or subsequent runs reintroduces the target leakage bug resolved in Phase 2.
    """

    model_config = ConfigDict(extra="forbid")

    diff_size: int = Field(
        ...,
        ge=0,
        description=(
            "Total lines of diff in the pull request (additions + deletions). "
            "Caller should compute this from the PR patch or GitHub API pull request summary."
        ),
    )
    files_changed: int = Field(
        ...,
        ge=0,
        description=(
            "Total count of distinct files modified, added, or deleted in the pull request. "
            "Caller should obtain this directly from the PR changed files count."
        ),
    )
    test_files_changed: int = Field(
        ...,
        ge=0,
        description=(
            "Count of test files changed in the pull request. "
            "Caller should count files whose normalized path matches test file patterns "
            "(e.g., contains 'test', 'tests/', '_test.py', '.test.js', etc.)."
        ),
    )
    touches_test_files: bool = Field(
        ...,
        description=(
            "Boolean flag indicating whether any test files were modified in the PR. "
            "Caller should set to True if test_files_changed > 0, otherwise False."
        ),
    )
    time_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description=(
            "Hour of the day in UTC (integer from 0 to 23) when the pull request was opened "
            "or the CI workflow run was triggered."
        ),
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description=(
            "Day of the week in UTC (integer from 0 to 6, where 0=Monday, 6=Sunday) "
            "when the pull request was opened or the CI workflow run was triggered."
        ),
    )
    is_weekend: bool = Field(
        ...,
        description=(
            "Boolean flag indicating whether the run timestamp occurs on a weekend. "
            "Caller should set to True if day_of_week is 5 (Saturday) or 6 (Sunday), otherwise False."
        ),
    )
    author_past_run_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of prior CI workflow runs by the PR author strictly prior to the current run. "
            "CRITICAL: Caller must count only historical runs with timestamp < current_run_timestamp. "
            "Never count the current run."
        ),
    )
    author_past_failure_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Historical CI failure rate (failures / total runs) of the PR author strictly prior "
            "to the current run. If author_past_run_count == 0, caller should set this to 0.0. "
            "CRITICAL: Must strictly exclude the current run from both numerator and denominator."
        ),
    )
    repo_recent_failure_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Recent rolling failure rate of the repository strictly prior to the current run "
            "(computed over up to the prior 20 workflow runs). If no prior runs exist, set to 0.0. "
            "CRITICAL: Must strictly exclude the current run from historical calculation."
        ),
    )


class PredictionResponse(BaseModel):
    """Prediction outcome and risk explanation for a pull request."""

    model_config = ConfigDict(extra="forbid")

    failure_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of CI run failure (0.0 to 1.0).",
    )
    risk_label: str = Field(
        ...,
        description=(
            "Categorical risk level: 'high' if failure_risk_score >= operating_threshold, "
            "'medium' if operating_threshold / 2 <= failure_risk_score < operating_threshold, "
            "and 'low' if failure_risk_score < operating_threshold / 2."
        ),
    )
    top_risk_factors: List[RiskFactor] = Field(
        ...,
        description="Top 3 features contributing to failure risk, ordered by contribution magnitude.",
    )
    model_version: str = Field(
        ...,
        description="Version string of the loaded model artifact from model_metadata.json.",
    )
