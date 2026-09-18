# Project History & Technical Changelog

This document tracks the technical evolution of `ci-failure-predictor`, key findings, bugs discovered and resolved, and the architectural reasoning behind major design decisions.

---

## Data Collection

- **GitHub API Client**: Built an authenticated GitHub client with exponential backoff for 5xx errors and dynamic rate-limit throttling (inspecting `X-RateLimit-Remaining` and `X-RateLimit-Reset`).
- **Data-Linkage Bug Discovery & Fix**:
  - Discovered that GitHub's REST API returns empty/blank `pull_requests` arrays on workflow runs associated with historical merged or closed PRs.
  - Developed a SHA-matching linkage algorithm (`link_runs_to_prs`) that cross-references `run['head_sha']` against `pr['head']['sha']` from paginated pull request queries, successfully recovering full linkage across all repositories.
- **Multi-Repository Collection**:
  - Collected 2,591 labeled workflow runs and pull request metadata across 8 distinct open-source repositories: `pallets/flask`, `tiangolo/fastapi`, `psf/requests`, `django/django`, `expressjs/express`, `scikit-learn/scikit-learn`, `huggingface/transformers`, and `vuejs/core`.
  - Filtered runs to pull request trigger events and classified conclusions into binary outcomes (399 failures, 2,192 successes; 15.4% dataset failure rate), dropping cancelled, skipped, and neutral runs.

---

## Feature Engineering

- **Leakage-Free Historical Features**:
  - Implemented historical author and repository metrics (`author_past_run_count`, `author_past_failure_rate`, `repo_recent_failure_rate`).
  - **Temporal Ordering Invariant**: Enforced strict timestamp sorting broken deterministically by `run_id`, shifting expanding and rolling windows by 1 within groups so no run's outcome is ever visible to itself or subsequent feature calculations.
  - **Cold-Start Fallback**: Implemented unbiased historical global prior fallback (0.1540) for authors on their initial run.
- **Diff & Code Heuristics**:
  - Extracted pull request diff statistics (`diff_size`, `files_changed`, `test_files_changed`, `touches_test_files`) and temporal markers (`time_of_day`, `day_of_week`, `is_weekend`).
  - Implemented regex heuristics (`is_test_file`) to identify test files across languages (`test_*.py`, `*.spec.ts`, `*_test.go`, etc.).
- **Leakage Invariant Testing**:
  - Validated temporal safety with dedicated unit tests in `tests/test_features.py`, including an adversarial test with identical timestamps to ensure tie-breaking cannot cause forward leakage.

---

## Baseline Models & Evaluation Methodology

- **Split Protocol Confounding & Diagnostic Discovery**:
  - Diagnosed that a naive global chronological 80/20 split confounded time with repository identity because target repositories had disparate collection windows (e.g., `pallets/flask` and `psf/requests` landed 100% in train, while `scikit-learn` dominated the test set).
  - Formalized a per-repository chronological split (`per_repo_time_split`) as the primary evaluation protocol (earliest ~80% of each repo's history in train, latest ~20% in test), ensuring every repository is represented in both sets while preserving temporal order.
- **5 Baseline Benchmarks**:
  - Evaluated 5 baseline models on the held-out primary test set (519 runs, 131 failures):
    1. **Dummy Classifier (class prior)**: 0.2524 PR-AUC (ROC-AUC: 0.5000)
    2. **Repo-Prior Classifier**: 0.3887 PR-AUC (95% CI: [0.3304, 0.4590], ROC-AUC: 0.7001)
    3. **Decision Stump (depth 1)**: 0.4140 PR-AUC (95% CI: [0.3577, 0.4730], ROC-AUC: 0.7457)
    4. **Recent-Rate Heuristic**: 0.5020 PR-AUC (95% CI: [0.4297, 0.5864], ROC-AUC: 0.7929)
    5. **Logistic Regression Pipeline**: 0.6591 PR-AUC (95% CI: [0.5789, 0.7597], ROC-AUC: 0.8647, F1: 0.6498 at threshold 0.62)

---

## XGBoost Model

- **Cross-Validation Protocol**:
  - Formulated `per_repo_time_series_cv`: splits time series independently within each repository and merges fold indices across repositories, preserving temporal causality without cross-repo leakage.
- **Hyperparameter Optimization & Training**:
  - Tuned hyperparameters using 5-fold cross-validation on the training set (best: `max_depth=6`, `learning_rate=0.0332`, `n_estimators=100`, `subsample=0.6`, `colsample_bytree=0.8`, `min_child_weight=9`), achieving mean CV PR-AUC of 0.5414.
  - Selected operating threshold of 0.73 based on out-of-fold F1 optimization (OOF F1: 0.5467).
- **Primary Protocol Test Set Evaluation**:
  - **PR-AUC**: 0.7044 (95% Bootstrap CI: [0.6251, 0.7745], ROC-AUC: 0.8439, F1: 0.6034).
  - **Paired Bootstrap Comparison (1,000 resamples vs. Logistic Regression)**:
    - Mean difference: +0.0379 PR-AUC.
    - 95% Confidence Interval: [-0.0223, +0.0976].
    - Win rate: 88.3%.
    - *Honest Reporting*: While XGBoost achieved the top headline score and won 88.3% of resamples, the confidence interval spanning zero demonstrated a modest rather than decisive advantage over simpler linear modeling.
- **Feature Importance (Gain)**:
  - Top features: `repo_recent_failure_rate` (31.35 gain), `author_past_failure_rate` (27.08 gain), `author_past_run_count` (12.30 gain), `files_changed` (11.96 gain), `diff_size` (11.54 gain).

---

## Recall/Threshold Experiments

- **Rigorous Train-Only Exploration**:
  - Evaluated four recall-enhancement experiments strictly via 5-fold cross-validation on train without test-set peeking:
    1. **Probability Calibration**: Isotonic calibration improved Brier score (0.1259 -> 0.0942), but degraded CV PR-AUC (0.5414 -> 0.5357) due to tied bins.
    2. **Ensembling**: Blending Logistic Regression with XGBoost underperformed standalone XGBoost across all blend weights.
    3. **Feature Variants**: Rolling 10-run failure rate showed a slight CV PR-AUC bump (+0.0170), but in a single permitted held-out test check failed to beat the baseline (0.7039 vs. 0.7044; 48.5% win rate).
    4. *Decision*: Documented as a rigorous negative result; production model artifacts were preserved as-is.
- **Operational High-Recall Characterization**:
  - Identified that threshold 0.37 achieves 76.9% CV recall (vs. 47.4% at threshold 0.73) with 35.2% precision.
  - Confirmed on held-out test data that threshold 0.37 detects 90.8% of all failures (119 of 131) with 42.5% precision (F2: 0.7400).
  - Kept production threshold at 0.73 to maintain precision and minimize false positive PR interruptions.

---

## Serving & Integration

- **FastAPI Inference Microservice**:
  - Built a production-grade inference service in `src/service/main.py` serving the serialized model at threshold 0.73.
  - Implemented Pydantic v2 schemas (`PRFeaturesRequest`, `PredictionResponse`) with strict range validation and anti-leakage field docstrings.
  - Integrated `shap.TreeExplainer` for dynamic, per-request log-odds risk factor attribution (with normalized gain fallback).
  - Verified endpoints (`GET /`, `GET /health`, `POST /predict`) and boundary threshold behavior (high >= 0.73, medium 0.365-0.73, low < 0.365).
- **Live Feature Computation (`src/features/compute_live_features.py`)**:
  - Designed live feature extraction querying the GitHub API at evaluation time.
  - Established the "now" (`datetime.now(timezone.utc)`) historical cutoff rule: evaluates history up to the evaluation moment rather than PR creation, capturing subsequent CI runs on multi-commit PRs while strictly preventing future outcome leakage.
- **GitHub Action Workflow (`.github/workflows/failure-risk-score.yml`)**:
  - Implemented an automated workflow on `pull_request: [opened, synchronize]` that starts the service, computes live features, requests predictions, and maintains an updated sticky comment with `<!-- ci-failure-predictor-comment -->`.
  - Added CLI auto-discovery (`find_a_live_open_pr`) and verified end-to-end against live GitHub PRs (`pallets/flask` PR #5918 and `tiangolo/fastapi` PR #16374).
- **Test Suite**:
  - Maintained 100% test pass rate across all 76 unit and integration tests (`tests/test_api.py`, `tests/test_compute_live_features.py`, `tests/test_baselines.py`, `tests/test_features.py`, `tests/test_github_client.py`, `tests/test_recall_experiments.py`, `tests/test_split.py`, `tests/test_xgboost_model.py`).
