# CI Failure Predictor

> [!IMPORTANT]
> **Project Memory & Conventions**: See [memory.md](file:///c:/Users/sriva/Desktop/ci-failure-predictor/memory.md) — read before making changes.

## Overview

CI Failure Predictor predicts whether a GitHub Actions CI workflow run will fail using metadata available at pull request open time (before the build executes). The system focuses on leak-free evaluation using time-ordered splits, real metrics (PR-AUC, recall), and deployment via a live FastAPI inference service integrated with a GitHub Action.

## Setup & Quickstart

### 1. Create a GitHub Personal Access Token (Classic)

To collect workflow run and pull request metadata without running into tight unauthenticated rate limits:
1. Go to GitHub -> **Settings** -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)** (or visit `https://github.com/settings/tokens`).
2. Click **Generate new token** -> **Generate new token (classic)**.
3. Provide a note (e.g. `ci-failure-predictor`).
4. **Scopes**: For public repositories, **no special scopes are required** (leave all checkboxes unchecked).
5. Generate the token and copy its value.

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your token and target repositories:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GITHUB_TOKEN=ghp_your_personal_access_token_here
TARGET_REPOS=owner/repo1,owner/repo2
```

### 3. Install Requirements

Install the project dependencies using pip:

```bash
pip install -r requirements.txt
```

*(On Windows, you can also run `py -m pip install -r requirements.txt`)*

### 4. Run Data Collection

Run the data collection script to fetch workflow runs and PR metadata for the configured repositories:

```bash
python -m src.collect.run_collection
```

To bypass cached responses under `data/raw/` and force fresh API downloads, use the `--force` flag:

```bash
python -m src.collect.run_collection --force
```

### 5. Running Tests

Run the test suite using pytest:

```bash
pytest tests/
```
