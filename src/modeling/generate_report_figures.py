"""Generate report visualization figures from metrics.json and serialized model artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.modeling.baselines import (
    FEATURE_COLS,
    build_logistic_regression_pipeline,
)
from src.modeling.split import per_repo_time_split


# Paths
DATA_PATH = Path("data/processed/combined_features.parquet")
MODEL_PKL_PATH = Path("src/modeling/model.pkl")
METRICS_JSON_PATH = Path("reports/metrics.json")
FIGURES_DIR = Path("reports/figures")


def setup_plot_style() -> None:
    """Set aesthetic defaults for crisp, publication-ready figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.2,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Generate and save annotated confusion matrix with class names and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)

    # Color map
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", alpha=0.85)

    # Values and annotations
    classes = ["Success (0)", "Failure (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight="medium")
    ax.set_yticklabels(classes, fontsize=11, fontweight="medium")

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_title(
        f"Test Set Confusion Matrix\n(Operating Threshold = {threshold:.2f}, N = {total})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # Cell text annotations
    cell_info = [
        [(tn, "True Negatives (TN)"), (fp, "False Positives (FP)")],
        [(fn, "False Negatives (FN)"), (tp, "True Positives (TP)")],
    ]

    for i in range(2):
        for j in range(2):
            count, label = cell_info[i][j]
            pct = count / total * 100
            # Choose white text for dark cells, dark text for light cells
            color = "white" if count > (total * 0.4) else "#0f172a"
            ax.text(
                j,
                i - 0.12,
                f"{count}",
                ha="center",
                va="center",
                color=color,
                fontsize=20,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.12,
                f"{label}\n({pct:.1f}% of total)",
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_feature_importance(
    feature_importances: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate and save horizontal bar chart of top gain-based feature importances."""
    top_features = feature_importances[:5]
    # Reverse order so highest rank is at top of chart
    top_features_reversed = list(reversed(top_features))

    names = [item["feature"] for item in top_features_reversed]
    gains = [item["gain"] for item in top_features_reversed]

    # Nicely formatted display labels
    name_display_map = {
        "repo_recent_failure_rate": "repo_recent_failure_rate\n(recent repo instability)",
        "author_past_failure_rate": "author_past_failure_rate\n(author failure history)",
        "author_past_run_count": "author_past_run_count\n(author tenure / experience)",
        "files_changed": "files_changed\n(PR blast radius)",
        "diff_size": "diff_size\n(lines added + deleted)",
    }
    display_names = [name_display_map.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    bars = ax.barh(display_names, gains, color="#3b82f6", edgecolor="#1d4ed8", height=0.6, alpha=0.9)

    ax.set_xlabel("Feature Importance Score (Gain)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Top 5 Features by Gain Importance (XGBoost)\nSource: reports/metrics.json",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Annotate values at the end of each bar
    max_gain = max(gains)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max_gain * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1e293b",
        )

    ax.set_xlim(0, max_gain * 1.15)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    xgb_scores: np.ndarray,
    lr_scores: np.ndarray,
    xgb_prauc: float,
    lr_prauc: float,
    xgb_thresh: float,
    output_path: Path,
) -> None:
    """Generate and save overlaid Precision-Recall curve for XGBoost vs Logistic Regression."""
    prec_xgb, rec_xgb, thresholds_xgb = precision_recall_curve(y_true, xgb_scores)
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, lr_scores)

    # Baseline prevalence
    prevalence = float(np.sum(y_true == 1) / len(y_true))

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)

    # Curves
    ax.plot(
        rec_xgb,
        prec_xgb,
        color="#2563eb",
        linewidth=2.5,
        label=f"XGBoost (tuned) — PR-AUC: {xgb_prauc:.4f}",
    )
    ax.plot(
        rec_lr,
        prec_lr,
        color="#10b981",
        linewidth=2.0,
        linestyle="-.",
        label=f"Logistic Regression — PR-AUC: {lr_prauc:.4f}",
    )
    ax.axhline(
        prevalence,
        color="#94a3b8",
        linestyle=":",
        linewidth=1.5,
        label=f"No-Skill Baseline (Prior: {prevalence:.3f})",
    )

    # Mark operating point for XGBoost
    # Find index of threshold closest to xgb_thresh
    thresh_idx = np.argmin(np.abs(thresholds_xgb - xgb_thresh))
    op_prec = prec_xgb[thresh_idx]
    op_rec = rec_xgb[thresh_idx]

    ax.plot(
        op_rec,
        op_prec,
        marker="o",
        markersize=9,
        color="#dc2626",
        markeredgecolor="white",
        markeredgewidth=2,
        label=f"Operating Point (Thresh = {xgb_thresh:.2f}, R={op_rec:.2f}, P={op_prec:.2f})",
        zorder=5,
    )

    ax.set_xlabel("Recall (Fraction of CI Failures Caught)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Precision-Recall Curve (Primary Protocol Test Set)\nDirect Comparison: XGBoost vs Logistic Regression",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.95)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Load model artifact and metrics.json, generate all figures, and save to reports/figures/."""
    setup_plot_style()

    print("=" * 80)
    print("GENERATING VISUALIZATION FIGURES (PHASE 7 CLOSE-OUT)")
    print("=" * 80)

    # 1. Verify required source files
    if not MODEL_PKL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PKL_PATH}. Run run_xgboost.py first.")
    if not METRICS_JSON_PATH.exists():
        raise FileNotFoundError(f"Metrics report not found at {METRICS_JSON_PATH}. Run run_xgboost.py first.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

    # 2. Load metrics.json as source of truth
    print(f"Loading metrics from {METRICS_JSON_PATH}...")
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    per_repo_metrics = metrics_data["per_repo"]
    xgb_prauc = per_repo_metrics["xgboost"]["metrics"]["pr_auc"]
    lr_prauc = per_repo_metrics["logistic_regression"]["metrics"]["pr_auc"]
    feature_importances = per_repo_metrics["xgboost"]["feature_importances"]

    # 3. Load model artifact
    print(f"Loading fitted model from {MODEL_PKL_PATH}...")
    artifact = joblib.load(MODEL_PKL_PATH)
    xgb_model = artifact["model"]
    operating_threshold = artifact["operating_threshold"]
    feature_names = artifact["feature_names"]

    # 4. Load dataset and reproduce primary test split
    print(f"Loading dataset from {DATA_PATH} and reproducing primary test split...")
    df = pd.read_parquet(DATA_PATH)
    train_df, test_df = per_repo_time_split(df, test_frac=0.2)

    y_test = test_df["label"].values
    xgb_scores = xgb_model.predict_proba(test_df[feature_names])[:, 1]
    xgb_preds = (xgb_scores >= operating_threshold).astype(int)

    # 5. Fit Logistic Regression pipeline on primary train set for overlaid curve
    print("Fitting Logistic Regression on train set to extract PR curve...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=42)
    lr_pipeline.fit(train_df[FEATURE_COLS], train_df["label"])
    lr_scores = lr_pipeline.predict_proba(test_df[FEATURE_COLS])[:, 1]

    # 6. Generate Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures in reports/figures/:")

    # Figure 1: Confusion Matrix
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=xgb_preds,
        threshold=operating_threshold,
        output_path=cm_path,
    )

    # Figure 2: Feature Importance
    fi_path = FIGURES_DIR / "feature_importance.png"
    plot_feature_importance(
        feature_importances=feature_importances,
        output_path=fi_path,
    )

    # Figure 3: Precision-Recall Curve
    pr_path = FIGURES_DIR / "pr_curve.png"
    plot_precision_recall_curve(
        y_true=y_test,
        xgb_scores=xgb_scores,
        lr_scores=lr_scores,
        xgb_prauc=xgb_prauc,
        lr_prauc=lr_prauc,
        xgb_thresh=operating_threshold,
        output_path=pr_path,
    )

    print("\n[SUCCESS] All 3 figures generated successfully under reports/figures/.")


if __name__ == "__main__":
    main()
