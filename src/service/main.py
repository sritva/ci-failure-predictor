"""FastAPI inference service for predicting CI pipeline failures on pull requests.

Serves the tuned XGBoost model from Phase 6 at its single production operating threshold (0.73).
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.service.schema import PRFeaturesRequest, PredictionResponse, RiskFactor

logger = logging.getLogger("ci_failure_service")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "src" / "modeling" / "model.pkl"
METADATA_PATH = BASE_DIR / "src" / "modeling" / "model_metadata.json"

app_state: Dict[str, Any] = {
    "model": None,
    "feature_names": None,
    "operating_threshold": None,
    "model_version": None,
    "explainer": None,
    "global_gains": None,
    "explainer_mode": None,
}


def load_artifacts() -> None:
    """Load model.pkl and model_metadata.json once, verifying feature order and schema."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found at {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    meta_features = metadata.get("feature_names")
    if not meta_features:
        raise ValueError(f"No feature_names found in {METADATA_PATH}")

    operating_threshold = float(metadata.get("operating_threshold", 0.73))
    model_version = str(metadata.get("model_version", "1.0.0-xgb"))

    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        artifact_features = artifact.get("feature_names")
    else:
        model = artifact
        artifact_features = meta_features

    if model is None:
        raise ValueError(f"Failed to extract fitted model from {MODEL_PATH}")

    if artifact_features != meta_features:
        raise ValueError(
            f"Feature order mismatch between model.pkl and model_metadata.json:\n"
            f"model.pkl features: {artifact_features}\n"
            f"metadata features: {meta_features}"
        )

    explainer = None
    explainer_mode = "gain_fallback"
    global_gains: Dict[str, float] = {}

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        explainer_mode = "shap"
        logger.info("Initialized SHAP TreeExplainer for per-request risk factor explanations.")
    except Exception as exc:
        logger.warning(
            "SHAP TreeExplainer could not be initialized (%s). Falling back to global gain * feature value.",
            exc,
        )
        explainer_mode = "gain_fallback"

    try:
        booster = model.get_booster()
        raw_score = booster.get_score(importance_type="gain")
        for idx, feat in enumerate(meta_features):
            gain = raw_score.get(feat, raw_score.get(f"f{idx}", 1.0))
            global_gains[feat] = float(gain)
    except Exception as exc:
        logger.warning("Could not extract feature gain scores from booster: %s", exc)
        global_gains = {feat: 1.0 for feat in meta_features}

    app_state["model"] = model
    app_state["feature_names"] = meta_features
    app_state["operating_threshold"] = operating_threshold
    app_state["model_version"] = model_version
    app_state["explainer"] = explainer
    app_state["global_gains"] = global_gains
    app_state["explainer_mode"] = explainer_mode

    logger.info(
        "Successfully loaded model version %s with %d features at operating threshold %.2f (explanation mode: %s).",
        model_version,
        len(meta_features),
        operating_threshold,
        explainer_mode,
    )


def ensure_loaded() -> None:
    """Ensure artifacts are loaded into state (handles lazy loading or testing contexts)."""
    if app_state["model"] is None:
        load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown hooks."""
    load_artifacts()
    yield


app = FastAPI(
    title="CI Pipeline Failure Predictor",
    description=(
        "Production inference service serving the tuned XGBoost model to predict "
        "GitHub Actions CI pipeline failure risk at PR-open time."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Info"])
def root_info() -> Dict[str, Any]:
    """Basic service information and available endpoint routes."""
    ensure_loaded()
    return {
        "name": "ci-failure-predictor",
        "description": "FastAPI inference service for predicting CI pipeline failures on pull requests",
        "model_version": app_state["model_version"],
        "operating_threshold": app_state["operating_threshold"],
        "active_explanation_method": app_state["explainer_mode"],
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Health check endpoint confirming model availability and service readiness."""
    ensure_loaded()
    return {
        "status": "ok",
        "model_version": app_state["model_version"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PRFeaturesRequest) -> PredictionResponse:
    """Predict CI failure risk for a pull request and return top explanatory factors.

    Constructs the feature vector strictly in the model's expected column order,
    evaluates probability via the tuned XGBoost classifier, and classifies risk
    using the calibrated 0.73 operating threshold:
      - High: failure_risk_score >= 0.73
      - Medium: 0.365 <= failure_risk_score < 0.73
      - Low: failure_risk_score < 0.365
    """
    ensure_loaded()

    model = app_state["model"]
    feature_names: List[str] = app_state["feature_names"]
    operating_threshold: float = app_state["operating_threshold"]
    model_version: str = app_state["model_version"]
    explainer = app_state["explainer"]
    explainer_mode: str = app_state["explainer_mode"]
    global_gains: Dict[str, float] = app_state["global_gains"]

    data_dict = request.model_dump()
    row_df = pd.DataFrame([data_dict])[feature_names]

    try:
        proba = float(model.predict_proba(row_df)[0, 1])
    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        )

    failure_risk_score = round(proba, 4)

    medium_threshold = operating_threshold / 2.0
    if failure_risk_score >= operating_threshold:
        risk_label = "high"
    elif failure_risk_score >= medium_threshold:
        risk_label = "medium"
    else:
        risk_label = "low"

    top_factors: List[RiskFactor] = []

    if explainer_mode == "shap" and explainer is not None:
        try:
            shap_values = explainer(row_df).values[0]
            raw_factors = [
                {"feature": feat, "contribution": round(float(val), 4)}
                for feat, val in zip(feature_names, shap_values)
            ]
            sorted_factors = sorted(raw_factors, key=lambda x: x["contribution"], reverse=True)
            top_factors = [RiskFactor(**item) for item in sorted_factors[:3]]
        except Exception as exc:
            logger.warning("SHAP explanation failed on input, using gain fallback: %s", exc)
            explainer_mode = "gain_fallback"

    if not top_factors:
        raw_contributions = {}
        for feat in feature_names:
            val = float(data_dict[feat])
            gain = global_gains.get(feat, 1.0)
            raw_contributions[feat] = val * gain

        total_mag = sum(abs(v) for v in raw_contributions.values()) or 1.0
        normalized_factors = [
            {"feature": feat, "contribution": round(raw_contributions[feat] / total_mag, 4)}
            for feat in feature_names
        ]
        sorted_factors = sorted(normalized_factors, key=lambda x: x["contribution"], reverse=True)
        top_factors = [RiskFactor(**item) for item in sorted_factors[:3]]

    return PredictionResponse(
        failure_risk_score=failure_risk_score,
        risk_label=risk_label,
        top_risk_factors=top_factors,
        model_version=model_version,
    )
