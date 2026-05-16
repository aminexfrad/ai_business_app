"""Central configuration for the AI Business Intelligence App."""
import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets (cloud deployment)."""
    val = os.getenv(key, "")
    if val:
        return val
    # Try Streamlit secrets (available when running on Streamlit Cloud)
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

# ─── DagsHub / MLflow ────────────────────────────────────────────────────────
DAGSHUB_USERNAME = _get("DAGSHUB_USERNAME")
DAGSHUB_TOKEN    = _get("DAGSHUB_TOKEN")
DAGSHUB_REPO     = _get("DAGSHUB_REPO_NAME", "ai_business_app")
MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)

# ─── Google Gemini ────────────────────────────────────────────────────────────
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
GEMINI_MODEL = _get("GEMINI_MODEL")  # optional; empty = try several models

# ─── Datasets ────────────────────────────────────────────────────────────────
DATASETS = {
    "Building Energy (Regression)": {
        "file": "building_energy_regression_realistic.csv",
        "target": "energy_consumption",
        "task": "regression",
        "description": "Predict building energy consumption based on surface, temperature, occupancy, etc.",
    },
    "Restaurant Revenue (Regression)": {
        "file": "restaurant_revenue_regression_realistic.csv",
        "target": "monthly_revenue",
        "task": "regression",
        "description": "Predict monthly restaurant revenue from customers, orders, marketing spend, etc.",
    },
    "Fraud Detection (Classification)": {
        "file": "fraud_detection_classification_realistic.csv",
        "target": "fraud",
        "task": "classification",
        "description": "Detect fraudulent transactions based on amount, frequency, risk scores, etc.",
    },
    "Streaming Churn (Classification)": {
        "file": "streaming_subscription_classification_realistic.csv",
        "target": "subscription_cancelled",
        "task": "classification",
        "description": "Predict subscription cancellation from watch time, cost, support requests, etc.",
    },
}

# ─── ML Models ────────────────────────────────────────────────────────────────
REGRESSION_MODELS = [
    "Linear Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "XGBoost Regressor",
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "XGBoost Classifier",
]
