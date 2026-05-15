"""
mlflow_tracking.py
──────────────────
Configure DagsHub as the MLflow remote tracking server.
Call `setup_mlflow(experiment_name)` before starting any run.
"""

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

_mlflow_configured = False


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def experiment_name_for_dataset(dataset_key: str) -> str:
    return f"AI_BI_{dataset_key}"


def get_dagshub_url() -> str:
    username = _env("DAGSHUB_USERNAME")
    repo = _env("DAGSHUB_REPO_NAME", "ai_business_app")
    if username and repo:
        return f"https://dagshub.com/{username}/{repo}"
    return ""


def configure_mlflow_tracking() -> str:
    """Configure MLflow tracking URI once (DagsHub or local). Returns tracking mode."""
    global _mlflow_configured
    username = _env("DAGSHUB_USERNAME")
    token = _env("DAGSHUB_TOKEN")
    repo = _env("DAGSHUB_REPO_NAME", "ai_business_app")

    if _mlflow_configured:
        return "dagshub" if username and token else "local"

    if username and token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        tracking_uri = f"https://dagshub.com/{username}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        _mlflow_configured = True
        return "dagshub"

    _use_local_tracking()
    _mlflow_configured = True
    return "local"


def setup_mlflow(experiment_name: str) -> None:
    """Initialise DagsHub + MLflow tracking URI and set the experiment."""
    mode = configure_mlflow_tracking()
    if mode == "dagshub":
        print(f"[MLflow] DagsHub tracking: {_env('DAGSHUB_USERNAME')}/{_env('DAGSHUB_REPO_NAME', 'ai_business_app')}")
    else:
        print("[MLflow] Local tracking (set DAGSHUB_USERNAME and DAGSHUB_TOKEN in .env)")

    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Experiment: '{experiment_name}'")


def _use_local_tracking() -> None:
    """Fallback: use a local MLflow tracking server."""
    from pathlib import Path

    mlruns_path = Path("mlruns").resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)
    local_uri = mlruns_path.as_uri()
    mlflow.set_tracking_uri(local_uri)
    print(f"[MLflow] Local tracking URI: {local_uri}")
