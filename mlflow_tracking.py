"""
mlflow_tracking.py
──────────────────
Configure DagsHub as the MLflow remote tracking server.
Call `setup_mlflow(experiment_name)` before starting any run.
"""

import os
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_TOKEN    = os.getenv("DAGSHUB_TOKEN", "")
DAGSHUB_REPO     = os.getenv("DAGSHUB_REPO_NAME", "ai_business_app")


def setup_mlflow(experiment_name: str) -> None:
    """Initialise DagsHub + MLflow tracking URI and set the experiment."""
    if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
        # Authentification via token (non-interactive)
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        try:
            dagshub.init(
                repo_owner=DAGSHUB_USERNAME,
                repo_name=DAGSHUB_REPO,
                mlflow=True,
            )
            print(f"[MLflow] DagsHub tracking activé pour {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
        except Exception as e:
            print(f"[MLflow] Impossible de se connecter à DagsHub : {e}")
            _use_local_tracking()
    else:
        print("[MLflow] Aucun token DagsHub trouvé — tracking local activé.")
        _use_local_tracking()

    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Expérience : '{experiment_name}'")


def _use_local_tracking() -> None:
    """Fallback: use a local MLflow tracking server."""
    local_uri = "mlruns"
    mlflow.set_tracking_uri(local_uri)
    print(f"[MLflow] Tracking URI local : {local_uri}/")
