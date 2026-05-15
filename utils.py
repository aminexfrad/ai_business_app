"""
utils.py
────────
Fonctions utilitaires :
- Logging des actions utilisateur dans un fichier CSV
- Chargement du modèle champion depuis MLflow (prioritaire) ou local
- Prédiction unitaire et batch
"""

import os
import json
import datetime
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from data_preprocessing import DATASETS, encode_for_prediction
from mlflow_tracking import configure_mlflow_tracking, experiment_name_for_dataset

CHAMPIONS_DIR = "models/champions"
LOG_FILE = "logs/actions.csv"

METRIC_KEYS = {
    "regression": ["R2", "RMSE", "MAE"],
    "classification": ["F1_Score", "Accuracy", "AUC"],
}


def log_action(action: str, details: str = "") -> None:
    """Enregistre une action dans le fichier CSV de logs."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "details": details,
    }

    file_exists = os.path.exists(LOG_FILE)
    df = pd.DataFrame([entry])
    df.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)


def get_action_logs(limit: int = 100) -> pd.DataFrame:
    """Retourne les dernières actions enregistrées."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "action", "details"])
    df = pd.read_csv(LOG_FILE)
    return df.tail(limit).iloc[::-1]


def _metrics_from_run_row(row, task: str) -> dict:
    metrics = {}
    for key in METRIC_KEYS.get(task, []):
        col = f"metrics.{key}"
        if col in row.index and pd.notna(row[col]):
            metrics[key] = float(row[col])
    return metrics


def _features_from_run_row(row) -> list:
    if "params.feature_names" not in row.index or pd.isna(row["params.feature_names"]):
        return []
    try:
        return json.loads(row["params.feature_names"])
    except (json.JSONDecodeError, TypeError):
        return []


def load_champion_from_mlflow(dataset_key: str):
    """
    Charge le modèle champion depuis MLflow (run tagué champion=true).
    Retourne (model, metadata_dict) ou (None, None).
    """
    configure_mlflow_tracking()
    experiment_name = experiment_name_for_dataset(dataset_key)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None, None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.champion = 'true'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None, None

    row = runs.iloc[0]
    run_id = row["run_id"]
    task = DATASETS[dataset_key]["task"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    meta = {
        "dataset": dataset_key,
        "model_name": row.get("params.model_name", "Unknown"),
        "task": task,
        "metrics": _metrics_from_run_row(row, task),
        "features": _features_from_run_row(row),
        "mlflow_run_id": run_id,
        "source": "mlflow",
    }
    return model, meta


def load_champion_local(dataset_key: str):
    """Charge le modèle champion depuis le dossier local."""
    path = os.path.join(CHAMPIONS_DIR, dataset_key)
    model_path = os.path.join(path, "model.joblib")
    meta_path = os.path.join(path, "metadata.json")

    if not os.path.exists(model_path):
        return None, None

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.setdefault("source", "local")
    return model, meta


def _merge_champion_metadata(mlflow_meta: dict, local_meta: dict | None) -> dict:
    """Complete les metadonnees MLflow avec le fichier local si besoin."""
    if not local_meta:
        return mlflow_meta
    if not mlflow_meta.get("features"):
        mlflow_meta["features"] = local_meta.get("features", [])
    if not mlflow_meta.get("metrics"):
        mlflow_meta["metrics"] = local_meta.get("metrics", {})
    return mlflow_meta


def load_champion(dataset_key: str):
    """
    Charge le modèle champion : MLflow en priorité, puis fallback local.
    Retourne (model, metadata_dict) ou (None, None).
    """
    local_model, local_meta = load_champion_local(dataset_key)
    model, meta = load_champion_from_mlflow(dataset_key)
    if model is not None and meta is not None:
        return model, _merge_champion_metadata(meta, local_meta)
    if local_model is not None:
        return local_model, local_meta
    return None, None


def get_champion_source(dataset_key: str) -> str:
    """Indique la source du champion : mlflow, local ou unavailable."""
    _, meta = load_champion(dataset_key)
    if meta is None:
        return "unavailable"
    return meta.get("source", "local")


def get_champion_metrics(dataset_key: str) -> dict | None:
    _, meta = load_champion(dataset_key)
    if meta is None:
        return None
    return meta.get("metrics")


def get_champion_name(dataset_key: str) -> str:
    _, meta = load_champion(dataset_key)
    if meta is None:
        return "Non entraine"
    return meta.get("model_name", "Inconnu")


def predict_single(
    dataset_key: str,
    input_dict: dict,
    model=None,
    meta=None,
) -> float | int:
    if model is None or meta is None:
        model, meta = load_champion(dataset_key)
    if model is None:
        raise RuntimeError(
            f"Aucun champion pour '{dataset_key}'. Lancez: python train_models.py"
        )

    expected_features = meta["features"]
    df_input = encode_for_prediction(pd.DataFrame([input_dict]), dataset_key, expected_features)
    prediction = model.predict(df_input)[0]

    log_action("prediction_single", f"dataset={dataset_key}, result={prediction}")
    return prediction


def predict_batch(
    dataset_key: str,
    df_batch: pd.DataFrame,
    model=None,
    meta=None,
) -> pd.DataFrame:
    if model is None or meta is None:
        model, meta = load_champion(dataset_key)
    if model is None:
        raise RuntimeError(
            f"Aucun champion pour '{dataset_key}'. Lancez: python train_models.py"
        )

    expected_features = meta["features"]
    target = DATASETS[dataset_key]["target"]

    if target in df_batch.columns:
        df_batch = df_batch.drop(columns=[target])

    df_aligned = encode_for_prediction(df_batch.copy(), dataset_key, expected_features)
    predictions = model.predict(df_aligned)

    df_result = df_batch.copy()
    df_result["prediction"] = predictions

    log_action("prediction_batch", f"dataset={dataset_key}, rows={len(df_result)}")
    return df_result


def get_all_champions_summary() -> pd.DataFrame:
    rows = []
    for key, info in DATASETS.items():
        _, meta = load_champion(key)
        if meta is not None:
            row = {
                "Dataset": key,
                "Tache": info["task"],
                "Champion": meta["model_name"],
                "Source": meta.get("source", "local"),
            }
            row.update(meta.get("metrics", {}))
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
