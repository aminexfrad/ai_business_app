"""Action logger — keeps traces of predictions, uploads, and user interactions."""
import json
import os
from datetime import datetime


LOG_FILE = "app_logs.jsonl"


def _log(event_type: str, data: dict):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **data,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_prediction(dataset: str, model: str, inputs: dict, result):
    _log("prediction", {
        "dataset": dataset,
        "model": model,
        "inputs": str(inputs),
        "result": str(result),
    })


def log_batch_upload(dataset: str, model: str, n_rows: int):
    _log("batch_upload", {
        "dataset": dataset,
        "model": model,
        "n_rows": n_rows,
    })


def log_training(dataset: str, models_trained: list, champion: str):
    _log("training", {
        "dataset": dataset,
        "models_trained": models_trained,
        "champion": champion,
    })


def log_gemini_query(dataset: str, question: str):
    _log("gemini_query", {
        "dataset": dataset,
        "question": question,
    })


def get_logs() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    with open(LOG_FILE) as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except Exception:
                pass
    return logs
