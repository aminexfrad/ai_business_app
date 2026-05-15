"""
utils.py
────────
Fonctions utilitaires :
- Logging des actions utilisateur dans un fichier CSV
- Chargement du modèle champion depuis le dossier local
- Prédiction unitaire et batch
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import joblib

from data_preprocessing import DATASETS, load_data, clean_data, encode_features

# ─── Constantes ────────────────────────────────────────────────────────────────
CHAMPIONS_DIR = "models/champions"
LOG_FILE = "logs/actions.csv"


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def log_action(action: str, details: str = "") -> None:
    """Enregistre une action dans le fichier CSV de logs."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action": action,
        "details": details,
    }

    file_exists = os.path.exists(LOG_FILE)
    df = pd.DataFrame([entry])
    df.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAMPION MODEL
# ═══════════════════════════════════════════════════════════════════════════════


def load_champion(dataset_key: str):
    """
    Charge le modèle champion et ses métadonnées.
    Retourne (model, metadata_dict) ou (None, None) si introuvable.
    """
    path = os.path.join(CHAMPIONS_DIR, dataset_key)
    model_path = os.path.join(path, "model.joblib")
    meta_path = os.path.join(path, "metadata.json")

    if not os.path.exists(model_path):
        return None, None

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def get_champion_metrics(dataset_key: str) -> dict | None:
    """Retourne uniquement les métriques du champion (ou None)."""
    _, meta = load_champion(dataset_key)
    if meta is None:
        return None
    return meta.get("metrics")


def get_champion_name(dataset_key: str) -> str:
    """Retourne le nom du modèle champion."""
    _, meta = load_champion(dataset_key)
    if meta is None:
        return "Non entraîné"
    return meta.get("model_name", "Inconnu")


# ═══════════════════════════════════════════════════════════════════════════════
#  PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════


def predict_single(dataset_key: str, input_dict: dict) -> float | int:
    """
    Prédiction unitaire à partir d'un dict de features brutes.
    Le dict est converti en DataFrame, encodé comme à l'entraînement,
    puis aligné sur les colonnes du champion.
    """
    model, meta = load_champion(dataset_key)
    if model is None:
        raise RuntimeError(f"Aucun champion trouvé pour '{dataset_key}'. Lancez train_models.py d'abord.")

    expected_features = meta["features"]

    # Construire un DataFrame à partir de l'input
    df_input = pd.DataFrame([input_dict])

    # Encoder les catégorielles (même one-hot que l'entraînement)
    target = DATASETS[dataset_key]["target"]
    df_input = encode_features(df_input, target)

    # Aligner les colonnes sur celles attendues par le modèle
    df_input = df_input.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(df_input)[0]

    log_action("prediction_single", f"dataset={dataset_key}, input={input_dict}, result={prediction}")
    return prediction


def predict_batch(dataset_key: str, df_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Prédiction batch : prend un DataFrame brut (mêmes colonnes que le CSV
    original, sans la colonne cible) et retourne le DataFrame enrichi
    d'une colonne 'prediction'.
    """
    model, meta = load_champion(dataset_key)
    if model is None:
        raise RuntimeError(f"Aucun champion trouvé pour '{dataset_key}'. Lancez train_models.py d'abord.")

    expected_features = meta["features"]
    target = DATASETS[dataset_key]["target"]

    # Supprimer la colonne cible si elle est présente (l'utilisateur peut l'avoir incluse)
    if target in df_batch.columns:
        df_batch = df_batch.drop(columns=[target])

    # Nettoyage + encodage
    df_clean = clean_data(df_batch.copy())
    df_encoded = encode_features(df_clean, target)

    # Aligner
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)

    predictions = model.predict(df_aligned)
    df_batch = df_batch.copy()
    df_batch["prediction"] = predictions

    log_action("prediction_batch", f"dataset={dataset_key}, rows={len(df_batch)}")
    return df_batch


def get_all_champions_summary() -> pd.DataFrame:
    """Retourne un DataFrame résumant tous les champions disponibles."""
    rows = []
    for key, info in DATASETS.items():
        model, meta = load_champion(key)
        if meta is not None:
            row = {
                "Dataset": key,
                "Tâche": info["task"],
                "Champion": meta["model_name"],
            }
            row.update(meta["metrics"])
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
