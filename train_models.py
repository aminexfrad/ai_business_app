"""
train_models.py
───────────────
Entraîne, évalue et enregistre dans MLflow les modèles suivants
pour chaque jeu de données :

Régression :    Linear Regression, Random Forest, XGBoost, Gradient Boosting
Classification : Logistic Regression, Random Forest, XGBoost, Gradient Boosting

Le meilleur modèle (R² pour régression, F1 pour classification) est
automatiquement tagué « champion » et sauvegardé localement.
"""

import os
import json
import joblib
import warnings
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from data_preprocessing import DATASETS, prepare_dataset
from mlflow_tracking import setup_mlflow

warnings.filterwarnings("ignore")

# ─── Dossier de sauvegarde local des champions ─────────────────────────────────
CHAMPIONS_DIR = "models/champions"
os.makedirs(CHAMPIONS_DIR, exist_ok=True)

# ─── Définition des modèles ───────────────────────────────────────────────────

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


# ─── Fonctions d'entraînement ─────────────────────────────────────────────────


def train_regression(dataset_key: str) -> dict:
    """Entraîne les 4 modèles de régression, log dans MLflow et retourne le résumé."""
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_dataset(dataset_key)
    results = {}
    best_r2, best_model_name = -np.inf, None

    for model_name, model in REGRESSION_MODELS.items():
        with mlflow.start_run(run_name=f"{dataset_key}__{model_name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            # Log des paramètres
            mlflow.log_param("dataset", dataset_key)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("task", "regression")
            params = model.get_params()
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            # Log des métriques
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # Log du modèle
            mlflow.sklearn.log_model(model, artifact_path="model")

            results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            print(f"  ✓ {model_name:<30s}  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_model_obj = model

    # Sauvegarde locale du champion
    _save_champion(dataset_key, best_model_name, best_model_obj, results[best_model_name], feature_names)
    return results


def train_classification(dataset_key: str) -> dict:
    """Entraîne les 4 modèles de classification, log dans MLflow et retourne le résumé."""
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_dataset(dataset_key)
    results = {}
    best_f1, best_model_name = -np.inf, None

    for model_name, model in CLASSIFICATION_MODELS.items():
        with mlflow.start_run(run_name=f"{dataset_key}__{model_name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:
                auc = 0.0

            mlflow.log_param("dataset", dataset_key)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("task", "classification")
            params = model.get_params()
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("AUC", auc)

            mlflow.sklearn.log_model(model, artifact_path="model")

            results[model_name] = {"Accuracy": acc, "F1_Score": f1, "AUC": auc}
            print(f"  ✓ {model_name:<30s}  F1={f1:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_model_obj = model

    _save_champion(dataset_key, best_model_name, best_model_obj, results[best_model_name], feature_names)
    return results


# ─── Utilitaire interne ───────────────────────────────────────────────────────


def _save_champion(dataset_key, model_name, model, metrics, feature_names):
    """Sauvegarde le modèle champion localement (joblib + métadonnées JSON)."""
    path = os.path.join(CHAMPIONS_DIR, dataset_key)
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, os.path.join(path, "model.joblib"))

    meta = {
        "dataset": dataset_key,
        "model_name": model_name,
        "task": DATASETS[dataset_key]["task"],
        "metrics": metrics,
        "features": feature_names,
    }
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  ★ Champion sauvegardé : {model_name} → {path}/")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def train_all():
    """Entraîne tous les modèles pour chaque dataset."""
    for key, info in DATASETS.items():
        experiment_name = f"AI_BI_{key}"
        setup_mlflow(experiment_name)
        print(f"\n{'='*60}")
        print(f" Dataset : {key}  |  Tâche : {info['task']}")
        print(f"{'='*60}")

        if info["task"] == "regression":
            train_regression(key)
        else:
            train_classification(key)


if __name__ == "__main__":
    train_all()
    print("\n✅ Entraînement terminé pour tous les datasets.")
