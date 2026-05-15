"""
train_models.py
───────────────
Entraîne, évalue et enregistre dans MLflow les modèles suivants
pour chaque jeu de données :

Régression :    Linear Regression, Random Forest, XGBoost, Gradient Boosting
Classification : Logistic Regression, Random Forest, XGBoost, Gradient Boosting

Le meilleur modèle (R² pour régression, F1 pour classification) est
tagué « champion » dans MLflow et sauvegardé localement.
"""

import os
import json
import joblib
import warnings
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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
from mlflow_tracking import setup_mlflow, experiment_name_for_dataset

warnings.filterwarnings("ignore")

CHAMPIONS_DIR = "models/champions"
os.makedirs(CHAMPIONS_DIR, exist_ok=True)

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": XGBClassifier(
        n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"
    ),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def _log_model_params(model, dataset_key: str, model_name: str, task: str) -> None:
    mlflow.log_param("dataset", dataset_key)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("task", task)
    for k, v in model.get_params().items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            pass


def _tag_champion_run(run_id: str, dataset_key: str) -> None:
    """Marque le run gagnant comme champion dans MLflow."""
    client = MlflowClient()
    experiment_name = experiment_name_for_dataset(dataset_key)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    for run in client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.champion = 'true'",
    ):
        client.delete_tag(run.info.run_id, "champion")

    client.set_tag(run_id, "champion", "true")
    client.set_tag(run_id, "status", "champion")


def _save_champion(dataset_key, model_name, model, metrics, feature_names, run_id=None):
    path = os.path.join(CHAMPIONS_DIR, dataset_key)
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, os.path.join(path, "model.joblib"))

    meta = {
        "dataset": dataset_key,
        "model_name": model_name,
        "task": DATASETS[dataset_key]["task"],
        "metrics": metrics,
        "features": feature_names,
        "mlflow_run_id": run_id,
        "source": "mlflow" if run_id else "local",
    }
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [*] Champion saved: {model_name} -> {path}/")


def train_regression(dataset_key: str) -> dict:
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_dataset(dataset_key)
    results = {}
    best_r2, best_model_name = -np.inf, None
    best_model_obj = None
    best_run_id = None
    best_metrics = None
    feature_names_json = json.dumps(feature_names)

    for model_name, model in REGRESSION_MODELS.items():
        with mlflow.start_run(run_name=f"{dataset_key}__{model_name}") as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            _log_model_params(model, dataset_key, model_name, "regression")
            mlflow.log_param("feature_names", feature_names_json)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            mlflow.sklearn.log_model(model, name="model")

            results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            print(f"  [OK] {model_name:<30s}  R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_model_obj = model
                best_run_id = run.info.run_id
                best_metrics = results[model_name]

    if best_run_id:
        _tag_champion_run(best_run_id, dataset_key)
    _save_champion(dataset_key, best_model_name, best_model_obj, best_metrics, feature_names, best_run_id)
    return results


def train_classification(dataset_key: str) -> dict:
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_dataset(dataset_key)
    results = {}
    best_f1, best_model_name = -np.inf, None
    best_model_obj = None
    best_run_id = None
    best_metrics = None
    feature_names_json = json.dumps(feature_names)

    for model_name, model in CLASSIFICATION_MODELS.items():
        with mlflow.start_run(run_name=f"{dataset_key}__{model_name}") as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:
                auc = 0.0

            _log_model_params(model, dataset_key, model_name, "classification")
            mlflow.log_param("feature_names", feature_names_json)
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("AUC", auc)
            mlflow.sklearn.log_model(model, name="model")

            results[model_name] = {"Accuracy": acc, "F1_Score": f1, "AUC": auc}
            print(f"  [OK] {model_name:<30s}  F1={f1:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_model_obj = model
                best_run_id = run.info.run_id
                best_metrics = results[model_name]

    if best_run_id:
        _tag_champion_run(best_run_id, dataset_key)
    _save_champion(dataset_key, best_model_name, best_model_obj, best_metrics, feature_names, best_run_id)
    return results


def train_all():
    for key, info in DATASETS.items():
        experiment_name = experiment_name_for_dataset(key)
        setup_mlflow(experiment_name)
        print(f"\n{'=' * 60}")
        print(f" Dataset: {key}  |  Task: {info['task']}")
        print(f"{'=' * 60}")

        if info["task"] == "regression":
            train_regression(key)
        else:
            train_classification(key)


if __name__ == "__main__":
    train_all()
    print("\n[DONE] Training completed for all datasets.")
