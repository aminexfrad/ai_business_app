"""
ML Pipeline — Training, preprocessing, MLflow tracking, champion selection.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

import config

# ─── MLflow / DagsHub init ───────────────────────────────────────────────────

def init_mlflow():
    """Authenticate with DagsHub and set MLflow tracking URI."""
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = config.DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = config.DAGSHUB_TOKEN
        dagshub.init(
            repo_owner=config.DAGSHUB_USERNAME,
            repo_name=config.DAGSHUB_REPO,
            mlflow=True
        )
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        return True, config.MLFLOW_TRACKING_URI
    except Exception as e:
        return False, str(e)


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target: str):
    """
    Clean data, handle missing values, encode categoricals.
    Returns X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols.
    """
    df = df.copy()
    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols


# ─── Model registry ─────────────────────────────────────────────────────────

def get_model(name: str):
    models = {
        "Linear Regression":               LinearRegression(),
        "Random Forest Regressor":         RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor":     GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor":               XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "Logistic Regression":             LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting Classifier":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost Classifier":              XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
    }
    return models.get(name)


# ─── Training + MLflow Tracking ──────────────────────────────────────────────

def train_and_track(df: pd.DataFrame, dataset_name: str, target: str,
                    task: str, model_names: list,
                    progress_callback=None) -> dict:
    """Train multiple models, track every run with MLflow on DagsHub."""
    init_mlflow()

    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = \
        preprocess(df, target)

    # Sanitize experiment name
    exp_name = dataset_name.split("(")[0].strip().replace(" ", "_") + f"_{task}"
    mlflow.set_experiment(exp_name)

    results = {}

    for i, model_name in enumerate(model_names):
        if progress_callback:
            progress_callback(i, len(model_names), model_name)

        model = get_model(model_name)
        if model is None:
            continue

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        with mlflow.start_run(run_name=model_name):
            # ── Log parameters
            mlflow.log_param("model_name",   model_name)
            mlflow.log_param("dataset",      dataset_name)
            mlflow.log_param("task",         task)
            mlflow.log_param("target",       target)
            mlflow.log_param("train_size",   len(X_train))
            mlflow.log_param("test_size",    len(X_test))
            mlflow.log_param("n_num_features", len(num_cols))
            mlflow.log_param("n_cat_features", len(cat_cols))
            mlflow.log_param("total_features", len(num_cols) + len(cat_cols))

            # ── Train
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            # ── Compute & log metrics
            if task == "regression":
                mse  = mean_squared_error(y_test, preds)
                rmse = float(np.sqrt(mse))
                mae  = float(mean_absolute_error(y_test, preds))
                r2   = float(r2_score(y_test, preds))
                mlflow.log_metric("mse",  float(mse))
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae",  mae)
                mlflow.log_metric("r2",   r2)
                metrics = {"rmse": rmse, "mae": mae, "r2": r2}
                champion_metric = r2   # higher is better

            else:
                acc  = float(accuracy_score(y_test, preds))
                f1   = float(f1_score(y_test, preds, average="weighted"))
                prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
                rec  = float(recall_score(y_test, preds, average="weighted", zero_division=0))
                try:
                    proba = pipeline.predict_proba(X_test)
                    auc = float(roc_auc_score(y_test, proba[:, 1])) \
                          if proba.shape[1] == 2 else \
                          float(roc_auc_score(y_test, proba, multi_class="ovr"))
                except Exception:
                    auc = 0.0
                mlflow.log_metric("accuracy",  acc)
                mlflow.log_metric("f1_score",  f1)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall",    rec)
                mlflow.log_metric("auc",       auc)
                metrics = {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc}
                champion_metric = f1

            # ── Log model artifact
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            run_id = mlflow.active_run().info.run_id

            results[model_name] = {
                "metrics": metrics,
                "champion_metric": champion_metric,
                "run_id": run_id,
                "pipeline": pipeline,
            }

    return results, X_test, y_test


# ─── Champion Management ─────────────────────────────────────────────────────

def get_champion(results: dict) -> tuple:
    """Return (name, info) for the best-scoring model."""
    if not results:
        return None, None
    return max(results.items(), key=lambda x: x[1]["champion_metric"])


def register_champion_in_mlflow(run_id: str, registered_name: str = "champion_model"):
    """
    Register the champion run in the MLflow Model Registry
    and tag it with the 'champion' alias so it can be loaded by URI.
    """
    try:
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, registered_name)
        client = mlflow.tracking.MlflowClient()
        try:
            client.set_registered_model_alias(registered_name, "champion", mv.version)
        except Exception:
            pass  # older MLflow versions don't support aliases
        return True, mv.version
    except Exception as e:
        return False, str(e)


def load_champion_from_mlflow(registered_name: str = "champion_model"):
    """
    Load champion model from MLflow Model Registry.
    Falls back to the best run across all experiments if registry fails.
    """
    init_mlflow()

    # Try alias first (MLflow >= 2.x)
    try:
        pipeline = mlflow.sklearn.load_model(f"models:/{registered_name}@champion")
        return pipeline, f"MLflow Registry — {registered_name}@champion"
    except Exception:
        pass

    # Try latest version
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(registered_name)
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            pipeline = mlflow.sklearn.load_model(f"models:/{registered_name}/{latest.version}")
            return pipeline, f"MLflow Registry — {registered_name} v{latest.version}"
    except Exception:
        pass

    # Last resort: best run by metric across all experiments
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        best_run, best_score = None, -float("inf")
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.r2 DESC"],
                max_results=5,
            )
            for run in runs:
                m = run.data.metrics
                score = m.get("r2", m.get("f1_score", -999))
                if score > best_score:
                    best_score = score
                    best_run = run
        if best_run:
            pipeline = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
            return pipeline, f"Best run: {best_run.info.run_id[:8]}… (score={best_score:.4f})"
    except Exception as e:
        return None, f"MLflow load failed: {e}"

    return None, "No model found in MLflow"


def save_champion_locally(pipeline, path: str = "champion_model.pkl"):
    """Persist champion to disk as backup."""
    joblib.dump(pipeline, path)


def load_champion_locally(path: str = "champion_model.pkl"):
    """Load champion from local disk (fallback when MLflow is unreachable)."""
    if os.path.exists(path):
        return joblib.load(path)
    return None
