"""
data_preprocessing.py
─────────────────────
Fonctions de chargement, nettoyage, encodage (One-Hot) et split train/test
pour les 4 jeux de données du projet AI Business Intelligence.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ─── Registre des datasets ────────────────────────────────────────────────────

DATASETS = {
    "building_energy": {
        "file": "data/building_energy_regression_realistic.csv",
        "target": "energy_consumption",
        "task": "regression",
        "description": (
            "🏢 Consommation énergétique des bâtiments\n\n"
            "Prédire la consommation d'énergie (kWh) d'un bâtiment en fonction "
            "de sa surface, la température extérieure, le taux d'occupation, "
            "l'utilisation des équipements, l'efficacité des panneaux solaires, "
            "le score d'isolation, le type de bâtiment et la ville."
        ),
    },
    "restaurant_revenue": {
        "file": "data/restaurant_revenue_regression_realistic.csv",
        "target": "monthly_revenue",
        "task": "regression",
        "description": (
            "🍽️ Chiffre d'affaires mensuel des restaurants\n\n"
            "Prédire le revenu mensuel d'un restaurant à partir du nombre "
            "de clients quotidiens, la valeur moyenne de commande, le budget "
            "marketing, les commandes de livraison, le nombre d'employés, "
            "la note client, le score réseaux sociaux, le type de restaurant et la ville."
        ),
    },
    "fraud_detection": {
        "file": "data/fraud_detection_classification_realistic.csv",
        "target": "fraud",
        "task": "classification",
        "description": (
            "🔒 Détection de fraude transactionnelle\n\n"
            "Classifier une transaction comme frauduleuse (1) ou légitime (0) "
            "selon le montant, la fréquence de transaction, l'âge du compte, "
            "le score de risque de l'appareil, les tentatives de connexion, "
            "le score marchand, le score d'utilisation de la carte, "
            "la méthode de paiement et le pays."
        ),
    },
    "streaming_subscription": {
        "file": "data/streaming_subscription_classification_realistic.csv",
        "target": "subscription_cancelled",
        "task": "classification",
        "description": (
            "📺 Résiliation d'abonnement streaming\n\n"
            "Prédire si un abonné va résilier (1) ou non (0) son abonnement "
            "selon ses heures de visionnage, le coût mensuel, les demandes "
            "de support, les jours depuis la dernière connexion, les clics "
            "de recommandation, le ratio d'utilisation mobile, le plan "
            "d'abonnement et la catégorie favorite."
        ),
    },
}


# ─── Fonctions ─────────────────────────────────────────────────────────────────


def load_data(dataset_key: str) -> pd.DataFrame:
    """Charge un CSV depuis le registre DATASETS et retourne un DataFrame."""
    info = DATASETS[dataset_key]
    filepath = info["file"]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage générique :
    - Suppression des doublons
    - Imputation des valeurs manquantes :
        • numériques → médiane
        • catégorielles → mode
    """
    df = df.drop_duplicates()

    # Colonnes numériques
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Colonnes catégorielles
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    One-Hot Encoding des colonnes catégorielles (hors target).
    Retourne le DataFrame encodé.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Sépare le DataFrame en X_train, X_test, y_train, y_test.
    Retourne aussi les noms de features pour les formulaires Streamlit.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def get_feature_info(dataset_key: str) -> dict:
    """
    Retourne un dictionnaire { col_name: dtype } pour les colonnes
    BRUTES (avant encodage) du dataset, hors colonne cible.
    Utile pour le formulaire de saisie dans Streamlit.
    """
    df = load_data(dataset_key)
    target = DATASETS[dataset_key]["target"]
    feature_cols = [c for c in df.columns if c != target]

    info = {}
    for col in feature_cols:
        if df[col].dtype == "object":
            info[col] = {"type": "categorical", "values": sorted(df[col].dropna().unique().tolist())}
        else:
            info[col] = {
                "type": "numerical",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
    return info


def get_categorical_columns(dataset_key: str) -> list[str]:
    """Colonnes catégorielles brutes (hors cible) pour un dataset."""
    df = load_data(dataset_key)
    target = DATASETS[dataset_key]["target"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cat_cols if c != target]


def encode_for_prediction(df: pd.DataFrame, dataset_key: str, expected_features: list[str]) -> pd.DataFrame:
    """
    Encode un DataFrame brut et aligne les colonnes sur celles du modèle entraîné.
    """
    target = DATASETS[dataset_key]["target"]
    if target in df.columns:
        df = df.drop(columns=[target])
    df = clean_data(df)
    df = encode_features(df, target)
    return df.reindex(columns=expected_features, fill_value=0)


def prepare_dataset(dataset_key: str):
    """
    Pipeline complet : chargement → nettoyage → encodage → split.
    Retourne (X_train, X_test, y_train, y_test, feature_names, df_encoded).
    """
    df = load_data(dataset_key)
    df = clean_data(df)
    target = DATASETS[dataset_key]["target"]
    df = encode_features(df, target)

    X_train, X_test, y_train, y_test = split_data(df, target)
    feature_names = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, feature_names, df
