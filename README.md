# AI Business Intelligence App

Application complète de **Machine Learning**, **MLOps** et **IA générative** pour l'analyse de données business.

[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194e2?logo=mlflow)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Repo-9B59B6)](https://dagshub.com)

**Liens du projet :**
- GitHub : https://github.com/aminexfrad/ai_business_app
- DagsHub : https://dagshub.com/aminexfrad/ai_business_app
- Présentation orale : voir [PRESENTATION.md](PRESENTATION.md)
- Livrables examen : voir [LIVRABLES.md](LIVRABLES.md)

---

## Description

La plateforme analyse **4 jeux de données business** et entraîne automatiquement des modèles ML pour chacun :

| Dataset | Tâche | Cible |
|---------|-------|-------|
| Énergie Bâtiments | Régression | `energy_consumption` |
| Revenus Restaurants | Régression | `monthly_revenue` |
| Détection Fraude | Classification | `fraud` |
| Résiliation Streaming | Classification | `subscription_cancelled` |

### Modèles comparés

- **Régression** : Linear Regression, Random Forest, XGBoost, Gradient Boosting
- **Classification** : Logistic Regression, Random Forest, XGBoost, Gradient Boosting

Le **modèle champion** (meilleur R² ou F1) est tagué dans MLflow et chargé automatiquement par l'application Streamlit.

---

## Fonctionnalités (conformité examen)

| Partie | Fonctionnalité | Fichier |
|--------|----------------|---------|
| **1 — ML & MLOps** | Nettoyage, imputation, encodage, split | `data_preprocessing.py` |
| | Entraînement multi-modèles + tracking MLflow/DagsHub | `train_models.py` |
| | Champion auto + tag `champion=true` | `train_models.py` |
| | Chargement champion depuis MLflow | `utils.py` |
| **2 — Streamlit** | 5 onglets (Présentation, Unitaire, Batch, Gemini, Dashboard) | `app.py` |
| **3 — Cloud** | Secrets `.env` / Streamlit | `.env.example`, `.streamlit/` |
| **4 — Logging** | Prédictions, uploads, interactions, Gemini | `logs/actions.csv` |

---

## Installation locale

```bash
git clone https://github.com/aminexfrad/ai_business_app.git
cd ai_business_app

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Remplir DAGSHUB_USERNAME, DAGSHUB_TOKEN, GOOGLE_API_KEY

python train_models.py
streamlit run app.py
```

---

## Structure du projet

```
ai_business_app/
├── data/                    # 4 datasets CSV
├── models/champions/        # Sauvegarde locale (fallback)
├── logs/                    # Journal MLOps (actions.csv)
├── app.py                   # Application Streamlit
├── data_preprocessing.py    # Pipeline données
├── train_models.py          # Entraînement + MLflow
├── mlflow_tracking.py       # Configuration DagsHub
├── utils.py                 # Champion, prédictions, logs
├── PRESENTATION.md          # Guide soutenance 5 min
├── LIVRABLES.md             # Liens et checklist examen
├── requirements.txt
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## Déploiement Streamlit Cloud

1. Pousser le code sur GitHub.
2. Exécuter `python train_models.py` pour enregistrer les runs sur DagsHub.
3. Déployer sur [share.streamlit.io](https://share.streamlit.io) avec `app.py`.
4. Configurer les secrets (voir `.streamlit/secrets.toml.example`).

L'application charge le champion depuis **MLflow/DagsHub** en priorité.

---

## Technologies

Python · Streamlit · scikit-learn · XGBoost · MLflow · DagsHub · Google Gemini · Plotly · Git/GitHub

---

Projet académique — Examen MLOps & IA Générative.
