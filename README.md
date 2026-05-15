# AI Business Intelligence App 🧠

Application complète de **Machine Learning** et **MLOps** pour l'analyse de données business.

[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194e2?logo=mlflow)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Repo-9B59B6)](https://dagshub.com)

---

## 📋 Description

Cette application analyse **4 jeux de données business** et entraîne automatiquement des modèles de Machine Learning pour chacun :

| Dataset | Tâche | Cible |
|---|---|---|
| 🏢 Énergie Bâtiments | Régression | `energy_consumption` |
| 🍽️ Revenus Restaurants | Régression | `monthly_revenue` |
| 🔒 Détection Fraude | Classification | `fraud` |
| 📺 Résiliation Streaming | Classification | `subscription_cancelled` |

### Modèles entraînés

- **Régression** : Linear Regression, Random Forest, XGBoost, Gradient Boosting
- **Classification** : Logistic Regression, Random Forest, XGBoost, Gradient Boosting

Le **meilleur modèle** (champion) est automatiquement sélectionné selon R² (régression) ou F1-Score (classification).

---

## 🚀 Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/<votre-user>/ai_business_app.git
cd ai_business_app

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
cp .env.example .env
# Remplir DAGSHUB_USERNAME, DAGSHUB_TOKEN, GOOGLE_API_KEY dans .env

# 5. Entraîner les modèles
python train_models.py

# 6. Lancer l'application
streamlit run app.py
```

---

## 📁 Structure du projet

```
ai_business_app/
├── data/
│   ├── building_energy_regression_realistic.csv
│   ├── restaurant_revenue_regression_realistic.csv
│   ├── fraud_detection_classification_realistic.csv
│   └── streaming_subscription_classification_realistic.csv
├── models/
│   └── champions/           # Modèles champions sauvegardés
├── logs/
│   └── actions.csv          # Logs des actions utilisateur
├── app.py                   # Application Streamlit (5 onglets)
├── data_preprocessing.py    # Chargement, nettoyage, encodage, split
├── train_models.py          # Entraînement + tracking MLflow
├── mlflow_tracking.py       # Configuration DagsHub / MLflow
├── utils.py                 # Logging, chargement champion, prédiction
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Technologies

- **Python 3.10+**
- **Streamlit** — Interface web interactive
- **scikit-learn / XGBoost** — Modèles ML
- **MLflow** — Tracking des expériences
- **DagsHub** — Hébergement MLflow distant
- **Google Gemini** — Analyse IA générative
- **Plotly** — Visualisations interactives

---

## ☁️ Déploiement Streamlit Cloud

1. Pousser le repo sur GitHub.
2. Aller sur [share.streamlit.io](https://share.streamlit.io).
3. Connecter le repo, spécifier `app.py`.
4. Ajouter les secrets dans **Settings → Secrets** :

```toml
DAGSHUB_USERNAME = "votre_username"
DAGSHUB_TOKEN = "votre_token"
DAGSHUB_REPO_NAME = "ai_business_app"
GOOGLE_API_KEY = "votre_clé_gemini"
```

---

## 📝 Licence

Projet académique — Examen MLOps.
