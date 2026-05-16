# 🧠 AI Business Intelligence — Guide en français

Plateforme **MLOps** (Machine Learning Operations) avec interface web **Streamlit**, suivi d’expériences **MLflow** sur **DagsHub**, et analyse en langage naturel via **Google Gemini**.

Ce document t’aide à **comprendre**, **utiliser** et **présenter** l’application.

---

## 1. À quoi sert cette application ?

L’application répond à un besoin métier classique : **transformer des données d’entreprise en prédictions utiles**, tout en gardant une trace professionnelle de chaque expérience ML.

En résumé, elle permet de :

| Fonction | Description |
|--------|-------------|
| **Explorer** | Voir les statistiques, graphiques et corrélations des données |
| **Entraîner** | Comparer plusieurs modèles ML sur le même jeu de données |
| **Tracer (MLOps)** | Enregistrer paramètres, métriques et modèles dans MLflow (DagsHub) |
| **Prédire** | Faire une prédiction pour **une ligne** ou pour **un fichier CSV entier** |
| **Analyser avec l’IA** | Poser des questions en français (ou autre langue) à Gemini sur les données |
| **Journaliser** | Garder un historique des actions (entraînement, prédictions, requêtes IA) |

---

## 2. Architecture (à expliquer à l’oral)

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE (Streamlit)                       │
│  6 onglets : Présentation | Entraînement | Prédiction | …   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Données CSV        Pipeline ML         Gemini AI
   (4 datasets)    (scikit-learn,      (analyse texte)
                    XGBoost)
        │                  │
        │                  ▼
        │            MLflow + DagsHub
        │         (tracking + registry)
        ▼
   champion_model.pkl  (sauvegarde locale du meilleur modèle)
```

**Fichiers principaux du code :**

| Fichier | Rôle |
|---------|------|
| `app.py` | Interface Streamlit (tous les onglets) |
| `config.py` | Configuration : datasets, clés API, DagsHub |
| `ml_pipeline.py` | Nettoyage, entraînement, MLflow, modèle « Champion » |
| `gemini_ai.py` | Appels à l’API Google Gemini |
| `logger.py` | Journal des actions (`app_logs.jsonl`) |

---

## 3. Les 4 jeux de données

Tu choisis le dataset dans la **barre latérale gauche**.

| Dataset | Type | Variable cible | Cas d’usage métier |
|---------|------|----------------|-------------------|
| 🏢 **Building Energy** | Régression | `energy_consumption` | Prédire la consommation énergétique d’un bâtiment |
| 🍽️ **Restaurant Revenue** | Régression | `monthly_revenue` | Prédire le chiffre d’affaires mensuel d’un restaurant |
| 🔍 **Fraud Detection** | Classification | `fraud` | Détecter une transaction frauduleuse (0/1) |
| 📺 **Streaming Churn** | Classification | `subscription_cancelled` | Prédire si un abonné va résilier |

Chaque fichier CSV est chargé automatiquement au démarrage.

---

## 4. Les 8 modèles Machine Learning

Selon le type de tâche, des modèles différents sont proposés :

**Régression** (valeur numérique à prédire) :
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**Classification** (catégorie / oui-non) :
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Après l’entraînement, le **modèle Champion** 🏆 est celui qui a la **meilleure métrique** :
- Régression → **R²** le plus élevé
- Classification → **F1-score** le plus élevé

---

## 5. Les 6 onglets de l’application

### 🏠 Présentation
- Page d’accueil pour la démo
- Explique le problème, l’architecture, les technologies
- Montre un résumé de chaque dataset

**Pour la présentation :** commence ici pour poser le contexte.

---

### ⚙️ Entraînement & MLflow
- Aperçu du dataset (tableau, stats, valeurs manquantes)
- Choix des modèles à entraîner
- Bouton : **« Lancer l’entraînement & Tracker avec MLflow »**

**Ce qui se passe en coulisses :**
1. Nettoyage : imputation des valeurs manquantes, encodage des variables catégorielles
2. Split 80 % train / 20 % test
3. Entraînement de chaque modèle sélectionné
4. Enregistrement dans MLflow (métriques + modèle)
5. Sélection du Champion + sauvegarde locale (`champion_model.pkl`)
6. Enregistrement du Champion dans le **Model Registry** MLflow

**Pour la présentation :** c’est l’onglet le plus important — montre le tableau comparatif et le graphique des métriques.

---

### 🎯 Prédiction Unitaire
- Saisie manuelle des features (nombres + listes déroulantes)
- Bouton **« Prédire »**
- Affiche le résultat (+ probabilités pour la classification)
- Option : interprétation par Gemini

**Prérequis :** un modèle Champion doit être chargé (après entraînement ou via le bouton sidebar).

---

### 📦 Prédiction Batch CSV
- Upload d’un fichier CSV avec les mêmes colonnes que le dataset (sans la cible)
- Téléchargement d’un **template** exemple
- Prédictions sur toutes les lignes + export CSV des résultats
- Graphique de distribution des prédictions

**Pour la présentation :** montre le flux « entreprise qui envoie un fichier de clients à scorer ».

---

### 🤖 Analyse IA Générative
- Questions en langage naturel sur le dataset
- Questions rapides prédéfinies (qualité des données, features importantes, etc.)
- Utilise **Google Gemini** (`gemini-2.0-flash`)

**Prérequis :** clé `GOOGLE_API_KEY` valide avec quota API disponible.

---

### 📊 Dashboard
- KPIs (nombre de lignes, features, etc.)
- Histogrammes, box plots, heatmap de corrélation
- Exploration feature vs cible
- Comparaison des modèles (si entraînement fait dans la session)
- **Journal des actions** (logs MLOps)

---

## 6. Barre latérale (sidebar)

| Élément | Utilité |
|---------|---------|
| **Select Dataset** | Changer de jeu de données |
| **Task / Target / Shape** | Infos sur le dataset actif |
| **Champion Active** | Indique si un modèle est prêt pour les prédictions |
| **Load Champion from MLflow** | Recharger le meilleur modèle depuis le cloud |
| **Liens DagsHub / GitHub** | Montrer le tracking distant pendant la démo |

---

## 7. Installation et lancement

### Prérequis
- Python 3.11+ (testé aussi sur 3.13)
- Compte DagsHub + token (pour MLflow)
- Clé API Google Gemini (pour l’onglet IA)

### Installation

```bash
cd ai_business_app
pip install -r requirements.txt
```

### Variables d’environnement

Crée un fichier `.env` à la racine du projet :

```env
DAGSHUB_USERNAME=ton_utilisateur
DAGSHUB_TOKEN=ton_token_dagshub
DAGSHUB_REPO_NAME=ai_business_app
GOOGLE_API_KEY=ta_cle_gemini
```

Sur **Streamlit Cloud**, les mêmes clés se configurent dans **Settings → Secrets**.

### Lancer l’app

```bash
streamlit run app.py
```

Ouvre : **http://localhost:8501**

---

## 8. Scénario de démo recommandé (5–10 minutes)

1. **Présentation** → expliquer le problème et les 4 datasets (30 s)
2. **Sidebar** → choisir *Fraud Detection* ou *Building Energy* (15 s)
3. **Entraînement** → sélectionner 2–4 modèles → lancer l’entraînement (2–3 min)
4. Montrer le **Champion** 🏆, le tableau des métriques, le lien **DagsHub MLflow**
5. **Prédiction Unitaire** → saisir des valeurs → **Prédire** → lire le résultat
6. **Dashboard** → montrer 1–2 graphiques (corrélation ou distribution cible)
7. *(Optionnel)* **Gemini** → poser une question business
8. **Dashboard → Logs** → montrer la traçabilité MLOps

---

## 9. Concepts MLOps à maîtriser pour la soutenance

| Terme | Explication simple |
|-------|-------------------|
| **MLflow** | Outil qui enregistre chaque expérience ML (comme un carnet de laboratoire) |
| **DagsHub** | Hébergeur distant pour MLflow + Git (collaboration et suivi) |
| **Run** | Une exécution d’entraînement d’un modèle |
| **Experiment** | Groupe de runs (ex. : un dataset + une tâche) |
| **Champion** | Le meilleur modèle retenu pour la production / les prédictions |
| **Model Registry** | Catalogue officiel des modèles versionnés |
| **Pipeline** | Chaîne automatique : nettoyage → modèle → prédiction |

---

## 10. Métriques affichées

**Régression :**
- **RMSE** — erreur moyenne (plus bas = mieux)
- **MAE** — erreur absolue moyenne
- **R²** — qualité d’ajustement (plus proche de 1 = mieux) → utilisé pour le Champion

**Classification :**
- **Accuracy** — % de bonnes prédictions
- **F1** — équilibre précision/rappel → utilisé pour le Champion
- **Precision / Recall / AUC**

---

## 11. Dépannage rapide

| Problème | Solution |
|----------|----------|
| « No champion loaded » | Entraîner dans l’onglet Entraînement, ou cliquer *Load Champion from MLflow* |
| Erreur MLflow / DagsHub | Vérifier `DAGSHUB_USERNAME` et `DAGSHUB_TOKEN` dans `.env` |
| Erreur Gemini 404 | Modèle mis à jour → `gemini-2.0-flash` dans `gemini_ai.py` |
| Erreur Gemini 429 | Quota API épuisé → attendre ou activer la facturation Google AI |
| Dataset introuvable | Lancer l’app depuis le dossier contenant les fichiers `.csv` |

---

## 12. Technologies utilisées

Python · Streamlit · pandas · scikit-learn · XGBoost · MLflow · DagsHub · Plotly · Google Gemini · joblib · python-dotenv

---

## 13. Liens utiles

- **GitHub :** https://github.com/aminexfrad/ai_business_app
- **DagsHub MLflow :** https://dagshub.com/aminexfrad/ai_business_app.mlflow
- **Google AI Studio (clé API) :** https://aistudio.google.com/

---

*Projet MLOps — Examen TP 2024 — aminexfrad*
