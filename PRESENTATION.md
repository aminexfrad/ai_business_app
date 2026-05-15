# Guide de présentation orale — AI Business Intelligence App

**Durée : 5 minutes** | Examen MLOps & IA Générative

---

## 1. Introduction — Le problème (30 secondes)

> *« Notre entreprise en transformation digitale doit exploiter ses données pour décider plus vite. J'ai développé une plateforme **AI Business Intelligence** qui combine Machine Learning, MLOps et IA générative. »*

**Problème traité :**
- Prédire des indicateurs business à partir de données réelles (énergie, revenus, fraude, churn).
- Comparer plusieurs algorithmes ML de façon reproductible.
- Déployer une interface web pour des prédictions interactives et des analyses assistées par IA.

**4 cas d'usage :**

| Dataset | Question métier | Type |
|---------|-----------------|------|
| Énergie bâtiments | Quelle consommation énergétique (kWh) ? | Régression |
| Revenus restaurants | Quel CA mensuel ? | Régression |
| Détection fraude | Transaction frauduleuse ou non ? | Classification |
| Streaming | L'abonné va-t-il résilier ? | Classification |

---

## 2. Architecture de la solution (1 minute)

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  CSV data/  │────▶│ data_preprocessing│────▶│ train_models │
└─────────────┘     │ clean · encode    │     │ 4 modèles/DS │
                    │ split 80/20       │     └──────┬──────┘
                    └──────────────────┘            │
                                                      ▼
                    ┌──────────────────┐     ┌─────────────┐
                    │   DagsHub        │◀────│   MLflow    │
                    │ (remote tracking)│     │ params·metrics│
                    └────────┬─────────┘     │ models·tags  │
                             │               └─────────────┘
                             │ champion=true
                             ▼
                    ┌──────────────────┐     ┌─────────────┐
                    │  app.py          │────▶│ logs/       │
                    │  Streamlit 5 tabs│     │ actions.csv │
                    └────────┬─────────┘     └─────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Prédiction      Gemini API     Dashboard
         unitaire/batch  (analyse IA)   Plotly
```

**Fichiers clés :**
- `data_preprocessing.py` — pipeline données
- `train_models.py` — entraînement + sélection champion
- `mlflow_tracking.py` — connexion DagsHub
- `utils.py` — chargement champion MLflow, prédictions, logs
- `app.py` — interface utilisateur

---

## 3. Outils utilisés (30 secondes)

| Technologie | Rôle |
|-------------|------|
| **Python** | Langage principal |
| **scikit-learn / XGBoost** | Modèles ML |
| **MLflow** | Tracking expériences (params, métriques, artefacts) |
| **DagsHub** | Hébergement distant du tracking MLflow |
| **Streamlit** | Application web professionnelle |
| **Google Gemini** | Analyse IA générative des résultats |
| **Plotly** | Visualisations interactives |
| **Git / GitHub** | Versionnement et déploiement |

---

## 4. Pipeline MLOps & système Champion (1 minute 30)

### Partie 1 — Machine Learning

1. **Chargement** des CSV (`data/`)
2. **Nettoyage** : doublons supprimés, valeurs manquantes (médiane / mode)
3. **Encodage** : One-Hot Encoding des variables catégorielles
4. **Split** train/test 80/20
5. **Entraînement** de 4 modèles par dataset :
   - Régression : Linear Regression, Random Forest, XGBoost, Gradient Boosting
   - Classification : Logistic Regression, Random Forest, XGBoost, Gradient Boosting
6. **Tracking MLflow** : chaque run enregistre paramètres, métriques (R², F1, RMSE…), modèle
7. **Champion** : le meilleur modèle reçoit le tag `champion=true` sur DagsHub

**Critère de sélection :**
- Régression → **R² maximal**
- Classification → **F1-Score maximal**

### Démonstration MLflow / DagsHub

1. Ouvrir https://dagshub.com/aminexfrad/ai_business_app
2. Onglet **Experiments** → voir `AI_BI_building_energy`, etc.
3. Montrer les runs comparés (4 modèles)
4. Filtrer le run avec tag **champion = true**
5. Expliquer : l'app Streamlit charge ce run automatiquement

### Démonstration Champion dans l'app

1. Sidebar : **Champion** + **Source: mlflow**
2. Onglet Présentation → tableau récapitulatif des 4 champions
3. Onglet Prédiction → badge « Linear Regression (mlflow) » + run ID

---

## 5. Démonstration live de l'application (1 minute 30)

### Onglet 1 — Présentation
- Description du dataset sélectionné
- Statistiques (lignes, colonnes)
- Aperçu des données + pipeline MLOps

### Onglet 2 — Prédiction unitaire
- Remplir le formulaire (surface, température, type de bâtiment…)
- Cliquer **Prédire** → résultat en kWh (régression) ou classe (classification)

### Onglet 3 — Prédiction batch
- Uploader un CSV (mêmes colonnes que le dataset, sans la cible)
- Lancer les prédictions → télécharger le CSV enrichi

### Onglet 4 — Analyse IA (Gemini)
- Contexte automatique (dataset, champion, métriques)
- Cliquer **Générer l'analyse** → interprétation en français par Gemini

### Onglet 5 — Dashboard
- Métriques du champion, histogrammes, corrélations, camemberts
- **Journal des actions** (prédictions, uploads, Gemini) → Partie 4 Logging

---

## 6. Déploiement & sécurité (30 secondes)

- **Local** : secrets dans `.env` (non versionné)
- **Cloud** : secrets dans Streamlit Cloud (DagsHub + Gemini)
- **GitHub** : code public, pas de clés API dans le repo
- **App déployée** : lien Streamlit Cloud (voir `LIVRABLES.md`)

---

## Script minute par minute (5 min)

| Min | Contenu |
|-----|---------|
| 0:00–0:30 | Problème + 4 datasets |
| 0:30–1:30 | Architecture + outils |
| 1:30–3:00 | MLflow/DagsHub + champion (écran DagsHub) |
| 3:00–4:30 | Démo Streamlit (prédiction + Gemini) |
| 4:30–5:00 | Déploiement cloud + conclusion |

---

## Phrase de conclusion

> *« En résumé, j'ai livré une application IA complète : pipeline ML reproductible, tracking MLOps sur DagsHub, sélection automatique du champion, interface Streamlit déployée sur le cloud, et analyse enrichie par Gemini. Merci. »*

---

## Questions fréquentes du jury

**Pourquoi MLflow + DagsHub ?**  
Pour tracer chaque expérience, comparer les modèles et recharger le champion sans re-entraîner.

**Comment l'app trouve le champion ?**  
`mlflow.search_runs(filter_string="tags.champion = 'true'")` puis `mlflow.sklearn.load_model`.

**Que se passe-t-il si DagsHub est indisponible ?**  
Fallback sur les modèles locaux dans `models/champions/`.

**Comment sont sécurisées les clés API ?**  
`.env` en local (gitignored), `st.secrets` sur Streamlit Cloud.
