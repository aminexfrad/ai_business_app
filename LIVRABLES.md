# Livrables obligatoires — Examen MLOps

## Checklist

| Livrable | Lien / statut |
|----------|----------------|
| Dépôt GitHub | https://github.com/aminexfrad/ai_business_app |
| DagsHub (MLflow) | https://dagshub.com/aminexfrad/ai_business_app |
| Application déployée | https://ai-business-app-aminexfrad.streamlit.app *(à activer après déploiement Streamlit Cloud)* |
| Datasets | Dossier `data/` (4 fichiers CSV) |
| Présentation orale | Voir `PRESENTATION.md` |

---

## Déploiement Streamlit Cloud (étapes)

1. Aller sur https://share.streamlit.io
2. **New app** → Repository : `aminexfrad/ai_business_app`
3. Branch : `main` | Main file : `app.py`
4. **Advanced settings → Secrets** — coller :

```toml
DAGSHUB_USERNAME = "aminexfrad"
DAGSHUB_TOKEN = "votre_token_dagshub"
DAGSHUB_REPO_NAME = "ai_business_app"
GOOGLE_API_KEY = "votre_cle_gemini"
```

5. **Deploy** → copier l'URL publique et la mettre à jour dans ce fichier.

---

## Avant la soutenance

```bash
python train_models.py    # Pousse les runs vers DagsHub
streamlit run app.py      # Test local
```

Vérifier sur DagsHub : 4 expériences, 16 runs par cycle, 4 tags `champion=true`.

---

## Fichiers du projet

| Fichier | Description |
|---------|-------------|
| `PRESENTATION.md` | Guide oral 5 minutes |
| `README.md` | Documentation technique |
| `app.py` | Application Streamlit |
| `train_models.py` | Entraînement + MLflow |
| `data/*.csv` | Jeux de données |
