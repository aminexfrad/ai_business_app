"""
AI Business Intelligence App
Full MLOps platform: Streamlit + MLflow + DagsHub + Gemini AI
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

import config
import ml_pipeline
import gemini_ai
import logger

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Font & Smooth Layout */
html, body, [class*="css"] { 
    font-family: 'Outfit', sans-serif; 
}

/* Beautiful Animated Background with Custom Generated Image Overlay */
.stApp {
    background-image: 
        linear-gradient(150deg, rgba(10, 8, 8, 0.94) 0%, rgba(18, 15, 17, 0.90) 40%, rgba(32, 12, 18, 0.94) 100%), 
        url("https://raw.githubusercontent.com/aminexfrad/ai_business_app/main/background.png");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: #f3f4f6;
}

/* Premium Glassmorphic Card Styling */
.card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(239, 68, 68, 0.15);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-2px);
    border-color: rgba(239, 68, 68, 0.3);
    box-shadow: 0 12px 40px 0 rgba(239, 68, 68, 0.1);
}

/* Cybernetic Glowing Champion Banner */
.champion-banner {
    background: linear-gradient(90deg, #b91c1c 0%, #ef4444 50%, #991b1b 100%);
    color: #ffffff;
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    font-weight: 700;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid rgba(239, 68, 68, 0.5);
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    animation: pulse 2.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
    50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
    100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
}

/* Sleek Tech Pills */
.pill {
    display: inline-block;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #fca5a5;
    margin: 0.25rem;
    transition: all 0.2s ease;
}
.pill:hover {
    background: rgba(239, 68, 68, 0.18);
    border-color: rgba(239, 68, 68, 0.4);
}

/* Beautiful Status Alerts (Glassmorphic Colors) */
.info  { background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 1rem; margin: 0.6rem 0; color: #93c5fd; font-weight: 500; }
.ok    { background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 1rem; margin: 0.6rem 0; color: #a7f3d0; font-weight: 500; }
.warn  { background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 1rem; margin: 0.6rem 0; color: #fde68a; font-weight: 500; }
.err   { background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 1rem; margin: 0.6rem 0; color: #fca5a5; font-weight: 500; }

/* Futuristic Digital Metric Widgets */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(239, 68, 68, 0.12);
    border-radius: 12px;
    padding: 0.9rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(239, 68, 68, 0.25);
    background: rgba(255, 255, 255, 0.04);
}
div[data-testid="metric-container"] label { 
    color: #9ca3af !important; 
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] { 
    color: #ef4444 !important; 
    font-weight: 700 !important;
    text-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
}

/* Premium Buttons with Glowing Micro-Transitions */
.stButton>button {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2) !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #ff6b6b 0%, #ef4444 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4) !important;
}
.stButton>button:active {
    transform: translateY(0px) !important;
}

/* Elegant Text Styling & Text Gradient Headers */
h1, h2, h3, h4 { 
    font-weight: 700 !important;
    background: linear-gradient(120deg, #ffffff 0%, #ffc5c5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem !important;
}
p, li, span { 
    color: #e5e7eb !important; 
}
.stSelectbox label, .stSlider label, .stNumberInput label { 
    color: #9ca3af !important; 
    font-weight: 500 !important;
}

/* Glassmorphic Cyber Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 8, 8, 0.85) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(239, 68, 68, 0.15) !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    background: linear-gradient(120deg, #ffffff 0%, #ff8b8b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Beautiful Glowing Interactive Tabs */
.stTabs [data-baseweb="tab-list"] { 
    gap: 8px; 
    background: transparent; 
    border-bottom: 2px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px 10px 0 0 !important;
    color: #9ca3af !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-weight: 500 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border-color: rgba(239, 68, 68, 0.2) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
    color: #ffffff !important;
    border-color: #ef4444 !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.25) !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [
    ("trained_results",   {}),
    ("champion_pipeline", None),
    ("champion_name",     None),
    ("champion_source",   None),
    ("champion_run_id",   None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AI Business Intelligence")
    st.markdown("---")
    st.markdown("### Select Dataset")
    dataset_choice = st.selectbox("Dataset", list(config.DATASETS.keys()), label_visibility="collapsed")
    ds = config.DATASETS[dataset_choice]

    # Load data
    df_path = ds["file"]
    if not os.path.exists(df_path):
        st.error(f"Dataset file not found: {df_path}")
        st.stop()
    df = pd.read_csv(df_path)

    st.markdown(f"**Task:** `{ds['task']}`")
    st.markdown(f"**Target:** `{ds['target']}`")
    st.markdown(f"**Shape:** `{df.shape[0]:,} × {df.shape[1]}`")
    st.markdown("---")

    # Champion status
    if st.session_state.champion_name:
        st.markdown(f'<div class="champion-banner">Champion Active<br>{st.session_state.champion_name}</div>', unsafe_allow_html=True)
        if st.session_state.champion_source:
            st.caption(f"Source: {st.session_state.champion_source}")
    else:
        st.markdown('<div class="warn">No champion loaded.<br>Go to <b>Training</b> tab and train models first.</div>', unsafe_allow_html=True)

    # Load champion from MLflow button
    if st.button("Load Champion from MLflow"):
        with st.spinner("Connecting to MLflow..."):
            pipe, source = ml_pipeline.load_champion_from_mlflow()
            if pipe:
                st.session_state.champion_pipeline = pipe
                st.session_state.champion_source   = source
                st.session_state.champion_name     = st.session_state.champion_name or "MLflow Champion"
                st.success(f"Loaded: {source}")
            else:
                st.warning("No model found in MLflow yet.")

    st.markdown("---")
    st.markdown("### Links")
    dagshub_url = f"https://dagshub.com/{config.DAGSHUB_USERNAME}/{config.DAGSHUB_REPO}"
    st.markdown(f"[DagsHub / MLflow]({dagshub_url}.mlflow)")
    st.markdown(f"[GitHub](https://github.com/{config.DAGSHUB_USERNAME}/{config.DAGSHUB_REPO})")
    st.markdown("---")
    st.caption("MLOps Exam · 2026 · aminexfrad")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Entraînement & MLflow",
    "Prédiction Unitaire",
    "Prédiction Batch CSV",
    "Analyse IA Générative",
    "Dashboard",
])




# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENTRAÎNEMENT & MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Entraînement des modèles & Tracking MLflow")

    task = ds["task"]
    target = ds["target"]
    available_models = config.REGRESSION_MODELS if task == "regression" else config.CLASSIFICATION_MODELS

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Configuration")
        selected_models = st.multiselect(
            "Modèles à entraîner",
            available_models,
            default=available_models,
        )

        st.markdown("**Statistiques descriptives :**")
        st.dataframe(df.describe().round(2), height=220, use_container_width=True)

    with col_right:
        st.markdown("### Aperçu du dataset")
        st.dataframe(df.head(8), use_container_width=True)

        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss):
            st.markdown(f'<div class="warn">Valeurs manquantes détectées : {miss.to_dict()}<br>→ Traitées automatiquement par imputation.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok">Aucune valeur manquante détectée.</div>', unsafe_allow_html=True)

        cat_cols_preview = df.select_dtypes(include="object").columns.tolist()
        if cat_cols_preview:
            st.markdown(f'<div class="info">Variables catégorielles encodées (One-Hot) : {", ".join(cat_cols_preview)}</div>', unsafe_allow_html=True)

    st.markdown("---")

    btn_col, _ = st.columns([2, 3])
    with btn_col:
        run_training = st.button("Lancer l'entraînement & Tracker avec MLflow", use_container_width=True)

    if run_training:
        if not selected_models:
            st.error("Sélectionnez au moins un modèle.")
        else:
            prog = st.progress(0)
            status = st.empty()

            def cb(i, total, name):
                prog.progress((i + 1) / total)
                status.markdown(f"⏳ Entraînement de **{name}** en cours...")

            try:
                with st.spinner("Connexion à DagsHub / MLflow..."):
                    results, X_test, y_test = ml_pipeline.train_and_track(
                        df=df, dataset_name=dataset_choice,
                        target=target, task=task,
                        model_names=selected_models,
                        progress_callback=cb,
                    )

                prog.empty(); status.empty()
                st.session_state.trained_results = results

                # ── Select champion ──
                champ_name, champ_info = ml_pipeline.get_champion(results)
                if champ_name:
                    st.session_state.champion_name     = champ_name
                    st.session_state.champion_pipeline = champ_info["pipeline"]
                    st.session_state.champion_run_id   = champ_info["run_id"]

                    # Save locally
                    ml_pipeline.save_champion_locally(champ_info["pipeline"])

                    # Register in MLflow Model Registry
                    ok, ver = ml_pipeline.register_champion_in_mlflow(champ_info["run_id"])
                    if ok:
                        source = f"MLflow Registry v{ver} (@champion)"
                    else:
                        source = "Local (MLflow registration failed)"
                    st.session_state.champion_source = source

                    logger.log_training(dataset_choice, selected_models, champ_name)

                st.markdown(f'<div class="ok">Entraînement terminé ! Champion : <b>{champ_name}</b><br>Source : {source}</div>', unsafe_allow_html=True)

                # ── Results table ──
                st.markdown("### Comparaison des modèles")
                rows = []
                for mname, minfo in results.items():
                    row = {"Modèle": mname}
                    row.update({k: round(v, 4) for k, v in minfo["metrics"].items()})
                    row["MLflow Run ID"] = minfo["run_id"][:8] + "…"
                    row["Champion"] = "*" if mname == champ_name else ""
                    rows.append(row)
                res_df = pd.DataFrame(rows)
                st.dataframe(res_df, use_container_width=True)

                # ── Chart ──
                metric_col = "r2" if task == "regression" else "f1"
                if metric_col in res_df.columns:
                    chart_df = res_df.sort_values(metric_col, ascending=False).copy()
                    colors = ["#b91c1c" if m == champ_name else "#fca5a5" for m in chart_df["Modèle"]]
                    fig = go.Figure(go.Bar(
                        x=chart_df["Modèle"], y=chart_df[metric_col],
                        marker_color=colors, text=chart_df[metric_col].round(4),
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title=f"Comparaison — {metric_col.upper()} (Champion)",
                        template="plotly_white", showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        yaxis_title=metric_col.upper(), xaxis_title="Modèle",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                dagshub_url = f"https://dagshub.com/{config.DAGSHUB_USERNAME}/{config.DAGSHUB_REPO}.mlflow"
                st.markdown(f'<div class="info">Voir les expériences sur <a href="{dagshub_url}"target="_blank"><b>DagsHub MLflow →</b></a></div>', unsafe_allow_html=True)

            except Exception as e:
                prog.empty(); status.empty()
                st.error(f"Erreur d'entraînement : {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRÉDICTION UNITAIRE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Prédiction Unitaire")

    # Load champion: session → local pkl → MLflow
    pipeline = st.session_state.champion_pipeline
    if pipeline is None:
        pipeline = ml_pipeline.load_champion_locally()
        if pipeline:
            st.session_state.champion_pipeline = pipeline
            st.session_state.champion_source   = "Local (champion_model.pkl)"

    if pipeline is None:
        st.markdown('<div class="warn">Aucun modèle champion disponible.<br>Allez dans l\'onglet <b>Entraînement</b> pour entraîner les modèles, ou cliquez sur <b>Load Champion from MLflow</b> dans la sidebar.</div>', unsafe_allow_html=True)
    else:
        src = st.session_state.champion_source or "Session"
        st.markdown(f'<div class="ok">Modèle Champion chargé : <b>{st.session_state.champion_name}</b><br> Source : {src}</div>', unsafe_allow_html=True)

        target2 = ds["target"]
        task2   = ds["task"]
        feat_cols = [c for c in df.columns if c != target2]
        num_feats = df[feat_cols].select_dtypes(include=np.number).columns.tolist()
        cat_feats = df[feat_cols].select_dtypes(include="object").columns.tolist()

        st.markdown("### Saisir les valeurs des features")
        user_input = {}

        # Numeric inputs — 3 per row
        for i in range(0, len(num_feats), 3):
            cols = st.columns(3)
            for j, col_name in enumerate(num_feats[i:i+3]):
                with cols[j]:
                    mn  = float(df[col_name].min())
                    mx  = float(df[col_name].max())
                    med = float(df[col_name].median())
                    user_input[col_name] = st.number_input(col_name, mn, mx, med, format="%.4f")

        # Categorical inputs
        if cat_feats:
            cat_cols_ui = st.columns(min(len(cat_feats), 3))
            for j, col_name in enumerate(cat_feats):
                with cat_cols_ui[j % 3]:
                    options = sorted(df[col_name].dropna().unique().tolist())
                    user_input[col_name] = st.selectbox(col_name, options)

        st.markdown("---")
        if st.button("Prédire", use_container_width=True):
            try:
                input_df   = pd.DataFrame([user_input])
                prediction = pipeline.predict(input_df)[0]

                st.markdown("### Résultat de la prédiction")
                r1, r2_col, r3 = st.columns([1, 2, 1])
                with r2_col:
                    if task2 == "regression":
                        st.metric(label=f"Valeur prédite — {target2}", value=f"{prediction:,.2f}")
                    else:
                        classes = {0: ("Négatif", "green"), 1: ("Positif / Fraude", "red")}
                        label, _ = classes.get(int(prediction), (f"Classe {int(prediction)}", "blue"))
                        st.metric(label=f"Prédiction — {target2}", value=label)
                        try:
                            proba = pipeline.predict_proba(input_df)[0]
                            st.markdown(f"**Confiance :** {max(proba)*100:.1f}%")
                            prob_df = pd.DataFrame({"Classe": [str(i) for i in range(len(proba))], "Probabilité": proba})
                            fig_p = px.bar(prob_df, x="Classe", y="Probabilité", template="plotly_white",
                                           color="Classe", color_discrete_sequence=["#dc2626","#991b1b"],
                                           title="Distribution des probabilités")
                            fig_p.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                            st.plotly_chart(fig_p, use_container_width=True)
                        except Exception:
                            pass

                logger.log_prediction(dataset_choice, st.session_state.champion_name or "champion", user_input, prediction)

                with st.expander("Interprétation par Gemini AI"):
                    with st.spinner("Gemini analyse..."):
                        try:
                            txt = gemini_ai.analyze_prediction_result(dataset_choice, target2, task2, user_input, prediction)
                            st.markdown(txt)
                            logger.log_gemini_query(dataset_choice, "prediction interpretation")
                        except Exception as e:
                            st.error(gemini_ai.user_error_message(e))

            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRÉDICTION BATCH CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Prédiction Batch — Upload CSV")

    pipeline_b = st.session_state.champion_pipeline or ml_pipeline.load_champion_locally()

    if pipeline_b is None:
        st.markdown('<div class="warn">Aucun modèle champion. Entraînez d\'abord les modèles.</div>', unsafe_allow_html=True)
    else:
        target3 = ds["target"]
        task3   = ds["task"]
        feat_cols3 = [c for c in df.columns if c != target3]

        st.markdown(f'<div class="ok">Champion : <b>{st.session_state.champion_name or "Chargé"}</b></div>', unsafe_allow_html=True)
        st.markdown(f"**Colonnes attendues** (hors cible `{target3}`) :")
        st.code(", ".join(feat_cols3))

        # Download template
        template_csv = df[feat_cols3].head(5).to_csv(index=False)
        st.download_button("⬇ Télécharger le template CSV", data=template_csv,
                           file_name=f"template_{ds['file']}", mime="text/csv")

        uploaded = st.file_uploader(" Importer un fichier CSV pour prédiction batch", type=["csv"])

        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                st.markdown(f"**{len(batch_df)} lignes** chargées.")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("Lancer la prédiction batch", use_container_width=True):
                    pred_input = batch_df.drop(columns=[target3], errors="ignore")
                    preds = pipeline_b.predict(pred_input)

                    result_df = batch_df.copy()
                    result_df["Prédiction"] = preds
                    try:
                        proba_b = pipeline_b.predict_proba(pred_input)
                        result_df["Confiance"] = proba_b.max(axis=1).round(4)
                    except Exception:
                        pass

                    st.markdown("### Résultats")
                    st.dataframe(result_df, use_container_width=True)

                    out_csv = result_df.to_csv(index=False)
                    st.download_button("⬇ Télécharger les prédictions CSV", data=out_csv,
                                       file_name="predictions_batch.csv", mime="text/csv")

                    # Distribution
                    fig_b = px.histogram(result_df, x="Prédiction", nbins=30,
                                         template="plotly_white", title="Distribution des prédictions",
                                         color_discrete_sequence=["#dc2626"])
                    fig_b.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_b, use_container_width=True)

                    logger.log_batch_upload(dataset_choice, st.session_state.champion_name or "champion", len(batch_df))

                    with st.expander("Analyse Gemini du batch"):
                        with st.spinner("Gemini analyse le batch..."):
                            try:
                                txt = gemini_ai.analyze_batch_results(dataset_choice, target3, task3, result_df)
                                st.markdown(txt)
                                logger.log_gemini_query(dataset_choice, "batch summary")
                            except Exception as e:
                                st.error(f"Gemini : {e}")

            except Exception as e:
                st.error(f"Erreur : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYSE IA GÉNÉRATIVE (GEMINI)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Analyse IA Générative — Google Gemini")

    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        st.markdown("### Aperçu statistique")
        st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown(f"**Variables catégorielles :**")
        cat_g = df.select_dtypes(include="object").columns.tolist()
        for c in cat_g:
            st.markdown(f"- `{c}` → {df[c].nunique()} valeurs uniques")

    with col_g2:
        st.markdown("### Posez votre question à Gemini")
        quick = [
            "Analyse complète du dataset",
            "Quelles features sont les plus importantes pour la cible ?",
            "Identifie les problèmes de qualité des données",
            "Quelles actions business recommandes-tu ?",
            "Propose des idées de feature engineering",
            "Compare les distributions des variables numériques",
        ]
        choice_q = st.selectbox(" Questions rapides", ["Question personnalisée..."] + quick)
        if choice_q == "Question personnalisée...":
            user_q = st.text_area("Votre question", placeholder="Ex: Y a-t-il des outliers dans ce dataset ?", height=100)
        else:
            user_q = choice_q
            st.markdown(f'<div class="info">Question sélectionnée : {user_q}</div>', unsafe_allow_html=True)

        if st.button("Analyser avec Gemini", use_container_width=True):
            with st.spinner("Gemini réfléchit..."):
                try:
                    response = gemini_ai.analyze_dataset(
                        df, dataset_choice, ds["target"], ds["task"],
                        user_question=user_q if user_q else None
                    )
                    st.markdown("### Réponse Gemini")
                    st.markdown(response)
                    logger.log_gemini_query(dataset_choice, user_q or "full analysis")
                except Exception as e:
                    st.markdown(f'<div class="err">{gemini_ai.user_error_message(e)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DASHBOARD & VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Dashboard & Visualisations")

    target4  = ds["target"]
    task4    = ds["task"]
    feat_c   = [c for c in df.columns if c != target4]
    num_c    = df[feat_c].select_dtypes(include=np.number).columns.tolist()
    cat_c    = df[feat_c].select_dtypes(include="object").columns.tolist()

    # ── Row 1: KPIs ──────────────────────────────────────────────────────────
    km1, km2, km3, km4 = st.columns(4)
    km1.metric("Lignes", f"{len(df):,}")
    km2.metric("Features", len(feat_c))
    km3.metric("Num. cols", len(num_c))
    km4.metric("Cat. cols", len(cat_c))

    st.markdown("---")

    # ── Row 2: Target distribution ───────────────────────────────────────────
    st.markdown("### Distribution de la variable cible")
    d1, d2 = st.columns(2)
    with d1:
        if task4 == "regression":
            fig_t = px.histogram(df, x=target4, nbins=40, template="plotly_white",
                                 color_discrete_sequence=["#dc2626"], title=f"Histogramme — {target4}")
        else:
            vc = df[target4].value_counts().reset_index()
            vc.columns = [target4, "count"]
            fig_t = px.pie(vc, names=target4, values="count", template="plotly_white",
                           title=f"Répartition des classes — {target4}",
                           color_discrete_sequence=["#dc2626","#b91c1c","#991b1b"])
        fig_t.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t, use_container_width=True)

    with d2:
        fig_box = px.box(df, y=target4, template="plotly_white",
                         color_discrete_sequence=["#dc2626"], title=f"Box Plot — {target4}")
        fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Row 3: Correlations ───────────────────────────────────────────────────
    st.markdown("### Matrice de corrélation")
    corr_df = df[num_c + [target4]].corr()
    fig_corr = px.imshow(corr_df, text_auto=".2f", template="plotly_white",
                         color_continuous_scale="RdBu_r", aspect="auto",
                         title="Heatmap des corrélations")
    fig_corr.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Row 4: Feature explorer ───────────────────────────────────────────────
    st.markdown("### Exploration Feature vs Cible")
    if num_c:
        feat_x = st.selectbox("Feature X", num_c, key="dash_feat_x")
        color_by = cat_c[0] if cat_c else None
        if task4 == "regression":
            fig_sc = px.scatter(df, x=feat_x, y=target4, color=color_by,
                                template="plotly_white", trendline="ols",
                                title=f"{feat_x} vs {target4}",
                                color_discrete_sequence=px.colors.qualitative.Bold)
        else:
            fig_sc = px.box(df, x=str(target4), y=feat_x,
                            color=str(target4), template="plotly_white",
                            title=f"{feat_x} par classe de {target4}",
                            color_discrete_sequence=["#dc2626","#991b1b"])
        fig_sc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Row 5: Categorical distributions ─────────────────────────────────────
    if cat_c:
        st.markdown("### Distribution des variables catégorielles")
        cat_sel = st.selectbox("Variable catégorielle", cat_c, key="dash_cat")
        fig_cat = px.histogram(df, x=cat_sel, color=cat_sel, template="plotly_white",
                               title=f"Distribution — {cat_sel}",
                               color_discrete_sequence=px.colors.qualitative.Bold)
        fig_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Row 6: Model comparison (if trained) ──────────────────────────────────
    if st.session_state.trained_results:
        st.markdown("### Comparaison des modèles (dernière session)")
        rows = []
        champ = st.session_state.champion_name
        for mname, minfo in st.session_state.trained_results.items():
            row = {"Modèle": mname}
            row.update({k: round(v, 4) for k, v in minfo["metrics"].items()})
            rows.append(row)
        comp_df = pd.DataFrame(rows)
        metric_cols = [c for c in comp_df.columns if c != "Modèle"]
        fig_comp = go.Figure()
        colors = px.colors.qualitative.Bold
        for i, mname in enumerate(comp_df["Modèle"]):
            vals = comp_df[comp_df["Modèle"] == mname][metric_cols].values.flatten()
            bar_kwargs = dict(
                name=mname,
                x=metric_cols,
                y=vals,
                marker_color=colors[i % len(colors)],
            )
            if mname == champ:
                bar_kwargs["marker_line_width"] = 2
                bar_kwargs["marker_line_color"] = "#b91c1c"
            fig_comp.add_trace(go.Bar(**bar_kwargs))
        fig_comp.update_layout(barmode="group", template="plotly_white",
                               title="Toutes les métriques — Comparaison des modèles",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Row 7: Action Log ─────────────────────────────────────────────────────
    st.markdown("### Journal des actions (Logs MLOps)")
    logs = logger.get_logs()
    if logs:
        log_df = pd.DataFrame(logs[::-1]).head(30)
        st.dataframe(log_df, use_container_width=True)
        log_csv = log_df.to_csv(index=False)
        st.download_button("⬇ Télécharger les logs", data=log_csv,
                           file_name="app_logs.csv", mime="text/csv")
    else:
        st.markdown('<div class="info">Aucune action enregistrée. Entraînez des modèles et faites des prédictions pour voir les logs ici.</div>', unsafe_allow_html=True)
