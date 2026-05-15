"""
app.py
──────
Application Streamlit – AI Business Intelligence App
5 onglets : Présentation · Prédiction unitaire · Prédiction batch
             Analyse IA (Gemini) · Dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from data_preprocessing import DATASETS, load_data, clean_data, encode_features, get_feature_info, prepare_dataset
from utils import (
    log_action,
    load_champion,
    predict_single,
    predict_batch,
    get_champion_metrics,
    get_champion_name,
    get_all_champions_summary,
)

load_dotenv()

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1.05rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #4c51bf; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }

    /* Champion badge */
    .champion-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #333;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }

    /* Info box */
    .info-box {
        background: #eef2ff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 AI Business BI")
    st.markdown("---")

    dataset_key = st.selectbox(
        "📊 Choisir un dataset",
        list(DATASETS.keys()),
        format_func=lambda k: {
            "building_energy": "🏢 Énergie Bâtiments",
            "restaurant_revenue": "🍽️ Revenus Restaurants",
            "fraud_detection": "🔒 Détection Fraude",
            "streaming_subscription": "📺 Résiliation Streaming",
        }.get(k, k),
    )

    st.markdown("---")
    task = DATASETS[dataset_key]["task"]
    st.markdown(f"**Tâche** : `{task}`")
    st.markdown(f"**Cible** : `{DATASETS[dataset_key]['target']}`")

    champion_name = get_champion_name(dataset_key)
    st.markdown(f"**Champion** : `{champion_name}`")

    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>Projet d'examen MLOps<br>"
        "Streamlit · MLflow · DagsHub · Gemini</small>",
        unsafe_allow_html=True,
    )

log_action("page_view", f"dataset={dataset_key}")

# ─── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """<div class="main-header">
        <h1>🧠 AI Business Intelligence App</h1>
        <p>Machine Learning · MLflow · Analyse IA Générative</p>
    </div>""",
    unsafe_allow_html=True,
)

# ─── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Présentation",
    "🎯 Prédiction unitaire",
    "📦 Prédiction batch",
    "🤖 Analyse IA (Gemini)",
    "📊 Dashboard",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Présentation
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("📋 Présentation du projet")

    col_desc, col_stats = st.columns([3, 2])

    with col_desc:
        st.markdown(DATASETS[dataset_key]["description"])

        st.markdown("### 🔧 Pipeline MLOps")
        st.markdown("""
        1. **Chargement** des données CSV
        2. **Nettoyage** (doublons, valeurs manquantes)
        3. **Encodage** One-Hot des variables catégorielles
        4. **Split** train / test (80 / 20)
        5. **Entraînement** de 4 modèles par dataset
        6. **Tracking** dans MLflow / DagsHub
        7. **Sélection** automatique du champion
        """)

    with col_stats:
        try:
            df_raw = load_data(dataset_key)
            st.markdown("### 📊 Statistiques du dataset")

            m1, m2, m3 = st.columns(3)
            m1.metric("Lignes", f"{len(df_raw):,}")
            m2.metric("Colonnes", f"{len(df_raw.columns)}")
            m3.metric("Catégorielles", f"{len(df_raw.select_dtypes(include='object').columns)}")

            with st.expander("👀 Aperçu des données", expanded=True):
                st.dataframe(df_raw.head(10), use_container_width=True)

            with st.expander("📈 Statistiques descriptives"):
                st.dataframe(df_raw.describe().T, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")

    # Résumé des champions
    st.markdown("---")
    st.markdown("### 🏆 Résumé des modèles champions")
    summary = get_all_champions_summary()
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun modèle entraîné. Exécutez `python train_models.py` pour lancer l'entraînement.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Prédiction unitaire
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🎯 Prédiction unitaire")
    st.markdown("Saisissez les caractéristiques ci-dessous pour obtenir une prédiction du modèle champion.")

    model, meta = load_champion(dataset_key)
    if model is None:
        st.warning("⚠️ Aucun champion trouvé. Lancez `python train_models.py` d'abord.")
    else:
        st.markdown(f'<span class="champion-badge">🏆 {meta["model_name"]}</span>', unsafe_allow_html=True)
        st.markdown("")

        feature_info = get_feature_info(dataset_key)

        # Formulaire dynamique
        with st.form("prediction_form"):
            input_values = {}
            cols = st.columns(2)

            for i, (feat, info) in enumerate(feature_info.items()):
                with cols[i % 2]:
                    if info["type"] == "numerical":
                        input_values[feat] = st.number_input(
                            feat,
                            min_value=info["min"],
                            max_value=info["max"],
                            value=round(info["mean"], 2),
                            key=f"input_{feat}",
                        )
                    else:
                        input_values[feat] = st.selectbox(
                            feat,
                            options=info["values"],
                            key=f"input_{feat}",
                        )

            submitted = st.form_submit_button("🔮 Prédire", use_container_width=True)

        if submitted:
            try:
                result = predict_single(dataset_key, input_values)

                st.markdown("---")
                if task == "regression":
                    st.success(f"### 📈 Prédiction : **{result:,.2f}**")
                else:
                    label = "✅ Positif (1)" if int(result) == 1 else "❌ Négatif (0)"
                    st.success(f"### 🏷️ Classe prédite : **{label}**")

                log_action("prediction_single_submitted", f"dataset={dataset_key}, result={result}")
            except Exception as e:
                st.error(f"Erreur : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Prédiction batch
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📦 Prédiction batch")
    st.markdown("Uploadez un fichier CSV contenant les mêmes features que le dataset original.")

    model, meta = load_champion(dataset_key)
    if model is None:
        st.warning("⚠️ Aucun champion trouvé. Lancez `python train_models.py` d'abord.")
    else:
        uploaded_file = st.file_uploader("📂 Choisir un fichier CSV", type=["csv"], key="batch_upload")

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.markdown(f"**{len(df_upload)} lignes** chargées.")

                with st.expander("👀 Aperçu des données uploadées"):
                    st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button("🚀 Lancer les prédictions", use_container_width=True):
                    with st.spinner("Prédiction en cours…"):
                        df_result = predict_batch(dataset_key, df_upload)

                    st.success(f"✅ {len(df_result)} prédictions effectuées !")

                    st.dataframe(df_result, use_container_width=True)

                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Télécharger les résultats (CSV)",
                        data=csv,
                        file_name=f"predictions_{dataset_key}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    log_action("batch_upload", f"dataset={dataset_key}, rows={len(df_result)}")
            except Exception as e:
                st.error(f"Erreur : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Analyse IA générative (Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("🤖 Analyse IA Générative (Gemini)")

    api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key:
        st.warning(
            "⚠️ Clé API Gemini non configurée.\n\n"
            "Ajoutez `GOOGLE_API_KEY=votre_clé` dans le fichier `.env` "
            "ou dans les secrets Streamlit Cloud."
        )
    else:
        champion_name = get_champion_name(dataset_key)
        metrics = get_champion_metrics(dataset_key)
        task_type = DATASETS[dataset_key]["task"]

        # Contexte automatique
        context_parts = [
            f"Dataset : {dataset_key}",
            f"Tâche : {task_type}",
            f"Modèle champion : {champion_name}",
        ]
        if metrics:
            context_parts.append(f"Métriques : {metrics}")

        default_prompt = (
            "Analyse les résultats du modèle champion pour ce dataset. "
            "Donne une interprétation des métriques, identifie les forces et faiblesses, "
            "et propose des recommandations concrètes pour améliorer les performances."
        )

        user_prompt = st.text_area(
            "💬 Votre question / demande d'analyse",
            value=default_prompt,
            height=120,
        )

        if st.button("🧠 Générer l'analyse", use_container_width=True):
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model_genai = genai.GenerativeModel("gemini-2.0-flash")

                full_prompt = (
                    "Tu es un expert Data Scientist / MLOps. Voici le contexte :\n"
                    + "\n".join(context_parts)
                    + "\n\nDemande de l'utilisateur :\n"
                    + user_prompt
                    + "\n\nRéponds en français avec des sections structurées (Markdown)."
                )

                with st.spinner("Gemini réfléchit… 🤔"):
                    response = model_genai.generate_content(full_prompt)

                st.markdown("---")
                st.markdown("### 📝 Analyse générée")
                st.markdown(response.text)

                log_action("gemini_analysis", f"dataset={dataset_key}, prompt_length={len(user_prompt)}")
            except Exception as e:
                st.error(f"Erreur Gemini : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Dashboard & Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("📊 Dashboard & Visualisations")

    try:
        df_raw = load_data(dataset_key)
        target = DATASETS[dataset_key]["target"]

        # ── Métriques du champion ──────────────────────────────────────────────
        metrics = get_champion_metrics(dataset_key)
        if metrics:
            st.markdown("### 🏆 Métriques du modèle champion")
            st.markdown(
                f'<span class="champion-badge">🏆 {get_champion_name(dataset_key)}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            metric_cols = st.columns(len(metrics))
            for col, (k, v) in zip(metric_cols, metrics.items()):
                col.markdown(
                    f'<div class="metric-card"><div class="value">{v:.4f}</div>'
                    f'<div class="label">{k}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

        # ── Distribution de la cible ───────────────────────────────────────────
        st.markdown("### 📈 Distribution de la variable cible")
        if task == "classification":
            fig_target = px.histogram(
                df_raw, x=target, color=target,
                color_discrete_sequence=["#667eea", "#f6ad55"],
                title=f"Distribution de '{target}'",
                barmode="group",
            )
        else:
            fig_target = px.histogram(
                df_raw, x=target,
                color_discrete_sequence=["#667eea"],
                title=f"Distribution de '{target}'",
                nbins=40,
            )
        fig_target.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_target, use_container_width=True)

        # ── Distributions des features numériques ──────────────────────────────
        num_cols = df_raw.select_dtypes(include="number").columns.drop(target, errors="ignore").tolist()

        if num_cols:
            st.markdown("### 📊 Distributions des features numériques")
            selected_features = st.multiselect(
                "Choisir les features à visualiser",
                num_cols,
                default=num_cols[:4],
            )
            if selected_features:
                n = len(selected_features)
                cols_per_row = min(n, 2)
                chart_cols = st.columns(cols_per_row)
                for i, feat in enumerate(selected_features):
                    with chart_cols[i % cols_per_row]:
                        fig = px.histogram(
                            df_raw, x=feat,
                            color_discrete_sequence=["#764ba2"],
                            title=feat,
                            nbins=30,
                        )
                        fig.update_layout(template="plotly_white", height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

        # ── Matrice de corrélation ─────────────────────────────────────────────
        st.markdown("### 🔗 Matrice de corrélation")
        corr = df_raw[num_cols + [target]].corr() if target in df_raw.select_dtypes(include="number").columns else df_raw[num_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Corrélations",
            aspect="auto",
        )
        fig_corr.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── Features catégorielles ─────────────────────────────────────────────
        cat_cols = df_raw.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.markdown("### 🏷️ Variables catégorielles")
            cat_cols_vis = st.columns(min(len(cat_cols), 2))
            for i, col in enumerate(cat_cols):
                with cat_cols_vis[i % len(cat_cols_vis)]:
                    value_counts = df_raw[col].value_counts()
                    fig_cat = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=col,
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    )
                    fig_cat.update_layout(height=350)
                    st.plotly_chart(fig_cat, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🧠 AI Business Intelligence App · "
    "MLflow · DagsHub · Gemini · Streamlit</small></center>",
    unsafe_allow_html=True,
)
