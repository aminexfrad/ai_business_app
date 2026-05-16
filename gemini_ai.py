"""Google Gemini integration for AI-powered data analysis."""
import google.generativeai as genai
import pandas as pd

import config

# Lighter models first — often still available when flash quota is 0 on free tier.
DEFAULT_GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]


def _model_candidates() -> list[str]:
    preferred = (config.GEMINI_MODEL or "").strip()
    if preferred:
        return [preferred] + [m for m in DEFAULT_GEMINI_MODELS if m != preferred]
    return DEFAULT_GEMINI_MODELS.copy()


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "quota", "resource_exhausted", "rate limit", "limit: 0")
    )


def _is_model_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "not supported" in msg


def user_error_message(exc: Exception) -> str:
    """Human-readable error for the Streamlit UI (French)."""
    if not config.GOOGLE_API_KEY:
        return (
            "Clé API absente. Ajoutez `GOOGLE_API_KEY` dans le fichier `.env` "
            "ou dans les secrets Streamlit Cloud, puis redémarrez l'application."
        )

    msg = str(exc).lower()
    if _is_retryable(exc):
        return (
            "**Quota Gemini épuisé** pour votre clé API (erreur 429).\n\n"
            "Ce n'est pas un bug de l'application. Le compte Google n'a plus de "
            "requêtes gratuites sur les modèles testés.\n\n"
            "**Que faire :**\n"
            "1. Attendre la réinitialisation du quota (parfois 1–24 h)\n"
            "2. Créer une **nouvelle clé** sur [Google AI Studio](https://aistudio.google.com/apikey)\n"
            "3. Activer la **facturation** sur le projet Google Cloud lié à la clé\n"
            "4. Définir un autre modèle dans `.env` : `GEMINI_MODEL=gemini-2.0-flash-lite`\n\n"
            f"*Détail :* {exc}"
        )

    if "api key" in msg or "invalid" in msg and "key" in msg:
        return (
            "Clé API invalide. Vérifiez `GOOGLE_API_KEY` dans `.env` ou les secrets Streamlit."
        )

    return f"Erreur Gemini : {exc}"


def generate_text(prompt: str) -> str:
    """Call Gemini with automatic model fallback on quota / 404 errors."""
    if not config.GOOGLE_API_KEY:
        raise ValueError(user_error_message(ValueError("missing key")))

    genai.configure(api_key=config.GOOGLE_API_KEY)
    last_error = None
    tried = []

    for model_name in _model_candidates():
        tried.append(model_name)
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", None) or ""
            if text.strip():
                return text
        except Exception as exc:
            last_error = exc
            if _is_retryable(exc) or _is_model_unavailable(exc):
                continue
            raise ValueError(user_error_message(exc)) from exc

    raise ValueError(
        user_error_message(last_error or RuntimeError("no model available"))
        + f"\n\n*Modèles essayés :* {', '.join(tried)}"
    )


def analyze_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
    task: str,
    user_question: str = None,
) -> str:
    """Ask Gemini to analyze a dataset and answer a question about it."""
    stats = df.describe(include="all").to_string()
    sample = df.head(5).to_string()
    shape = df.shape
    nulls = df.isnull().sum().to_string()

    base_prompt = f"""
You are a senior data scientist and business intelligence analyst.
You are analyzing the **{dataset_name}** dataset for a business AI platform.

**Task type:** {task}
**Target variable:** {target}
**Dataset shape:** {shape[0]} rows × {shape[1]} columns

**Column statistics:**
{stats}

**Missing values:**
{nulls}

**Sample rows:**
{sample}
"""

    if user_question:
        prompt = base_prompt + (
            f"\n**User question:** {user_question}\n\n"
            "Answer concisely, with actionable business insights."
        )
    else:
        prompt = base_prompt + """
Please provide:
1. **Executive summary** (2-3 sentences about what this dataset represents)
2. **Key patterns & correlations** you observe
3. **Business insights** (top 3 actionable recommendations)
4. **Data quality notes** (any issues to be aware of)
5. **Model recommendations** (which ML approach suits this data best and why)

Use markdown formatting. Be concise and business-focused.
"""

    return generate_text(prompt)


def analyze_prediction_result(
    dataset_name: str,
    target: str,
    task: str,
    input_data: dict,
    prediction,
) -> str:
    """Ask Gemini to interpret a single prediction result."""
    prompt = f"""
You are a business analyst. A machine learning model just made a prediction.

**Dataset:** {dataset_name}
**Task:** {task}
**Target:** {target}
**Input features:** {input_data}
**Model prediction:** {prediction}

In 3-4 sentences, explain:
- What this prediction means in plain business language
- What the key input factors likely influenced this prediction
- One actionable recommendation based on this result

Be concise and avoid technical jargon.
"""
    return generate_text(prompt)


def analyze_batch_results(
    dataset_name: str,
    target: str,
    task: str,
    predictions_df: pd.DataFrame,
) -> str:
    """Ask Gemini to summarize batch prediction results."""
    summary_stats = (
        predictions_df["Prediction"].describe().to_string()
        if task == "regression"
        else predictions_df["Prediction"].value_counts().to_string()
    )

    prompt = f"""
You are a business analyst reviewing batch ML predictions.

**Dataset:** {dataset_name}
**Task:** {task}
**Target:** {target}
**Number of predictions:** {len(predictions_df)}
**Prediction summary:**
{summary_stats}

In 3-5 sentences, provide:
- A summary of the batch prediction results
- Notable patterns or outliers
- Business implications and recommended actions

Be concise and business-focused.
"""
    return generate_text(prompt)
