"""Google Gemini integration for AI-powered data analysis."""
import google.generativeai as genai
import pandas as pd
import config


def init_gemini():
    """Initialize Gemini client."""
    genai.configure(api_key=config.GOOGLE_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")


def analyze_dataset(df: pd.DataFrame, dataset_name: str,
                    target: str, task: str, user_question: str = None) -> str:
    """
    Ask Gemini to analyze a dataset and answer a question about it.
    """
    model = init_gemini()

    # Build a rich context
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
        prompt = base_prompt + f"\n**User question:** {user_question}\n\nAnswer concisely, with actionable business insights."
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

    response = model.generate_content(prompt)
    return response.text


def analyze_prediction_result(dataset_name: str, target: str,
                               task: str, input_data: dict,
                               prediction) -> str:
    """Ask Gemini to interpret a single prediction result."""
    model = init_gemini()
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
    response = model.generate_content(prompt)
    return response.text


def analyze_batch_results(dataset_name: str, target: str, task: str,
                           predictions_df: pd.DataFrame) -> str:
    """Ask Gemini to summarize batch prediction results."""
    model = init_gemini()
    summary_stats = predictions_df["Prediction"].describe().to_string() \
        if task == "regression" else \
        predictions_df["Prediction"].value_counts().to_string()

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
    response = model.generate_content(prompt)
    return response.text
