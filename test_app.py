"""Smoke tests for SmartBI App."""
import warnings
warnings.filterwarnings("ignore")


def main():
    errors = []

    try:
        import streamlit  # noqa: F401
        import pandas as pd
        import numpy as np
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
        import mlflow  # noqa: F401
        import dagshub  # noqa: F401
        import plotly.express as px
        import joblib  # noqa: F401
        import statsmodels  # noqa: F401
        print("OK imports")
    except Exception as e:
        errors.append(f"imports: {e}")

    try:
        import config
        import ml_pipeline
        import gemini_ai  # noqa: F401
        import logger  # noqa: F401
        print("OK modules")
    except Exception as e:
        errors.append(f"modules: {e}")
        return errors

    import os
    import pandas as pd

    for name, ds in config.DATASETS.items():
        if not os.path.exists(ds["file"]):
            errors.append(f"missing file: {ds['file']}")
            continue
        df = pd.read_csv(ds["file"])
        if ds["target"] not in df.columns:
            errors.append(f"target missing in {ds['file']}")
            continue
        ml_pipeline.preprocess(df, ds["target"])
        print(f"OK preprocess {ds['file']}")

    df = pd.read_csv("building_energy_regression_realistic.csv")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    feat_x = [c for c in num_cols if c != "energy_consumption"][0]
    px.scatter(df, x=feat_x, y="energy_consumption", trendline="ols")
    print("OK plotly OLS trendline")

    champ = ml_pipeline.load_champion_locally()
    print(f"OK champion load: {champ is not None}")

    if errors:
        print("FAILED:", *errors, sep="\n")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
