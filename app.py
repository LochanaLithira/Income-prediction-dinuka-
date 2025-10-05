"""Interactive Streamlit frontend for the Income Prediction project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TOOLS_DIR = BASE_DIR / "tools"

st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="💰",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    """Load the processed dataset used during model training."""
    return pd.read_csv(DATA_DIR / "cleaned.csv")


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the original dataset to source categorical value ranges."""
    df = pd.read_csv(DATA_DIR / "income.csv")
    # Align with preprocessing choices made in the notebooks
    return df.replace("?", "Unknown")


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[object, object, object | None]:
    """Load the trained model alongside supporting transformers."""
    model_path = MODELS_DIR / "best_xgboost_model.pkl"
    encoder_path = TOOLS_DIR / "education_encoder.pkl"
    scaler_path = TOOLS_DIR / "hours_scaler.pkl"

    model = joblib.load(model_path)
    education_encoder = joblib.load(encoder_path)

    try:
        hours_scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        hours_scaler = None

    return model, education_encoder, hours_scaler


@st.cache_resource(show_spinner=False)
def get_feature_columns() -> List[str]:
    """Return the feature ordering used during training."""
    clean_df = load_clean_data()
    return [column for column in clean_df.columns if column != "income"]


def build_category_options(raw_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Collect sorted categorical values to power Streamlit widgets."""
    categorical_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
        "gender",
    ]
    options: Dict[str, List[str]] = {}
    for column in categorical_cols:
        values = raw_df[column].dropna().unique().tolist()
        options[column] = sorted(values)
    return options


def preprocess_input(
    user_input: pd.DataFrame,
    feature_columns: List[str],
    education_encoder,
    hours_scaler,
) -> pd.DataFrame:
    """Mirror the notebook preprocessing pipeline for fresh user inputs."""
    prepared = user_input.copy()

    # Encode education levels
    prepared["education"] = education_encoder.transform(prepared["education"])

    # Apply the same scaling and transformations as training
    if hours_scaler is not None:
        prepared[["hours-per-week"]] = hours_scaler.transform(prepared[["hours-per-week"]])

    for column in ["capital-gain", "capital-loss"]:
        prepared[column] = np.log1p(prepared[column].astype(float))

    prepared = pd.get_dummies(
        prepared,
        columns=[
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country",
            "gender",
        ],
        drop_first=False,
    )

    # Align with the training feature space
    missing_cols = set(feature_columns) - set(prepared.columns)
    for column in missing_cols:
        prepared[column] = 0

    prepared = prepared[feature_columns]
    return prepared


def evaluate_model(clean_df: pd.DataFrame, model) -> Dict[str, float]:
    """Provide quick reference metrics using the saved training dataset."""
    X = clean_df.drop(columns=["income"])
    y = clean_df["income"]
    y_pred = model.predict(X)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
    }
    return metrics


def main() -> None:
    clean_df = load_clean_data()
    raw_df = load_raw_data()
    model, education_encoder, hours_scaler = load_artifacts()
    feature_columns = get_feature_columns()
    category_options = build_category_options(raw_df)

    st.title("Income Prediction Assistant")
    st.caption(
        "Interact with the trained XGBoost model to estimate whether an individual's income exceeds $50K."
    )

    with st.sidebar:
        st.header("Project snapshot")
        st.metric("Total cleaned samples", f"{len(clean_df):,}")
        st.metric("Feature count", f"{len(feature_columns)}")

        if hours_scaler is None:
            st.info(
                "Hours-per-week values are used as-is because no scaler artifact was found."
            )

        st.divider()
        st.subheader("Model performance")
        metrics = evaluate_model(clean_df, model)
        for name, value in metrics.items():
            st.write(f"**{name}:** {value:.3f}")

        st.divider()
        with st.expander("Peek at cleaned data", expanded=False):
            st.dataframe(clean_df.head(), width="stretch")

    st.subheader("Provide applicant details")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", min_value=17, max_value=90, value=35)
        education = col1.selectbox("Education", options=list(education_encoder.classes_), index=0)
        hours_per_week = col1.slider("Hours per week", min_value=1, max_value=99, value=40)

        capital_gain = col2.number_input("Capital gain", min_value=0, value=0, step=100)
        capital_loss = col2.number_input("Capital loss", min_value=0, value=0, step=100)
        workclass = col2.selectbox("Workclass", options=category_options["workclass"], index=0)

        marital_status = col3.selectbox("Marital status", options=category_options["marital-status"], index=0)
        occupation = col3.selectbox("Occupation", options=category_options["occupation"], index=0)
        relationship = col3.selectbox("Relationship", options=category_options["relationship"], index=0)
        race = col3.selectbox("Race", options=category_options["race"], index=0)
        native_country = col3.selectbox("Native country", options=category_options["native-country"], index=0)
        gender = col3.selectbox("Gender", options=category_options["gender"], index=0)

        submitted = st.form_submit_button("Predict income")

    if submitted:
        user_input = pd.DataFrame(
            [
                {
                    "age": age,
                    "education": education,
                    "capital-gain": capital_gain,
                    "capital-loss": capital_loss,
                    "hours-per-week": hours_per_week,
                    "workclass": workclass,
                    "marital-status": marital_status,
                    "occupation": occupation,
                    "relationship": relationship,
                    "race": race,
                    "native-country": native_country,
                    "gender": gender,
                }
            ]
        )

        try:
            prepared = preprocess_input(user_input, feature_columns, education_encoder, hours_scaler)
            prediction = model.predict(prepared)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba_values = model.predict_proba(prepared)[0]
                proba = float(proba_values[int(prediction)])

            label = ">=50K" if prediction == 1 else "<=50K"

            st.success(f"Predicted Income Class: **{label}**")
            if proba is not None:
                st.write(f"Model confidence: **{proba:.1%}**")

            with st.expander("See the model-ready row", expanded=False):
                st.dataframe(prepared, width="stretch")
        except Exception as exc:  # pragma: no cover - surfaced directly to the UI
            st.error("An error occurred while preparing the input. Please review your entries.")
            st.exception(exc)

    st.divider()
    st.subheader("Understand the dataset")
    st.write(
        "The cleaned dataset mirrors the preprocessing performed in the notebooks. "
        "Categorical values are label/one-hot encoded, income is binary (1 for >50K)."
    )
    st.dataframe(clean_df.describe(include="all").transpose(), width="stretch")


if __name__ == "__main__":
    main()
