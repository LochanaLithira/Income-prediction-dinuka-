# Income Prediction Project

This repository houses an end-to-end workflow for predicting whether an individual's annual income exceeds $50K. It includes:

- Data cleaning and feature engineering notebooks
- Model selection and tuning for multiple classifiers, with an XGBoost model retained as the best performer
- Evaluation and interactive prediction notebooks
- A Streamlit frontend that mirrors the preprocessing pipeline and serves real-time predictions

## Project structure

```
.
├── app.py                   # Streamlit dashboard
├── data/                    # Raw and cleaned datasets
├── models/                  # Persisted trained models
├── notebooks/               # Exploratory, preprocessing, training, and evaluation notebooks
├── src/                     # (Reserved for future scripted pipeline components)
├── tools/                   # Saved preprocessing artifacts (encoders, scalers)
└── requirements.txt         # Python dependencies
```

## Getting started

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Run the Streamlit app

With dependencies installed and the persisted artifacts in place (`models/best_xgboost_model.pkl`, `tools/education_encoder.pkl`, and optionally `tools/hours_scaler.pkl`), launch the dashboard:

```bash
streamlit run app.py
```

The app provides:

- Sidebar metrics summarising the training data and model performance
- A guided form to capture demographic and financial attributes
- On-submit preprocessing that matches the notebook pipeline (label encoding, one-hot encoding, log transforms, and optional scaling)
- Predicted income class with model confidence, plus a peek at the model-ready feature vector

## Reproducing the pipeline

The notebooks under `notebooks/` document the full lifecycle:

1. `preprocess.ipynb` – Cleans the raw Census Income dataset, handles missing values, encodes categorical features, and exports `data/cleaned.csv` alongside helper artifacts.
2. `train.ipynb` – Trains multiple models, performs XGBoost hyper-parameter search, and saves `models/best_xgboost_model.pkl`.
3. `evaluation.ipynb` – Reports model metrics on a hold-out split.
4. `predict.ipynb` – Demonstrates inference with the persisted model and preprocessing artifacts.

Running these notebooks sequentially regenerates every artifact consumed by the Streamlit interface.

## Troubleshooting

- **Missing scaler**: If `tools/hours_scaler.pkl` is absent, the app falls back to using raw `hours-per-week` values (matching the cleaned dataset). A sidebar hint will confirm this state.
- **Encoding mismatches**: Ensure new categorical selections exist in the raw dataset or extend the preprocessing notebooks to accommodate fresh categories before inference.

Feel free to adapt the UI, extend the feature set, or embed the prediction logic into other applications.
