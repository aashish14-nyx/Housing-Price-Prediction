# Housing Price Prediction — End to End ML System

An end-to-end machine learning system for predicting housing prices across 30 US metro cities using XGBoost. The project covers the full ML engineering lifecycle: data preprocessing, feature engineering, model training, hyperparameter tuning, REST API serving, and a live interactive dashboard.

## Live Demo

**Streamlit Dashboard:** [https://housing-price-prediction-ge3cgmmzc6ex99fogegsan.streamlit.app/](https://housing-price-prediction-ge3cgmmzc6ex99fogegsan.streamlit.app/)

> Note: The API backend runs on Render's free tier and may take 30–50 seconds to wake up after inactivity. This is expected — wait a moment and try again if predictions are slow to load.

---

## Architecture

```
Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Serve
```

### Core Modules

- **`src/feature_pipeline/`** — Data loading, preprocessing, feature engineering
  - `load.py`: Time-aware data splitting (train <2020, eval 2020–21, holdout ≥2022)
  - `preprocess.py`: City normalization, lat/lng merging, deduplication, outlier removal
  - `feature_engineering.py`: Date features, frequency encoding (zipcode), target encoding (city_full)

- **`src/training_pipeline/`** — Model training and hyperparameter optimization
  - `train.py`: Baseline XGBoost training
  - `tune.py`: Optuna-based hyperparameter tuning with MLflow experiment tracking
  - `eval.py`: Model evaluation and metrics

- **`src/inference_pipeline/`** — Production inference
  - `inference.py`: Applies saved encoders and aligns schema to training feature order before prediction

- **`src/batch/`** — Batch processing
  - `run_monthly.py`: Generates monthly predictions on holdout data

- **`src/api/`** — FastAPI backend
  - `main.py`: REST API with Cloudflare R2 integration, health checks, prediction and batch endpoints

- **`app.py`** — Streamlit dashboard
  - Interactive filtering by year, month, and region
  - Predictions vs actuals table with MAE, RMSE, and % Error metrics
  - Yearly trend chart with selected month highlighted

---

## Cloud Infrastructure

| Component | Service |
|---|---|
| Model & Data Storage | Cloudflare R2 (S3-compatible) |
| API Backend | Render (free tier) |
| Dashboard | Streamlit Community Cloud |
| Version Control | GitHub |

Environment variables (R2 credentials, API URL) are managed securely on each platform — no secrets are stored in the codebase.

---

## Model Performance

- **Algorithm:** XGBoost Regressor
- **R²:** 0.9222 on holdout data
- **Avg % Error:** ~9% on holdout data
- **Coverage:** 30 US metro cities, time-based train/eval/holdout splits

---

## Data Leakage Prevention

- Time-based splits (not random) — train < 2020, eval 2020–21, holdout ≥ 2022
- Encoders fitted only on training data, applied to eval/holdout
- Leakage-prone columns (raw city names, zipcodes) dropped before training
- Schema and feature order enforced at inference time using the model's own booster feature names

---

## Local Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Create a `.env` file in the project root (never commit this):

```
S3_BUCKET=your-bucket-name
AWS_REGION=auto
S3_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### 3. Upload data and model to R2

```bash
python upload_data.py
```

### 4. Run the pipeline

```bash
# Preprocess
python -m src.feature_pipeline.preprocess

# Feature engineering
python -m src.feature_pipeline.feature_engineering

# Train
python src/training_pipeline/train.py

# Tune (with MLflow)
python src/training_pipeline/tune.py

# Evaluate
python src/training_pipeline/eval.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 6. Start the dashboard

```bash
streamlit run app.py
```

---

## Testing

```bash
# Run all tests
pytest

# Verbose
pytest -v
```

---

## Tech Stack

**ML/Data:** Python, XGBoost, scikit-learn, pandas, NumPy, category-encoders

**API & Dashboard:** FastAPI, Uvicorn, Streamlit, Plotly

**Experiment Tracking:** MLflow, Optuna

**Storage:** Cloudflare R2 (boto3/S3-compatible)

**Deployment:** Render (API), Streamlit Community Cloud (dashboard)

**Tools:** Git, GitHub, Jupyter, joblib, python-dotenv