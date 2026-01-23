# 🏠 House Price Prediction Project

An end-to-end machine learning system for predicting residential property prices in Amsterdam, built to mirror a realistic applied ML workflow—from data collection and experimentation to production-ready inference.

The project evolves from exploratory notebooks into a **modular, reproducible, production-oriented codebase**, with a strong emphasis on **transparent model selection, train–inference parity, and decision traceability**.

It covers **data collection, preprocessing, feature engineering, model training, hyperparameter tuning, production model selection, inference API, and a Vite+React frontend**.


**Motivation:**
The project was initiated to independently estimate residential property values and assess how listing prices compare to model-driven valuations, using publicly available listing information.

---

## ⭐ Key Features / Highlights

- Config-driven preprocessing and feature engineering
- Leakage-safe, fold-wise cross-validation
- Support for baseline and extended/engineered features
- Hyperparameter optimization with Optuna for XGBoost, Random Forest, and Linear Regression
- Unified evaluation interface (`ModelEvaluator`) for train/val/test metrics
- MLflow integration for experiment tracking, metrics, hyperparameters, and model artifacts
- Explicit production model selection with audit trail (`approved_model.yaml`)
- Serverless inference with AWS Lambda and lazy pipeline initialization
- Vite + React frontend for prediction consumption

---

## 📌 Project Overview

The repository implements a complete applied ML workflow:

1. **Web scraping** of [Funda.nl](https://www.funda.nl) listings
2. **Config-driven preprocessing & feature engineering**
3. **Model benchmarking & hyperparameter tuning** ([Optuna](https://optuna.org))
4. **Experiment tracking & model versioning** ([MLflow](https://mlflow.org))
5. **Explicit production model selection**
6. **Model loading for inference**
7. **Deployment interfaces** (API / AWS Lambda / Frontend)

Exploratory analysis is performed in notebooks, later **refactored into scripts and reusable modules**.

---

## ⚠️ Data Disclaimer

- All data is collected from Funda.nl for **educational and research purposes only**.
- **Do not redistribute raw scraped data**.
- Tools and explanation of how to collect data are shared, but the data itself is not shared.
- This project **does not endorse commercial scraping** or usage outside personal learning.

---

## 🗂️ Repository Structure

```text
house_price_prediction_project/
│
├── config/
│ ├── preprocessing_config.yaml
│ └── model_config.yaml
│
├── data/
│ └── parsed_json/ # Parsed listing-level data
│
├── logs/
│ └── mlruns/ # MLflow tracking
│
├── notebooks/
│ └── [01_modelling_main.ipynb](notebooks/01_modelling_main.ipynb)
│
├── scripts/
│ ├── [run_optuna.py](scripts/run_optuna.py)
│ ├── [train_best_model.py](scripts/train_best_model.py)
│ ├── [select_and_train_best_model.py](scripts/select_and_train_best_model.py)
│ └── [load_production_model.py](scripts/load_production_model.py)
│
├── src/
│ ├── scraper/ # Data collection (Web scraping)
│ ├── data_loading/ # Parsing & preprocessing
│ ├── features/ # Feature engineering
│ ├── model/ # Training, evaluation, MLflow
│ ├── utils/ # Shared utilities
│ ├── api/ # Inference API
│ └── aws_lambda/ # Lambda deployment
│
├── frontend/ # Vite + React frontend
├── scrape_funda_url_for_data.py
└── README.md
```
---

## 🔍 Data Collection (`src/scraper`)

- Modular, reusable web scraping components
- Separation between URL discovery, HTML parsing, and data extraction
- Rate-limit and failure tolerant
- Outputs structured JSON

> `scrape_funda_url_for_data.py` is a utility script for generating listing URLs.
> Not part of production inference, but useful for retraining or exploration.
> There is a README.md file for this folder going deeper.

---

## 🧼 Data Loading & Preprocessing (`src/data_loading`)

Responsibilities:

- Loading raw JSON listings
- Schema normalization & type coercion
- Missing value handling
- Preprocessing for modeling
- Cache management (`CacheManager`)

Principles:

- No model assumptions
- Fully config-driven
- Reusable across experiments & inference

> Note: On AWS Lambda, the preprocessing pipeline uses lazy initialization and can load a cached geolocation and amenities dataset for fast inference, avoiding large cold-start overheads.

---

## 🧠 Feature Engineering (`src/features`)

- Centralized feature preparation logic
- Supports baseline and extended/engineered features
- CV-aware train/validation/test splits
- Behavior controlled via `model_config.yaml`

Benefits:

- Fair feature comparisons
- Reproducible experiments
- Minimal code duplication

> `inference_meta.pkl` ensures production predictions align with training-time feature schema.
> There is a README.md file for this folder going deeper.

---

## 🤖 Modeling & Evaluation (`src/model`)

### ModelEvaluator
- Unified interface supporting sklearn & XGBoost
- Train / validation / test metrics
- Optional target transformation (e.g., log-scale)
- Early-stopping support for XGBoost

### Hyperparameter Optimization
- Optuna objectives with leakage-safe K-Fold CV
- Supports Random Forest, Linear Regression, and XGBoost
- Fold-wise feature engineering & optional geo/amenities enrichment

### MLflow Logging
- Centralized logging via `MLFlowLogger`
- Metrics, hyperparameters, model artifacts
- Local tracking in `logs/mlruns`

> Note: ModelEvaluator handles metrics for train/validation/test splits, target transformations, and early stopping. MLflow logging ensures reproducibility.
> ModelEvaluator ensures all training and evaluation metrics are consistent, while MLflow provides a transparent experiment history.

---

## 📊 Experimentation & Model Selection

### Notebooks
- [`notebooks/01_modelling_main.ipynb`](notebooks/01_modelling_main.ipynb)
  - Model benchmarking
  - Feature addition experiments
  - Hyperparameter tuning
  - Overfitting analysis
  - Final model comparison

Other notebooks are exploratory or testing.
- [`notebooks/06_trying_new_feature_expansion.ipynb`](notebooks/06_trying_new_feature_expansion.ipynb) generates `df_with_lat_lon_encoded.csv`, `amsterdam_amenities.csv`, etc. If new data is scraped, these must be regenerated.

### Scripts

| Script | Purpose |
|--------|---------|
| [`run_optuna.py`](scripts/run_optuna.py) | Hyperparameter tuning |
| [`train_best_model.py`](scripts/train_best_model.py) | Train fixed-config models |
| [`select_and_train_best_model.py`](scripts/select_and_train_best_model.py) | Select & retrain production model |
| [`load_production_model.py`](scripts/load_production_model.py) | Load production model |

---

## 🧪 Model Selection Strategy

- Production model is **not chosen solely by test metric**
- Selection criteria:
  - Test performance (RMSE, MAE, MAPE)
  - Train–test gap
  - Stability across folds
  - Overfitting behavior

- Explicitly defined in:
  - `config/model_config.yaml`
  - `src/model/approved_model.yaml`
  - `notebooks/01_modelling_main.ipynb`

Ensures deterministic loading and clear audit trail.

---

## 🚀 Inference & Deployment

### `src/api`
- Lightweight inference layer
- Designed for backend or frontend consumption
- Decoupled from training logic
> There is a README.md file for this folder going deeper (src/api/core).

### `src/aws_lambda`
- AWS Lambda-compatible wrapper
- Lazy pipeline initialization for fast cold-start
- Serverless deployment
> There is a README.md file for this folder going deeper.

### `frontend/`
- Vite + React application consuming prediction API
- Separate from ML training code

> Note: To run the Vite+React frontend locally, run the following three commands in the terminal:

- cd frontend
- npm install
- npm run dev

> I did not push to github node modules or build output. Nor the secrets/api keys.
---

## 🖼 Pipeline Diagram

```text
Funda.nl URLs
│
▼
Scraper (src/scraper)
│
▼
Data Loading & Preprocessing (src/data_loading)
│
▼
Feature Engineering (src/features)
│
▼
Model Training & Tuning (src/model + Optuna)
│
▼
MLflow Logging & Selection
│
▼
Production Model (model_config.yaml/approved_model.yaml)
│
▼
Inference API / AWS Lambda (src/api + aws_lambda)
│
▼
Frontend (Vite + React)
```

---

## ▶️ Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place Funda URLs in config/house_pages_scraped.txt

Example URL:
https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-75-b/43179082/

### 3. Choose workflow

#### a) Exploration (recommended)

- Run notebooks/01_modelling_main.ipynb
- Discover outliers & feature trade-offs
- Identify best production model

#### b) Automated

- Run scripts sequentially:
- 01_train_baselines.py
- 02_run_optuna.py
- 03_select_and_train_best_model.py
- 04_load_production_model.py

Update production model references in:

config/model_config.yaml

src/model/approved_model.yaml

Pipeline: Scraper → Feature Pipeline → Training Scripts → MLflow → Production Model → Lambda API

### 4. Design Principles

- Config over hardcoding
- Explicit pipelines over magic
- Notebooks for discovery, scripts for execution
- Reproducibility first
- Production realism without over-engineering

### 5. 🧰 CI/CD & Dev Tools

- .github/workflows/python-tests.yml contains CI/CD pipelines for automated testing, linting and formatting
- Dockerfile for containerized execution (used in AWS Lambda)
- .pre-commit-config.yaml for code quality enforcement
- requirements-dev.txt for development dependencies
- buildspec.yaml for AWS build pipelines

Included to improve reproducibility, testing, and deployment, though the system can also run locally without them.

### 6. Tests

- Tests exist in the tests/ folder

### 📬 Notes

- System scales without rewriting core logic
- Prioritizes clarity, traceability, and correctness

### Contact

For questions, collaboration, or discussion, feel free to reach out.
