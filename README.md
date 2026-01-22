# 🏠 House Price Prediction Project

An end-to-end machine learning system for predicting residential property prices in Amsterdam, covering **data collection, preprocessing, feature engineering, model training, hyperparameter tuning, model selection, production-ready model loading, API, and Vite+React frontend**.

The project intentionally evolves from exploratory notebooks into a **modular, reproducible, and production-oriented codebase**, while preserving experimentation history and decision-making transparency.

---

## 📌 Project Overview

This repository implements a complete applied ML workflow:

1. **Web scraping** of real-estate listings from Funda.nl
2. **Config-driven preprocessing & feature engineering**
3. **Model benchmarking and hyperparameter tuning** (Optuna)
4. **Experiment tracking & model versioning** (MLflow)
5. **Explicit production model selection**
6. **Model loading for inference**
7. **Deployment interfaces** (API / AWS Lambda / Frontend)

Exploratory analysis was performed in notebooks and later **refactored into scripts and reusable modules**.

---

## 🗂️ Repository Structure

house_price_prediction_project/
│
├── config/
│ ├── preprocessing_config.yaml
│ ├── model_config.yaml
│
├── data/
│ └── parsed_json/ # Parsed listing-level data
│
├── logs/
│ └── mlruns/ # MLflow tracking directory
│
├── notebooks/
│ └── 02_modelling.ipynb # Exploration & model comparison
│
├── scripts/
│ ├── run_optuna.py
│ ├── train_best_model.py
│ ├── select_and_train_best_model.py
│ └── load_production_model.py
│
├── src/
│ ├── scraper/ # Data collection
│ ├── data_loading/ # Parsing & preprocessing
│ ├── features/ # Feature engineering
│ ├── model/ # Training, evaluation, MLflow
│ ├── utils/ # Shared utilities
│ ├── api/ # Inference API
│ └── aws_lambda/ # Lambda deployment
│
├── frontend/ # Vite + React frontend
│
├── scrape_funda_url_for_data.py # Helper scraping script
│
└── README.md


---

## 🔍 Data Collection (`src/scraper`)

- Modular, reusable web scraping components
- Separation between:
  - URL discovery
  - HTML parsing
  - Data extraction
- Rate-limit and failure tolerant
- Outputs structured JSON

### Notes
`scrape_funda_url_for_data.py` is a **utility script** used to generate and scrape listing URLs.
It is **not part of the production ML pipeline**. But can be included if we would need retraining or for other reasons.

---

## 🧼 Data Loading & Preprocessing (`src/data_loading`)

Responsibilities:
- Loading raw JSON listings
- Schema normalization
- Type coercion
- Missing value handling
- Preprocessing of raw dat
- CacheManager class for cache management

Key principles:
- No model assumptions
- Fully config-driven
- Reusable across experiments and inference

---

## 🧠 Feature Engineering (`src/features`)

- Centralized feature preparation logic
- Supports:
  - baseline features
  - extended / engineered features
- CV-aware train / validation / test splits
- Behavior controlled via `model_config.yaml`

This allows:
- fair feature comparisons
- reproducible experiments
- minimal code duplication

---

## 🤖 Modeling & Evaluation (`src/model`)

### ModelEvaluator
A unified evaluation interface supporting:
- sklearn estimators
- `xgboost.train` with early stopping
- train / validation / test metrics
- optional target transformation (e.g. log-scale)

### Hyperparameter Optimization
- Optuna objectives abstracted into reusable functions
- Supports:
  - Random Forest
  - XGBoost
  - XGBoost with early stopping
- Integrated with MLflow logging

### MLflow Logging
Centralized logging via `MLFlowLogger`:
- Metrics
- Hyperparameters
- Model artifacts
- File-based local tracking (`logs/mlruns`)

---

## 📊 Experimentation & Model Selection

### Notebooks
`notebooks/01_modelling_main.ipynb` contains:
- Model benchmarking
- Feature addition experiments
- Hyperparameter tuning
- Overfitting analysis
- Final model comparison

Other notebooks have self-explanatory names. Most are for some personal tests.

### Scripts

| Script | Purpose |
|------|--------|
| `run_optuna.py` | Hyperparameter tuning |
| `train_best_model.py` | Train fixed-config models |
| `select_and_train_best_model.py` | Select & retrain best MLflow run |
| `load_production_model.py` | Load production model |

---

## 🧪 Model Selection Strategy

The **production model is not chosen solely by best test metric**. It comes from one of the models in the 01_modelling_main.ipynb notebook.

Selection criteria include:
- Test performance
- Train–test gap
- Stability across folds
- Overfitting behavior

The chosen production model is explicitly defined in last line of:

config/model_config.yaml

and

src/model/approved_model.yaml.

This ensures:
- Deterministic loading
- No accidental retraining
- Clear audit trail

---

## 🚀 Inference & Deployment

### `src/api`
- Lightweight inference layer
- Designed for backend or frontend consumption
- Decoupled from training logic

Some code is deprecated in src/api. That is mentioned in the .py files docstring at the top if deprecated.

### `src/aws_lambda`
- AWS Lambda-compatible inference wrapper
- Enables serverless deployment

### `frontend/`
- Vite + React application
- Consumes prediction API
- Separate from ML training code

---

## ▶️ How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a list of funda urls

Put the list of funda urls in "config/house_pages_scraped.txt". Example url: https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-75-b/43179082/

I have scraped funda sitemap for all the existing urls from Amsterdam. Doing that would give someone trying recreate the most data.

### 3. Now there is a choice, either the automated model selection, or exploration. I will outline both.

#### 3.a. Exploration (my followed path and here highly prefered due to outlier issue)

Run the notebooks/01_modelling_main.ipynb to get the best model with best hyperparameters. It is an orchestrator of the code from src. But there is more exploration and trade-off explanation.

#### 3.b. Automated

To have automated best test RMSE model, one can run scripts 01 to 04.

01_train_baselines.py trains baselines for comparison.
02_run_optuna.py runs hyperparamater tuning for xgboost and random forest models.
03_select_and_train_best_model.py selects and trains production model.
04_load_production_model.py loads production model.

If this path is used, for now, one has to change the explanation and documentation of production model in:

config/model_config.yaml
and
src/model/approved_model.yaml.

### 4. Dockerization and CI/CD

I have a Dockerfile at root used to build a container for AWS Lambda. I do not build the container locally. I build it using AWS CodeBuild.

For AWS Lambda, we have buildspec.yml at root for creation.
I am also using

### 5. Design Principles

- Config over hardcoding
- Explicit pipelines over magic
- Notebooks for discovery, scripts for execution
- Reproducibility first
- Production realism without over-engineering

### Notes

CI/CD and cloud orchestration are intentionally minimal

The system is designed to scale without rewriting core logic

This repository prioritizes clarity, traceability, and correctness

### Contact

For questions, collaboration, or discussion — feel free to reach out.
