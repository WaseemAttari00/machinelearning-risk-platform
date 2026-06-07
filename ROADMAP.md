# Intelligent Risk Analytics Platform — Project Roadmap

## Project Overview

An end-to-end machine learning platform for risk prediction across two domains
(credit risk and network intrusion) using a unified, production-grade ML workflow.

---

## Milestone Map

```
Phase 1: Foundation          [Week 1]   Repository, environment, datasets, EDA
Phase 2: Feature Engineering [Week 2]   Cleaning, encoding, feature creation
Phase 3: Modeling            [Week 3]   Training, tuning, evaluation, SHAP
Phase 4: MLOps               [Week 4]   MLflow tracking, model registry
Phase 5: Deployment          [Week 5]   FastAPI + Streamlit + Docker
Phase 6: Polish              [Week 6]   Tests, README, final GitHub push
```

---

## Phase 1 — Foundation & EDA

### Goals
- Set up the repository and Python environment
- Download and inspect both datasets
- Perform exploratory data analysis (EDA) for each domain
- Understand class imbalance, distributions, and data quality

### Deliverables
- `notebooks/01_eda_credit_risk.ipynb`
- `notebooks/02_eda_network_intrusion.ipynb`
- Raw data placed in `data/raw/`
- `src/data/ingestion.py` — data loading utilities
- `src/data/validation.py` — data quality checks

---

## Phase 2 — Feature Engineering

### Goals
- Clean and preprocess raw data
- Engineer meaningful features
- Build reusable preprocessing pipelines using scikit-learn

### Deliverables
- `src/features/credit_features.py`
- `src/features/network_features.py`
- `src/data/preprocessing.py`
- `data/processed/` — serialized processed datasets

---

## Phase 3 — Modeling, Tuning & Explainability

### Goals
- Train baseline models (Logistic Regression)
- Train primary model (XGBoost)
- Tune hyperparameters with Optuna (50 trials, TPE sampler)
- Evaluate with full metrics suite and optimal threshold tuning
- Generate SHAP explanations

### Deliverables
- `src/models/train.py`
- `src/models/evaluate.py`
- `src/models/predict.py`
- `src/explainability/shap_analysis.py`
- `notebooks/03_model_experiments.ipynb`
- Saved models and SHAP plots in `models/`

---

## Phase 4 — MLOps with MLflow

### Goals
- Track every experiment (parameters, metrics, artifacts)
- Version trained models in the MLflow Model Registry
- Make experiments reproducible

### Deliverables
- MLflow tracking integrated into `src/models/train.py`
- Registered models visible in MLflow UI (`mlflow ui`)
- `mlruns/` directory committed as experiment evidence

---

## Phase 5 — API & Frontend Deployment

### Goals
- Wrap models in a FastAPI REST API
- Build a Streamlit interactive dashboard
- Containerize everything with Docker

### Deliverables
- `api/main.py` — FastAPI application
- `api/routes/predict.py` — prediction endpoints
- `api/schemas/` — Pydantic request/response schemas
- `frontend/app.py` — Streamlit dashboard
- `docker/docker-compose.yml`

---

## Phase 6 — Polish & GitHub Presentation

### Goals
- Write unit and integration tests
- Write professional README
- Final GitHub push with clean commit history

### Deliverables
- `tests/` — pytest suite (unit + integration)
- `README.md` — portfolio README with results table
- Clean git history

---

## Datasets

### Credit Risk — UCI Default of Credit Card Clients
- **URL**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Size**: 30,000 records, 23 features
- **Target**: `default` (1 = defaulted on next month's payment)
- **Class balance**: ~22% positive — moderate imbalance

### Network Intrusion — NSL-KDD
- **URL**: https://www.unb.ca/cic/datasets/nsl.html (free, no login)
- **Size**: 125,973 train / 22,544 test records, 41 features
- **Target**: `label` (normal vs. attack types → binarized to 0/1)
- **Notable**: test set contains novel attack types absent from training data

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Data Layer                        │
│  data/raw/ → ingestion.py → validation.py           │
│       → preprocessing.py → data/processed/         │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                 Feature Layer                       │
│  credit_features.py    network_features.py          │
│  scikit-learn Pipelines (fit on train only)         │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  Model Layer                        │
│  train.py → evaluate.py → shap_analysis.py         │
│  MLflow experiment tracking + model registry        │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│               Deployment Layer                      │
│  FastAPI (port 8000) ◄──► Streamlit (port 8501)    │
│  Docker Compose orchestrates both services          │
└─────────────────────────────────────────────────────┘
```
