# Intelligent Risk Analytics Platform — Project Roadmap

## Project Overview

A portfolio-quality, end-to-end machine learning platform that predicts risk across two
domains (credit risk and network intrusion) using a unified, production-grade ML workflow.

---

## Milestone Map

```
Phase 1: Foundation         [Week 1]     Repository, environment, datasets, EDA
Phase 2: Feature Engineering [Week 2]    Cleaning, encoding, feature creation
Phase 3: Modeling           [Week 3]     Training, tuning, evaluation, SHAP
Phase 4: MLOps              [Week 4]     MLflow tracking, model registry
Phase 5: Deployment         [Week 5]     FastAPI + Streamlit + Docker
Phase 6: Polish             [Week 6]     Tests, README, final GitHub push
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

### Key Concepts
- **Class imbalance**: Both datasets will have far more "safe" samples than "risky"
  ones. This is the normal reality in fraud/risk domains.
- **Data leakage**: Understanding which features are safe to use vs. which ones would
  only be available after the event we are predicting (cheating).
- **Distribution analysis**: Checking for skewness, outliers, missing values.

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
- `notebooks/03_feature_engineering.ipynb`

### Key Concepts
- **Pipelines**: Chaining transformations so train/test transforms are always consistent
- **Imputation**: Strategies for missing values (median, mode, model-based)
- **Scaling**: StandardScaler vs. RobustScaler (robust is better with outliers)
- **Encoding**: One-hot vs. ordinal vs. target encoding for categorical features
- **Feature selection**: Dropping highly correlated features, low-variance features

---

## Phase 3 — Modeling, Tuning & Explainability

### Goals
- Train baseline models (Logistic Regression, Decision Tree)
- Train advanced model (XGBoost)
- Tune hyperparameters with Optuna
- Evaluate with full metrics suite
- Generate SHAP explanations

### Deliverables
- `src/models/train.py`
- `src/models/evaluate.py`
- `src/models/predict.py`
- `src/explainability/shap_analysis.py`
- `notebooks/04_model_experiments.ipynb`
- Saved models in `models/`

### Key Concepts
- **Baseline first**: Always establish a simple baseline before complex models
- **Cross-validation**: K-fold CV to get reliable performance estimates
- **Threshold tuning**: For imbalanced problems, the default 0.5 threshold is rarely optimal
- **SHAP**: Model-agnostic explanations that show each feature's contribution per sample

---

## Phase 4 — MLOps with MLflow

### Goals
- Track every experiment (parameters, metrics, artifacts)
- Version trained models in the MLflow Model Registry
- Make experiments reproducible

### Deliverables
- MLflow tracking integrated into `src/models/train.py`
- Registered models visible in MLflow UI (`mlflow ui`)
- `mlruns/` directory auto-populated

### Key Concepts
- **Experiment tracking**: Logging hyperparameters and metrics so you can compare runs
- **Model registry**: Promoting models through Staging → Production lifecycle
- **Reproducibility**: Logging dataset versions and random seeds alongside models

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
- `frontend/app.py` — Streamlit app
- `docker/Dockerfile.api`
- `docker/Dockerfile.frontend`
- `docker/docker-compose.yml`

### Key Concepts
- **REST API design**: Clean endpoint naming, versioning (`/api/v1/`), error handling
- **Pydantic validation**: Input validation at the API boundary
- **Docker multi-stage builds**: Keeping image sizes small
- **docker-compose**: Orchestrating multiple containers locally

---

## Phase 6 — Polish & GitHub Presentation

### Goals
- Write tests
- Write professional README with badges, architecture diagram, usage instructions
- Final GitHub push

### Deliverables
- `tests/` — unit + integration tests
- `README.md` — polished portfolio README
- Clean git history with meaningful commit messages

---

## Datasets

### Credit Risk — "Give Me Some Credit" (Kaggle)
- **URL**: https://www.kaggle.com/c/GiveMeSomeCredit
- **Size**: ~150,000 records, 11 features
- **Target**: `SeriousDlqin2yrs` (1 = defaulted within 2 years)
- **Why this dataset**: Well-sized for a portfolio project, real financial features
  (revolving utilization, debt ratio, income, age), well-known benchmark, binary
  classification with natural class imbalance (~7% positive class).

### Network Intrusion — NSL-KDD
- **URL**: https://www.unb.ca/cic/datasets/nsl.html (free, no login)
- **Size**: ~125,000 records, 41 features
- **Target**: `class` (normal vs. attack types → binarized to 0/1)
- **Why this dataset**: Free direct download, classic benchmark that interviewers
  recognize, manageable size, demonstrates different feature types (continuous +
  categorical + binary), well-documented.

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
│  scikit-learn Pipelines (fit on train only)        │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  Model Layer                        │
│  train.py → evaluate.py → shap_analysis.py         │
│  MLflow experiment tracking + model registry       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│               Deployment Layer                      │
│  FastAPI (port 8000) ◄──► Streamlit (port 8501)    │
│  Docker Compose orchestrates both services         │
└─────────────────────────────────────────────────────┘
```

---

## What Makes a Strong Final GitHub Portfolio

### 1. Professional README
- Project banner/title
- Badges (Python version, license, Docker)
- Clear problem statement (1 paragraph)
- Architecture diagram
- Quick-start instructions (clone → docker-compose up)
- Screenshots of the Streamlit app
- Results table (model comparison)
- Link to MLflow experiment results

### 2. Clean Code Structure
- Every Python module has a clear single responsibility
- No notebook code mixed into src/
- Config-driven (YAML configs, not hardcoded values)
- Consistent logging (not print statements)

### 3. Notebooks Tell a Story
- Numbered sequentially (01_, 02_, ...)
- Each has a markdown intro explaining what and why
- All cells executed top-to-bottom with clean output
- Professional visualizations (titles, labels, color palette)

### 4. MLflow Evidence
- Screenshot of MLflow UI showing multiple experiment runs
- Model comparison table in README
- Shows you understand experiment management

### 5. Tests
- Even basic tests show professionalism
- Test data loading, preprocessing, model prediction
- `pytest` runnable with one command

### 6. Docker Works First Try
- `docker-compose up` should start both API and frontend
- Health check endpoints in the API
- `.env.example` for any environment variables

### 7. Git History
- Meaningful commit messages ("feat: add XGBoost training pipeline")
- Logical progression of commits
- No large binary files committed (models saved via MLflow or .gitignored)

---

## Interview Discussion Points

### "Walk me through your ML pipeline"
> "I designed the pipeline around scikit-learn Pipeline objects so that all
> transformations are fit exclusively on training data and applied identically to
> test data, eliminating leakage. Data flows from raw → validated → processed →
> feature-engineered, with each stage persisted to disk for reproducibility."

### "How did you handle class imbalance?"
> "I used class_weight='balanced' in tree models and SMOTE as an alternative
> experiment. I also tuned the decision threshold rather than relying on the default
> 0.5, optimizing for F1 score on the positive class."

### "Why MLflow?"
> "MLflow gives me a structured way to compare experiments — every run logs its
> hyperparameters, metrics, and artifacts atomically. This means I can always
> reproduce any historical result and promote the best model to the registry
> without touching code."

### "How would you scale this to production?"
> "The current Docker Compose setup would map to a Kubernetes deployment. The
> FastAPI app is already stateless so horizontal scaling is straightforward.
> The MLflow registry abstracts model storage so switching from local to S3
> is one config change."
