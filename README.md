# Intelligent Risk Analytics Platform

> An end-to-end machine learning platform for risk prediction across multiple domains,
> demonstrating the complete ML lifecycle from raw data to deployed API.

---

## Use Cases

| Domain | Task | Dataset |
|---|---|---|
| **Credit Risk** | Predict probability of loan default | Give Me Some Credit (Kaggle) |
| **Network Intrusion** | Detect malicious vs. benign traffic | NSL-KDD |

---

## Architecture

```
Data Ingestion → Validation → Feature Engineering → Model Training
     ↓                                                     ↓
  MLflow Experiment Tracking ←──────────────────────────────
     ↓
  FastAPI REST API  ←──→  Streamlit Dashboard
     ↓
  Docker Compose (local deployment)
```

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone <your-repo-url>
cd ml-risk-platform
docker-compose -f docker/docker-compose.yml up --build
```

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501
- MLflow UI: http://localhost:5000

### Option 2 — Local Python

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (see data/README.md for instructions)

# 4. Run the full pipeline
python src/models/train.py --domain credit_risk
python src/models/train.py --domain network_intrusion

# 5. Start MLflow UI
mlflow ui

# 6. Start the API
uvicorn api.main:app --reload --port 8000

# 7. Start the frontend
streamlit run frontend/app.py
```

---

## Model Results

### Credit Risk (Give Me Some Credit)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

### Network Intrusion (NSL-KDD)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

*Results will be filled in after training is complete.*

---

## Repository Structure

```
.
├── src/                    # All source code (importable Python modules)
│   ├── data/               # Data ingestion, validation, preprocessing
│   ├── features/           # Domain-specific feature engineering
│   ├── models/             # Training, evaluation, prediction
│   ├── explainability/     # SHAP analysis
│   └── utils/              # Shared utilities (logging, config loading)
├── data/
│   ├── raw/                # Original, unmodified data (gitignored if large)
│   │   ├── credit_risk/
│   │   └── network_intrusion/
│   └── processed/          # Cleaned and feature-engineered data
├── notebooks/              # Jupyter notebooks (EDA, experiments)
├── models/                 # Serialized trained models
├── api/                    # FastAPI application
│   ├── routes/             # Endpoint definitions
│   └── schemas/            # Pydantic request/response models
├── frontend/               # Streamlit dashboard
├── tests/                  # Unit and integration tests
├── docker/                 # Dockerfiles and docker-compose
├── configs/                # YAML configuration files
├── mlruns/                 # MLflow experiment tracking (auto-generated)
├── requirements.txt
└── ROADMAP.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Pandas, NumPy, Scikit-learn |
| Modeling | Scikit-learn, XGBoost |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |

---

## Author

Waseem Attari — built as a portfolio project demonstrating ML engineering practices.
