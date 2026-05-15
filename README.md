# ML Pipeline

A supervised ML pipeline that works on any CSV dataset. It auto-detects whether the task is classification or regression, trains several models in parallel, tracks everything with MLflow, and exposes a Streamlit dashboard + a FastAPI.

![CI](https://github.com/adammed31/ml-pipeline/actions/workflows/ci.yml/badge.svg)

**Stack:** Python · Scikit-learn · XGBoost · LightGBM · MLflow · FastAPI · Streamlit · Docker

## Quickstart

Requires [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/).

```bash
git clone https://github.com/adammed31/projet_ML.git
cd projet_ML
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |

## What it does

Upload any CSV, pick a target column, and the pipeline handles the rest:

- **Preprocessing** — missing value imputation (median, mean, KNN), categorical encoding (OneHot or Ordinal), StandardScaler
- **Training** — Logistic/Linear Regression, Random Forest, XGBoost, LightGBM, Voting ensemble. Optional cross-validation, hyperparameter tuning (GridSearch / RandomSearch), feature selection, and probability calibration
- **Explainability** — feature importances, SHAP (tree models), permutation importance, PDP, learning curves
- **Analysis** — adversarial validation to check train/test split quality, data drift detection (KS test, Chi-square, PSI), calibration curves
- **MLflow tracking** — all runs logged, comparison view, automatic best-model recommendation

## Project structure

```
projet_ML/
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── prediction.py
│   ├── drift.py
│   └── report.py
├── api/main.py
├── app/streamlit_app.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
└── docker-compose.yml
```

## Tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

33 tests covering preprocessing, training, and API endpoints.
