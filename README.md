# Salary Predictor

Predicts Australian job salaries from job listings using data from the Adzuna API. Built as a production ML system with automated retraining, experiment tracking, gated deployment, and a live prediction API.

**Live demo:** [Frontend](https://seanchrismurphy.github.io/salary_prediction_project/)
**API endpoint:** [API Endpoint](https://salary-predictor-app.wittysky-2997a03d.australiaeast.azurecontainerapps.io/predict)

---

## Architecture

```
Adzuna API → Pipeline (collect → engineer → scrape → train)
                ↓ logs metrics + registers model
           MLflow (Azure ML)
                ↓ loads latest Production model on cold start
           FastAPI (Azure Container Apps) ← Frontend (GitHub Pages)
```

The pipeline runs on a schedule via Azure Container Apps Jobs. New models are only promoted to Production if validation MAE is within 5% of the previous run.

---

## Tech Stack

- **Data:** Adzuna Jobs API, BeautifulSoup scraper
- **Modelling:** scikit-learn (Ridge regression, TF-IDF, TruncatedSVD, OneHotEncoder)
- **Experiment tracking:** MLflow with SQLite backend (local) / Azure ML (production)
- **API:** FastAPI + Pydantic, containerised with Docker
- **Deployment:** Azure Container Apps, Azure Container Registry
- **Frontend:** Vanilla JS, hosted on GitHub Pages

---

## Project Structure

```
pipeline/               # Scheduled retraining pipeline
    collect_data.py     # Adzuna API collection
    engineer_features.py
    scrape_descriptions.py
    train_model.py      # Time-based train/val split, MLflow logging, gated deployment
    run_pipeline.py     # Orchestrator with lockfile, MLflow run management
    utils.py            # Shared utilities (atomic saves, project root)

src/
    api/
        main.py         # FastAPI app
        schemas.py      # Pydantic request/response models
    models/
        predict.py      # Loads latest Production model from MLflow registry

notebooks/              # Exploration and experimentation (not production code)
config/                 # Model configuration
docs/                   # Frontend
```

---

## Running Locally

**Prerequisites:** Python 3.12, virtualenv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
cd pipeline
python run_pipeline.py           # full run
python run_pipeline.py --test    # test run (limited data)
```

**Start MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**Run the API:**
```bash
uvicorn src.api.main:app --reload
```

**Run tests:**
```bash
pytest tests/
```

---

## Model

The prediction pipeline combines three feature types:

- **Job title:** TF-IDF (10k features, bigrams)
- **Full description:** TF-IDF + TruncatedSVD (100 components)
- **Structured features:** contract type, contract time, category, location hierarchy, coordinates

Final model: Ridge regression. Validation MAE ~$20,700 on a time-based holdout (most recent 20% of data).

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Time-based train/val split | Simulates real deployment conditions — model evaluated on future data |
| Gated deployment | New model only promoted if val MAE within 5% of previous run |
| Live model loading | API pulls latest Production model from MLflow registry on cold start; decouples retraining from Docker deploys |
| Atomic file saves | Prevents data corruption on interrupted pipeline runs |
| Pipeline lockfile | Prevents race conditions from concurrent runs |
| pyfunc model wrapper | Registers full pipeline bundle (transformers + model) rather than bare estimator |

