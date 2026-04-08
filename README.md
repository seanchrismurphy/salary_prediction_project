# Salary Predictor
Predicts Australian job salaries from job listings using data from the Adzuna API. Built as a production ML system with automated retraining, experiment tracking, gated deployment, and a live prediction API.

**Live demo:** [Frontend](https://seanchrismurphy.github.io/salary_prediction_project/)
**API endpoint:** [API Endpoint](https://salary-predictor-app.wittysky-2997a03d.australiaeast.azurecontainerapps.io/predict)

---

## Architecture
```
Adzuna API → Pipeline (collect → engineer → scrape → train)
                ↓ writes data to Azure Blob Storage
           Blob Storage (salaryprdata / pipeline-data)
                ↓ logs metrics + registers model
           MLflow (Azure ML Workspace)
                ↓ loads latest Production model on cold start
           FastAPI (Azure Container Apps) ← Frontend (GitHub Pages)
```

The pipeline runs daily on a CRON schedule via Azure Container Apps Jobs. New models are only promoted to Production if validation MAE is within 5% of the previous run.

---

## Tech Stack

- **Data:** Adzuna Jobs API, BeautifulSoup scraper
- **Storage:** Azure Blob Storage (Parquet format)
- **Modelling:** scikit-learn, XGBoost (TF-IDF, TruncatedSVD, OneHotEncoder)
- **Experiment tracking:** MLflow 2.22.0 pinned against Azure ML workspace
- **API:** FastAPI + Pydantic, containerised with Docker
- **Deployment:** Azure Container Apps (API), Azure Container Apps Jobs (pipeline), Azure Container Registry
- **Frontend:** Vanilla JS, hosted on GitHub Pages

---

## Project Structure
```
pipeline/
    collect_data.py         # Adzuna API collection with deduplication
    engineer_features.py    # Feature engineering + location parsing
    scrape_descriptions.py  # BeautifulSoup scraper for full job descriptions
    train_model.py          # Time-based train/val split, MLflow logging, gated deployment
    run_pipeline.py         # Orchestrator with MLflow run management
    utils.py                # Blob storage read/write helpers (JSON + Parquet)
    requirements.txt        # Pipeline-specific dependencies
src/
    api/
        main.py             # FastAPI app with CORS
        schemas.py          # Pydantic request/response models
    models/
        predict.py          # Loads latest Production model from MLflow registry
model_wrapper.py            # SalaryPipelineWrapper (pyfunc) — shared by pipeline and API
notebooks/                  # Exploration and experimentation
dockerfile_api              # API container
Dockerfile.pipeline         # Pipeline container
LOGGING.txt                 # Logging reference — where to find logs and useful queries
```

---

## Running Locally

**Prerequisites:** Python 3.11+, virtualenv, Azure CLI (`az login` required for blob storage and MLflow access)

```bash
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
```

**Run the pipeline:**
```bash
python pipeline/run_pipeline.py           # full run
python pipeline/run_pipeline.py --test    # test run (limited data)
```

**Run the API:**
```bash
uvicorn src.api.main:app --reload
```

**View experiment logs:**
MLflow experiments are tracked in Azure ML. Access at ml.azure.com under Experiments > salary-predictor.

Container logs are available in Log Analytics (workspace-salarypredictorrgxnpB). See LOGGING.txt for queries.

---

## Model

The prediction pipeline combines three feature types:

- **Job title:** TF-IDF (10k features, bigrams)
- **Job description:** TF-IDF + TruncatedSVD (100 components)
- **Structured features:** contract type, contract time, category, location hierarchy, coordinates

Final model: XGBoost (n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=10). Validation MAE ~$17,800 on a time-based holdout (most recent 20% of data), improved from Ridge baseline of ~$20,600.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Time-based train/val split | Simulates real deployment — model evaluated on future data |
| Gated deployment | New model only promoted if val MAE within 5% of previous Production model |
| Live model loading | API pulls latest Production model from MLflow registry on cold start; decouples retraining from Docker deploys |
| Parquet storage | Schema enforcement and compression over CSV; pipeline data stored in Azure Blob Storage |
| Blob storage for all pipeline data | Ephemeral container filesystem means local file writes don't persist across runs |
| Separate pipeline and API containers | Independent deploy cycles; pipeline image includes Azure CLI for local credential testing |
| pyfunc model wrapper | Registers full pipeline bundle (transformers + model) rather than bare estimator |
| MLflow pinned to 2.22.0 | Azure ML implements MLflow 2.x API only; 3.x breaks at log_model |
| azure-common==1.1.28 installed separately | pip resolver silently drops it otherwise; required by azureml-mlflow at runtime |
| Chunked blob uploads | WSL2 network layer drops large single-PUT uploads; manual stage_block/commit_block_list workaround |

---

## Azure Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Resource group | salary-predictor-rg | All resources |
| Container Apps environment | salary-predictor-env | Shared environment for API and job |
| Container App | salary-predictor-app | FastAPI prediction API |
| Container Apps Job | salary-predictor-pipeline-job | Scheduled daily pipeline (0 18 * * * UTC) |
| Container Registry | salarypredictorscm.azurecr.io | Docker images |
| Storage account | salaryprdata | Pipeline data (Parquet) |
| Azure ML workspace | salary-predictor-ml | MLflow tracking + model registry |
| Log Analytics workspace | workspace-salarypredictorrgxnpB | Container logs |

---

## Next Steps

**CI/CD pipeline (GitHub Actions)**
Automate the build/tag/push/deploy sequence on commits to main. Currently done manually after each code change. Would require a service principal with ACR push and Container Apps update roles, stored as GitHub secrets.

**Model error over time (frontend)**
Surface val MAE trend from MLflow experiment history on the frontend. Shows the system is live and improving as data accumulates — high portfolio signal, low implementation effort.

**Seniority and recruiter flags**
Known model weakness: bigram TF-IDF cannot learn "Senior Data Scientist" as a phrase, and sparse training data for specific senior titles limits accuracy. Extracting explicit seniority (senior/lead/principal/head) and recruiter flags from titles would likely improve MAE meaningfully.

**SHAP feature importance dashboard**
Visualise which features drive predictions. Useful for understanding model behaviour and explaining predictions to non-technical users.

**Data quality checks**
Validate pipeline inputs before training — salary distribution bounds, minimum new record count, description scrape success rate. Fail fast with a clear error rather than training on bad data silently.

