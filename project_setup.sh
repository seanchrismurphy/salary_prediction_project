#!/bin/bash
set -e

PROJECT="salary_prediction_project"

echo "Creating project: $PROJECT"
mkdir -p "$PROJECT"
cd "$PROJECT"

# Directory structure
mkdir -p config
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p notebooks
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/api
mkdir -p src/monitoring
mkdir -p tests

# ── Git & repo files ──────────────────────────────────────

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Data (too large for git)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Trained models (too large for git)
models/*
!models/.gitkeep

# Environment
.env
*.env

# OS
.DS_Store
Thumbs.db
EOF

touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep

cat > README.md << 'EOF'
# Salary Predictor

Predicts salary from job descriptions and structured features using a LinkedIn dataset.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train the model
make train

# Run the API
make serve

# Run tests
make test
```

## Project Structure

- `src/data/` — data collection and preprocessing
- `src/features/` — feature engineering (text + structured)
- `src/models/` — training and prediction
- `src/api/` — FastAPI serving layer
- `notebooks/` — exploration and experimentation (not production code)
- `config/` — configuration files
- `models/` — serialized trained models (git-ignored, regenerate via `make train`)
EOF

# ── Requirements ──────────────────────────────────────────

cat > requirements.txt << 'EOF'
# Core
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3

# NLP
nltk>=3.8

# API
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0

# Config
pyyaml>=6.0

# Notebooks
jupyter>=1.0
matplotlib>=3.7
seaborn>=0.12

# Testing
pytest>=7.4
httpx>=0.24

# Linting (optional but recommended)
ruff>=0.1
EOF

# ── Makefile ──────────────────────────────────────────────

cat > Makefile << 'EOF'
.PHONY: train serve test lint clean

train:
	python -m src.models.train

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
EOF

# ── Dockerfile ────────────────────────────────────────────

cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# ── Config ────────────────────────────────────────────────

cat > config/model_config.yaml << 'EOF'
data:
  raw_path: data/raw
  processed_path: data/processed
  test_size: 0.2
  random_state: 42

features:
  text:
    max_features: 5000
    ngram_range: [1, 2]
  structured:
    categorical_cols: []
    numerical_cols: []

model:
  type: ridge
  params:
    alpha: 1.0
  output_path: models/model.joblib

api:
  host: 0.0.0.0
  port: 8000
EOF

# ── Python package init files ─────────────────────────────

cat > src/__init__.py << 'EOF'
EOF

cat > src/data/__init__.py << 'EOF'
EOF

cat > src/features/__init__.py << 'EOF'
EOF

cat > src/models/__init__.py << 'EOF'
EOF

cat > src/api/__init__.py << 'EOF'
EOF

cat > src/monitoring/__init__.py << 'EOF'
EOF

# ── Placeholder source files (empty, you'll build these) ─

touch src/data/collector.py
touch src/data/preprocessor.py
touch src/features/text_features.py
touch src/features/structured_features.py
touch src/models/train.py
touch src/models/predict.py
touch src/monitoring/drift.py

# ── API boilerplate (thin enough to be useful scaffolding) ─

cat > src/api/main.py << 'EOF'
from fastapi import FastAPI

from src.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Salary Predictor",
    description="Predict salary from job descriptions and structured features.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # TODO: load model, generate prediction
    raise NotImplementedError("Prediction not yet implemented")
EOF

cat > src/api/schemas.py << 'EOF'
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    job_title: str = ""
    job_description: str = ""
    location: str = ""

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "job_title": "Senior Data Scientist",
                "job_description": "We are looking for a senior data scientist...",
                "location": "Sydney, NSW",
            }
        ]
    }}


class PredictionResponse(BaseModel):
    predicted_salary: float
    confidence_interval: list[float] | None = None
EOF

# ── Test boilerplate ──────────────────────────────────────

cat > tests/__init__.py << 'EOF'
EOF

cat > tests/test_preprocessor.py << 'EOF'
EOF

cat > tests/test_predict.py << 'EOF'
EOF

cat > tests/test_api.py << 'EOF'
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
EOF

echo ""
echo "Done. Project scaffolded at $(pwd)"
echo ""
echo "Next steps:"
echo "  cd $PROJECT"
echo "  python -m venv venv && source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  git init && git add -A && git commit -m 'initial scaffold'"
