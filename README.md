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
