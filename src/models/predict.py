import joblib
import numpy as np
from pathlib import Path

_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        model_path = Path(__file__).resolve().parents[2] / "models" / "salary_pipeline.joblib"
        _bundle = joblib.load(model_path)
    return _bundle


def predict(
    job_title: str,
    description: str,
    contract_type: str | None,
    contract_time: str | None,
    category_label: str,
    location_area_length: int,
    location_state: str,
    location_region: str,
    location_city: str,
    missing_long_lat: bool,
    longitude: float,
    latitude: float,
) -> float:
    b = _load_bundle()

    title_features = b['title_tfidf'].transform([job_title or ''])

    desc_sparse = b['desc_tfidf'].transform([description or ''])
    desc_features = b['desc_svd'].transform(desc_sparse)

    cat_input = [[
        contract_type,
        contract_time,
        category_label,
        location_area_length,
        location_state,
        location_region,
        location_city,
        missing_long_lat,
    ]]
    cat_features = b['ohe'].transform(cat_input)

    num_input = [[longitude, latitude]]
    num_imputed = b['imputer'].transform(num_input)
    num_features = b['scaler'].transform(num_imputed)

    X = np.hstack([
        title_features.toarray(),
        desc_features,
        cat_features,
        num_features,
    ])

    return round(float(b['model'].predict(X)[0]), 2)