import os
import mlflow
import pandas as pd

_model = None


def _load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        _model = mlflow.pyfunc.load_model("models:/salary-predictor/Production")
    return _model


def predict(request) -> float:
    model = _load_model()
    
    df = pd.DataFrame([{
        'title': request.job_title,
        'full_description': request.description,
        'contract_type': request.contract_type,
        'contract_time': request.contract_time,
        'category_label': request.category_label,
        'location_area_length': float(request.location_area_length),
        'location_state': request.location_state,
        'location_region_abridged': request.location_region,
        'location_city_abridged': request.location_city,
        'missing_long_lat': float(request.missing_long_lat),
        'longitude': float(request.longitude) if request.longitude is not None else None,
        'latitude': float(request.latitude) if request.latitude is not None else None
    }])
    
    result = model.predict(df)
    return round(float(result[0]), 2)