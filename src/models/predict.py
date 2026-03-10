import os
import mlflow
import pandas as pd

_model = None


def _load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        _model = mlflow.pyfunc.load_model("models:/salary-predictor@production")
    return _model


def predict(request) -> float:
    model = _load_model()
    
    df = pd.DataFrame([{
        'title': request.job_title,
        'full_description': request.description,
        'contract_type': request.contract_type,
        'contract_time': request.contract_time,
        'category.label': request.category_label,
        'location.area_length': request.location_area_length,
        'location_state': request.location_state,
        'location_region_abridged': request.location_region,
        'location_city_abridged': request.location_city,
        'missing_long_lat': request.missing_long_lat,
        'longitude': request.longitude,
        'latitude': request.latitude,
    }])
    
    result = model.predict(df)
    return round(float(result[0]), 2)