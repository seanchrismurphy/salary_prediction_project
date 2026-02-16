from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse
from src.models.predict import predict


app = FastAPI(
    title="Salary Predictor",
    description="Predict Australian job salaries from job descriptions and structured features.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_salary(request: PredictionRequest):
    try:
        salary = predict(
            job_title=request.job_title,
            description=request.description,
            contract_type=request.contract_type,
            contract_time=request.contract_time,
            category_label=request.category_label,
            location_area_length=request.location_area_length,
            location_state=request.location_state,
            location_region=request.location_region,
            location_city=request.location_city,
            missing_long_lat=request.missing_long_lat,
            longitude=request.longitude,
            latitude=request.latitude,
        )
        return PredictionResponse(predicted_salary=salary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))