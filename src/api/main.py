from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictionRequest, PredictionResponse
from src.models.predict import predict

app = FastAPI(
    title="Salary Predictor",
    description="Predict Australian job salaries from job descriptions and structured features.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_salary(request: PredictionRequest):
    try:
        salary = predict(request)
        return PredictionResponse(predicted_salary=salary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))