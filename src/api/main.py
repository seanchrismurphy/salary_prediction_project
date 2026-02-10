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
