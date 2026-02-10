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
