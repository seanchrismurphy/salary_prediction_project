from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    job_title: str = Field(..., example="Senior Data Scientist")
    description: str = Field(default="")
    contract_type: Optional[str] = Field(default=None, example="permanent")
    contract_time: Optional[str] = Field(default=None, example="full_time")
    category_label: str = Field(..., example="IT Jobs")
    location_area_length: int = Field(..., example=3)
    location_state: str = Field(..., example="New South Wales")
    location_region: str = Field(..., example="Sydney")
    location_city: str = Field(..., example="Sydney")
    missing_long_lat: bool = Field(default=False)
    longitude: float = Field(default=0.0, example=151.2093)
    latitude: float = Field(default=0.0, example=-33.8688)


class PredictionResponse(BaseModel):
    predicted_salary: float
    currency: str = "AUD"