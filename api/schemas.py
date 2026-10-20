from pydantic import BaseModel, Field


class TopK(BaseModel):
    class_name: str = Field(serialization_alias="class")
    probability: float


class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    top_k: list[TopK]
    inference_time_ms: float
    model_version: str