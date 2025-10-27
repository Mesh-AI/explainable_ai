# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from energy_forecast import predict as core_predict

app = FastAPI(title="Energy Forecast API", version="0.1.0")


class PredictRequest(BaseModel):
    row: Dict[str, Any] = Field(
        ..., description="Single feature row (include date_time if you want auto hour/dow/doy)"
    )


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        out = core_predict(req.model_dump())
        return PredictResponse(**out)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
