import time
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#Connect to MLFlow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_URI = "models:/trust-safety-classifier@champion"

#global model container for startup population
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model artifact from {MODEL_URI}...")
    ml_models["champion"] = mlflow.sklearn.load_model(MODEL_URI)
    print("Champion model loaded and ready for inference")
    #clear model container on shut-down
    yield
    ml_models.clear()


#define app
app = FastAPI(
    title="Trust & Safety Real-Time Risk Scoring API",
    description="Serves calibrated LightGBM fraud/risk predictions from MLFlow Registry.",
    version="1.0.0",
    lifespan=lifespan,
)

# Strict Pydantic Input Schema matching the 30 creditcard features
class TransactionFeatures(BaseModel):
    Time: float = Field(..., example=406.0)
    V1: float = Field(..., example=-2.3122)
    V2: float = Field(..., example=1.9519)
    V3: float = Field(..., example=-1.6098)
    V4: float = Field(..., example=3.9979)
    V5: float = Field(..., example=-0.5221)
    V6: float = Field(..., example=-1.4265)
    V7: float = Field(..., example=-2.5373)
    V8: float = Field(..., example=1.3916)
    V9: float = Field(..., example=-2.7700)
    V10: float = Field(..., example=-2.7722)
    V11: float = Field(..., example=3.2020)
    V12: float = Field(..., example=-2.8999)
    V13: float = Field(..., example=-0.5952)
    V14: float = Field(..., example=-4.2892)
    V15: float = Field(..., example=0.3897)
    V16: float = Field(..., example=-1.1407)
    V17: float = Field(..., example=-2.8300)
    V18: float = Field(..., example=-0.0168)
    V19: float = Field(..., example=0.4169)
    V20: float = Field(..., example=0.1269)
    V21: float = Field(..., example=0.5172)
    V22: float = Field(..., example=-0.0350)
    V23: float = Field(..., example=-0.4652)
    V24: float = Field(..., example=0.3201)
    V25: float = Field(..., example=0.0445)
    V26: float = Field(..., example=0.1778)
    V27: float = Field(..., example=0.2611)
    V28: float = Field(..., example=-0.1432)
    Amount: float = Field(..., ge=0.0, example=0.0)

class RiskPredictionResponse(BaseModel):
    model_alias: str
    calibrated_risk_probability: float
    risk_decision: Literal["APPROVE", "STEP_UP_REVIEW", "BLOCK"]
    inference_latency_ms: float

def route_decision(prob: float, amount: float) -> Literal["APPROVE", "STEP_UP_REVIEW", "BLOCK"]:
    #Applies slice-aware thresholds based on offline eval
    block_threshold = 0.5
    review_threshold = 0.0057 #1% FPR operating point

    #review microtransactions to avoid high FPR
    if (amount <= 10.0) & (0.50 <= prob <= 0.75):
        return "STEP_UP_REVIEW"
    elif prob >= block_threshold:
        return "BLOCK"
    elif prob >= review_threshold:
        return "STEP_UP_REVIEW"
    return "APPROVE"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "champion" in ml_models,
        "model_uri": MODEL_URI, #obviously do not include in production env
    }

@app.post("/predict", response_model = RiskPredictionResponse)
def predict_risk(payload: TransactionFeatures):
    if "champion" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

    #convert validated Pydantic model to single-row DataFrame preserving column order
    input_df = pd.DataFrame([payload.model_dump()])
    prob = float(ml_models["champion"].predict_proba(input_df)[0, 1])
    decision = route_decision(prob, payload.Amount)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return RiskPredictionResponse(
        model_alias = "champion",
        calibrated_risk_probability=round(prob, 6),
        risk_decision=decision,
        inference_latency_ms=round(elapsed_ms, 3)
    )