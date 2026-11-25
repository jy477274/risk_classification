from typing import Literal
import os
import bentoml
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = os.getenv("MODEL_URI", "models:/trust-safety-classifier@champion")

class TransactionFeatures(BaseModel):
    Time: float = Field(..., examples=[406.0])
    V1: float = Field(..., examples=[-2.3122])
    V2: float = Field(..., examples=[1.9519])
    V3: float = Field(..., examples=[-1.6098])
    V4: float = Field(..., examples=[3.9979])
    V5: float = Field(..., examples=[-0.5221])
    V6: float = Field(..., examples=[-1.4265])
    V7: float = Field(..., examples=[-2.5373])
    V8: float = Field(..., examples=[1.3916])
    V9: float = Field(..., examples=[-2.7700])
    V10: float = Field(..., examples=[-2.7722])
    V11: float = Field(..., examples=[3.2020])
    V12: float = Field(..., examples=[-2.8999])
    V13: float = Field(..., examples=[-0.5952])
    V14: float = Field(..., examples=[-4.2892])
    V15: float = Field(..., examples=[0.3897])
    V16: float = Field(..., examples=[-1.1407])
    V17: float = Field(..., examples=[-2.8300])
    V18: float = Field(..., examples=[-0.0168])
    V19: float = Field(..., examples=[0.4169])
    V20: float = Field(..., examples=[0.1269])
    V21: float = Field(..., examples=[0.5172])
    V22: float = Field(..., examples=[-0.0350])
    V23: float = Field(..., examples=[-0.4652])
    V24: float = Field(..., examples=[0.3201])
    V25: float = Field(..., examples=[0.0445])
    V26: float = Field(..., examples=[0.1778])
    V27: float = Field(..., examples=[0.2611])
    V28: float = Field(..., examples=[-0.1432])
    Amount: float = Field(..., ge=0.0, examples=[0.0])


class RiskPredictionResponse(BaseModel):
    model_alias: str
    calibrated_risk_probability: float
    risk_decision: Literal["APPROVE", "STEP_UP_REVIEW", "BLOCK"]


def route_decision(prob: float, amount: float) -> Literal["APPROVE", "STEP_UP_REVIEW", "BLOCK"]:
    block_threshold = 0.5
    review_threshold = 0.0057  # 1% FPR operating point

    # Route microtransactions to step-up review to avoid high FPR
    if (amount <= 10.0) and (0.50 <= prob <= 0.75):
        return "STEP_UP_REVIEW"
    elif prob >= block_threshold:
        return "BLOCK"
    elif prob >= review_threshold:
        return "STEP_UP_REVIEW"
    return "APPROVE"

#BentoML Service class replaces FastAPI app + lifespan + global model dict
@bentoml.service(
    name = "trust_safety_risk_scorer",
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TrustSafetyRiskService:
    def __init__(self) -> None:
        #runs once per worker process at startup
        print(f"Loading model artifact from {MODEL_URI}...")
        self.model = mlflow.sklearn.load_model(MODEL_URI)
        print("Champion model loaded and ready for inference")

    @bentoml.on_shutdown
    def cleanup(self) -> None:
        self.model = None

    #adaptive batching groups concurrent HTTP requests into single dataframe inference
    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=64,
        max_latency_ms=10,
    )
    def predict(self, payloads: list[TransactionFeatures]) -> list[RiskPredictionResponse]:
        #vectorized batch scoring across all grouped requests
        input_df = pd.DataFrame([p.model_dump() for p in payloads])
        probs = self.model.predict_proba(input_df)[:, 1]

        return [
            RiskPredictionResponse(
                model_alias = "champion",
                calibrated_risk_probability = round(float(prob), 6),
                risk_decision = route_decision(float(prob), payload.Amount),
            )
            for payload, prob in zip(payloads, probs)
        ]