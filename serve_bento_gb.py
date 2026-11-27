import asyncio
import logging
from typing import Literal
import os
import bentoml
import pandas as pd
from growthbook import GrowthBook
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

#setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("risk_scorer")

#GrowthBook SDK config
GB_API_HOST = os.getenv("GB_API_HOST", "http://host.docker.internal:3100")
GB_CLIENT_KEY = os.getenv("GB_CLIENT_KEY")

#Custom Prometheus metrics
PREDICTION_SCORE_HIST = Histogram(
    "risk_prediction_score_distribution",
    "Distribution of calibrated fraud risk scores for online drift detection",
    labelnames=["model_variant"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)
SHADOW_DISAGREEMENT_COUNTER = Counter(
    "risk_shadow_decision_disagreement_total",
    "Count of requests where v2 shadow decision differed from v1 live decision",
    labelnames=["v1_decision", "v2_decision"]
)
MODEL_TRAFFIC_COUNTER = Counter(
    "risk_model_served_total",
    "Total live requests served by model variant and rollout mode",
    labelnames=["served_variant", "rollout_mode"]
)

MODEL_URI = os.getenv("MODEL_URI", "models:/trust-safety-classifier@champion")

class TransactionFeatures(BaseModel):
    transaction_id: str = Field(default="tx_default_001", examples=["tx_987654"])
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

#add transaction id and other fields for growthbook experiment
class RiskPredictionResponse(BaseModel):
    transaction_id: str
    served_model_variant: Literal["v1_champion", "v2_challenger"]
    rollout_mode: str
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
        self.v1_model = bentoml.sklearn.load_model("trust_safety_champion:latest")
        self.v2_model = bentoml.sklearn.load_model("trust_safety_challenger:latest")

        self.gb = GrowthBook(
            api_host=GB_API_HOST,
            client_key=GB_CLIENT_KEY,
        )
        self.gb.load_features()

    def _evaluate_rollout_flag(self, transaction_id: str) -> str:
        #evaluates growthbook feature flag ("shadow", "canary", "v1_only")
        self.gb.set_attributes({"id": transaction_id})
        return self.gb.get_feature_value("risk-model-rollout-mode", "v1_only")

    async def _run_shadow_v2(self, transaction_id: str, input_df: pd.DataFrame, amount: float, v1_prob: float, v1_decision: str) -> None:
        #non-blocking background run of v2. Logs disagreement rate
        prob_v2 = await asyncio.to_thread(self.v2_model.predict_proba, input_df)
        v2_prob = float(prob_v2[0,1])
        v2_decision = route_decision(v2_prob, amount)

        PREDICTION_SCORE_HIST.labels(model_variant="v2_challenger").observe(v2_prob)

        disagreed = v1_decision != v2_decision
        if disagreed:
            SHADOW_DISAGREEMENT_COUNTER.labels(
                v1_decision = v1_decision, v2_decision = v2_decision
            ).inc()

        #log run information
        logger.info(
            f'{{"event": "shadow_eval", "tx_id": "{transaction_id}", '
            f'"v1_prob": {v1_prob:.6f}, "v2_prob": {v2_prob:.6f}, '
            f'"v1_decision": "{v1_decision}", "v2_decision": "{v2_decision}", '
            f'"disagreed": {str(disagreed).lower()}}}'
        )

    @bentoml.on_shutdown
    def cleanup(self) -> None:
        self.v1_model = None
        self.v2_model = None


    @bentoml.api
    async def predict(self, payload: TransactionFeatures) -> RiskPredictionResponse:
        feature_dict = payload.model_dump(exclude={"transaction_id"})
        input_df = pd.DataFrame([feature_dict])

        rollout_mode = self._evaluate_rollout_flag(payload.transaction_id)

        if rollout_mode == "canary":
            #10% Canary Split: v2 serves the live response directly
            prob = float(self.v2_model.predict_proba(input_df)[0,1])
            decision = route_decision(prob, payload.Amount)
            served_variant = "v2_challenger"
        else:
            #Default / Shadow: V1 serves the live traffic immediately
            prob = float(self.v1_model.predict_proba(input_df)[0,1])
            decision = route_decision(prob, payload.Amount)
            served_variant = "v1_champion"

            if rollout_mode == "shadow":
                #fire-and-forget async background task for v2 shadow scoring
                asyncio.create_task(
                    self._run_shadow_v2(
                        payload.transaction_id, input_df, payload.Amount, prob, decision
                    )
                )

        #record online score drift histogram & traffic volume
        PREDICTION_SCORE_HIST.labels(model_variant=served_variant).observe(prob)
        MODEL_TRAFFIC_COUNTER.labels(
            served_variant = served_variant, rollout_mode = rollout_mode
        ).inc()

        return RiskPredictionResponse(
            transaction_id=payload.transaction_id,
            served_model_variant=served_variant,
            rollout_mode=rollout_mode,
            calibrated_risk_probability=round(prob, 6),
            risk_decision=decision,
        )
