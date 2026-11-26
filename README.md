# Real-Time Trust & Safety Risk Scoring Service

End-to-end production ML pipeline for real-time payment fraud and risk decisioning—covering imbalanced classifier training, probability calibration, slice-based fairness/error guardrails, progressive delivery (shadow + canary rollouts), and Prometheus observability.

## System Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline Training & Governance (MLflow 3)"]
        A[(284k Imbalanced<br/>Transactions)] --> B[LightGBM +<br/>Platt Calibration]
        B --> C[Slice Audit &<br/>Guardrail Checks]
        C -->|Pass <1.5% FPR| D[(MLflow Model Registry<br/>@champion / @challenger)]
    end

    subgraph Release["CI/CD Packaging"]
        D -->|Build-Time Export| E[BentoML Model Store]
        E --> F[Slim Docker Image<br/>344 MB / ~234 MiB RAM]
    end

    subgraph Online["Production Serving & Progressive Delivery"]
        Client[Payment Gateway] -->|POST /predict| G[BentoML Async Service<br/>Pydantic v2 Validation]
        GB[(GrowthBook<br/>Feature Flags)] -.->|90% Shadow / 10% Canary| G
        G -->|Live Decision| Client
        G -.->|Async Shadow Pass| H[v2 Challenger Scorer]
        H -.->|Disagreement Logs| I[(Structured JSON Logs)]
        G & H -->|Scraped via Sidecar| J[Prometheus /metrics<br/>Latency & Score Drift]
    end