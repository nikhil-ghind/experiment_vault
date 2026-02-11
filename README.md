# Experiment Vault

MLflow-backed experiment registry with automated retraining triggers, PSI/KS drift detection, alert routing, and a FastAPI model-serving endpoint.

## Architecture

```mermaid
flowchart TB
    NEW["new production data<br/>reference_df vs current_df"] --> DD["detect_feature_drift<br/>PSI over 10 bins + two-sample KS per column"]
    DD --> DR["drift_results<br/>per column: psi, ks_stat, p_value, drifted"]
    DR --> AL["AlertManager.check_and_alert<br/>channels from config: log, email via SMTP"]
    DR --> TRG

    subgraph trg["RetrainingTrigger.should_retrain — first match wins"]
        TRG{"any condition?"}
        T1["days_since_last >= always_retrain_after_days"]
        T2["samples seen >= min_samples_since_last"]
        T3["any column flagged drifted"]
        T4["metric degraded past metric_threshold<br/>relative to the last recorded metric"]
    end
    TRG --> T1
    TRG --> T2
    TRG --> T3
    TRG --> T4

    TRG -->|"no condition met"| SKIP["no retraining, reason logged"]
    TRG -->|"triggered"| RUN["RetrainingPipeline._run_retrain"]

    RUN --> UF["user function registered via @pipeline.register_train<br/>returns (model, metrics, params)"]
    UF --> MT["MLflowTracker<br/>log_params, log_metrics, log_model<br/>sklearn / pytorch / pyfunc flavors"]
    MT --> MLF[("MLflow tracking server + artifacts")]
    UF --> ES["ExperimentStore.save<br/>SQLite row with params, metrics, run_id"]
    ES --> SQL[("experiments.db<br/>list() and best() queries")]
    RUN --> REC["trigger.record_metric(val_loss)<br/>becomes the next comparison baseline"]

    REG["MLflowTracker.register<br/>register_model + transition_model_version_stage<br/>available but not called by _run_retrain"] -.-> MLF

    subgraph srv["Serving — src/serving/model_server.py"]
        API["POST /predict {model_name, stage, data}"] --> CACHE["_load_model with an in-process cache<br/>keyed by (name, stage)"]
        CACHE --> LOAD["mlflow.pyfunc.load_model('models:/name/stage')"]
        LOAD --> PRED["predictions"]
        DEL["DELETE /cache clears the cache"] --> CACHE
    end
    MLF --> LOAD
```

## Overview

- **ExperimentStore** — lightweight SQLite-backed store indexed by MLflow run IDs; `best()` returns the champion run by any metric
- **MLflowTracker** — thin wrapper + `@tracked` decorator that auto-opens/closes MLflow runs, logs params/metrics, registers model versions
- **Drift detection** — PSI (Population Stability Index) + KS two-sample test per numeric feature; prediction-distribution monitoring
- **RetrainingTrigger** — fires on: metric degradation beyond threshold, data drift, N new samples accumulated, or scheduled interval
- **RetrainingPipeline** — decorator-based orchestration: register your `train()` and `eval()` functions, call `.check()` on new data
- **Model server** — FastAPI endpoint `/predict` backed by `mlflow.pyfunc.load_model` with model-version routing

## Tech Stack

Python 3.11 · MLflow · FastAPI · scikit-learn · scipy · SQLite · pandas · pytest

## Quickstart

```bash
pip install -r requirements.txt
mlflow ui &   # http://localhost:5000

# Run tests
pytest tests/

# Start model server
uvicorn src.serving.model_server:app --reload
```

## Usage Example

```python
from src.retraining.pipeline import RetrainingPipeline

pipeline = RetrainingPipeline("configs/config.yaml")

@pipeline.register_train
def train():
    from sklearn.linear_model import LogisticRegression
    # ... load data, train model ...
    return model, {"val_loss": 0.12, "val_acc": 0.92}, {"C": 1.0, "model_type": "lr"}

# Check for drift / trigger retraining
pipeline.check(reference_df=ref, current_df=cur, current_metric=0.18, days_since_last=35)
```

## Evaluation

The platform tracks model and data-health metrics rather than computing task-specific metrics directly. User-supplied `train()` and `eval()` functions return a metrics dict that is logged to MLflow and persisted in `ExperimentStore`. Champion selection uses `ExperimentStore.best(metric, mode)`.

Built-in monitoring metrics computed by `src/monitoring/drift_detector.py`:

| Metric | What it measures | How to compute |
|--------|------------------|----------------|
| PSI (Population Stability Index) | Distribution shift per numeric feature between reference and current | `detect_feature_drift(ref_df, cur_df)` returns per-column PSI; threshold from `configs/config.yaml` |
| KS two-sample statistic + p-value | Non-parametric drift per numeric feature | Same call; uses `scipy.stats.ks_2samp` |
| Prediction-distribution drift | PSI on model output distribution | Pass model predictions as a single-column frame to the same detector |
| Metric degradation | Drop in champion metric vs. baseline | `RetrainingTrigger.should_retrain(current_metric=...)` with `metric_drop_threshold` from config |

Tests under `tests/test_drift.py` exercise the PSI/KS implementation with synthetic shifted distributions; `tests/test_trigger.py` covers retraining trigger logic; `tests/test_store.py` covers champion selection.

## Project Structure

```
experiment_vault/
├── src/
│   ├── registry/     # experiment_store.py, mlflow_tracker.py
│   ├── monitoring/   # drift_detector.py, alerts.py
│   ├── retraining/   # trigger.py, pipeline.py
│   └── serving/      # model_server.py (FastAPI)
├── configs/          # config.yaml
└── tests/            # pytest suite
```
