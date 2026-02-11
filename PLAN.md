# Experiment Vault

## Project Overview
MLflow experiment tracking pipeline that logs hyperparameters, metrics, and artifacts across training runs. Implements automated retraining triggered by performance degradation detection. Transforms data science prototypes into reproducible, versioned ML pipelines.

## Tech Stack
- **Tracking:** MLflow 2.x
- **Training:** PyTorch 2.x, scikit-learn
- **Data:** Pandas, DVC (data versioning)
- **Pipeline:** MLflow Projects, Makefiles
- **Serving:** MLflow Model Serving
- **Container:** Docker

## Architecture Overview
```
┌──────────────┐    MLflow API    ┌──────────────┐
│ Training     │ ──────────────► │ MLflow Server │
│ Scripts      │   log metrics   │ (Tracking)    │
└──────────────┘                 └──────┬───────┘
                                        │
                                 ┌──────▼───────┐
                                 │ Artifact      │
                                 │ Store (S3)    │
                                 └──────────────┘
┌──────────────┐
│ Monitor      │ ─── detect degradation ──► trigger retrain
│ (scheduled)  │
└──────────────┘
```

## Phase 1: MLflow Setup & Basic Experiment Tracking
**Goal:** Set up MLflow tracking server and instrument training scripts.

### Tasks
1. Project structure:
   ```
   mlExperimentTrackingPipeline/
   ├── src/
   │   ├── train.py             # Main training script
   │   ├── evaluate.py          # Model evaluation
   │   ├── preprocess.py        # Data preprocessing
   │   ├── models/
   │   │   ├── classifier.py    # PyTorch classifier
   │   │   └── sklearn_model.py # Scikit-learn baseline
   │   ├── monitor/
   │   │   ├── drift_detector.py   # Performance degradation detection
   │   │   └── retrain_trigger.py  # Automated retraining
   │   └── utils/
   │       ├── metrics.py       # Custom metric calculations
   │       └── data_loader.py
   ├── configs/
   │   ├── experiment.yaml      # Experiment configs
   │   └── model_configs/
   │       ├── resnet.yaml
   │       └── xgboost.yaml
   ├── MLproject                # MLflow Project definition
   ├── Dockerfile
   ├── docker-compose.yml       # MLflow server + DB
   └── Makefile
   ```
2. `docker-compose.yml`:
   - `mlflow-server`: `mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...`
   - `postgres`: metadata store
   - MinIO (S3-compatible): artifact store
3. `src/train.py`:
   ```python
   with mlflow.start_run(run_name=config.run_name):
       mlflow.log_params({"lr": lr, "batch_size": bs, "epochs": epochs, "model": model_name, "optimizer": opt_name})
       for epoch in range(epochs):
           train_loss = train_one_epoch(model, train_loader, optimizer)
           val_loss, val_acc = evaluate(model, val_loader)
           mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)
       mlflow.pytorch.log_model(model, "model")
       mlflow.log_artifact("configs/experiment.yaml")
       mlflow.set_tag("model_type", model_name)
   ```
4. `src/evaluate.py`:
   - Load model from MLflow run
   - Compute: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
   - Log all metrics + confusion matrix plot as artifact

## Phase 2: Experiment Comparison & Model Registry
**Goal:** Build tooling for comparing runs and promoting models.

### Tasks
1. `src/utils/metrics.py`:
   - `compare_runs(run_ids: list[str]) -> pd.DataFrame`:
     - Fetch params + metrics for each run from MLflow API
     - Return comparison table sorted by target metric
   - `plot_metric_history(run_id, metric_name) -> Figure`
   - `plot_parallel_coordinates(experiment_id) -> Figure` — hyperparameter search visualization
2. MLflow Model Registry workflow:
   - After training: `mlflow.register_model(model_uri, "production_classifier")`
   - Stages: `None` → `Staging` → `Production` → `Archived`
   - `promote_model(model_name, version, stage)` — transition with validation
3. `configs/experiment.yaml`:
   ```yaml
   experiment_name: image_classifier
   model: resnet18
   hyperparameters:
     lr: [0.001, 0.003, 0.01]
     batch_size: [32, 64]
     optimizer: [adam, sgd]
     weight_decay: [0.0, 0.0001]
   data:
     train_split: 0.8
     val_split: 0.1
     test_split: 0.1
   ```
4. Add hyperparameter sweep script: grid search or Optuna integration logging each trial as MLflow run.

## Phase 3: Reproducible ML Pipelines
**Goal:** Package training as MLflow Projects with dependency management.

### Tasks
1. `MLproject`:
   ```yaml
   name: experiment_pipeline
   docker_env:
     image: ml-pipeline:latest
   entry_points:
     preprocess:
       command: "python src/preprocess.py --config {config}"
     train:
       parameters:
         config: {type: str, default: "configs/experiment.yaml"}
         lr: {type: float, default: 0.001}
         epochs: {type: int, default: 50}
       command: "python src/train.py --config {config} --lr {lr} --epochs {epochs}"
     evaluate:
       parameters:
         run_id: {type: str}
       command: "python src/evaluate.py --run-id {run_id}"
     full_pipeline:
       command: "python src/pipeline.py --config {config}"
   ```
2. `src/pipeline.py`:
   - Orchestrates: preprocess → train → evaluate → register model (if better than current production)
   - `mlflow.run(".", entry_point="preprocess")` → `mlflow.run(".", entry_point="train")` → etc.
3. `Makefile`:
   - `train`: `mlflow run . -e train --experiment-name $(EXP_NAME) -P lr=$(LR)`
   - `sweep`: runs grid of hyperparameters
   - `evaluate`: `mlflow run . -e evaluate -P run_id=$(RUN_ID)`
   - `serve`: `mlflow models serve -m models:/production_classifier/Production -p 5001`
4. `Dockerfile`: Python 3.11, PyTorch, scikit-learn, MLflow, all deps pinned.

## Phase 4: Automated Retraining Pipeline
**Goal:** Detect model performance degradation and trigger automated retraining.

### Tasks
1. `src/monitor/drift_detector.py`:
   - `class PerformanceDriftDetector`:
     - `__init__(model_name, metric_name, threshold, window_size)`
     - `check_degradation() -> bool`:
       - Query MLflow for production model's logged inference metrics (last N days)
       - Compute moving average of target metric
       - If moving avg drops below threshold → return True
     - `get_baseline_metrics() -> dict` — fetch metrics from production model's training run
2. `src/monitor/retrain_trigger.py`:
   - `class RetrainTrigger`:
     - `evaluate_and_retrain()`:
       1. Run `PerformanceDriftDetector.check_degradation()`
       2. If degraded: launch `mlflow run . -e full_pipeline` with latest data
       3. Compare new model metrics vs production
       4. If new model better: promote to Staging, notify for review
       5. Log retraining event with reason
     - `schedule(interval_hours)` — run on cron schedule
3. `scripts/monitor_cron.py`:
   - Runs hourly: checks drift, triggers retrain if needed
   - Sends Slack/email notification on retraining
4. Tests:
   - Mock degraded metrics → verify retrain triggered
   - Mock improved new model → verify promotion to Staging
   - Mock no degradation → verify no action taken
