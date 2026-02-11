from __future__ import annotations
import yaml
import mlflow
import logging
from typing import Callable, Dict, Any, Optional

from src.registry.experiment_store import ExperimentStore, Experiment
from src.registry.mlflow_tracker import MLflowTracker
from src.monitoring.drift_detector import detect_feature_drift, detect_prediction_drift
from src.monitoring.alerts import AlertManager
from src.retraining.trigger import RetrainingTrigger, TriggerConfig

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Orchestrates: drift detection → trigger evaluation → retrain → register.
    """

    def __init__(self, cfg_path: str = "configs/config.yaml"):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.store   = ExperimentStore(self.cfg.get("db_path", "experiments.db"))
        self.tracker = MLflowTracker(
            self.cfg.get("experiment_name", "experiment_vault"),
            self.cfg.get("mlflow_tracking_uri"),
        )
        self.alerts  = AlertManager(self.cfg.get("alert_channels", ["log"]))
        self.trigger = RetrainingTrigger(
            TriggerConfig(**self.cfg.get("trigger", {})),
            retrain_fn=self._run_retrain,
        )
        self._train_fn: Optional[Callable] = None
        self._eval_fn:  Optional[Callable] = None

    def register_train(self, fn: Callable) -> Callable:
        self._train_fn = fn
        return fn

    def register_eval(self, fn: Callable) -> Callable:
        self._eval_fn = fn
        return fn

    def _run_retrain(self):
        if self._train_fn is None:
            raise RuntimeError("No training function registered. Use @pipeline.register_train.")
        logger.info("Starting retraining run ...")
        with self.tracker as t:
            t.start(run_name="auto_retrain")
            model, metrics, params = self._train_fn()
            t.log_params(params)
            t.log_metrics(metrics)
            t.log_model(model, flavor=self.cfg.get("model_flavor", "sklearn"))
            exp = Experiment(
                name=self.cfg.get("experiment_name", "experiment_vault"),
                model_type=params.get("model_type", "unknown"),
                params=params,
                metrics=metrics,
                artifact_path=self.cfg.get("artifact_path", "model"),
                run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
            )
            self.store.save(exp)
            self.trigger.record_metric(metrics.get("val_loss", 0.0))
        logger.info(f"Retraining complete. Metrics: {metrics}")

    def check(self, reference_df=None, current_df=None,
               current_metric: float = None, days_since_last: int = 0):
        drift_results = {}
        if reference_df is not None and current_df is not None:
            drift_results = detect_feature_drift(reference_df, current_df)
            self.alerts.check_and_alert(drift_results, self.cfg.get("experiment_name", ""))

        self.trigger.maybe_retrain(
            current_metric=current_metric,
            drift_results=drift_results,
            days_since_last=days_since_last,
        )
