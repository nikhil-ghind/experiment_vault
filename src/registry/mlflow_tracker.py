from __future__ import annotations
import functools
import mlflow
from typing import Any, Callable, Dict, Optional


def tracked(experiment_name: str, run_name: str = None):
    """Decorator: wrap a training function with MLflow run tracking."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name or fn.__name__):
                result = fn(*args, **kwargs)
                return result
        return wrapper
    return decorator


class MLflowTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self._run = None

    def start(self, run_name: str = None, tags: Dict = None):
        self._run = mlflow.start_run(run_name=run_name, tags=tags)
        return self

    def end(self):
        if self._run:
            mlflow.end_run()
            self._run = None

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model", flavor: str = "sklearn"):
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif flavor == "pyfunc":
            mlflow.pyfunc.log_model(artifact_path, python_model=model)

    def register(self, model_name: str, artifact_path: str = "model",
                 stage: str = "Staging"):
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(model_name, result.version, stage)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end()
