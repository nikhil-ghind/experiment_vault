from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    metric_threshold: float = 0.10
    metric_name: str = "val_loss"
    lower_is_better: bool = True
    drift_psi_threshold: float = 0.2
    min_samples_since_last: int = 1000
    always_retrain_after_days: int = 30


class RetrainingTrigger:
    def __init__(self, config: TriggerConfig, retrain_fn: Callable):
        self.cfg = config
        self.retrain_fn = retrain_fn
        self._last_metric: Optional[float] = None
        self._samples_seen: int = 0

    def record_metric(self, value: float):
        self._last_metric = value

    def record_samples(self, n: int):
        self._samples_seen += n

    def should_retrain(self, current_metric: float = None,
                        drift_results: dict = None,
                        days_since_last: int = 0) -> tuple[bool, str]:
        if days_since_last >= self.cfg.always_retrain_after_days:
            return True, f"scheduled: {days_since_last} days since last training"

        if self._samples_seen >= self.cfg.min_samples_since_last:
            return True, f"new data: {self._samples_seen} new samples"

        if drift_results:
            drifted = [k for k, v in drift_results.items() if v.get("drifted")]
            if drifted:
                return True, f"drift detected: {', '.join(drifted)}"

        if current_metric is not None and self._last_metric is not None:
            if self.cfg.lower_is_better:
                degraded = current_metric > self._last_metric * (1 + self.cfg.metric_threshold)
            else:
                degraded = current_metric < self._last_metric * (1 - self.cfg.metric_threshold)
            if degraded:
                return True, (f"metric degradation: {self.cfg.metric_name} "
                              f"{self._last_metric:.4f} → {current_metric:.4f}")

        return False, "no trigger condition met"

    def maybe_retrain(self, **kwargs) -> bool:
        should, reason = self.should_retrain(**kwargs)
        if should:
            logger.info(f"Triggering retraining: {reason}")
            self.retrain_fn()
            self._samples_seen = 0
            return True
        logger.debug(f"No retraining: {reason}")
        return False
