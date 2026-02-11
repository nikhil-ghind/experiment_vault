import pytest
from src.retraining.trigger import RetrainingTrigger, TriggerConfig


def test_no_trigger():
    fired = []
    t = RetrainingTrigger(TriggerConfig(metric_threshold=0.10, min_samples_since_last=9999),
                          retrain_fn=lambda: fired.append(1))
    t.record_metric(0.5)
    t.maybe_retrain(current_metric=0.52, days_since_last=5)
    assert len(fired) == 0


def test_metric_degradation_triggers():
    fired = []
    t = RetrainingTrigger(TriggerConfig(metric_threshold=0.05, min_samples_since_last=9999),
                          retrain_fn=lambda: fired.append(1))
    t.record_metric(0.5)
    t.maybe_retrain(current_metric=0.60, days_since_last=5)
    assert len(fired) == 1


def test_sample_count_triggers():
    fired = []
    t = RetrainingTrigger(TriggerConfig(min_samples_since_last=100),
                          retrain_fn=lambda: fired.append(1))
    t.record_samples(150)
    t.maybe_retrain(days_since_last=0)
    assert len(fired) == 1


def test_scheduled_retrain():
    fired = []
    t = RetrainingTrigger(TriggerConfig(always_retrain_after_days=30, min_samples_since_last=9999),
                          retrain_fn=lambda: fired.append(1))
    t.maybe_retrain(days_since_last=31)
    assert len(fired) == 1
