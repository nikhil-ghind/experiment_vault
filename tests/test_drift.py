import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift_detector import psi, ks_test, detect_feature_drift


def test_psi_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    assert psi(x, x) < 0.05


def test_psi_different_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(5, 1, 1000)
    assert psi(ref, cur) > 0.2


def test_ks_same():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 500)
    _, p = ks_test(x, x)
    assert p > 0.05


def test_feature_drift_detection():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.uniform(0, 1, 500)})
    cur = pd.DataFrame({"a": rng.normal(10, 1, 500), "b": rng.uniform(0, 1, 500)})
    results = detect_feature_drift(ref, cur)
    assert results["a"]["drifted"] is True
    assert results["b"]["drifted"] is False
