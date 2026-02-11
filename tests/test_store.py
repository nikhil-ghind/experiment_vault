import tempfile
import os
import pytest
from src.registry.experiment_store import ExperimentStore, Experiment


def make_store():
    tmp = tempfile.mktemp(suffix=".db")
    return ExperimentStore(tmp), tmp


def test_save_and_list():
    store, tmp = make_store()
    exp = Experiment(name="test_exp", model_type="lr",
                     params={"lr": 0.01}, metrics={"val_loss": 0.3},
                     artifact_path="model")
    store.save(exp)
    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["name"] == "test_exp"
    os.unlink(tmp)


def test_best():
    store, tmp = make_store()
    for loss in [0.5, 0.3, 0.4]:
        store.save(Experiment(name="e", model_type="lr",
                              params={}, metrics={"val_loss": loss},
                              artifact_path="m"))
    best = store.best(metric="val_loss", lower_is_better=True)
    import json
    assert json.loads(best["metrics"])["val_loss"] == pytest.approx(0.3)
    os.unlink(tmp)
