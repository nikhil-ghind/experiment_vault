from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional
import mlflow.pyfunc
import numpy as np

app = FastAPI(title="ExperimentVault Model Server")


class PredictRequest(BaseModel):
    data: List[List[float]]
    model_name: Optional[str] = "champion"
    model_version: Optional[str] = "Production"


class PredictResponse(BaseModel):
    predictions: List[Any]
    model_name: str
    model_version: str


_model_cache: dict = {}


def _load_model(name: str, stage: str):
    key = f"{name}:{stage}"
    if key not in _model_cache:
        uri = f"models:/{name}/{stage}"
        _model_cache[key] = mlflow.pyfunc.load_model(uri)
    return _model_cache[key]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = _load_model(req.model_name, req.model_version)
        preds = model.predict(np.array(req.data))
        return PredictResponse(
            predictions=preds.tolist() if hasattr(preds, "tolist") else list(preds),
            model_name=req.model_name,
            model_version=req.model_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.delete("/cache")
def clear_cache():
    _model_cache.clear()
    return {"cleared": True}
