from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — detects feature distribution shift."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float) + 1e-6
    act_counts = np.histogram(actual,   bins=breakpoints)[0].astype(float) + 1e-6
    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    stat, pvalue = stats.ks_2samp(reference, current)
    return float(stat), float(pvalue)


def detect_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                          psi_threshold: float = 0.2,
                          ks_pvalue_threshold: float = 0.05) -> Dict[str, Dict]:
    results = {}
    for col in reference_df.select_dtypes(include=np.number).columns:
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        psi_val = psi(ref, cur)
        ks_stat, ks_p = ks_test(ref, cur)
        drifted = psi_val > psi_threshold or ks_p < ks_pvalue_threshold
        results[col] = {
            "psi": psi_val,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "drifted": drifted,
        }
    return results


def detect_prediction_drift(reference_preds: np.ndarray,
                              current_preds: np.ndarray,
                              threshold: float = 0.1) -> Dict:
    ref_mean, cur_mean = reference_preds.mean(), current_preds.mean()
    shift = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9)
    ks_stat, ks_p = ks_test(reference_preds, current_preds)
    return {
        "mean_shift_pct": float(shift * 100),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "drifted": shift > threshold or ks_p < 0.05,
    }
