"""Image metrics for shader evaluation."""

from __future__ import annotations

import numpy as np


def compute_variance(img: np.ndarray) -> float:
    return float(np.var(img))


def temporal_delta(img_a: np.ndarray, img_b: np.ndarray) -> float:
    return float(np.mean(np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))))


def nan_inf_detected(img: np.ndarray) -> bool:
    return bool(np.isnan(img).any() or np.isinf(img).any())
