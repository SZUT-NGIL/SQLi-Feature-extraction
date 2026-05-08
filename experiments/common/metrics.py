from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def to_probabilities(scores: Any) -> np.ndarray:
    """将模型输出统一成 [0, 1] 概率。"""
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if np.all((arr >= 0.0) & (arr <= 1.0)):
        return arr
    return _sigmoid(arr)


def scores_from_model(model: Any, features: Any) -> np.ndarray:
    """兼容 predict_proba / decision_function / predict 三类接口。"""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        scores = model.predict(features)
    return to_probabilities(scores)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """计算论文常用的二分类指标。"""
    probs = to_probabilities(scores)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    auc = math.nan
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, probs))

    return {
        "acc": float(accuracy_score(y_true, preds)),
        "prec": float(precision_score(y_true, preds, zero_division=0)),
        "rec": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def summarize_latency_ms(values_ms: list[float]) -> dict[str, float]:
    """汇总均值/中位数/P95，便于直接写表。"""
    arr = np.asarray(values_ms, dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
