from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def official_metrics(y_true, probability):
    """Return Zindi F1@0.5, ROC-AUC, and weighted competition score."""
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    f1 = f1_score(y_true, prediction)
    auc = roc_auc_score(y_true, probability)
    score = 0.60 * f1 + 0.40 * auc
    return {"f1": float(f1), "auc": float(auc), "score": float(score)}
