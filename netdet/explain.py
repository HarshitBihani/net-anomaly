"""Explain an alert: which fields of this connection made it look suspicious?

Method (works for any model, and is easy to say out loud): for each field, swap this connection's value for what is typical in
NORMAL traffic (the median, or the most common category) and see how much the suspicion score drops. The fields that drop it
the most are the reasons. It is an occlusion explanation: if removing a field's evidence calms the alert, that field was driving it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data import CATEGORICAL, COLUMNS


def normal_baseline(train: pd.DataFrame) -> dict:
    normal = train[train.is_attack == 0]
    return {c: (normal[c].mode().iloc[0] if c in CATEGORICAL else normal[c].median()) for c in COLUMNS}


def explain_alert(detector, row: pd.Series, baseline: dict, top: int = 5) -> list[tuple[str, object, float]]:
    """Returns the `top` fields as (field, this connection's value, score drop when replaced by the normal value)."""
    base = pd.DataFrame([row[COLUMNS].to_dict()]).infer_objects()
    s0 = float(detector.score(base)[0])
    rows = []
    for c in COLUMNS:
        alt = base.copy()
        alt[c] = baseline[c]
        rows.append(alt)
    scores = detector.score(pd.concat(rows, ignore_index=True))
    drops = s0 - scores
    order = np.argsort(-drops)[:top]
    return [(COLUMNS[i], row[COLUMNS[i]], float(drops[i])) for i in order if drops[i] > 0]
