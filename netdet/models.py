"""Detectors. All of them return a score where HIGHER means MORE SUSPICIOUS, so they can be compared and combined.

Two families:
  supervised   learns from examples of attacks AND normal traffic (logistic regression, random forest)
  unsupervised learns only what NORMAL traffic looks like, and flags what does not fit (Isolation Forest, PCA reconstruction error)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL, COLUMNS, NUMERIC

HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]


def preprocessor() -> ColumnTransformer:
    """One-hot the three text columns, log-squash the heavy-tailed counts, standardise everything numeric."""
    log = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("log", log, HEAVY_TAILED),
        ("num", StandardScaler(), [c for c in NUMERIC if c not in HEAVY_TAILED]),
    ])


class Detector:
    """Wraps a model so every detector exposes fit(df) and score(df)."""
    def __init__(self, name: str, kind: str):
        self.name, self.kind = name, kind
        self.pipe: Pipeline | None = None

    def fit(self, df: pd.DataFrame):
        X, y = df[COLUMNS], df["is_attack"]
        if self.kind == "logistic":
            self.pipe = Pipeline([("p", preprocessor()), ("m", LogisticRegression(max_iter=1000, C=1.0))]).fit(X, y)
        elif self.kind == "forest":
            self.pipe = Pipeline([("p", preprocessor()), ("m", RandomForestClassifier(
                n_estimators=150, min_samples_leaf=2, n_jobs=2, random_state=0))]).fit(X, y)
        elif self.kind == "isolation":
            normal = df[df.is_attack == 0]
            self.pipe = Pipeline([("p", preprocessor()), ("m", IsolationForest(
                n_estimators=200, max_samples=512, random_state=0, n_jobs=2))]).fit(normal[COLUMNS])
        elif self.kind == "pca":
            normal = df[df.is_attack == 0]
            self.pipe = Pipeline([("p", preprocessor()), ("m", PCA(n_components=15, random_state=0))]).fit(normal[COLUMNS])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[COLUMNS]
        if self.kind in ("logistic", "forest"):
            return self.pipe.predict_proba(X)[:, 1]
        if self.kind == "isolation":
            return -self.pipe.score_samples(X)                  # sklearn: lower score = more abnormal, so flip it
        Z = self.pipe["p"].transform(X)                         # pca: how badly can normal-traffic patterns rebuild this row?
        rebuilt = self.pipe["m"].inverse_transform(self.pipe["m"].transform(Z))
        return ((Z - rebuilt) ** 2).sum(axis=1)


def all_detectors() -> list[Detector]:
    return [Detector("Logistic regression (supervised)", "logistic"), Detector("Random forest (supervised)", "forest"),
            Detector("Isolation Forest (normal traffic only)", "isolation"), Detector("PCA reconstruction (normal traffic only)", "pca")]


def family_classifier(train: pd.DataFrame) -> Pipeline:
    """Says WHICH kind of attack a connection looks like (DoS, Probe, R2L, U2R). Used to rank alerts by severity."""
    att = train[train.is_attack == 1]
    return Pipeline([("p", preprocessor()), ("m", RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, n_jobs=2, random_state=0, class_weight="balanced"))]).fit(att[COLUMNS], att["family"])
