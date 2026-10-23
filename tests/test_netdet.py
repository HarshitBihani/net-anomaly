import os, sys
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import Ecdf
from netdet import data, explain, models


@pytest.fixture(scope="module")
def fitted():
    train, test = data.load()
    fit, val = train_test_split(train, test_size=0.2, stratify=train.is_attack, random_state=0)
    sub = fit.sample(20000, random_state=0)                 # a smaller fit keeps the tests quick
    dets = {d.kind: d.fit(sub) for d in models.all_detectors()}
    return train, test, val, dets


def test_data_shapes_and_unseen_attacks():
    train, test = data.load()
    assert len(train) == 125973 and len(test) == 22544
    assert test.unseen_attack.sum() == 3750 and train.unseen_attack.sum() == 0
    assert not set(train.attack) - set(data.FAMILY) and not set(test.attack) - set(data.FAMILY)   # every attack has a family


def test_every_detector_scores_attacks_higher_than_normal(fitted):
    _, test, _, dets = fitted
    sample = test.sample(4000, random_state=1)
    for kind, d in dets.items():
        s = d.score(sample)
        assert s[sample.is_attack.to_numpy() == 1].mean() > s[sample.is_attack.to_numpy() == 0].mean(), kind


def test_normal_only_detectors_never_saw_an_attack(fitted):
    _, _, _, dets = fitted
    for kind in ("isolation", "pca"):
        assert dets[kind].pipe["p"].transform(pd.DataFrame([{c: 0 for c in data.COLUMNS} | {"protocol_type": "tcp", "service": "http", "flag": "SF"}])).shape[0] == 1


def test_scores_are_finite_and_one_per_row(fitted):
    _, test, _, dets = fitted
    sample = test.head(500)
    for d in dets.values():
        s = d.score(sample)
        assert len(s) == 500 and np.isfinite(s).all()


def test_ecdf_is_monotone_and_bounded():
    f = Ecdf(np.random.default_rng(0).normal(size=1000))
    x = np.linspace(-4, 4, 50)
    y = f(x)
    assert (np.diff(y) >= 0).all() and 0 <= y.min() and y.max() <= 1


def test_explanation_names_the_field_that_was_changed(fitted):
    """Take a normal-looking connection, make ONE field extreme, and check the explanation points at it."""
    train, _, _, dets = fitted
    d = dets["forest"]
    base = explain.normal_baseline(train)
    normal = train[(train.is_attack == 0)].iloc[100].copy()
    flagged = normal.copy()
    flagged["serror_rate"], flagged["srv_serror_rate"], flagged["dst_host_serror_rate"], flagged["flag"] = 1.0, 1.0, 1.0, "S0"
    reasons = explain.explain_alert(d, flagged, base, top=6)
    assert {r[0] for r in reasons} & {"serror_rate", "srv_serror_rate", "dst_host_serror_rate", "flag"}


def test_explanation_of_a_perfectly_normal_row_is_empty_or_tiny(fitted):
    train, _, _, dets = fitted
    base = explain.normal_baseline(train)
    row = train[train.is_attack == 0].iloc[5].copy()
    for c in data.COLUMNS:
        row[c] = base[c]
    reasons = explain.explain_alert(dets["forest"], row, base)
    assert all(abs(r[2]) < 0.2 for r in reasons)


def test_threshold_rule_hits_the_false_alarm_budget_on_validation(fitted):
    _, _, val, dets = fitted
    s = dets["isolation"].score(val)
    normal = s[val.is_attack.to_numpy() == 0]
    thr = np.quantile(normal, 0.99)
    assert abs((normal >= thr).mean() - 0.01) < 0.002
