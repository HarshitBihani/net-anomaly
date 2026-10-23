"""Train every detector, tune each alarm threshold on held-out normal traffic, and score on the official test set.

    python evaluate.py        (about 3 minutes)
Alarm threshold rule (same for every model, so the comparison is fair): choose the score where only 1% of
held-out NORMAL connections would raise an alarm. An analyst team can only handle so many false alarms, so this is the budget.
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from netdet import data, explain, models

SEED = 0
FPR_BUDGET = 0.01
SEVERITY = {"U2R": 4.0, "R2L": 3.0, "Probe": 2.0, "DoS": 1.0}      # an ASSUMPTION: gaining root access matters more than a flood
SHIFT_SIZE = 1000                                                   # connections per simulated analyst shift
ALERTS_PER_SHIFT = 40                                               # what the team can read in a shift


class Ecdf:
    """Turns a raw score into 'what fraction of normal traffic scores lower than this', so different models can be averaged."""
    def __init__(self, normal_scores):
        self.s = np.sort(normal_scores)

    def __call__(self, x):
        return np.searchsorted(self.s, x, side="right") / len(self.s)


def summarise(name, score, thr, test):
    y, alarm = test.is_attack.to_numpy(), score >= thr
    out = dict(name=name, roc_auc=float(roc_auc_score(y, score)), pr_auc=float(average_precision_score(y, score)),
               recall=float(alarm[y == 1].mean()), false_alarm_rate=float(alarm[y == 0].mean()),
               precision=float(y[alarm].mean()) if alarm.any() else 0.0)
    out["f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-9)
    seen, unseen = (test.is_attack == 1) & (test.unseen_attack == 0), test.unseen_attack == 1
    out["recall_seen_attacks"] = float(alarm[seen].mean())
    out["recall_unseen_attacks"] = float(alarm[unseen].mean())
    normal = test.is_attack == 0
    out["roc_auc_unseen_only"] = float(roc_auc_score(np.r_[np.zeros(normal.sum()), np.ones(unseen.sum())],
                                                     np.r_[score[normal], score[unseen]]))
    out["recall_by_family"] = {f: float(alarm[test.family == f].mean()) for f in ["DoS", "Probe", "R2L", "U2R"]}
    # ranking quality with the threshold set on the TEST normal traffic itself (not something you could do live):
    # this removes the effect of mis-set thresholds and compares how well each model orders the connections
    for q, tag in ((0.99, "1pct"), (0.95, "5pct")):
        t = np.quantile(score[normal], q)
        out[f"recall_at_{tag}_false_alarms"] = float((score[y == 1] > t).mean())
        out[f"unseen_recall_at_{tag}_false_alarms"] = float((score[unseen] > t).mean())
    return out


def main():
    t0 = time.time()
    os.makedirs("outputs", exist_ok=True)
    train, test = data.load()
    fit, val = train_test_split(train, test_size=0.2, stratify=train.is_attack, random_state=SEED)
    print(f"train {len(fit):,} | validation {len(val):,} | test {len(test):,}  ({int(test.unseen_attack.sum()):,} test attacks are of types never seen in training)")

    dets = models.all_detectors()
    scores_val, scores_test, thr = {}, {}, {}
    for d in dets:
        d.fit(fit)
        scores_val[d.name], scores_test[d.name] = d.score(val), d.score(test)
        thr[d.name] = float(np.quantile(scores_val[d.name][val.is_attack.to_numpy() == 0], 1 - FPR_BUDGET))
        print(f"  trained {d.name}  ({time.time() - t0:.0f}s)")

    # ensemble: average of the random forest and the isolation forest, each put on the same 0-1 scale
    f_name, i_name = "Random forest (supervised)", "Isolation Forest (normal traffic only)"
    vn = val.is_attack.to_numpy() == 0
    ef, ei = Ecdf(scores_val[f_name][vn]), Ecdf(scores_val[i_name][vn])
    ens = "Ensemble (forest + isolation forest)"
    scores_val[ens] = (ef(scores_val[f_name]) + ei(scores_val[i_name])) / 2
    scores_test[ens] = (ef(scores_test[f_name]) + ei(scores_test[i_name])) / 2
    thr[ens] = float(np.quantile(scores_val[ens][vn], 1 - FPR_BUDGET))

    res = {name: summarise(name, scores_test[name], thr[name], test) for name in scores_test}

    # ---------- alert prioritisation with the ensemble ----------
    fam = models.family_classifier(fit)
    classes = list(fam.named_steps["m"].classes_)
    fam_pred = np.array(classes)[fam.predict_proba(test[data.COLUMNS]).argmax(axis=1)]
    sev = np.array([SEVERITY.get(f, 1.0) for f in fam_pred])
    base_score = scores_test[ens]
    risk = base_score * sev
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(test))                              # simulated arrival order (the data has no timestamps)
    prio = {"by suspicion score only": [], "by suspicion x severity": []}
    shifts = []
    for s in range(0, len(order) - SHIFT_SIZE + 1, SHIFT_SIZE):
        idx = order[s:s + SHIFT_SIZE]
        flagged = idx[base_score[idx] >= thr[ens]]
        row = dict(shift=s // SHIFT_SIZE + 1, connections=SHIFT_SIZE, true_attacks=int(test.is_attack.to_numpy()[idx].sum()),
                   alarms=len(flagged))
        for label, key in [("by suspicion score only", base_score), ("by suspicion x severity", risk)]:
            top = flagged[np.argsort(-key[flagged])][:ALERTS_PER_SHIFT]
            att = test.is_attack.to_numpy()[top]
            hi = test.family.iloc[top].isin(["R2L", "U2R"]).to_numpy()
            prio[label].append(dict(read=len(top), real=int(att.sum()), high_severity_found=int(hi.sum())))
            row["real_in_top_" + ("score" if "only" in label else "risk")] = int(att.sum())
            row["rarest_in_top_" + ("score" if "only" in label else "risk")] = int(hi.sum())
        shifts.append(row)
    prioritisation = {k: dict(alerts_read=sum(x["read"] for x in v), real_attacks=sum(x["real"] for x in v),
                              high_severity_found=sum(x["high_severity_found"] for x in v)) for k, v in prio.items()}
    total_hi = int(sum(test.family.iloc[order[:len(shifts) * SHIFT_SIZE]].isin(["R2L", "U2R"])))
    prioritisation["_total_R2L_U2R_in_replayed_data"] = total_hi

    # ---------- explanations ----------
    forest = next(d for d in dets if d.kind == "forest")
    baseline = explain.normal_baseline(fit)
    flagged_unseen = test[(test.unseen_attack == 1) & (scores_test[ens] >= thr[ens])].sample(3, random_state=1)
    false_pos = test[(test.is_attack == 0) & (scores_test[ens] >= thr[ens])].sample(2, random_state=1)
    explanations = []
    for kind, frame in [("unseen attack, caught", flagged_unseen), ("false alarm", false_pos)]:
        for _, r in frame.iterrows():
            explanations.append(dict(kind=kind, attack=r["attack"], service=r["service"],
                                     reasons=[(f, str(v), round(dr, 3)) for f, v, dr in explain.explain_alert(forest, r, baseline)]))

    out = dict(data=dict(train=len(fit), validation=len(val), test=len(test), test_attacks=int(test.is_attack.sum()),
                         test_unseen_attacks=int(test.unseen_attack.sum()),
                         unseen_types=sorted(set(test[test.unseen_attack == 1].attack)), seen_types=int(train[train.is_attack == 1].attack.nunique())),
               fpr_budget=FPR_BUDGET, results=res, thresholds=thr, prioritisation=prioritisation, shifts=shifts,
               alerts_per_shift=ALERTS_PER_SHIFT, shift_size=SHIFT_SIZE, severity_assumption=SEVERITY, explanations=explanations,
               seconds=round(time.time() - t0))
    json.dump(out, open("outputs/results.json", "w"), indent=1)
    pickle.dump(dict(test_score=scores_test, thr=thr, y=test.is_attack.to_numpy(), unseen=test.unseen_attack.to_numpy(),
                     family=test.family.to_numpy()), open("outputs/scores.pkl", "wb"))
    for n, r in res.items():
        print(f"{n:<44} AUC {r['roc_auc']:.3f}  recall {r['recall']:.2f}  FAR {r['false_alarm_rate']:.3f}  precision {r['precision']:.2f}  "
              f"seen {r['recall_seen_attacks']:.2f}  unseen {r['recall_unseen_attacks']:.2f}")
    print(prioritisation)


if __name__ == "__main__":
    main()
