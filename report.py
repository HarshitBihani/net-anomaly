"""Build outputs/report.html from outputs/results.json and outputs/scores.pkl."""
import json
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from sklearn.metrics import roc_curve

r = json.load(open("outputs/results.json"))
s = pickle.load(open("outputs/scores.pkl", "rb"))
R = r["results"]
names = list(R)
pal = ["#9ca3af", "#2563eb", "#d97706", "#7c3aed", "#0d9488"]
pct = lambda v: f"{v * 100:.1f}%"

roc = go.Figure()
for n, c in zip(names, pal):
    fpr, tpr, _ = roc_curve(s["y"], s["test_score"][n])
    roc.add_scatter(x=fpr, y=tpr, name=f"{n} (AUC {R[n]['roc_auc']:.3f})", line=dict(color=c))
roc.add_vline(x=0.01, line_dash="dot", line_color="#dc2626", annotation_text="1% false alarms")
roc.update_layout(template="plotly_white", height=420, xaxis=dict(range=[0, 0.2], title="false alarm rate (normal traffic flagged)"),
                  yaxis_title="attacks caught", title="Attacks caught vs false alarms (zoomed to the first 20% false alarms)",
                  legend=dict(orientation="h", y=-0.3))

bar = go.Figure()
for tag, label, col in [("unseen_recall_at_1pct_false_alarms", "1% false alarms", "#2563eb"), ("unseen_recall_at_5pct_false_alarms", "5% false alarms", "#93c5fd")]:
    bar.add_bar(x=[n.split(" (")[0] for n in names], y=[R[n][tag] for n in names], name=label, marker_color=col,
                text=[pct(R[n][tag]) for n in names], textposition="outside")
bar.update_layout(template="plotly_white", barmode="group", height=360, yaxis=dict(range=[0, 1], title="share caught"),
                  title="Attacks the model has NEVER seen in training: how many are caught?", legend=dict(orientation="h", y=-0.25))

rows = "".join(
    f"<tr><td>{n}</td><td>{R[n]['roc_auc']:.3f}</td><td>{pct(R[n]['recall'])}</td><td>{pct(R[n]['false_alarm_rate'])}</td><td>{pct(R[n]['precision'])}</td>"
    f"<td>{pct(R[n]['recall_seen_attacks'])}</td><td>{pct(R[n]['recall_unseen_attacks'])}</td></tr>" for n in names)
fam_rows = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{pct(R[n]['recall_by_family'][f])}</td>" for f in ["DoS", "Probe", "R2L", "U2R"]) + "</tr>" for n in names)
pr = r["prioritisation"]
a, b = pr["by suspicion score only"], pr["by suspicion x severity"]
shifts = r["shifts"]
sh = go.Figure()
sh.add_bar(x=[x["shift"] for x in shifts], y=[x["alarms"] for x in shifts], name="alarms raised", marker_color="#e5e7eb")
sh.add_bar(x=[x["shift"] for x in shifts], y=[r["alerts_per_shift"]] * len(shifts), name="alerts the team can read", marker_color="#2563eb")
sh.add_scatter(x=[x["shift"] for x in shifts], y=[x["true_attacks"] for x in shifts], name="real attacks in the shift", mode="lines+markers", line=dict(color="#dc2626"))
sh.update_layout(template="plotly_white", barmode="overlay", height=330, xaxis_title="simulated shift (1,000 connections each)",
                 title="Replay: far more alarms than a team can read, so ranking matters", legend=dict(orientation="h", y=-0.3))
expl = "".join(
    f"<tr><td>{e['kind']}</td><td>{e['attack']} / {e['service']}</td><td>" + "; ".join(f"{f}={v}" for f, v, _ in e["reasons"][:4]) + "</td></tr>" for e in r["explanations"])
d = r["data"]
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Network anomaly detection</title><script>{get_plotlyjs()}</script>
<style>body{{font:15px/1.55 Segoe UI,Arial,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;color:#1f2937}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:6px 8px;border-bottom:1px solid #e5e7eb;text-align:left}}h2{{margin-top:30px}}
.k{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px 16px}}</style></head><body>
<h1>Network intrusion detection: known attacks vs attacks nobody has seen</h1>
<p>NSL-KDD: {d['train']:,} training connections, {d['test']:,} official test connections. The test set contains {d['test_unseen_attacks']:,} attacks of
{len(d['unseen_types'])} types that never appear in training. (The traffic is from the late 1990s: a standard benchmark, not a picture of today's networks.)</p>
<div class="k"><b>Main finding.</b> At the same false-alarm rate, models trained only on normal traffic caught far more brand-new attacks than models trained on labelled attacks.
At 1% false alarms: Isolation Forest caught {pct(R[names[2]]['unseen_recall_at_1pct_false_alarms'])} of unseen attacks, random forest {pct(R[names[1]]['unseen_recall_at_1pct_false_alarms'])}.</div>
<h2>Attacks caught vs false alarms</h2>{roc.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False))}
{bar.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False))}
<p>Both bars use the same rule for every model: set the alarm level so that 1% (or 5%) of normal test traffic would trigger it. This compares how well each model orders connections, independent of threshold setting.</p>
<h2>What happens with a threshold set in advance</h2>
<p>In practice you must pick the alarm level before you see tomorrow's traffic. Each threshold below was set so that 1% of held-out normal training traffic raises an alarm.</p>
<table><tr><th>Model</th><th>AUC</th><th>Attacks caught</th><th>Actual false alarm rate on test</th><th>Alarm precision</th><th>Known attacks caught</th><th>Unseen attacks caught</th></tr>{rows}</table>
<p>The supervised models drifted the most: tuned for 1%, the random forest raised false alarms on {pct(R[names[1]]['false_alarm_rate'])} of normal test traffic, because the test traffic differs from training.
The normal-only models stayed close to the budget.</p>
<h2>By attack family (threshold set in advance)</h2>
<table><tr><th>Model</th><th>DoS (floods)</th><th>Probe (scans)</th><th>R2L (remote break-ins)</th><th>U2R (privilege escalation)</th></tr>{fam_rows}</table>
<p>Remote break-ins (R2L) are the hardest for every model: they look much like normal logins.</p>
<h2>Alert prioritisation</h2>{sh.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False))}
<p>The ensemble raises about 300+ alarms per 1,000 connections, but a team can read {r['alerts_per_shift']}. Reading the top {r['alerts_per_shift']} per shift across {len(shifts)} simulated shifts ({a['alerts_read']} alerts):</p>
<table><tr><th>Ranking</th><th>Alerts that were real attacks</th><th>R2L and U2R attacks found</th></tr>
<tr><td>by suspicion score only</td><td>{a['real_attacks']} of {a['alerts_read']}</td><td>{a['high_severity_found']}</td></tr>
<tr><td>by suspicion x severity</td><td>{b['real_attacks']} of {b['alerts_read']}</td><td>{b['high_severity_found']}</td></tr></table>
<p>Ranking by score alone fills the list with easy floods and never reaches the stealthy attacks. Weighting by the predicted attack family (U2R 4, R2L 3, Probe 2, DoS 1, an assumption of mine) trades
{a['real_attacks'] - b['real_attacks']} real alerts for {b['high_severity_found']} high-severity ones. Even so, {b['high_severity_found']} of the {pr['_total_R2L_U2R_in_replayed_data']:,} R2L and U2R attacks in the replayed data were surfaced: the budget is the limit.
The data has no timestamps, so the arrival order is a random shuffle.</p>
<h2>Why was this alert raised?</h2>
<table><tr><th>Case</th><th>Type</th><th>Fields that drove the alert (value)</th></tr>{expl}</table>
<p>Method: replace one field at a time with its normal value and see how much the score falls. The fields with the biggest fall are the reasons.</p>
</body></html>"""
open("outputs/report.html", "w", encoding="utf-8").write(html)
print("wrote outputs/report.html")
