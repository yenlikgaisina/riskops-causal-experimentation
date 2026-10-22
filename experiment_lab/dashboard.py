"""Create a dependency-free, accessible HTML portfolio demo report."""
from __future__ import annotations

from html import escape
from pathlib import Path


def render_dashboard(report: dict, output: Path) -> None:
    methods = report["estimates"]
    e = methods["ipw_cuped"]
    decision = report["simulation_decision"]
    pp = lambda n: f"{100 * n:+.2f} pp"  # noqa: E731
    def bar(name: str, value: float) -> str:
        percent = max(0.0, min(100.0, float(value) * 100))
        return (f'<div class="group"><div class="group-top"><span>{escape(name)}</span><b>{percent:.2f}%</b></div>'
                f'<div class="track"><div class="fill" style="width:{percent:.3f}%"></div></div></div>')
    rows = "\n".join(
        f'<tr><td>{escape(name.replace("_", " ").upper())}</td><td>{pp(item["absolute_uplift"])}</td>'
        f'<td>{pp(item["ci_lower"])} to {pp(item["ci_upper"])}</td>'
        f'<td>{item["standard_error"] * 100:.2f} pp</td></tr>'
        for name, item in methods.items()
    )
    issues = "".join(f"<li>{escape(reason)}</li>" for reason in decision["reasons"])
    limitations = "".join(f"<li>{escape(l)}</li>" for l in report["limitations"])
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RiskOps — Experimentation Intelligence</title><style>
:root{{--bg:#0b1220;--card:#172136;--line:#2c3b55;--text:#eef5ff;--muted:#a7b9cf;--teal:#50e0c0;--yellow:#ffd18b}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif;background:var(--bg);color:var(--text)}}
main{{max-width:1100px;padding:36px 20px 72px;margin:auto}}.eyebrow{{color:var(--teal);text-transform:uppercase;font-size:12px;letter-spacing:.16em;font-weight:800}}
h1{{font-size:clamp(2rem,5vw,3.9rem);line-height:1.1;margin:12px 0}}h2{{font-size:1.12rem;margin:0 0 14px}}
p,li{{color:var(--muted)}}.lead{{max-width:750px;font-size:1.12rem}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:30px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px}}.value{{font-size:2.1rem;font-weight:750;margin:8px 0 0;font-variant-numeric:tabular-nums}}
.small{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.group{{margin:20px 0}}
.group-top{{display:flex;justify-content:space-between;margin-bottom:8px}}.track{{height:12px;background:#2c3b55;border-radius:20px;overflow:hidden}}.fill{{height:100%;background:var(--teal);border-radius:20px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:12px 8px}}td:first-child{{font-weight:700;color:white}}
.status{{color:var(--yellow);font-weight:700}}.caution{{border-left:3px solid var(--yellow);padding:6px 0 6px 15px}}
footer{{margin-top:30px;color:var(--muted);font-size:13px}}@media(max-width:760px){{.grid,.row{{grid-template-columns:1fr}}table{{font-size:12px}}}}
</style></head><body><main>
<div class="eyebrow">Portfolio case study · Synthetic data only</div>
<h1>RiskOps<br>Experimentation Intelligence</h1>
<p class="lead">Does assisted triage increase the share of eligible risk reports resolved within SLA? Randomised simulation, missing-feedback correction and human-safety guardrails.</p>
<div class="grid"><section class="card"><div class="small">Simulated eligible reports</div><div class="value">{report["eligible_units"]:,}</div></section>
<section class="card"><div class="small">Adjusted uplift (CUPED + IPW)</div><div class="value">{pp(e["absolute_uplift"])}</div></section>
<section class="card"><div class="small">Simulation-only review</div><div class="value" style="font-size:1.55rem">{escape(decision["status"].replace("_", " ").upper())}</div></section></div>
<div class="row"><section class="card"><h2>Estimated SLA success by arm</h2>
{bar('Control',e['control_mean'])}{bar('Assisted triage',e['treatment_mean'])}<p>Rates are estimated from generated outcomes; no actual user reports or service measurements.</p></section>
<section class="card"><h2>Observation coverage</h2>
{bar('Control feedback captured',report['observation_rate']['control'])}{bar('Treatment feedback captured',report['observation_rate']['treatment'])}
<p>Selection on observed covariates is corrected with simulated known propensity weights.</p></section></div>
<section class="card" style="margin-top:16px"><h2>Estimators and approximate 95% intervals</h2><div style="overflow-x:auto"><table><thead><tr><th>Estimator</th><th>Uplift</th><th>Confidence interval</th><th>Std. error</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<div class="row" style="margin-top:16px"><section class="card"><h2>Safety decision</h2><p class="status">PRODUCTION LAUNCH: NEVER AUTHORISED</p><ul>{issues}</ul><p class="caution">Protected reports cannot enter treatment, and mandatory human review remains in place.</p></section>
<section class="card"><h2>Limitations</h2><ul>{limitations}</ul></section></div>
<footer>Research/portfolio demonstration · Generated from synthetic events · Audit and model artefacts available in repository · No live deployment</footer>
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
