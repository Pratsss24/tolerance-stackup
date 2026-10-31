"""Single self-contained HTML report for a tolerance stack-up."""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

from stackup import Contributor, MonteCarlo, Stack, Statistical, WorstCase

CSS = """
:root {
  --bg: #f4f6f8; --card: #ffffff; --ink: #1b2733; --muted: #5d6f7e;
  --rule: #dde4ea; --good: #2f7d4f; --warn: #8a6d1f; --bad: #b23a2e;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--ink);
  font: 15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif; }
.wrap { max-width: 1020px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 0 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 24px; }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 8px;
  padding: 18px 20px; margin: 0 0 18px; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: right; padding: 7px 9px; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; font-family: ui-monospace, "Cascadia Mono", monospace; }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
tbody tr:last-child td { border-bottom: none; }
.num { font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
.pill.ok { background: #e3f1e8; color: var(--good); }
.pill.warn { background: #f6efdb; color: var(--warn); }
.pill.bad { background: #f7e3e0; color: var(--bad); }
img { width: 100%; display: block; margin: 14px 0 0; border: 1px solid var(--rule); border-radius: 6px; }
.finding { margin: 12px 0 0; padding: 11px 13px; border-left: 3px solid var(--warn); background: #fbf8ef; font-size: 13.5px; }
.finding.bad { border-left-color: var(--bad); background: #fdf4f3; }
.finding.ok { border-left-color: var(--good); background: #f3f9f5; }
.finding b { font-weight: 600; }
.kpi { display: flex; gap: 26px; flex-wrap: wrap; margin: 4px 0 2px; }
.kpi div { min-width: 120px; }
.kpi .v { font-size: 22px; font-variant-numeric: tabular-nums; }
.kpi .l { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
footer { color: var(--muted); font-size: 12.5px; margin-top: 40px; }
@media print { body { background: #fff; padding: 0; } .card { break-inside: avoid; } }
"""


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _cpk_pill(cpk: float) -> str:
    if cpk < 1.0:
        return f'<span class="pill bad">{cpk:.2f}</span>'
    if cpk < 1.33:
        return f'<span class="pill warn">{cpk:.2f}</span>'
    return f'<span class="pill ok">{cpk:.2f}</span>'


def build(
    stack: Stack,
    wc: WorstCase,
    rss_result: Statistical,
    mc: MonteCarlo,
    ranked: list[tuple[Contributor, float, float]],
    allocation: dict[str, object],
    figures: dict[str, Path],
    out_path: Path,
) -> Path:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    u = stack.unit
    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Tolerance Stack-up Report</title>",
        f"<style>{CSS}</style></head><body><div class='wrap'>",
        f"<h1>{html.escape(stack.name)}</h1>",
        f"<p class='sub'>{html.escape(stack.description)}<br>"
        f"Closing dimension <b>{html.escape(stack.closing_dimension)}</b>, "
        f"specification {stack.lsl} to {stack.usl} {u} &middot; "
        f"{len(stack.contributors)} contributors &middot; "
        f"tolerance = {stack.sigma_per_tolerance:g}&sigma; &middot; generated {generated}</p>",
    ]

    # Headline
    parts.append("<div class='card'><h2>Result</h2><div class='kpi'>")
    for label, value in (
        ("nominal gap", f"{stack.nominal_gap:.3f} {u}"),
        ("MC mean", f"{mc.mean:.3f} {u}"),
        ("MC sigma", f"{mc.sigma:.4f} {u}"),
        ("yield", f"{mc.yield_fraction * 100:.2f}%"),
        ("scrap", f"{mc.ppm:,.0f} ppm"),
    ):
        parts.append(f"<div><div class='v'>{value}</div><div class='l'>{label}</div></div>")
    parts.append(f"<div><div class='v'>{_cpk_pill(mc.cpk)}</div><div class='l'>Cpk (MC)</div></div>")
    parts.append("</div>")

    verdict_cls = "ok" if mc.cpk >= 1.33 else ("warn" if mc.cpk >= 1.0 else "bad")
    wc_note = (
        "Worst case fits, so the stack is safe by the strictest reading and the "
        "statistical methods are only confirming headroom."
        if wc.fits else
        f"<b>Worst case does not fit</b>: the extremes run {wc.minimum:.3f} to "
        f"{wc.maximum:.3f} {u} against a {stack.lsl} to {stack.usl} {u} "
        "specification. That alone is not a reason to change the design &mdash; "
        "the probability of every part sitting at its limit together is "
        "negligible &mdash; but it does mean the stack has to be justified "
        "statistically rather than by inspection."
    )
    parts.append(f"<div class='finding {verdict_cls}'>{wc_note}</div>")

    delta = (mc.ppm - rss_result.ppm)
    if abs(delta) > max(rss_result.ppm * 0.25, 50):
        direction = "worse" if delta > 0 else "better"
        parts.append(
            f"<div class='finding'><b>Monte Carlo disagrees with RSS.</b> "
            f"RSS predicts {rss_result.ppm:,.0f} ppm, Monte Carlo {mc.ppm:,.0f} ppm "
            f"&mdash; {direction} by {abs(delta):,.0f} ppm. RSS assumes every "
            "contributor is normal and centred; the stack declares a skewed "
            "bearing width, a uniform shim and a mean shift on the bore, and "
            "that is the whole of the difference.</div>"
        )
    parts.append("</div>")

    # Methods
    parts.append("<div class='card'><h2>Three methods</h2><table>")
    parts.append("<thead><tr><th>Method</th><th>Mean</th><th>Half-band / 3&sigma;</th>"
                 "<th>Low</th><th>High</th><th>Predicted scrap</th><th>Cpk</th></tr></thead><tbody>")
    parts.append(
        f"<tr><td>Worst case</td><td class='num'>{(wc.minimum + wc.maximum) / 2:.4f}</td>"
        f"<td class='num'>{wc.half_band:.4f}</td><td class='num'>{wc.minimum:.4f}</td>"
        f"<td class='num'>{wc.maximum:.4f}</td><td class='num'>&mdash;</td>"
        f"<td class='num'>&mdash;</td></tr>"
    )
    parts.append(
        f"<tr><td>RSS</td><td class='num'>{rss_result.mean:.4f}</td>"
        f"<td class='num'>{3 * rss_result.sigma:.4f}</td>"
        f"<td class='num'>{rss_result.mean - 3 * rss_result.sigma:.4f}</td>"
        f"<td class='num'>{rss_result.mean + 3 * rss_result.sigma:.4f}</td>"
        f"<td class='num'>{rss_result.ppm:,.0f} ppm</td>"
        f"<td class='num'>{rss_result.cpk:.2f}</td></tr>"
    )
    parts.append(
        f"<tr><td>Monte Carlo</td><td class='num'>{mc.mean:.4f}</td>"
        f"<td class='num'>{3 * mc.sigma:.4f}</td>"
        f"<td class='num'>{mc.percentiles[1]:.4f}</td>"
        f"<td class='num'>{mc.percentiles[99]:.4f}</td>"
        f"<td class='num'>{mc.ppm:,.0f} ppm</td>"
        f"<td class='num'>{mc.cpk:.2f}</td></tr>"
    )
    parts.append("</tbody></table>")
    parts.append(f"<img alt='method comparison' src='data:image/png;base64,"
                 f"{_b64(figures['comparison'])}'>")
    parts.append(f"<img alt='monte carlo distribution' src='data:image/png;base64,"
                 f"{_b64(figures['distribution'])}'>")
    if mc.below or mc.above:
        parts.append(
            f"<div class='finding'>Monte Carlo rejects split "
            f"<b>{mc.below:,} low</b> ({mc.below / mc.n * 1e6:,.0f} ppm) and "
            f"<b>{mc.above:,} high</b> ({mc.above / mc.n * 1e6:,.0f} ppm). "
            "Which tail dominates decides the fix: a lopsided split is a nominal "
            "problem, an even split is a variation problem.</div>"
        )
    parts.append("</div>")

    # Contributors
    parts.append("<div class='card'><h2>Contributors</h2><table>")
    parts.append("<thead><tr><th>Name</th><th>Nominal</th><th>Tolerance</th><th>Dir</th>"
                 "<th>Distribution</th><th>&sigma;</th><th>Variance share</th>"
                 "<th>Source</th></tr></thead><tbody>")
    k = stack.sigma_per_tolerance
    for c, variance, frac in ranked:
        dist = c.distribution
        if c.distribution == "triangular":
            dist += f" (mode {c.mode:+.3f})"
        if c.mean_shift:
            dist += f", shift {c.mean_shift:+.2g}&sigma;"
        parts.append(
            f"<tr><td>{html.escape(c.name)}</td>"
            f"<td class='num'>{c.nominal:.3f}</td>"
            f"<td class='num'>{c.lower:+.3f} / {c.upper:+.3f}</td>"
            f"<td class='num'>{c.direction:+d}</td>"
            f"<td style='text-align:left'>{dist}</td>"
            f"<td class='num'>{variance ** 0.5:.4f}</td>"
            f"<td class='num'>{frac * 100:.1f}%</td>"
            f"<td style='text-align:left;color:var(--muted)'>{html.escape(c.source)}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(f"<img alt='variance pareto' src='data:image/png;base64,"
                 f"{_b64(figures['pareto'])}'>")

    top, _, top_frac = ranked[0]
    parts.append(
        f"<div class='finding'><b>{html.escape(top.name)} carries "
        f"{top_frac * 100:.0f}% of the variance.</b> Tolerances add as squares, so "
        "effort spent anywhere else is close to wasted until this one moves: "
        f"halving its tolerance removes about {top_frac * 75:.0f}% of the total "
        "variance on its own."
        + ("" if top.adjustable else
           " It is a purchased part, so the lever is supplier selection or a "
           "tighter class, not a drawing change.")
        + "</div>"
    )
    parts.append("</div>")

    # Allocation
    parts.append("<div class='card'><h2>Tolerance allocation</h2>")
    if allocation.get("feasible"):
        scale = float(allocation["scale"])
        parts.append(
            f"<div class='finding ok'>Reaching <b>Cpk {allocation['target_cpk']}</b> needs the "
            f"in-house tolerances scaled by <b>&times;{scale:.3f}</b> "
            f"(total &sigma; budget {float(allocation['sigma_needed']):.4f} {u}).</div>"
        )
        parts.append("<table><thead><tr><th>Contributor</th><th>Current</th>"
                     "<th>Required</th></tr></thead><tbody>")
        for name, (lo, hi) in allocation["new_tolerances"].items():
            current = next(c for c in stack.contributors if c.name == name)
            parts.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td class='num'>{current.lower:+.4f} / {current.upper:+.4f}</td>"
                f"<td class='num'>{lo:+.4f} / {hi:+.4f}</td></tr>"
            )
        parts.append("</tbody></table>")
        if allocation.get("fixed"):
            parts.append(
                "<p class='sub' style='margin:10px 0 0'>Held fixed (purchased): "
                + ", ".join(html.escape(n) for n in allocation["fixed"]) + "</p>"
            )
    else:
        parts.append(
            f"<div class='finding bad'><b>Not reachable by tightening in-house "
            f"tolerances.</b> {html.escape(str(allocation.get('reason', '')))}. "
            "The options are a different nominal, a tighter bearing class, or a "
            "selective-assembly or shim-to-fit step &mdash; not a tighter drawing.</div>"
        )
    parts.append("</div>")

    parts.append(
        "<footer>Worst case sums the extreme limits. RSS assumes each contributor "
        f"is normal and centred with its tolerance band spanning "
        f"{stack.sigma_per_tolerance:g}&sigma; each side. Monte Carlo samples the "
        f"declared distributions ({mc.n:,} runs, fixed seed). All sensitivity "
        "coefficients are &plusmn;1: this is a linear 1D stack, so no geometric "
        "or kinematic effects are modelled. Form and orientation tolerances, "
        "temperature and assembly-induced deflection are outside the model."
        "</footer>"
    )
    parts.append("</div></body></html>")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
