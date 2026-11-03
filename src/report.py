"""Build the stack-up report as a printable PDF.

`paper.py` owns the page furniture; the findings below are this
project's own.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from paper import Paper
from stackup import Contributor, MonteCarlo, Stack, Statistical, WorstCase

FOOTER = (
    "Worst case sums the extreme limits. RSS assumes each contributor is "
    "normal and centred. Monte Carlo samples the declared distributions "
    "with a fixed seed. All sensitivity coefficients are ±1: this is "
    "a linear 1D stack, so no geometric or kinematic effects are "
    "modelled. Form and orientation tolerances, temperature and "
    "assembly-induced deflection are outside the model.")


def _methods(stack: Stack, wc: WorstCase, rss: Statistical,
             mc: MonteCarlo) -> tuple[list[str], list[list[str]]]:
    rows = [
        ["worst case", f"{(wc.minimum + wc.maximum) / 2:.4f}",
         f"{wc.half_band:.4f}", f"{wc.minimum:.4f}", f"{wc.maximum:.4f}",
         "—", "—"],
        ["RSS", f"{rss.mean:.4f}", f"{3 * rss.sigma:.4f}",
         f"{rss.mean - 3 * rss.sigma:.4f}",
         f"{rss.mean + 3 * rss.sigma:.4f}", f"{rss.ppm:,.0f} ppm",
         f"{rss.cpk:.2f}"],
        ["Monte Carlo", f"{mc.mean:.4f}", f"{3 * mc.sigma:.4f}",
         f"{mc.percentiles[1]:.4f}", f"{mc.percentiles[99]:.4f}",
         f"{mc.ppm:,.0f} ppm", f"{mc.cpk:.2f}"],
    ]
    return (["method", "mean", "half-band / 3s", "low", "high",
             "predicted scrap", "Cpk"], rows)


def _contributors(ranked) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one, variance, share in ranked:
        dist = one.distribution
        if one.distribution == "triangular":
            dist += f" (mode {one.mode:+.3f})"
        if one.mean_shift:
            dist += f", shift {one.mean_shift:+.2g}s"
        rows.append([one.name, f"{one.nominal:.3f}",
                     f"{one.lower:+.3f} / {one.upper:+.3f}",
                     f"{one.direction:+d}", dist,
                     f"{variance ** 0.5:.4f}", f"{share * 100:.1f}%",
                     one.source])
    return (["name", "nominal", "tolerance", "dir", "distribution", "s",
             "variance share", "source"], rows)


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
    unit = stack.unit
    top, _, top_share = ranked[0]
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, stack.name,
               f"{stack.description} · closing dimension "
               f"{stack.closing_dimension}, specification {stack.lsl} to "
               f"{stack.usl} {unit} · {len(stack.contributors)} "
               f"contributors · tolerance = "
               f"{stack.sigma_per_tolerance:g}s · generated "
               f"{generated}") as paper:

        if wc.fits:
            verdict = ("Worst case fits, so the stack is safe by the "
                       "strictest reading and the statistical methods are "
                       "only confirming headroom.")
        else:
            verdict = (f"Worst case does not fit: the extremes run "
                       f"{wc.minimum:.3f} to {wc.maximum:.3f} {unit} "
                       f"against a {stack.lsl} to {stack.usl} {unit} "
                       f"specification. That alone is not a reason to "
                       f"change the design — the probability of "
                       f"every part sitting at its limit together is "
                       f"negligible — but it does mean the stack has "
                       f"to be justified statistically rather than by "
                       f"inspection.")

        paper.cover(
            headline=f"Cpk {mc.cpk:.2f}, {mc.ppm:,.0f} ppm predicted scrap",
            small=f"Monte Carlo over {mc.n:,} runs: mean {mc.mean:.3f} "
                  f"{unit}, sigma {mc.sigma:.4f} {unit}, yield "
                  f"{mc.yield_fraction * 100:.2f}%. Nominal gap "
                  f"{stack.nominal_gap:.3f} {unit}.",
            tiles=[("nominal gap", f"{stack.nominal_gap:.3f}"),
                   ("MC sigma", f"{mc.sigma:.4f}"),
                   ("yield", f"{mc.yield_fraction * 100:.2f}%"),
                   ("Cpk (MC)", f"{mc.cpk:.2f}")],
            blurb=verdict,
            figure_path=figures["comparison"])

        delta = mc.ppm - rss_result.ppm
        disagree = ""
        if abs(delta) > max(rss_result.ppm * 0.25, 50):
            disagree = (
                f"Monte Carlo disagrees with RSS. RSS predicts "
                f"{rss_result.ppm:,.0f} ppm, Monte Carlo {mc.ppm:,.0f} ppm "
                f"— {'worse' if delta > 0 else 'better'} by "
                f"{abs(delta):,.0f} ppm. RSS assumes every contributor is "
                f"normal and centred; the stack declares a skewed bearing "
                f"width, a uniform shim and a mean shift on the bore, and "
                f"that is the whole of the difference.")

        # The tail split rides along here rather than taking a section of
        # its own: it is two numbers, and it is read against the table
        # directly above it.
        tails = ""
        if mc.below or mc.above:
            tails = (f"Rejects split {mc.below:,} low "
                     f"({mc.below / mc.n * 1e6:,.0f} ppm) and {mc.above:,} "
                     f"high ({mc.above / mc.n * 1e6:,.0f} ppm). Which tail "
                     f"dominates decides the fix: a lopsided split is a "
                     f"nominal problem, an even split is a variation "
                     f"problem.")

        paper.section(
            "Three methods",
            warn=disagree,
            figure_path=figures["distribution"],
            table=_methods(stack, wc, rss_result, mc),
            after=tails)

        paper.section(
            "Contributors",
            warn=f"{top.name} carries {top_share * 100:.0f}% of the "
                 f"variance. Tolerances add as squares, so effort spent "
                 f"anywhere else is close to wasted until this one moves: "
                 f"halving its tolerance removes about "
                 f"{top_share * 75:.0f}% of the total variance on its own."
                 + ("" if top.adjustable else
                    " It is a purchased part, so the lever is supplier "
                    "selection or a tighter class, not a drawing change."),
            figure_path=figures["pareto"], table=_contributors(ranked))

        if allocation.get("feasible"):
            scale = float(allocation["scale"])
            rows = []
            for name, (low, high) in allocation["new_tolerances"].items():
                current = next(one for one in stack.contributors
                               if one.name == name)
                rows.append([name,
                             f"{current.lower:+.4f} / {current.upper:+.4f}",
                             f"{low:+.4f} / {high:+.4f}"])
            fixed = allocation.get("fixed")
            paper.section(
                "Tolerance allocation",
                body=f"Reaching Cpk {allocation['target_cpk']} needs the "
                     f"in-house tolerances scaled by "
                     f"×{scale:.3f} (total sigma budget "
                     f"{float(allocation['sigma_needed']):.4f} {unit}).",
                table=(["contributor", "current", "required"], rows),
                after=(("Held fixed (purchased): " + ", ".join(fixed) + " "
                        if fixed else "") + FOOTER))
        else:
            paper.section(
                "Tolerance allocation",
                warn=f"Not reachable by tightening in-house tolerances. "
                     f"{allocation.get('reason', '')}. The options are a "
                     f"different nominal, a tighter bearing class, or a "
                     f"selective-assembly or shim-to-fit step — not "
                     f"a tighter drawing.",
                after=FOOTER)

    return out
