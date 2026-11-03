"""CLI: stack definition -> worst case, RSS, Monte Carlo, allocation, HTML report.

    python run.py                                  # the bundled example stack
    python run.py --stack data/my_stack.json
    python run.py --runs 1000000 --target-cpk 1.67
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts  # noqa: E402
import report  # noqa: E402
from stackup import Stack, allocate, monte_carlo, rss, sensitivity, worst_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", type=Path, default=ROOT / "data" / "shaft_endfloat.json")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--runs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--target-cpk", type=float, default=1.33)
    args = ap.parse_args()

    stack = Stack.from_json(args.stack)
    args.out.mkdir(parents=True, exist_ok=True)

    wc = worst_case(stack)
    rss_result = rss(stack)
    mc = monte_carlo(stack, n=args.runs, seed=args.seed)
    ranked = sensitivity(stack)
    allocation = allocate(stack, target_cpk=args.target_cpk)

    u = stack.unit
    print(f"\n{stack.name}")
    print(f"specification: {stack.lsl} to {stack.usl} {u}   "
          f"nominal {stack.closing_dimension}: {stack.nominal_gap:.4f} {u}\n")

    print(f"{'method':<14}{'mean':>10}{'low':>10}{'high':>10}{'scrap ppm':>12}{'Cpk':>7}")
    print("-" * 63)
    print(f"{'worst case':<14}{(wc.minimum + wc.maximum) / 2:>10.4f}{wc.minimum:>10.4f}"
          f"{wc.maximum:>10.4f}{'-':>12}{'-':>7}")
    print(f"{'RSS':<14}{rss_result.mean:>10.4f}{rss_result.mean - 3 * rss_result.sigma:>10.4f}"
          f"{rss_result.mean + 3 * rss_result.sigma:>10.4f}{rss_result.ppm:>12,.0f}"
          f"{rss_result.cpk:>7.2f}")
    print(f"{'Monte Carlo':<14}{mc.mean:>10.4f}{mc.percentiles[1]:>10.4f}"
          f"{mc.percentiles[99]:>10.4f}{mc.ppm:>12,.0f}{mc.cpk:>7.2f}")

    print(f"\nworst case fits: {'yes' if wc.fits else 'NO'}")
    print(f"MC yield: {mc.yield_fraction * 100:.3f}%  "
          f"({mc.below:,} below LSL, {mc.above:,} above USL, n={mc.n:,})")

    print("\nvariance contribution")
    print("-" * 63)
    for c, variance, frac in ranked:
        tag = "" if c.adjustable else "  (purchased - cannot tighten)"
        print(f"  {c.name:<22}{frac * 100:>6.1f}%   sigma {variance ** 0.5:.4f}{tag}")

    print(f"\nallocation to Cpk {args.target_cpk}")
    print("-" * 63)
    if allocation.get("feasible"):
        print(f"  scale in-house tolerances by x{float(allocation['scale']):.3f}")
        for name, (lo, hi) in allocation["new_tolerances"].items():
            print(f"    {name:<22}{lo:+.4f} / {hi:+.4f}")
    else:
        print(f"  NOT FEASIBLE - {allocation.get('reason')}")

    figures = {
        "comparison": charts.comparison_png(stack, wc, rss_result, mc,
                                            args.out / "comparison.png"),
        "distribution": charts.distribution_png(stack, mc, rss_result,
                                                args.out / "distribution.png"),
        "pareto": charts.pareto_png(stack, ranked, args.out / "pareto.png"),
    }
    report_path = report.build(stack, wc, rss_result, mc, ranked, allocation,
                               figures, args.out / "report.pdf")
    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
