"""Checks against closed-form values, not against previous output.

Run with `python tests/test_stackup.py` (no pytest needed) or `pytest tests/`.

The anchors that matter: for n identical contributors worst case grows as n*t
while RSS grows as t*sqrt(n), and a centred stack at exactly +/-3 sigma must
predict 2700 ppm. If the arithmetic drifts, those break.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stackup import (  # noqa: E402
    Contributor,
    Stack,
    allocate,
    monte_carlo,
    normal_cdf,
    rss,
    sensitivity,
    worst_case,
)


def _uniform_stack(n: int, tol: float, distribution: str = "normal",
                   lsl: float = -1.0, usl: float = 1.0, k: float = 3.0) -> Stack:
    """n identical contributors, nominal 0, symmetric tolerance, all adding."""
    return Stack(
        name="test",
        lsl=lsl,
        usl=usl,
        sigma_per_tolerance=k,
        contributors=[
            Contributor(name=f"C{i}", nominal=0.0, lower=-tol, upper=tol,
                        direction=1, distribution=distribution)
            for i in range(n)
        ],
    )


def test_worst_case_grows_linearly() -> None:
    """Worst case adds tolerances arithmetically: n contributors -> n*t."""
    for n in (1, 2, 5, 10):
        wc = worst_case(_uniform_stack(n, 0.05))
        assert abs(wc.half_band - n * 0.05) < 1e-12, n
        assert abs(wc.maximum - n * 0.05) < 1e-12
        assert abs(wc.minimum + n * 0.05) < 1e-12


def test_rss_grows_as_square_root() -> None:
    """RSS adds in quadrature: n contributors -> t*sqrt(n) at the same k."""
    k, tol = 3.0, 0.05
    for n in (1, 2, 5, 10):
        result = rss(_uniform_stack(n, tol, k=k))
        expected_sigma = (tol / k) * math.sqrt(n)
        assert abs(result.sigma - expected_sigma) < 1e-12, n
        # And the 3-sigma half-band is t*sqrt(n) when k is 3.
        assert abs(3 * result.sigma - tol * math.sqrt(n)) < 1e-12


def test_worst_case_exceeds_rss_for_any_real_stack() -> None:
    """The over-design worth quantifying: WC/RSS = sqrt(n) for equal parts."""
    for n in (2, 4, 9, 16):
        stack = _uniform_stack(n, 0.05)
        ratio = worst_case(stack).half_band / (3 * rss(stack).sigma)
        assert abs(ratio - math.sqrt(n)) < 1e-9, n


def test_direction_signs_subtract() -> None:
    """A -1 contributor must subtract, and must widen the band, not narrow it."""
    stack = Stack(
        name="t", lsl=-1.0, usl=1.0,
        contributors=[
            Contributor("A", 10.0, -0.1, 0.1, direction=1),
            Contributor("B", 4.0, -0.1, 0.1, direction=-1),
        ],
    )
    assert abs(stack.nominal_gap - 6.0) < 1e-12
    wc = worst_case(stack)
    assert abs(wc.maximum - 6.2) < 1e-12  # A high, B low
    assert abs(wc.minimum - 5.8) < 1e-12
    # Variance is direction-independent.
    assert abs(rss(stack).sigma - math.sqrt(2) * (0.1 / 3)) < 1e-12


def test_three_sigma_centred_predicts_2700_ppm() -> None:
    """A stack centred with the spec exactly at +/-3 sigma: 0.27% outside."""
    tol, k = 0.05, 3.0
    sigma = tol / k
    stack = _uniform_stack(1, tol, lsl=-3 * sigma, usl=3 * sigma, k=k)
    result = rss(stack)
    assert abs(result.cpk - 1.0) < 1e-12
    assert abs(result.ppm - 2699.796) < 0.01


def test_uniform_variance_is_band_squared_over_twelve() -> None:
    c = Contributor("U", 0.0, -0.02, 0.02, distribution="uniform")
    assert abs(c.variance(3.0) - (0.04**2) / 12.0) < 1e-18


def test_triangular_mean_and_variance_match_the_closed_form() -> None:
    a, b, m = -0.03, 0.03, 0.012
    c = Contributor("T", 5.0, a, b, distribution="triangular", mode=m)
    assert abs(c.mean(3.0) - (5.0 + (a + b + m) / 3.0)) < 1e-15
    expected = (a * a + b * b + m * m - a * b - a * m - b * m) / 18.0
    assert abs(c.variance(3.0) - expected) < 1e-18


def test_declared_distributions_are_sampled_as_declared() -> None:
    """Sampling must reproduce the closed-form mean and variance of each shape.

    This is the link between the analytic sensitivity numbers and the Monte
    Carlo histogram -- if it breaks, the two halves of the report disagree.
    """
    rng = np.random.default_rng(1)
    for c in (
        Contributor("N", 1.0, -0.03, 0.03, distribution="normal"),
        Contributor("U", 1.0, -0.03, 0.03, distribution="uniform"),
        Contributor("T", 1.0, -0.03, 0.03, distribution="triangular", mode=0.012),
    ):
        drawn = c.sample(rng, 400_000, 3.0)
        assert abs(drawn.mean() - c.mean(3.0)) < 5e-5, c.distribution
        assert abs(drawn.var(ddof=1) - c.variance(3.0)) / c.variance(3.0) < 0.02, c.distribution


def test_monte_carlo_matches_rss_when_assumptions_hold() -> None:
    """All-normal, all-centred: the two methods must agree to sampling noise."""
    stack = _uniform_stack(5, 0.05, lsl=-0.2, usl=0.2)
    analytic = rss(stack)
    mc = monte_carlo(stack, n=400_000, seed=3)

    assert abs(mc.mean - analytic.mean) < 1e-3
    assert abs(mc.sigma - analytic.sigma) / analytic.sigma < 0.01
    assert abs(mc.yield_fraction - analytic.yield_fraction) < 2e-3


def test_monte_carlo_diverges_from_rss_when_they_do_not() -> None:
    """A uniform contributor has a wider sigma than the tolerance=3sigma
    convention assumes, so Monte Carlo must come out worse, not merely
    different.

    The spec has to sit inside the arithmetic limit for this to mean anything:
    four uniform contributors of +/-0.05 are bounded at exactly +/-0.2, so a
    +/-0.2 spec would give both stacks zero fallout and prove nothing.
    """
    normal = _uniform_stack(4, 0.05, distribution="normal", lsl=-0.08, usl=0.08)
    uniform = _uniform_stack(4, 0.05, distribution="uniform", lsl=-0.08, usl=0.08)

    mc_normal = monte_carlo(normal, n=200_000, seed=5)
    mc_uniform = monte_carlo(uniform, n=200_000, seed=5)
    assert mc_uniform.sigma > mc_normal.sigma
    assert mc_uniform.ppm > mc_normal.ppm


def test_mean_shift_moves_the_mean_and_costs_cpk() -> None:
    plain = _uniform_stack(3, 0.05, lsl=-0.2, usl=0.2)
    shifted = _uniform_stack(3, 0.05, lsl=-0.2, usl=0.2)
    for c in shifted.contributors:
        c.mean_shift = 1.0  # one sigma each, all in the same direction

    base, moved = rss(plain), rss(shifted)
    assert abs(moved.mean - (base.mean + 3 * 0.05 / 3)) < 1e-12
    assert abs(moved.sigma - base.sigma) < 1e-12  # a shift is not spread
    assert moved.cpk < base.cpk


def test_sensitivity_shares_sum_to_one_and_rank_correctly() -> None:
    stack = Stack(
        name="t", lsl=-1.0, usl=1.0,
        contributors=[
            Contributor("SMALL", 1.0, -0.01, 0.01),
            Contributor("BIG", 1.0, -0.10, 0.10),
            Contributor("MID", 1.0, -0.05, 0.05),
        ],
    )
    ranked = sensitivity(stack)
    assert [c.name for c, _, _ in ranked] == ["BIG", "MID", "SMALL"]
    assert abs(sum(frac for _, _, frac in ranked) - 1.0) < 1e-12
    # Variance share scales with the square of the tolerance: 100:25:1.
    shares = [frac for _, _, frac in ranked]
    assert abs(shares[0] / shares[2] - 100.0) < 1e-9


def test_allocation_round_trips_to_the_target() -> None:
    """Applying the returned scale must actually produce the requested Cpk."""
    stack = _uniform_stack(4, 0.05, lsl=-0.2, usl=0.2)
    result = allocate(stack, target_cpk=1.33)
    assert result["feasible"]

    scale = float(result["scale"])
    for c in stack.contributors:
        c.lower *= scale
        c.upper *= scale
    assert abs(rss(stack).cpk - 1.33) < 1e-9


def test_allocation_reports_infeasible_when_fixed_parts_blow_the_budget() -> None:
    """The honest answer when a purchased part is the problem."""
    stack = Stack(
        name="t", lsl=-0.05, usl=0.05,
        contributors=[
            # Nominals cancel, so the mean gap is zero and the only problem
            # left is the purchased part's spread.
            Contributor("BOUGHT", 1.0, -0.10, 0.10, direction=1, adjustable=False),
            Contributor("MADE", 1.0, -0.01, 0.01, direction=-1, adjustable=True),
        ],
    )
    result = allocate(stack, target_cpk=1.33)
    assert result["feasible"] is False
    assert "budget" in str(result["reason"]), result["reason"]
    # Shrinking the in-house tolerance to nothing still would not be enough.
    assert result["fixed_sigma"] > result["sigma_needed"]


def test_invalid_definitions_are_rejected() -> None:
    for kwargs, reason in (
        (dict(distribution="lognormal"), "unknown distribution"),
        (dict(lower=0.1, upper=-0.1), "inverted band"),
        (dict(direction=0), "bad direction"),
        (dict(distribution="triangular", mode=0.5), "mode outside band"),
        (dict(distribution="uniform", mean_shift=1.0), "shift on non-normal"),
    ):
        base = dict(name="X", nominal=1.0, lower=-0.1, upper=0.1)
        base.update(kwargs)
        try:
            Contributor(**base)
        except ValueError:
            continue
        raise AssertionError(f"accepted an invalid contributor: {reason}")


def test_normal_cdf_matches_known_quantiles() -> None:
    assert abs(normal_cdf(0.0) - 0.5) < 1e-12
    assert abs(normal_cdf(1.96) - 0.9750021048) < 1e-9


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}  {exc}")
        else:
            print(f"pass  {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
