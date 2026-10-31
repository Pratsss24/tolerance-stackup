"""Tolerance stack-up analysis: worst case, RSS, and Monte Carlo.

The three methods answer the same question with different assumptions, and the
gap between them is the point of the tool:

* **Worst case** assumes every part sits at its worst limit simultaneously. It
  guarantees fit and routinely over-designs, because that combination has a
  vanishing probability on a stack of any length.
* **RSS** (root-sum-square) assumes every contributor is normal, centred in its
  tolerance band, and that each tolerance equals k sigma. It is the textbook
  statistical method and it is optimistic exactly where those assumptions fail.
* **Monte Carlo** samples the distributions as declared -- skewed, uniform,
  mean-shifted -- so it can disagree with RSS, and when it does, the assumption
  that broke is named in the stack definition rather than hidden in the method.

Convention: a contributor's tolerance is a deviation from its nominal, and
`direction` is +1 if growing that dimension opens the closing gap, -1 if it
closes it. The closing dimension is sum(direction * dimension).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DISTRIBUTIONS = ("normal", "uniform", "triangular")


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


@dataclass
class Contributor:
    name: str
    nominal: float
    lower: float
    upper: float
    direction: int = 1
    distribution: str = "normal"
    mode: float | None = None
    mean_shift: float = 0.0
    source: str = ""
    adjustable: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"{self.name}: unknown distribution {self.distribution!r}, "
                f"expected one of {DISTRIBUTIONS}"
            )
        if self.upper < self.lower:
            raise ValueError(f"{self.name}: upper {self.upper} is below lower {self.lower}")
        if self.direction not in (1, -1):
            raise ValueError(f"{self.name}: direction must be +1 or -1")
        if self.distribution == "triangular":
            if self.mode is None:
                self.mode = (self.lower + self.upper) / 2.0
            if not self.lower <= self.mode <= self.upper:
                raise ValueError(f"{self.name}: triangular mode {self.mode} is outside the band")
        if self.distribution != "normal" and self.mean_shift:
            # A shift is expressed in sigma, which only has an agreed meaning for
            # the normal case here. Refuse rather than guess.
            raise ValueError(f"{self.name}: mean_shift is only supported for normal contributors")

    @property
    def band(self) -> float:
        """Full tolerance band width."""
        return self.upper - self.lower

    @property
    def half_band(self) -> float:
        return self.band / 2.0

    def sigma_rss(self, k: float) -> float:
        """Sigma under the RSS convention: the tolerance band spans 2k sigma."""
        return self.band / (2.0 * k)

    def mean(self, k: float) -> float:
        """True mean of the declared distribution, as an absolute dimension."""
        mid = self.nominal + (self.lower + self.upper) / 2.0
        if self.distribution == "normal":
            return mid + self.mean_shift * self.sigma_rss(k)
        if self.distribution == "uniform":
            return mid
        return self.nominal + (self.lower + self.upper + self.mode) / 3.0

    def variance(self, k: float) -> float:
        """True variance of the declared distribution."""
        if self.distribution == "normal":
            return self.sigma_rss(k) ** 2
        if self.distribution == "uniform":
            return self.band**2 / 12.0
        a, b, c = self.lower, self.upper, self.mode
        return (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0

    def sample(self, rng: np.random.Generator, n: int, k: float) -> np.ndarray:
        if self.distribution == "normal":
            sigma = self.sigma_rss(k)
            mid = self.nominal + (self.lower + self.upper) / 2.0
            return rng.normal(mid + self.mean_shift * sigma, sigma, n)
        if self.distribution == "uniform":
            return rng.uniform(self.nominal + self.lower, self.nominal + self.upper, n)
        return rng.triangular(
            self.nominal + self.lower,
            self.nominal + self.mode,
            self.nominal + self.upper,
            n,
        )


@dataclass
class Stack:
    name: str
    lsl: float
    usl: float
    contributors: list[Contributor]
    unit: str = "mm"
    description: str = ""
    closing_dimension: str = "gap"
    sigma_per_tolerance: float = 3.0

    @property
    def nominal_gap(self) -> float:
        return sum(c.direction * c.nominal for c in self.contributors)

    @property
    def mid_spec(self) -> float:
        return (self.lsl + self.usl) / 2.0

    @classmethod
    def from_json(cls, path: Path) -> "Stack":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in Contributor.__dataclass_fields__}
        contributors = []
        for entry in raw["contributors"]:
            unknown = set(entry) - known
            if unknown:
                raise ValueError(
                    f"contributor {entry.get('name', '?')}: unknown keys {sorted(unknown)}"
                )
            contributors.append(Contributor(**entry))
        if not contributors:
            raise ValueError("stack has no contributors")
        if raw["lsl"] >= raw["usl"]:
            raise ValueError("lsl must be below usl")
        return cls(
            name=raw["name"],
            lsl=float(raw["lsl"]),
            usl=float(raw["usl"]),
            contributors=contributors,
            unit=raw.get("unit", "mm"),
            description=raw.get("description", ""),
            closing_dimension=raw.get("closing_dimension", "gap"),
            sigma_per_tolerance=float(raw.get("sigma_per_tolerance", 3.0)),
        )


@dataclass
class WorstCase:
    minimum: float
    maximum: float
    half_band: float
    fits: bool


@dataclass
class Statistical:
    mean: float
    sigma: float
    cpk: float
    cp: float
    yield_fraction: float
    ppm: float


@dataclass
class MonteCarlo:
    samples: np.ndarray
    mean: float
    sigma: float
    cpk: float
    yield_fraction: float
    ppm: float
    below: int
    above: int
    n: int
    percentiles: dict[int, float] = field(default_factory=dict)


def worst_case(stack: Stack) -> WorstCase:
    """Both extreme combinations. For direction +1 the gap opens at the upper
    limit; for direction -1 it opens at the lower limit, since that dimension
    is being subtracted."""
    gap_max = sum(
        c.direction * (c.nominal + (c.upper if c.direction > 0 else c.lower))
        for c in stack.contributors
    )
    gap_min = sum(
        c.direction * (c.nominal + (c.lower if c.direction > 0 else c.upper))
        for c in stack.contributors
    )
    return WorstCase(
        minimum=gap_min,
        maximum=gap_max,
        half_band=(gap_max - gap_min) / 2.0,
        fits=gap_min >= stack.lsl and gap_max <= stack.usl,
    )


def _capability(mean: float, sigma: float, lsl: float, usl: float) -> tuple[float, float, float]:
    """Returns (cp, cpk, yield). Sigma of zero means a deterministic stack."""
    if sigma <= 0:
        inside = lsl <= mean <= usl
        return math.inf, math.inf, 1.0 if inside else 0.0
    cp = (usl - lsl) / (6.0 * sigma)
    cpk = min(usl - mean, mean - lsl) / (3.0 * sigma)
    fraction = normal_cdf((usl - mean) / sigma) - normal_cdf((lsl - mean) / sigma)
    return cp, cpk, fraction


def rss(stack: Stack) -> Statistical:
    """Textbook RSS: every contributor normal, tolerance band = 2k sigma.

    Note the mean still carries any declared mean shift -- leaving it out is
    what makes a naive RSS look better than the process really is.
    """
    k = stack.sigma_per_tolerance
    variance = sum(c.sigma_rss(k) ** 2 for c in stack.contributors)
    sigma = math.sqrt(variance)
    mean = sum(c.direction * c.mean(k) for c in stack.contributors)
    cp, cpk, fraction = _capability(mean, sigma, stack.lsl, stack.usl)
    return Statistical(
        mean=mean, sigma=sigma, cpk=cpk, cp=cp,
        yield_fraction=fraction, ppm=(1.0 - fraction) * 1e6,
    )


def monte_carlo(stack: Stack, n: int = 200_000, seed: int = 20261031) -> MonteCarlo:
    rng = np.random.default_rng(seed)
    k = stack.sigma_per_tolerance

    total = np.zeros(n)
    for c in stack.contributors:
        total += c.direction * c.sample(rng, n, k)

    below = int(np.sum(total < stack.lsl))
    above = int(np.sum(total > stack.usl))
    inside = n - below - above
    sigma = float(total.std(ddof=1))
    mean = float(total.mean())
    _, cpk, _ = _capability(mean, sigma, stack.lsl, stack.usl)

    return MonteCarlo(
        samples=total,
        mean=mean,
        sigma=sigma,
        cpk=cpk,
        yield_fraction=inside / n,
        ppm=(below + above) / n * 1e6,
        below=below,
        above=above,
        n=n,
        percentiles={p: float(np.percentile(total, p)) for p in (1, 5, 50, 95, 99)},
    )


def sensitivity(stack: Stack) -> list[tuple[Contributor, float, float]]:
    """Variance contribution per contributor, using the declared distributions.

    Returns (contributor, variance, fraction of total variance), largest first.
    For a linear stack every sensitivity coefficient is +/-1, so the variance
    share is the whole story -- halving the tolerance of the top contributor
    buys more than halving everything below it.
    """
    k = stack.sigma_per_tolerance
    variances = [(c, c.variance(k)) for c in stack.contributors]
    total = sum(v for _, v in variances)
    if total <= 0:
        return [(c, 0.0, 0.0) for c, _ in variances]
    ranked = sorted(variances, key=lambda item: item[1], reverse=True)
    return [(c, v, v / total) for c, v in ranked]


def allocate(stack: Stack, target_cpk: float = 1.33) -> dict[str, object]:
    """Scale the adjustable tolerances to reach a target Cpk on the closing gap.

    Solves analytically under the RSS convention: sigma^2 = s^2*A + B, where A
    comes from the adjustable contributors and B from the fixed ones. If the
    fixed contributors alone already exceed the variance budget, no scaling of
    the in-house tolerances can rescue the stack, and the result says so --
    which is the honest answer when the problem is a purchased part.
    """
    k = stack.sigma_per_tolerance
    base = rss(stack)
    margin = min(stack.usl - base.mean, base.mean - stack.lsl)

    adjustable = [c for c in stack.contributors if c.adjustable]
    fixed = [c for c in stack.contributors if not c.adjustable]
    a = sum(c.sigma_rss(k) ** 2 for c in adjustable)
    b = sum(c.sigma_rss(k) ** 2 for c in fixed)

    if margin <= 0:
        return {"feasible": False, "reason": "the mean gap is already outside the specification"}

    sigma_needed = margin / (3.0 * target_cpk)
    budget = sigma_needed**2 - b

    if a <= 0:
        return {"feasible": False, "reason": "no adjustable contributors to tighten"}
    if budget <= 0:
        return {
            "feasible": False,
            "reason": (
                f"the fixed contributors alone need sigma {math.sqrt(b):.4f} {stack.unit}, "
                f"above the {sigma_needed:.4f} {stack.unit} budget for Cpk {target_cpk:.2f}"
            ),
            "fixed_sigma": math.sqrt(b),
            "sigma_needed": sigma_needed,
        }

    scale = math.sqrt(budget / a)
    return {
        "feasible": True,
        "scale": scale,
        "target_cpk": target_cpk,
        "sigma_needed": sigma_needed,
        "new_tolerances": {
            c.name: (c.lower * scale, c.upper * scale) for c in adjustable
        },
        "fixed": [c.name for c in fixed],
    }
