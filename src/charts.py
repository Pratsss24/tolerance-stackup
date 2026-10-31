"""Chart rendering: Monte Carlo distribution, variance Pareto, method comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stackup import Contributor, MonteCarlo, Stack, Statistical, WorstCase

INK = "#1b2733"
GRID = "#d9e0e6"
BAR = "#cfe0ec"
BAR_EDGE = "#7fa6c0"
LINE = "#20638f"
LIMIT = "#b23a2e"
CENTRE = "#2f7d4f"
FIXED = "#8a6d1f"


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def distribution_png(stack: Stack, mc: MonteCarlo, rss_result: Statistical, out: Path) -> Path:
    """Monte Carlo histogram against the spec limits, with the RSS normal
    overlaid so the disagreement between the two is visible rather than
    stated."""
    fig, ax = plt.subplots(figsize=(9.2, 3.8))

    lo = min(mc.samples.min(), stack.lsl)
    hi = max(mc.samples.max(), stack.usl)
    pad = (hi - lo) * 0.06
    bins = np.linspace(lo - pad, hi + pad, 140)

    counts, _, _ = ax.hist(mc.samples, bins=bins, color=BAR, edgecolor=BAR_EDGE,
                           linewidth=0.3, zorder=3)

    grid = np.linspace(lo - pad, hi + pad, 500)
    if rss_result.sigma > 0:
        pdf = np.exp(-0.5 * ((grid - rss_result.mean) / rss_result.sigma) ** 2)
        ax.plot(grid, pdf / pdf.max() * counts.max(), color=LINE, linewidth=1.5,
                zorder=5, label="RSS normal assumption")

    for limit, label in ((stack.lsl, "LSL"), (stack.usl, "USL")):
        ax.axvline(limit, color=LIMIT, linewidth=1.6, zorder=6)
        ax.text(limit, counts.max() * 0.97, f" {label}", color=LIMIT, fontsize=8, va="top")

    # Shade the out-of-spec tails so the scrap is a visible area, not a number.
    ax.axvspan(grid[0], stack.lsl, color=LIMIT, alpha=0.07, zorder=1)
    ax.axvspan(stack.usl, grid[-1], color=LIMIT, alpha=0.07, zorder=1)

    ax.axvline(mc.mean, color=CENTRE, linewidth=1.2, linestyle="--", zorder=6)
    ax.text(mc.mean, counts.max() * 0.72, " MC mean", color=CENTRE, fontsize=8, va="top")

    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel(f"{stack.closing_dimension} [{stack.unit}]", color=INK, fontsize=9)
    ax.set_ylabel(f"samples (n={mc.n:,})", color=INK, fontsize=9)
    ax.set_title("Monte Carlo distribution of the closing dimension",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)

    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def pareto_png(stack: Stack, ranked: list[tuple[Contributor, float, float]], out: Path) -> Path:
    """Variance share per contributor, fixed ones marked.

    Variance share, not tolerance size, is what tells you where to spend money:
    halving a tolerance quarters its variance contribution.
    """
    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(ranked) + 1.6))

    names = [c.name for c, _, _ in ranked]
    shares = [frac * 100 for _, _, frac in ranked]
    colors = [LINE if c.adjustable else FIXED for c, _, _ in ranked]
    y = np.arange(len(names))[::-1]

    ax.barh(y, shares, color=colors, zorder=3, height=0.62)
    for yi, share, (c, _, _) in zip(y, shares, ranked):
        tag = "" if c.adjustable else "  (purchased)"
        ax.text(share + max(shares) * 0.015, yi, f"{share:.1f}%{tag}",
                va="center", fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, fontfamily="monospace")
    ax.set_xlim(0, max(shares) * 1.28)
    ax.set_xlabel("share of total variance [%]", color=INK, fontsize=9)
    ax.set_title("Where the variation comes from", color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)

    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def comparison_png(stack: Stack, wc: WorstCase, rss_result: Statistical,
                   mc: MonteCarlo, out: Path) -> Path:
    """The three methods as intervals against the spec band."""
    fig, ax = plt.subplots(figsize=(9.2, 2.9))

    rows = [
        ("Worst case", wc.minimum, wc.maximum, FIXED),
        ("RSS (+/-3 sigma)", rss_result.mean - 3 * rss_result.sigma,
         rss_result.mean + 3 * rss_result.sigma, LINE),
        ("Monte Carlo (1-99%)", mc.percentiles[1], mc.percentiles[99], CENTRE),
    ]

    ax.axvspan(stack.lsl, stack.usl, color=CENTRE, alpha=0.10, zorder=1)
    ax.axvline(stack.lsl, color=LIMIT, linewidth=1.5, zorder=4)
    ax.axvline(stack.usl, color=LIMIT, linewidth=1.5, zorder=4)

    for i, (label, lo, hi, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=color, linewidth=7, solid_capstyle="butt", zorder=3)
        ax.text(lo, y + 0.28, f"{lo:+.3f}", fontsize=7.5, color=INK, ha="center")
        ax.text(hi, y + 0.28, f"{hi:+.3f}", fontsize=7.5, color=INK, ha="center")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _, _ in rows][::-1], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    span = max(wc.maximum, stack.usl) - min(wc.minimum, stack.lsl)
    ax.set_xlim(min(wc.minimum, stack.lsl) - span * 0.08,
                max(wc.maximum, stack.usl) + span * 0.08)
    ax.set_xlabel(f"{stack.closing_dimension} [{stack.unit}]  "
                  f"(green band = specification {stack.lsl} to {stack.usl})",
                  color=INK, fontsize=9)
    ax.set_title("Three methods, same stack", color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)

    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
