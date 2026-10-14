"""
Visualization: geometry three-view, constraint dashboard, robustness figures.

Headless (matplotlib Agg backend) — safe on servers and WSL without displays.
Every function consumes already-computed results (no optimizer or NeuralFoil
work happens here), returns a matplotlib ``Figure``, and optionally saves it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never require a display

import matplotlib.pyplot as plt
import numpy as onp

from design_agent.constraints import ConstraintRecord
from design_agent.parameterization import DesignVector
from design_agent.uncertainty import RobustnessReport

_BAR_CAP = 3.0  # display cap for the margin-ratio axis
_PASS_COLOR = "#2ca02c"
_FAIL_COLOR = "#d62728"


def plot_geometry(design: DesignVector, path: str | Path | None = None):
    """Three-view drawing of the design's airplane."""
    airplane = design.to_airplane()
    axs = airplane.draw_three_view(show=False)
    fig = onp.ravel(axs)[0].get_figure()
    s_ref = design.span * design.root_chord * (1 + design.taper) / 2
    fig.suptitle(
        f"span {design.span:.1f} m · S {s_ref:.1f} m² · "
        f"S_h {design.htail_area:.1f} m² · S_v {design.vtail_area:.1f} m² · "
        f"P {design.power_w / 1e3:.0f} kW"
    )
    fig.tight_layout()
    return _maybe_save(fig, path)


def plot_constraints(
    records: list[ConstraintRecord],
    path: str | Path | None = None,
    title: str = "constraint verification",
):
    """Bar dashboard: margin ratio per constraint (≤ 1 passes), red/green.

    Ratio convention: upper → value/limit, lower → limit/value,
    equality → |value − limit| / |limit| (relative residual).
    Bars are capped at 3× for readability; the true ratio is annotated.
    """
    recs = sorted(records, key=lambda r: (r.passed, r.name))  # failures on top
    ratios = onp.array([_margin_ratio(r) for r in recs])
    shown = onp.minimum(ratios, _BAR_CAP)
    colors = [_PASS_COLOR if r.passed else _FAIL_COLOR for r in recs]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(recs) + 1.5)))
    ax.barh([r.name for r in recs], shown, color=colors)
    for i, ratio in enumerate(ratios):
        if ratio > _BAR_CAP:
            ax.text(_BAR_CAP + 0.03, i, f"{ratio:.2g}×", va="center", fontsize=8)
    ax.axvline(1.0, color="k", lw=1.0, ls="--", label="limit")
    ax.set_xlim(0, _BAR_CAP * 1.15)
    ax.set_xlabel("margin ratio (≤ 1 passes)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.text(
        0.01, 0.005,
        "upper: value/limit · lower: limit/value · equality: relative residual",
        fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _maybe_save(fig, path)


def plot_robustness(report: RobustnessReport, path: str | Path | None = None):
    """Two-panel Monte-Carlo figure: dispersed mass + constraint reliability."""
    fig, (axm, axp) = plt.subplots(1, 2, figsize=(12, 4.8))

    masses = onp.array(
        [s.mass_total_kg for s in report.samples], dtype=float
    )
    masses = masses[onp.isfinite(masses)]
    if masses.size:
        axm.hist(masses, bins=min(12, max(3, masses.size // 2)),
                 color="#1f77b4", alpha=0.8)
        mp = report.mass_percentiles()
        axm.axvline(mp["p50"], color="k", lw=1.5, label=f"p50 = {mp['p50']:.0f} kg")
        axm.axvline(mp["p5"], color="k", lw=1.0, ls="--", label="p5 / p95")
        axm.axvline(mp["p95"], color="k", lw=1.0, ls="--")
        axm.legend()
    axm.set_xlabel("gross mass [kg]")
    axm.set_ylabel("samples")
    axm.set_title(f"dispersed mass (n = {report.n_samples})")

    binding = report.binding_constraints()  # weakest first
    names = [n for n, _ in binding][::-1]  # barh plots bottom-up
    probs = [p for _, p in binding][::-1]
    axp.barh(names, probs,
             color=[plt.cm.RdYlGn(0.25 + 0.75 * p) for p in probs])
    axp.set_xlim(0, 1.0)
    axp.set_xlabel("pass probability")
    axp.set_title("constraint reliability (weakest on top)")
    for i, p in enumerate(probs):
        if p < 1.0:
            axp.text(min(p + 0.02, 0.9), i, f"{p:.2f}", va="center", fontsize=8)

    fig.suptitle(
        f"Monte-Carlo robustness — p(all constraints) = {report.p_success:.2f}"
    )
    fig.tight_layout()
    return _maybe_save(fig, path)


def _margin_ratio(r: ConstraintRecord) -> float:
    if r.kind == "upper":
        if r.limit > 0:
            return r.value / r.limit
        return r.value - r.limit
    if r.kind == "lower":
        if r.value > 0:
            return r.limit / r.value
        return float("inf") if r.limit > 0 else 1.0
    return abs(r.value - r.limit) / max(abs(r.limit), 1.0)  # equality residual


def _maybe_save(fig: Any, path: str | Path | None):
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
