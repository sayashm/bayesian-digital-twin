"""
plots.py — Thesis-quality figures
===================================
Reusable plotting functions, factored out of the Day-5 one-off figure
script (experiments/make_figure_engine77.py) so later figures (Chapter 4:
RMSE comparison across engines, ESS timeseries, baseline comparison) can
reuse the same two building blocks instead of copy-pasting matplotlib
calls.

Two functions cover everything Day 5 needed:
  plot_health_tracking()   -- observed HI vs filtered posterior (top panel
                               of the engine-summary figure)
  plot_rul_distribution()  -- predicted RUL vs implied ground truth
                               (bottom panel)
  engine_summary_figure()  -- combines both into the 2-panel figure used
                               for one representative engine.

Each plotting function takes a matplotlib Axes to draw on (rather than
creating its own figure), so callers can compose multi-panel layouts --
e.g. a grid of several engines -- without this module knowing about grid
layout at all.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from particle_twin.analysis.metrics import trajectory_ground_truth


def plot_health_tracking(ax, cycles, observed_hi, particle_p5, particle_p50, particle_p95,
                          obs_cycles=None):
    """
    Draw the health-tracking panel: raw observed Health Index vs the
    particle filter's posterior median with a 5th-95th percentile band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    cycles : array -- cycle number for each filter step (len = n_cycles).
    observed_hi : array -- raw sensor-derived HI, [0,1] scale.
    particle_p5, particle_p50, particle_p95 : arrays -- particle
        percentiles at each cycle, [0,1] scale.
    obs_cycles : array or None -- cycle numbers for observed_hi, if they
        don't line up 1:1 with `cycles` (defaults to `cycles`).
    """
    obs_cycles = cycles if obs_cycles is None else obs_cycles
    ax.scatter(obs_cycles, observed_hi, s=10, color="gray", alpha=0.5,
               label="Observed HI (sensor-derived)")
    ax.fill_between(cycles, particle_p5, particle_p95, color="steelblue", alpha=0.25,
                     label="Particle 5th-95th pct.")
    ax.plot(cycles, particle_p50, color="steelblue", linewidth=1.8,
            label="Posterior median health")
    ax.axhline(0.0, color="firebrick", linestyle=":", linewidth=1, label="Failure threshold")
    ax.set_ylabel("Health Index (0 = failed, 1 = healthy)")
    ax.set_title("Filtered health state vs. raw observations")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.05, 1.05)
    return ax


def plot_rul_distribution(ax, cycles, rul_median, rul_p5, rul_p95, true_rul, n_cycles):
    """
    Draw the RUL-distribution panel: predicted median + 90% credible
    interval vs the implied ground-truth countdown.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    cycles, rul_median, rul_p5, rul_p95 : arrays -- one entry per observed
        cycle (e.g. straight from rul.extract_rul_trajectory()).
    true_rul : float -- ground truth from RUL_FD001.txt (at truncation).
    n_cycles : int -- number of observed cycles for this engine.
    """
    ground_truth = trajectory_ground_truth(cycles, true_rul, n_cycles)
    ax.fill_between(cycles, rul_p5, rul_p95, color="darkorange", alpha=0.2,
                     label="90% credible interval")
    ax.plot(cycles, rul_median, color="darkorange", linewidth=1.8, label="Predicted median RUL")
    ax.plot(cycles, ground_truth, color="black", linewidth=1.5, linestyle="--",
            label="Implied true RUL")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("Predicted RUL distribution vs. implied ground truth")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def engine_summary_figure(engine_id, cycles, observed_hi, particle_p5, particle_p50, particle_p95,
                           rul_cycles, rul_median, rul_p5, rul_p95, true_rul, n_cycles,
                           obs_cycles=None, extra_title: str = "", figsize=(9, 8)):
    """
    Build the full 2-panel engine-summary figure (health tracking on top,
    RUL distribution below), sharing the x-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure -- caller is responsible for savefig().
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    title = f"Engine {engine_id} — health tracking and RUL prediction"
    if extra_title:
        title += f"\n{extra_title}"
    fig.suptitle(title, fontsize=12)

    plot_health_tracking(ax1, cycles, observed_hi, particle_p5, particle_p50, particle_p95,
                          obs_cycles=obs_cycles)
    plot_rul_distribution(ax2, rul_cycles, rul_median, rul_p5, rul_p95, true_rul, n_cycles)

    plt.tight_layout()
    return fig
