"""
make_figures_fullfleet.py — Day 10/11 thesis figures for the similarity
library result (full 100-engine FD001 fleet).

Two figures:
1. fig_library_fullfleet_comparison.png — bar chart of mean final-cycle
   abs. RUL error (+/- 1 std) and total PHM08 score (log scale), pooled
   vs one-shot vs periodic, full fleet (experiments/day10/full_fleet_results.json).
2. fig_engine78_library_before_after.png — qualitative example on engine 78
   (already referenced in the thesis text as the ci_coverage=0 example,
   true_rul=107): health tracking + RUL trajectory, pooled-only vs one-shot,
   side by side, to visually show the fix.

Run from repo root: python make_figures_fullfleet.py
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from experiments.day10.similarity_library_prototype import (  # noqa: E402
    run_streaming_pf, _load_or_build_pooled_and_library, WARMUP_CYCLES, N_PARTICLES, SIGMA_V, K_TOP, MAX_HORIZON,
)
from particle_twin.analysis.rul import extract_rul_trajectory  # noqa: E402
from particle_twin.analysis.metrics import trajectory_ground_truth  # noqa: E402
from particle_twin.visualization.plots import plot_health_tracking, plot_rul_distribution  # noqa: E402

OUT_DIR = "thesis/Fig"


def fig1_comparison():
    with open("experiments/day10/full_fleet_results.json") as f:
        d = json.load(f)
    res = d["results"]
    variants = ["pooled_only", "one_shot", "periodic"]
    labels = ["Pooled\n(fleet-wide rate)", "One-shot\n(match once)", "Periodic\n(re-match every 20)"]
    abs_err = {v: np.array([res[uid][v]["final_abs_error"] for uid in res]) for v in variants}
    phm08_total = {v: sum(res[uid][v]["phm08_score"] for uid in res) for v in variants}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    means = [abs_err[v].mean() for v in variants]
    stds = [abs_err[v].std() for v in variants]
    colors = ["#a6a6a6", "#2c7fb8", "#41ab5d"]
    ax1.bar(labels, means, yerr=stds, capsize=4, color=colors)
    ax1.set_ylabel("Final-cycle RUL abs. error (cycles)")
    ax1.set_title("Mean abs. error, full 100-engine\nFD001 test fleet (±1 std)")
    for i, mv in enumerate(means):
        ax1.text(i, mv + stds[i] + 5, f"{mv:.0f}", ha="center", fontsize=9)

    totals = [phm08_total[v] for v in variants]
    ax2.bar(labels, totals, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("Total PHM08 score (log scale, lower is better)")
    ax2.set_title("Fleet PHM08 total score")
    for i, tv in enumerate(totals):
        ax2.text(i, tv * 1.3, f"{tv:.2g}", ha="center", fontsize=9)

    fig.suptitle("Pooled degradation rate vs. per-engine reference library, full FD001 fleet", fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_library_fullfleet_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"[OK] wrote {out}")


def fig2_engine_example(uid=78):
    pooled_model, library, test, rul_true = _load_or_build_pooled_and_library()
    engine_df = test[test["unit_id"] == uid].sort_values("cycle")
    n_cycles = len(engine_df)
    true_val = float(rul_true.loc[rul_true["unit_id"] == uid, "true_rul"].values[0])
    y_obs = pooled_model.observe(df=engine_df).to_numpy()
    cycles = engine_df["cycle"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for col, (variant_name, rematch_at) in enumerate([
        ("Pooled (fleet-wide rate)", []),
        ("One-shot (matched library rate)", [min(WARMUP_CYCLES, n_cycles - 1)]),
    ]):
        np.random.seed(42)
        pf, _ = run_streaming_pf(pooled_model, engine_df, library, rematch_at,
                                  n_particles=N_PARTICLES, sigma_v=SIGMA_V, k_top=K_TOP)
        traj = extract_rul_trajectory(pf, failure_threshold=0.0, max_horizon=MAX_HORIZON)

        particles_arr = np.array([h["particles"] for h in pf.history[1:]])  # skip t=0; already [0,1] scale
        p5 = np.percentile(particles_arr, 5, axis=1)
        p50 = np.percentile(particles_arr, 50, axis=1)
        p95 = np.percentile(particles_arr, 95, axis=1)

        plot_health_tracking(axes[0, col], cycles, y_obs, p5, p50, p95, obs_cycles=cycles)
        axes[0, col].set_title(variant_name)

        rul_cycles = np.array([e["cycle"] for e in traj])
        rul_median = np.array([e["rul_median"] for e in traj])
        rul_p5 = np.array([e["rul_p5"] for e in traj])
        rul_p95 = np.array([e["rul_p95"] for e in traj])
        plot_rul_distribution(axes[1, col], rul_cycles, rul_median, rul_p5, rul_p95, true_val, n_cycles)
        final_err = abs(rul_median[-1] - true_val)
        axes[1, col].set_title(f"Predicted RUL (final abs. error = {final_err:.0f} cycles)")

    fig.suptitle(f"Engine {uid} (true RUL = {true_val:.0f} cycles) — pooled rate vs. matched library rate", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"fig_engine{uid}_library_before_after.png")
    fig.savefig(out, dpi=150)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    fig1_comparison()
    fig2_engine_example(78)
