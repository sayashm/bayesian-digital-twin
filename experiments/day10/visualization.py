"""
visualization.py — per-engine RUL prediction vs. true RUL, with 90% CI.

Reads exp_id=24 (Day 5's full 100-engine FD001 run, pooled degradation
model) from results.db and, for every engine, draws predicted median RUL
+ the 90% credible interval (rul_p5/rul_p95) against the implied
ground-truth countdown, using the existing plot_rul_distribution()
building block from particle_twin/visualization/plots.py rather than
duplicating any plotting logic.

Writes one PNG per engine into particle_twin/visualization/ (e.g.
rul_engine_001.png ... rul_engine_100.png).

Run from the repo root:
    python experiments/day10/visualization.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_twin.data.database import ResultsDB
from particle_twin.visualization.plots import plot_rul_distribution

EXP_ID = 24
DATASET = "FD001"
OUT_DIR = os.path.join("particle_twin", "visualization")


def main():
    db = ResultsDB("results.db")

    engine_ids = [
        row["engine_id"]
        for row in db._conn.execute(
            "SELECT DISTINCT engine_id FROM rul_estimates WHERE exp_id = ? AND dataset = ? "
            "ORDER BY engine_id",
            (EXP_ID, DATASET),
        ).fetchall()
    ]
    print(f"[OK] exp_id={EXP_ID}: {len(engine_ids)} engines with stored RUL trajectories")

    os.makedirs(OUT_DIR, exist_ok=True)

    for engine_id in engine_ids:
        engine_row = db._conn.execute(
            "SELECT n_cycles, true_rul FROM engines WHERE engine_id = ? AND dataset = ? AND split = 'test'",
            (engine_id, DATASET),
        ).fetchone()
        n_cycles = engine_row["n_cycles"]
        true_rul = engine_row["true_rul"]

        trajectory = db.get_rul_timeseries(exp_id=EXP_ID, engine_id=engine_id, dataset=DATASET)
        cycles = [r["cycle"] for r in trajectory]
        rul_median = [r["rul_median"] for r in trajectory]
        rul_p5 = [r["rul_p5"] for r in trajectory]
        rul_p95 = [r["rul_p95"] for r in trajectory]

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_rul_distribution(ax, cycles, rul_median, rul_p5, rul_p95, true_rul, n_cycles)
        fig.suptitle(f"Engine {engine_id} — RUL prediction vs. true RUL (exp_id={EXP_ID})", fontsize=12)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"rul_engine_{engine_id:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  [OK] engine {engine_id:>3}: n_cycles={n_cycles:>4}  true_rul={true_rul:>6.1f}  -> {out_path}")

    db.close()
    print(f"\n[OK] Wrote {len(engine_ids)} figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()