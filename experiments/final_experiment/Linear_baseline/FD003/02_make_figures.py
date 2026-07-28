"""
02_make_figures.py — Linear regression baseline, FD003: fleet-level
predicted-vs-true RUL scatter.

Identical in structure and rationale to
Linear_baseline/FD001/02_make_figures.py (see that script's module
docstring for why a single fleet scatter replaces the per-engine
trajectory panels PF/library scripts produce -- LinearRULBaseline has no
per-cycle trajectory or credible interval to plot).

Run from the repo root:
    python experiments/final_experiment/Linear_baseline/FD003/02_make_figures.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.baseline import LinearRULBaseline
from particle_twin.analysis import metrics as m

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD003"

CONFIG_JSON = "experiments/final_experiment/Linear_baseline/FD003/config.json"
FIG_DIR = "experiments/final_experiment/Linear_baseline/FD003/figures"


def _load_config() -> dict:
    with open(CONFIG_JSON) as f:
        return json.load(f)


def main():
    config = _load_config()
    if "exp_id" not in config:
        raise RuntimeError(f"{CONFIG_JSON} missing 'exp_id' -- run 01_run_experiment.py first.")

    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)
    all_engines = sorted(int(u) for u in test["unit_id"].unique())

    baseline = LinearRULBaseline(sensor_cols=SENSOR_COLS).fit(train)

    true_vals, pred_vals = [], []
    for engine_id in all_engines:
        engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
        true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])
        pred_val = baseline.predict_engine(engine_df)
        true_vals.append(true_val)
        pred_vals.append(pred_val)

    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    fleet_rmse = m.rmse(pred_vals, true_vals)
    fleet_mae = m.mae(pred_vals, true_vals)

    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_vals, pred_vals, s=30, color="steelblue", alpha=0.7,
               edgecolor="black", linewidth=0.3, label="Test engines")

    lo = min(true_vals.min(), pred_vals.min())
    hi = max(true_vals.max(), pred_vals.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray",
             linewidth=1.5, label="y = x (perfect prediction)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{DATASET} — Linear regression baseline: predicted vs. true RUL\n"
                 f"(fleet, n={len(all_engines)}, RMSE={fleet_rmse:.1f}, MAE={fleet_mae:.1f})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "rul_fleet_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
