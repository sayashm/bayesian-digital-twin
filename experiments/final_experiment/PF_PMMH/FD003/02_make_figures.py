"""
02_make_figures.py — PF PMMH experiment, FD003: per-engine health index +
RUL figure, read straight back out of results.db.

============================================================================
FIGURE LAYOUT (one PNG per FD003 test engine, 100 total)
============================================================================
    Row 1 (Health)   predicted vs. actual Health Index
    Row 2 (RUL)      predicted vs. actual RUL (90% credible interval)

Identical figure-building code to PF_Fixed_Parameters/FD003/02_make_figures.py
(particle_twin.visualization.plots.engine_summary_figure) -- ONE column,
because this experiment also applies a single pooled model to every test
engine (dynamics come from the PMMH library instead of OLS, but the
architecture -- one model, no per-engine matching -- is identical).

Everything plotted here comes from results.db (health_median/health_p5/
health_p95/rul_median/rul_p5/rul_p95 per cycle), EXCEPT the observed
Health Index line, which is deterministic given the fitted pooled model
and the raw test-engine sensor rows -- recomputed here rather than stored,
same convention as every other final_experiment figure step.

Run from the repo root:
    python experiments/final_experiment/PF_PMMH/FD003/02_make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.visualization.plots import engine_summary_figure

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD003"
POOLED_SIGMA_V = 0.5
SIGMA_0 = 0.05

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/PF_PMMH/FD003/config.json"
FIG_DIR = "experiments/final_experiment/PF_PMMH/FD003/figures"


def _load_config() -> dict:
    with open(CONFIG_JSON) as f:
        return json.load(f)


def main():
    config = _load_config()
    if "exp_id" not in config:
        raise RuntimeError(f"{CONFIG_JSON} missing 'exp_id' -- run 01_run_experiment.py first.")
    exp_id = config["exp_id"]

    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)
    all_engines = sorted(int(u) for u in test["unit_id"].unique())

    # Only the fitted HI builder is needed here (to recompute observed_hi) --
    # same pooled-model config as 01_run_experiment.py.
    pooled_model = DegradationModel(
        hi_method="weighted", degradation_model="exponential",
        measurement_method="gaussian", sigma_v=POOLED_SIGMA_V, sigma_0=SIGMA_0,
        sensor_cols=SENSOR_COLS,
    )
    pooled_model.fit(train_df=train)

    db = ResultsDB(DB_PATH)
    os.makedirs(FIG_DIR, exist_ok=True)

    for engine_id in all_engines:
        engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
        n_cycles = len(engine_df)
        true_rul = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])
        observed_hi = pooled_model.observe(engine_df).to_numpy()
        obs_cycles = engine_df["cycle"].to_numpy()

        rows = db.get_rul_timeseries(exp_id=exp_id, engine_id=engine_id, dataset=DATASET)
        if not rows:
            raise RuntimeError(f"engine {engine_id}: no rul_estimates rows for exp_id={exp_id} -- "
                               f"did 01_run_experiment.py finish?")
        cycles = [r["cycle"] for r in rows]

        fig = engine_summary_figure(
            engine_id, cycles, observed_hi,
            particle_p5=[r["health_p5"] for r in rows],
            particle_p50=[r["health_median"] for r in rows],
            particle_p95=[r["health_p95"] for r in rows],
            rul_cycles=cycles,
            rul_median=[r["rul_median"] for r in rows],
            rul_p5=[r["rul_p5"] for r in rows],
            rul_p95=[r["rul_p95"] for r in rows],
            true_rul=true_rul, n_cycles=n_cycles,
            obs_cycles=obs_cycles,
            extra_title=f"PMMH-calibrated pooled model — n_cycles={n_cycles}, "
                        f"true RUL at truncation={true_rul:.0f}",
        )
        out_path = os.path.join(FIG_DIR, f"engine_{engine_id:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] engine {engine_id:>3} -> {out_path}")

    db.close()
    print(f"\n[OK] Wrote {len(all_engines)} figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
