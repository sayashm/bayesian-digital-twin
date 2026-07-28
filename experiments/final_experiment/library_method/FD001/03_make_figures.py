"""
03_make_figures.py — Final experiment, step 3: per-engine 2x2 comparison
figure (health index + RUL, one-shot vs. periodic), read straight back
out of results.db (not from anything held in memory by step 2).

============================================================================
FIGURE LAYOUT (one PNG per FD001 test engine, 100 total)
============================================================================
                     Column 1: one-shot          Column 2: periodic
    Row 1:      predicted vs. actual        predicted vs. actual
    (Health)    Health Index                Health Index
    Row 2:      predicted vs. actual RUL    predicted vs. actual RUL
    (RUL)       (90% credible interval)     (90% credible interval)

Row 1 reuses particle_twin.visualization.plots.plot_health_tracking();
row 2 reuses plot_rul_distribution() -- both already validated,
thesis-quality building blocks (Day 5), not reimplemented here. Column 1
reads exp_id=config["one_shot_exp_id"], column 2 reads
config["periodic_exp_id"] -- both written by 02_run_experiment.py.

Everything plotted here comes from results.db (health_median/health_p5/
health_p95/rul_median/rul_p5/rul_p95 per cycle), EXCEPT the observed
Health Index line, which is deterministic given the fitted pooled model
and the raw test-engine sensor rows -- recomputed here rather than stored,
since it costs nothing (no particle filter involved) and doesn't depend
on which similarity strategy was used.

Run from the repo root:
    python experiments/final_experiment/03_make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.visualization.plots import plot_health_tracking, plot_rul_distribution

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
POOLED_SIGMA_V = 0.5
SIGMA_0 = 0.05

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/config.json"
FIG_DIR = "experiments/final_experiment/figures"

STRATEGY_LABELS = {"one_shot": "One-shot", "periodic": "Periodic"}


def _load_config() -> dict:
    with open(CONFIG_JSON) as f:
        return json.load(f)


def _engine_figure(engine_id: int, observed_hi, obs_cycles, true_rul: float, n_cycles: int,
                    strategy_rows: dict) -> plt.Figure:
    """
    Build the 2x2 comparison figure for one engine.

    strategy_rows : {"one_shot": [db rows...], "periodic": [db rows...]}
        Each list is get_rul_timeseries()'s output (already ordered by cycle).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")

    for col, strategy in enumerate(("one_shot", "periodic")):
        rows = strategy_rows[strategy]
        cycles = [r["cycle"] for r in rows]

        # -- Row 1: health index (predicted vs. observed) --
        plot_health_tracking(
            axes[0, col], cycles, observed_hi,
            particle_p5=[r["health_p5"] for r in rows],
            particle_p50=[r["health_median"] for r in rows],
            particle_p95=[r["health_p95"] for r in rows],
            obs_cycles=obs_cycles,
        )
        axes[0, col].set_title(f"{STRATEGY_LABELS[strategy]} — Health Index")

        # -- Row 2: RUL (predicted vs. implied ground truth) --
        plot_rul_distribution(
            axes[1, col], cycles,
            rul_median=[r["rul_median"] for r in rows],
            rul_p5=[r["rul_p5"] for r in rows],
            rul_p95=[r["rul_p95"] for r in rows],
            true_rul=true_rul, n_cycles=n_cycles,
        )
        axes[1, col].set_title(f"{STRATEGY_LABELS[strategy]} — RUL")

    fig.suptitle(f"Engine {engine_id} — health index & RUL, one-shot vs. periodic similarity matching "
                 f"(n_cycles={n_cycles}, true RUL at truncation={true_rul:.0f})", fontsize=13)
    plt.tight_layout()
    return fig


def main():
    config = _load_config()
    for key in ("library_exp_id", "one_shot_exp_id", "periodic_exp_id"):
        if key not in config:
            raise RuntimeError(f"config.json missing {key!r} -- run 01_build_library.py and "
                               f"02_run_experiment.py first.")

    strategy_exp_ids = {"one_shot": config["one_shot_exp_id"], "periodic": config["periodic_exp_id"]}

    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)
    all_engines = sorted(int(u) for u in test["unit_id"].unique())

    # Only the fitted HI builder is needed here (to recompute observed_hi) --
    # cheap, and avoids this figure step depending on the library's
    # degradation_learner (which the streaming filter swaps per engine).
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

        strategy_rows = {
            strategy: db.get_rul_timeseries(exp_id=exp_id, engine_id=engine_id, dataset=DATASET)
            for strategy, exp_id in strategy_exp_ids.items()
        }
        for strategy, rows in strategy_rows.items():
            if not rows:
                raise RuntimeError(f"engine {engine_id}: no rul_estimates rows for strategy="
                                   f"{strategy!r} (exp_id={strategy_exp_ids[strategy]}) -- did "
                                   f"02_run_experiment.py finish?")

        fig = _engine_figure(engine_id, observed_hi, obs_cycles, true_rul, n_cycles, strategy_rows)
        out_path = os.path.join(FIG_DIR, f"engine_{engine_id:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] engine {engine_id:>3} -> {out_path}")

    db.close()
    print(f"\n[OK] Wrote {len(all_engines)} figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()