"""
04_make_tables.py — FD003 final experiment, step 4: fleet-level summary
tables for one-shot vs. periodic similarity matching, one table for
Health Index and one for RUL, both read back from results.db. Same
script as experiments/final_experiment/04_make_tables.py, repointed at
the FD003 pipeline's own config.json/tables directory.

============================================================================
WHAT "FLEET-LEVEL" MEANS FOR EACH TABLE, AND WHY THE GRANULARITY DIFFERS
============================================================================
Health Index table (TRAJECTORY-level metrics):
    Health Index has an "actual" value at EVERY observed cycle (the
    sensor-derived observation, particle_twin.models.state_space.
    DegradationModel.observe()) -- there is no single external ground
    truth the way RUL_FD001.txt gives RUL, so the natural per-engine
    metric is the full-trajectory RMSE/MAE (predicted median vs. observed
    HI at every cycle), then averaged across the fleet. CI coverage is
    computed by POOLING every (engine, cycle) point together (~13,000
    points), not per-engine -- with this many points per engine, pooling
    gives a much more stable coverage estimate than one point per engine.

RUL table (FINAL-CYCLE metrics, matching every prior RUL number in this
    project's session logs and thesis chapters):
    RUL_FD001.txt gives exactly ONE ground-truth value per engine, at
    the final (truncated) observed cycle -- so fleet_rmse/fleet_mae/
    fleet_mape/ci_coverage/phm08_total are all computed from ONE point
    per engine (the final-cycle prediction), via
    particle_twin.analysis.metrics.evaluate_engine/evaluate_fleet, EXACTLY
    as Day 5/9/11's fleet numbers were computed -- so these numbers are
    directly comparable to day11_fullfleet_decision.md's table. Trajectory
    RMSE (mean/median across the fleet) is also included as a secondary
    column, same dual reporting convention metrics.py already documents.

OUTPUT
============================================================================
  - experiments/final_experiment/library_method/FD003/tables/health_index_summary.csv  (2 rows: one_shot, periodic)
  - experiments/final_experiment/library_method/FD003/tables/rul_summary.csv            (2 rows: one_shot, periodic)
  - experiments/final_experiment/library_method/FD003/tables/health_index_per_engine.csv (100 rows x 2 strategies, supplementary detail)
  - experiments/final_experiment/library_method/FD003/tables/rul_per_engine.csv          (100 rows x 2 strategies, supplementary detail)
  - both summary tables also printed to stdout.

Run from the repo root (requires FD003/01_build_library.py and
FD003/02_run_experiment.py to have completed first):
    python experiments/final_experiment/library_method/FD003/04_make_tables.py
"""

import json
import os

import numpy as np
import pandas as pd

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.analysis import metrics as m

SENSOR_COLS = ['T24', 'T30', 'T50', 'Ps30', 'BPR', 'htBleed', 'W31', 'W32']
DATASET = "FD003"
POOLED_SIGMA_V = 0.5
SIGMA_0 = 0.05
DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/library_method/FD003/config.json"
TABLE_DIR = "experiments/final_experiment/library_method/FD003/tables"

STRATEGIES = ["one_shot", "periodic"]


def _load_config() -> dict:
    with open(CONFIG_JSON) as f:
        return json.load(f)


def main():
    config = _load_config()
    for key in ("library_exp_id", "one_shot_exp_id", "periodic_exp_id"):
        if key not in config:
            raise RuntimeError(f"config.json missing {key!r} -- run 01_build_library.py and "
                               f"02_run_experiment.py first.")
    strategy_exp_ids = {"one_shot": config["one_shot_exp_id"], "periodic": config["periodic_exp_id"]}

    # Observed HI is deterministic given the fitted pooled model + raw
    # sensor rows -- recomputed here (cheap, no particle filter), exactly
    # as 03_make_figures.py does, so both scripts stay consistent without
    # needing to persist a third copy of the same numbers.
    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)
    all_engines = sorted(int(u) for u in test["unit_id"].unique())

    pooled_model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                                     measurement_method="gaussian", sigma_v=POOLED_SIGMA_V,
                                     sigma_0=SIGMA_0, sensor_cols=SENSOR_COLS)
    pooled_model.fit(train_df=train)

    db = ResultsDB(DB_PATH)

    health_summary_rows = []
    health_per_engine_rows = []
    rul_summary_rows = []
    rul_per_engine_rows = []

    for strategy in STRATEGIES:
        exp_id = strategy_exp_ids[strategy]

        # -------------------- Health Index (trajectory-level) --------------------
        all_pred, all_true, all_p5, all_p95 = [], [], [], []
        for engine_id in all_engines:
            rows = db.get_rul_timeseries(exp_id=exp_id, engine_id=engine_id, dataset=DATASET)
            engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
            observed_hi = pooled_model.observe(engine_df).to_numpy()

            pred = np.array([r["health_median"] for r in rows])
            p5 = np.array([r["health_p5"] for r in rows])
            p95 = np.array([r["health_p95"] for r in rows])
            assert len(pred) == len(observed_hi), (
                f"engine {engine_id}: {len(pred)} predicted health cycles vs. "
                f"{len(observed_hi)} observed -- should always match 1:1."
            )

            engine_rmse = m.rmse(pred, observed_hi)
            engine_mae = m.mae(pred, observed_hi)
            engine_coverage = m.ci_coverage(observed_hi, p5, p95)
            health_per_engine_rows.append({
                "strategy": strategy, "engine_id": engine_id, "n_cycles": len(pred),
                "health_rmse": engine_rmse, "health_mae": engine_mae, "health_ci_coverage": engine_coverage,
            })

            all_pred.append(pred); all_true.append(observed_hi)
            all_p5.append(p5); all_p95.append(p95)

        pooled_pred = np.concatenate(all_pred)
        pooled_true = np.concatenate(all_true)
        pooled_p5 = np.concatenate(all_p5)
        pooled_p95 = np.concatenate(all_p95)
        per_engine_rmse = [r["health_rmse"] for r in health_per_engine_rows if r["strategy"] == strategy]
        per_engine_mae = [r["health_mae"] for r in health_per_engine_rows if r["strategy"] == strategy]

        health_summary_rows.append({
            "strategy": strategy,
            "mean_trajectory_rmse": float(np.mean(per_engine_rmse)),
            "median_trajectory_rmse": float(np.median(per_engine_rmse)),
            "mean_trajectory_mae": float(np.mean(per_engine_mae)),
            "median_trajectory_mae": float(np.median(per_engine_mae)),
            "pooled_rmse": m.rmse(pooled_pred, pooled_true),
            "pooled_mae": m.mae(pooled_pred, pooled_true),
            "ci_coverage": m.ci_coverage(pooled_true, pooled_p5, pooled_p95),
            "n_engines": len(all_engines), "n_cycles_total": len(pooled_pred),
        })

        # -------------------- RUL (final-cycle, same convention as every --------
        # -------------------- prior fleet number in this project) --------------
        engine_evals = []
        for engine_id in all_engines:
            rows = db.get_rul_timeseries(exp_id=exp_id, engine_id=engine_id, dataset=DATASET)
            n_cycles = len(rows)
            true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])
            ev = m.evaluate_engine(rows, true_rul=true_val, n_cycles=n_cycles, n_particles=500)
            ev["final_rul_p5"] = rows[-1]["rul_p5"]
            ev["final_rul_p95"] = rows[-1]["rul_p95"]
            engine_evals.append(ev)
            rul_per_engine_rows.append({
                "strategy": strategy, "engine_id": engine_id, "n_cycles": n_cycles,
                "true_rul": true_val, "final_rul_median": ev["final_rul_median"],
                "final_abs_error": ev["final_abs_error"], "trajectory_rmse": ev["trajectory_rmse"],
            })

        fleet = m.evaluate_fleet(engine_evals)
        final_preds = np.array([e["final_rul_median"] for e in engine_evals])
        final_trues = np.array([e["true_rul"] for e in engine_evals])
        final_p5 = np.array([e["final_rul_p5"] for e in engine_evals])
        final_p95 = np.array([e["final_rul_p95"] for e in engine_evals])

        rul_summary_rows.append({
            "strategy": strategy,
            "fleet_rmse": fleet["fleet_rmse"],
            "fleet_mae": m.mae(final_preds, final_trues),
            "fleet_mape": fleet["fleet_mape"],
            "mean_trajectory_rmse": fleet["mean_trajectory_rmse"],
            "median_trajectory_rmse": fleet["median_trajectory_rmse"],
            "ci_coverage": m.ci_coverage(final_trues, final_p5, final_p95),
            "phm08_total": m.phm08_total_score(final_preds, final_trues),
            "n_engines": fleet["n_engines"],
        })

    db.close()

    # ---------------------------------------------------------------------
    # Write + print
    # ---------------------------------------------------------------------
    os.makedirs(TABLE_DIR, exist_ok=True)

    health_df = pd.DataFrame(health_summary_rows).set_index("strategy")
    rul_df = pd.DataFrame(rul_summary_rows).set_index("strategy")
    health_df.to_csv(os.path.join(TABLE_DIR, "health_index_summary.csv"))
    rul_df.to_csv(os.path.join(TABLE_DIR, "rul_summary.csv"))
    pd.DataFrame(health_per_engine_rows).to_csv(
        os.path.join(TABLE_DIR, "health_index_per_engine.csv"), index=False)
    pd.DataFrame(rul_per_engine_rows).to_csv(
        os.path.join(TABLE_DIR, "rul_per_engine.csv"), index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 78)
    print("TABLE 1 — Health Index (trajectory-level, pooled across all cycles/engines)")
    print("=" * 78)
    print(health_df.to_string())
    print(f"\n[OK] Wrote {TABLE_DIR}/health_index_summary.csv "
          f"(+ health_index_per_engine.csv, {len(health_per_engine_rows)} rows)")

    print()
    print("=" * 78)
    print("TABLE 2 — RUL (final-cycle, FD003, same convention as day11_fullfleet_decision.md)")
    print("=" * 78)
    print(rul_df.to_string())
    print(f"\n[OK] Wrote {TABLE_DIR}/rul_summary.csv "
          f"(+ rul_per_engine.csv, {len(rul_per_engine_rows)} rows)")


if __name__ == "__main__":
    main()