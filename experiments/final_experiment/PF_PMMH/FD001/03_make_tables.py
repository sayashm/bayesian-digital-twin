"""
03_make_tables.py — PF PMMH experiment, FD001: fleet-level summary tables
for Health Index and RUL, read back from results.db.

Same table schema, columns, and metric definitions as
PF_Fixed_Parameters/FD001/03_make_tables.py / library_method/FD001/
04_make_tables.py's TABLE 1/TABLE 2 (see those scripts' module docstrings
for the trajectory-level-vs-final-cycle granularity rationale) -- a
"strategy" column is kept, holding the single value "pmmh_pooled" (one
PMMH-informed pooled parameter set applied to the whole fleet, no
per-engine matching), so this table can be concatenated directly with
PF_Fixed_Parameters' and library_method's for a side-by-side comparison.

OUTPUT
============================================================================
  - tables/health_index_summary.csv     (1 row: pmmh_pooled)
  - tables/rul_summary.csv               (1 row: pmmh_pooled)
  - tables/health_index_per_engine.csv   (100 rows, supplementary detail)
  - tables/rul_per_engine.csv            (100 rows, supplementary detail)
  - both summary tables also printed to stdout.

Run from the repo root:
    python experiments/final_experiment/PF_PMMH/FD001/03_make_tables.py
"""

import json
import os

import numpy as np
import pandas as pd

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.analysis import metrics as m

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
POOLED_SIGMA_V = 0.5
SIGMA_0 = 0.05
DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/PF_PMMH/FD001/config.json"
TABLE_DIR = "experiments/final_experiment/PF_PMMH/FD001/tables"

STRATEGY = "pmmh_pooled"


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

    pooled_model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                                     measurement_method="gaussian", sigma_v=POOLED_SIGMA_V,
                                     sigma_0=SIGMA_0, sensor_cols=SENSOR_COLS)
    pooled_model.fit(train_df=train)

    db = ResultsDB(DB_PATH)

    # -------------------- Health Index (trajectory-level) --------------------
    all_pred, all_true, all_p5, all_p95 = [], [], [], []
    health_per_engine_rows = []
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
            "strategy": STRATEGY, "engine_id": engine_id, "n_cycles": len(pred),
            "health_rmse": engine_rmse, "health_mae": engine_mae, "health_ci_coverage": engine_coverage,
        })

        all_pred.append(pred); all_true.append(observed_hi)
        all_p5.append(p5); all_p95.append(p95)

    pooled_pred = np.concatenate(all_pred)
    pooled_true = np.concatenate(all_true)
    pooled_p5 = np.concatenate(all_p5)
    pooled_p95 = np.concatenate(all_p95)
    per_engine_rmse = [r["health_rmse"] for r in health_per_engine_rows]
    per_engine_mae = [r["health_mae"] for r in health_per_engine_rows]

    health_summary_rows = [{
        "strategy": STRATEGY,
        "mean_trajectory_rmse": float(np.mean(per_engine_rmse)),
        "median_trajectory_rmse": float(np.median(per_engine_rmse)),
        "mean_trajectory_mae": float(np.mean(per_engine_mae)),
        "median_trajectory_mae": float(np.median(per_engine_mae)),
        "pooled_rmse": m.rmse(pooled_pred, pooled_true),
        "pooled_mae": m.mae(pooled_pred, pooled_true),
        "ci_coverage": m.ci_coverage(pooled_true, pooled_p5, pooled_p95),
        "n_engines": len(all_engines), "n_cycles_total": len(pooled_pred),
    }]

    # -------------------- RUL (final-cycle, same convention as every -----
    # -------------------- prior fleet number in this project) -----------
    engine_evals = []
    rul_per_engine_rows = []
    for engine_id in all_engines:
        rows = db.get_rul_timeseries(exp_id=exp_id, engine_id=engine_id, dataset=DATASET)
        n_cycles = len(rows)
        true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])
        ev = m.evaluate_engine(rows, true_rul=true_val, n_cycles=n_cycles, n_particles=500)
        ev["final_rul_p5"] = rows[-1]["rul_p5"]
        ev["final_rul_p95"] = rows[-1]["rul_p95"]
        engine_evals.append(ev)
        rul_per_engine_rows.append({
            "strategy": STRATEGY, "engine_id": engine_id, "n_cycles": n_cycles,
            "true_rul": true_val, "final_rul_median": ev["final_rul_median"],
            "final_abs_error": ev["final_abs_error"], "trajectory_rmse": ev["trajectory_rmse"],
        })

    fleet = m.evaluate_fleet(engine_evals)
    final_preds = np.array([e["final_rul_median"] for e in engine_evals])
    final_trues = np.array([e["true_rul"] for e in engine_evals])
    final_p5 = np.array([e["final_rul_p5"] for e in engine_evals])
    final_p95 = np.array([e["final_rul_p95"] for e in engine_evals])

    rul_summary_rows = [{
        "strategy": STRATEGY,
        "fleet_rmse": fleet["fleet_rmse"],
        "fleet_mae": m.mae(final_preds, final_trues),
        "fleet_mape": fleet["fleet_mape"],
        "mean_trajectory_rmse": fleet["mean_trajectory_rmse"],
        "median_trajectory_rmse": fleet["median_trajectory_rmse"],
        "ci_coverage": m.ci_coverage(final_trues, final_p5, final_p95),
        "phm08_total": m.phm08_total_score(final_preds, final_trues),
        "n_engines": fleet["n_engines"],
    }]

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
    print("TABLE 2 — RUL (final-cycle, PMMH-calibrated pooled PF)")
    print("=" * 78)
    print(rul_df.to_string())
    print(f"\n[OK] Wrote {TABLE_DIR}/rul_summary.csv "
          f"(+ rul_per_engine.csv, {len(rul_per_engine_rows)} rows)")


if __name__ == "__main__":
    main()
