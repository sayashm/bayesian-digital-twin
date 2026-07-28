"""
03_make_tables.py — Linear regression baseline, FD001: fleet-level
summary table for RUL, matching PF_Fixed_Parameters/library_method's
table schema where applicable.

Same metric definitions as
PF_Fixed_Parameters/FD001/03_make_tables.py's TABLE 2 (particle_twin.
analysis.metrics), minus everything that requires a credible interval or
a per-cycle trajectory -- neither exists for a point-estimate OLS model:

  - no ci_coverage column (a point estimate has no interval to cover).
  - no trajectory_rmse column (LinearRULBaseline is not scored per
    cycle -- see 02_make_figures.py's docstring for why).

Refits LinearRULBaseline rather than reading predictions back from
results.db, same "cheap and deterministic -> just recompute" convention
as 02_make_figures.py.

OUTPUT
============================================================================
  - tables/rul_summary.csv     (1 row: fleet_rmse, fleet_mae, fleet_mape,
                                 phm08_total, n_engines)
  - tables/rul_per_engine.csv  (100 rows: engine_id, n_cycles, true_rul,
                                 pred_rul, final_abs_error)
  - summary table also printed to stdout.

Run from the repo root:
    python experiments/final_experiment/Linear_baseline/FD001/03_make_tables.py
"""

import json
import os

import numpy as np
import pandas as pd

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.baseline import LinearRULBaseline
from particle_twin.analysis import metrics as m

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"

CONFIG_JSON = "experiments/final_experiment/Linear_baseline/FD001/config.json"
TABLE_DIR = "experiments/final_experiment/Linear_baseline/FD001/tables"


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

    per_engine_rows = []
    for engine_id in all_engines:
        engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])
        pred_val = baseline.predict_engine(engine_df)
        final_abs_error = float(m.absolute_error(pred_val, true_val))

        per_engine_rows.append({
            "engine_id": engine_id, "n_cycles": n_cycles,
            "true_rul": true_val, "pred_rul": pred_val,
            "final_abs_error": final_abs_error,
        })

    true_vals = np.array([r["true_rul"] for r in per_engine_rows])
    pred_vals = np.array([r["pred_rul"] for r in per_engine_rows])

    summary_rows = [{
        "fleet_rmse": m.rmse(pred_vals, true_vals),
        "fleet_mae": m.mae(pred_vals, true_vals),
        "fleet_mape": m.mape(pred_vals, true_vals),
        "phm08_total": m.phm08_total_score(pred_vals, true_vals),
        "n_engines": len(all_engines),
    }]

    os.makedirs(TABLE_DIR, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLE_DIR, "rul_summary.csv"), index=False)
    pd.DataFrame(per_engine_rows).to_csv(
        os.path.join(TABLE_DIR, "rul_per_engine.csv"), index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 78)
    print(f"RUL summary — Linear regression baseline ({DATASET})")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print(f"\n[OK] Wrote {TABLE_DIR}/rul_summary.csv "
          f"(+ rul_per_engine.csv, {len(per_engine_rows)} rows)")


if __name__ == "__main__":
    main()
