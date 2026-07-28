"""
01_run_experiment.py — Linear regression baseline, FD001: fit
particle_twin.models.baseline.LinearRULBaseline (ordinary least squares,
pooled across the training fleet) and predict one point RUL estimate per
test engine from its last observed cycle.

============================================================================
WHY THIS RE-RUN EXISTS
============================================================================
thesis/chapt4.tex's Table~\\ref{tab:baseline-comparison} (Section
"Results: Comparison with Baseline") carried an explicit caveat: the
baseline numbers on record (RMSE 32.0 / MAPE 0.53 / PHM08 16,121) were
"computed against the earlier RUL definition and have not yet been re-run
on the current RUL target used for the particle-filter results" -- flagged
as "an open item for the final revision". This script is that re-run.

Concretely: nothing about LinearRULBaseline itself changed (it does not
call particle_twin.analysis.rul at all -- it is a flat OLS regression,
no state-space model, no filtering). What changed since the number 32.0
was computed is the RUL v2 rewrite of rul.py (calibrated failure
threshold, threshold uncertainty, rul_cap=350) that PF_Fixed_Parameters/,
PF_PMMH/, and library_method/ now all evaluate against. The ONE thing
that must match for the baseline to be "directly comparable to the final
results" is the ground-truth RUL target used for scoring, i.e.
CMAPSSLoader.load_rul(DATASET) -- the exact same RUL_FD001.txt read by
every sibling final_experiment/ script (e.g.
PF_Fixed_Parameters/FD001/01_run_experiment.py, line
"rul_true = loader.load_rul(DATASET)"). This script uses that same call,
so the "old RUL target" caveat no longer applies once this is run.
rul.py itself is not imported or modified here.

============================================================================
WHAT THIS SCRIPT DOES
============================================================================
  1. Loads FD001 train/test/rul via CMAPSSLoader (data_dir="data/cmapss"),
     identical to every other final_experiment/ script.
  2. Fits ONE LinearRULBaseline on the pooled training fleet, using the
     SAME ten sensors PF_Fixed_Parameters/PF_PMMH use for their health
     index (SENSOR_COLS below) -- a fair comparison, per baseline.py's
     own docstring.
  3. For every FD001 test engine, predicts a single point RUL from that
     engine's LAST observed cycle (LinearRULBaseline.predict_engine) --
     matching exactly what the particle filter sees (data up to the last
     observed cycle, nothing from the future).
  4. Registers the engines + per-engine result rows in results.db under a
     fresh exp_id, for the same provenance/queryability every other
     final_experiment result has (thesis/chapt4.tex's Reproducibility
     paragraph: "every experiment reported in this chapter is stored in
     results.db").

No particle filter, no health index, no credible interval -- so unlike
the PF/library scripts, run_results' n_particles/sigma_v/sigma_w columns
do not apply to this model; they are stored as 0/0.0/0.0 placeholders,
with a note in extra_params. run_results.rmse is schema-documented as
"RMSE of final RUL estimate vs. true RUL" -- for a single point that
formula reduces exactly to |pred - true|, so storing the per-engine
absolute error there is not a repurposing, it is what the column already
means for a one-number-per-engine model.

Deliberately NOT done here (see particle_twin/models/baseline.py's
module docstring for the "why", Day 9 decision): no piecewise-linear RUL
capping of the training target, and no clipping of negative predictions
at test time. Both are honest, reportable properties of a flat linear
model, not bugs to silently patch in this re-run.

Same seed (42) as every other final_experiment/ script, for consistency
with the documented reproducibility convention -- though OLS itself
(numpy.linalg.lstsq under the hood) is deterministic given fixed data, so
the seed has no actual effect on this particular model's output.

OUTPUT
============================================================================
  - results.db: one experiments row (exp_id), 100 engines rows, 100
    run_results rows.
  - experiments/final_experiment/Linear_baseline/FD001/config.json:
    records the exp_id for 02_make_figures.py / 03_make_tables.py.

Run from the repo root:
    python experiments/final_experiment/Linear_baseline/FD001/01_run_experiment.py
"""

import json
import os
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.baseline import LinearRULBaseline
from particle_twin.analysis import metrics as m

# ---------------------------------------------------------------------------
# Configuration -- SENSOR_COLS matches PF_Fixed_Parameters/FD001 and
# PF_PMMH/FD001 exactly (the "same ten selected sensors" the health index
# is built from), so the baseline sees identical information to the PF.
# ---------------------------------------------------------------------------
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
SEED = 42

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/Linear_baseline/FD001/config.json"


def _load_config() -> dict:
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def main():
    np.random.seed(SEED)
    config = _load_config()

    # ------------------------------------------------------------------
    # 1. Load data, fit the ONE pooled OLS model used for every test
    #    engine -- same loader/dataset/train-test split as every PF and
    #    library final_experiment script.
    # ------------------------------------------------------------------
    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)
    all_test_engines = sorted(int(u) for u in test["unit_id"].unique())
    print(f"[OK] Loaded {DATASET}: {len(train)} train rows, {len(test)} test rows, "
          f"{len(all_test_engines)} test engines")

    t0 = time.perf_counter()
    baseline = LinearRULBaseline(sensor_cols=SENSOR_COLS).fit(train)
    fit_time = time.perf_counter() - t0
    print(f"[OK] LinearRULBaseline fit on pooled training fleet ({fit_time:.3f}s)")

    # ------------------------------------------------------------------
    # 2. One dedicated exp_id -- created once, reused across resumed runs.
    # ------------------------------------------------------------------
    db = ResultsDB(DB_PATH)

    if "exp_id" in config:
        exp_id = config["exp_id"]
        print(f"[OK] Resuming existing experiment, exp_id={exp_id}")
    else:
        exp_id = db.create_experiment(
            dataset=DATASET, n_particles=0, sigma_v=0.0, sigma_w=0.0,
            extra_params={
                "step": "final_experiment/Linear_baseline/FD001/01_run_experiment",
                "model": "LinearRULBaseline", "sensor_cols": SENSOR_COLS, "seed": SEED,
                "note": "Point-estimate OLS baseline -- no state-space model, no "
                        "filtering, no credible interval. n_particles/sigma_v/sigma_w "
                        "are not applicable and stored as 0/0.0/0.0 placeholders. "
                        "Evaluated against the same RUL_FD001.txt ground truth "
                        "(loader.load_rul) used by PF_Fixed_Parameters/PF_PMMH/"
                        "library_method, closing thesis/chapt4.tex's 'matched re-run' "
                        "caveat on Table tab:baseline-comparison.",
            },
            description="Final experiment (Linear regression baseline, FD001) — OLS "
                        "pooled across the training fleet, same ten sensors as the PF, "
                        "one point RUL prediction per test engine from its last "
                        "observed cycle, no state-space model / filtering / interval.",
        )
        config["exp_id"] = exp_id
        _save_config(config)
        print(f"[OK] Registered experiment: exp_id={exp_id}")

    # ------------------------------------------------------------------
    # 3. Predict for every test engine, store engine + per-engine result.
    # ------------------------------------------------------------------
    for engine_id in all_test_engines:
        engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])

        pred_val = baseline.predict_engine(engine_df)
        abs_error = float(m.absolute_error(pred_val, true_val))
        pct_error = float(m.percentage_error(pred_val, true_val))

        db.register_engine(engine_id=engine_id, dataset=DATASET, split="test",
                            n_cycles=n_cycles, true_rul=true_val)
        db.store_run_result(exp_id=exp_id, engine_id=engine_id, dataset=DATASET,
                             rmse=abs_error, mape=pct_error, mean_ess=None, runtime_s=None)

    db.commit()
    db.close()
    print(f"[OK] Stored {len(all_test_engines)} engine predictions under exp_id={exp_id}")
    print(f"[OK] Done. exp_id={exp_id}")


if __name__ == "__main__":
    main()
