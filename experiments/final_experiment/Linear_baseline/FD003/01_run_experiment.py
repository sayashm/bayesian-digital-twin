"""
01_run_experiment.py — Linear regression baseline, FD003: fit
particle_twin.models.baseline.LinearRULBaseline (ordinary least squares,
pooled across the training fleet) and predict one point RUL estimate per
test engine from its last observed cycle.

Identical in structure and rationale to
Linear_baseline/FD001/01_run_experiment.py (see that script's module
docstring for the full "why this re-run exists" background: closing
thesis/chapt4.tex's "matched re-run" caveat on
Table~\\ref{tab:baseline-comparison} by scoring against the same
RUL_FD003.txt ground truth -- via CMAPSSLoader.load_rul -- that
PF_Fixed_Parameters/FD003, PF_PMMH/FD003, and library_method/FD003 all
evaluate against). Kept as its own copy rather than a shared import so
this directory is self-contained, same convention as every other
FD001/FD003 pair in final_experiment/ (e.g. PF_Fixed_Parameters/FD001 vs.
FD003, library_method/FD001 vs. FD003).

SENSOR_COLS is the SAME ten-sensor list as FD001 -- matching
PF_Fixed_Parameters/FD003/01_run_experiment.py and
PF_PMMH/FD003/01_run_experiment.py (both use the full ten), not
library_method/FD003's reduced eight-sensor set (that reduction is
specific to the similarity-matching/MMD step and is out of scope here --
the task is to match the PF's ten-sensor health index, per baseline.py's
own docstring).

rul.py is not imported or modified here -- LinearRULBaseline has no
state-space model, so nothing in the v2 RUL-extraction rewrite applies to
it directly; only the ground-truth target it is scored against needs to
match.

Run from the repo root:
    python experiments/final_experiment/Linear_baseline/FD003/01_run_experiment.py
"""

import json
import os
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.baseline import LinearRULBaseline
from particle_twin.analysis import metrics as m

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD003"
SEED = 42

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/Linear_baseline/FD003/config.json"


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

    db = ResultsDB(DB_PATH)

    if "exp_id" in config:
        exp_id = config["exp_id"]
        print(f"[OK] Resuming existing experiment, exp_id={exp_id}")
    else:
        exp_id = db.create_experiment(
            dataset=DATASET, n_particles=0, sigma_v=0.0, sigma_w=0.0,
            extra_params={
                "step": "final_experiment/Linear_baseline/FD003/01_run_experiment",
                "model": "LinearRULBaseline", "sensor_cols": SENSOR_COLS, "seed": SEED,
                "note": "Point-estimate OLS baseline -- no state-space model, no "
                        "filtering, no credible interval. n_particles/sigma_v/sigma_w "
                        "are not applicable and stored as 0/0.0/0.0 placeholders. "
                        "Evaluated against the same RUL_FD003.txt ground truth "
                        "(loader.load_rul) used by PF_Fixed_Parameters/PF_PMMH/"
                        "library_method, closing thesis/chapt4.tex's 'matched re-run' "
                        "caveat on Table tab:baseline-comparison.",
            },
            description="Final experiment (Linear regression baseline, FD003) — OLS "
                        "pooled across the training fleet, same ten sensors as the PF, "
                        "one point RUL prediction per test engine from its last "
                        "observed cycle, no state-space model / filtering / interval.",
        )
        config["exp_id"] = exp_id
        _save_config(config)
        print(f"[OK] Registered experiment: exp_id={exp_id}")

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
