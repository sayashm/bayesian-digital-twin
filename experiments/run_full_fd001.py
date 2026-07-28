"""
run_full_fd001.py — Day 5: full FD001 test-set run (all 100 engines)
========================================================================
Same model configuration as experiments/test_rul.py (Day 4), extended
from 4 engines to the full FD001 test split, with metrics.py (Day 5)
doing the RMSE / MAPE / ESS aggregation instead of ad-hoc inline code.

Two output paths, on purpose:
  1. results.db (via ResultsDB) — the normal path. Run this script from
     Sajjad's machine so this works; see Day 4's note in test_rul.py
     about the sandbox's SQLite locking limitation.
  2. experiments/fd001_full_results.json — ALWAYS written, regardless of
     whether the results.db write succeeds. This is what lets the full
     run happen once (the expensive part — PF + RUL forward simulation
     over 100 engines) and be reused: for the thesis figure, the
     hardest-engine analysis, and a quick local DB import
     (import_fd001_results.py) that doesn't need to redo the compute.

Run from the repo root:
    python experiments/run_full_fd001.py
"""

import json
import os
import sys
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.analysis.rul import extract_rul_trajectory, to_db_row
from particle_twin.analysis import metrics as m

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
MAX_HORIZON = 300  # see test_rul.py -- smaller caps censor before the model's
                     # own (already-too-shallow) curve can finish from healthy.

JSON_OUT = "experiments/fd001_full_results.json"


def _json_default(o):
    """rul.py's dicts carry numpy scalar types (np.int64 / np.float64 from
    entry['cycle_number'] and np.median/np.percentile); json.dump doesn't
    know about those, so coerce anything numpy-numeric to a plain Python
    scalar here rather than hunting down every call site."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data, fit the fleet-pooled model (identical setup to test_rul.py)
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train(DATASET)
test = loader.load_test(DATASET)
rul_true = loader.load_rul(DATASET)
all_engines = sorted(test['unit_id'].unique().tolist())
print(f"[OK] Loaded {DATASET} train ({len(train)} rows) / test ({len(test)} rows), "
      f"{len(all_engines)} test engines")

model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                          measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=0.05,
                          sensor_cols=SENSOR_COLS)
model.fit(train)
sigma_w = model.measurement_learner.params.get("sigma", float("nan"))
print(f"[OK] DegradationModel.fit() complete "
      f"(degradation R^2={model.degradation_learner.fit_quality['r_squared']:.3f}, sigma_w={sigma_w:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 2. Run filter + RUL extraction + metrics for every test engine
#
# Resumable by design: CHUNK_LIMIT (env var) caps how many NEW engines this
# invocation processes; already-done engines (present in JSON_OUT from a
# prior invocation) are skipped. This lets the full 100-engine run happen
# across several short calls instead of needing one long-running process
# (useful in sandboxes where background processes don't survive between
# tool calls).
# ══════════════════════════════════════════════════════════════════════════
CHUNK_LIMIT = int(os.environ.get("CHUNK_LIMIT", "0")) or None  # None = no cap, do all

engine_records = {}
if os.path.exists(JSON_OUT) and os.path.getsize(JSON_OUT) > 0:
    try:
        with open(JSON_OUT) as f:
            prior = json.load(f)
        # JSON keys are strings; normalise back to int engine ids
        engine_records = {int(k): v for k, v in prior.get('engines', {}).items()}
        print(f"[OK] Resuming: {len(engine_records)} engines already done in {JSON_OUT}")
    except json.JSONDecodeError:
        print(f"[WARN] {JSON_OUT} exists but isn't valid JSON (likely a crashed prior run) -- "
              f"starting fresh; it will be overwritten at the end of this run.")

todo = [uid for uid in all_engines if uid not in engine_records]
if CHUNK_LIMIT:
    todo = todo[:CHUNK_LIMIT]
print(f"[OK] Processing {len(todo)} engines this run ({len(all_engines) - len(engine_records) - len(todo)} left after)")

t_start = time.time()
skipped = []

for n, uid in enumerate(todo, start=1):
    engine_df = test[test['unit_id'] == uid]
    true_row = rul_true.loc[rul_true['unit_id'] == uid, 'true_rul'].values
    if len(true_row) == 0:
        skipped.append(uid)
        continue
    true_val = float(true_row[0])
    n_cycles = len(engine_df)

    t0 = time.time()
    pf = BootstrapPF(model, n_particles=N_PARTICLES, ess_threshold=0.5)
    pf.run(engine_df)
    trajectory = extract_rul_trajectory(pf, failure_threshold=FAILURE_THRESHOLD, max_horizon=MAX_HORIZON)
    runtime_s = time.time() - t0

    assert len(trajectory) == n_cycles, f"engine {uid}: trajectory length {len(trajectory)} != {n_cycles}"

    ev = m.evaluate_engine(trajectory, true_rul=true_val, n_cycles=n_cycles, n_particles=N_PARTICLES)
    ev['runtime_s'] = runtime_s

    engine_records[int(uid)] = {
        'true_rul': true_val,
        'n_cycles': n_cycles,
        'metrics': ev,
        'trajectory': trajectory,
    }

    if n % 5 == 0 or n == len(todo):
        elapsed = time.time() - t_start
        print(f"[chunk {n:>3}/{len(todo)}, total {len(engine_records):>3}/{len(all_engines)}] "
              f"engine {uid:>3}  n_cycles={n_cycles:>3}  final_err={ev['final_abs_error']:>7.1f}  "
              f"traj_rmse={ev['trajectory_rmse']:>7.1f}  elapsed={elapsed:>6.1f}s", flush=True)

if skipped:
    print(f"[WARN] {len(skipped)} engines had no RUL_FD001.txt entry, skipped: {skipped}")

fleet_evals = [rec['metrics'] for rec in engine_records.values()]
fleet_summary = m.evaluate_fleet(fleet_evals)
print("\n[OK] Fleet summary:", json.dumps(fleet_summary, indent=2))

# ══════════════════════════════════════════════════════════════════════════
# 3. Always write the portable JSON dump (sandbox-safe)
# ══════════════════════════════════════════════════════════════════════════
json_payload = {
    'config': {
        'dataset': DATASET, 'n_particles': N_PARTICLES, 'sigma_v': SIGMA_V, 'sigma_w': sigma_w,
        'failure_threshold': FAILURE_THRESHOLD, 'max_horizon': MAX_HORIZON,
        'hi_method': model.hi_method, 'degradation_model': model.degradation_model,
        'measurement_method': model.measurement_method,
        'degradation_r_squared': model.degradation_learner.fit_quality['r_squared'],
    },
    'fleet_summary': fleet_summary,
    'engines': engine_records,
}
with open(JSON_OUT, "w") as f:
    json.dump(json_payload, f, indent=2, default=_json_default)
print(f"[OK] Wrote portable results -> {JSON_OUT}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Try results.db (works on Sajjad's machine; sandbox SQLite is known-broken
#    -- see Day 4 note in test_rul.py -- so this is allowed to fail here).
#    Only attempted once every engine is done, to avoid partial experiment
#    rows when this script is run in resumable chunks.
# ══════════════════════════════════════════════════════════════════════════
if len(engine_records) < len(all_engines):
    print(f"[OK] {len(engine_records)}/{len(all_engines)} engines done -- "
          f"skipping results.db write until the full run is complete.")
    sys.exit(0)

try:
    db = ResultsDB("results.db")
    exp_id = db.create_experiment(
        dataset=DATASET, n_particles=N_PARTICLES, sigma_v=SIGMA_V, sigma_w=sigma_w,
        extra_params=json_payload['config'],
        description="Day 5 -- full FD001 test-set run (all engines), metrics.py",
    )
    for uid, rec in engine_records.items():
        db.register_engine(engine_id=uid, dataset=DATASET, split="test",
                            n_cycles=rec['n_cycles'], true_rul=rec['true_rul'])
        for estimate in rec['trajectory']:
            # to_db_row(): store_rul_timeseries() takes explicit kwargs (no
            # **kwargs), so a v2 estimate dict (rul_bayes, frac_censored, ...)
            # would otherwise raise TypeError on unpacking -- see rul.py's
            # "BACKWARD COMPATIBILITY" docstring section.
            db.store_rul_timeseries(exp_id=exp_id, engine_id=uid, dataset=DATASET,
                                     **to_db_row(estimate))
        db.commit()
        ev = rec['metrics']
        db.store_run_result(exp_id=exp_id, engine_id=uid, dataset=DATASET,
                             rmse=ev['final_abs_error'], mape=ev['final_pct_error'],
                             mean_ess=ev['mean_ess'], runtime_s=ev['runtime_s'])
    db.close()
    print(f"[OK] Stored full run in results.db (exp_id={exp_id})")
except Exception as exc:
    print(f"[WARN] results.db write failed ({exc!r}). This is the known sandbox SQLite "
          f"limitation from Day 4 if you're running inside Cowork's sandbox -- the JSON "
          f"dump above has everything; run import_fd001_results.py locally to populate "
          f"results.db without redoing the computation.")

print(f"\n[DONE] Total wall-clock time: {time.time() - t_start:.1f}s")
