"""
run_fd001_linear.py — Day 9: FD001 full run with a LINEAR degradation model
=============================================================================
Tests whether the exponential mean-dynamics shape is the source of the
severe over-prediction bias found in exp_id=24 (fleet_rmse=244.0,
phm08_total~4.7e17, ci_coverage=0.000 -- see run_fd001_metrics.py). The
exponential fit needs ~960 cycles to reach failure from a healthy start
(Day 4 finding); a linear fit reaches failure sooner from the same
starting point, so if the exponential SHAPE (not sigma_v, not the
pooling-by-cycle approach itself) is the main driver of the bias, this
run should show a smaller bias.

Everything else is held IDENTICAL to the exp_id=24 / run_fd001_metrics.py
configuration -- same sensors, same N, same sigma_v, same HI method, same
measurement model -- so this is a clean one-variable comparison
(degradation_model: exponential -> linear), not a different experiment.

degradation_learner.py already supports degradation_model="linear" (see
models/degradation_learner.py) -- no new modelling code needed here, just
a fresh fit + fleet run with that option selected.

Same resumable-chunk design as run_full_fd001.py / run_fd003.py, same
three metric families (standard rmse/mape/ess, PHM08 score, CI coverage),
same two output paths (JSON always written; results.db best-effort).

Run from the repo root:
    python experiments/day9/run_fd001_linear.py
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
from particle_twin.analysis.rul import extract_rul_trajectory
from particle_twin.analysis import metrics as m

np.random.seed(42)

# ---- Identical to exp_id=24 / run_fd001_metrics.py except degradation_model. ----
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
MAX_HORIZON = 300
DEGRADATION_MODEL = "linear"   # <-- the one variable being changed vs. exp_id=24

JSON_OUT = "experiments/day9/fd001_linear_results.json"


def _json_default(o):
    """Same numpy-scalar coercion as the other Day 9 scripts."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════════
# 1. Load data, fit the fleet-pooled model with the LINEAR degradation shape
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train(DATASET)
test = loader.load_test(DATASET)
rul_true = loader.load_rul(DATASET)
all_engines = sorted(test['unit_id'].unique().tolist())
print(f"[OK] Loaded {DATASET} train ({len(train)} rows) / test ({len(test)} rows), "
      f"{len(all_engines)} test engines")

model = DegradationModel(hi_method="weighted", degradation_model=DEGRADATION_MODEL,
                          measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=0.05,
                          sensor_cols=SENSOR_COLS)
model.fit(train)
sigma_w = model.measurement_learner.params.get("sigma", float("nan"))
print(f"[OK] DegradationModel.fit() complete on {DATASET} (degradation_model={DEGRADATION_MODEL}) "
      f"(degradation R^2={model.degradation_learner.fit_quality['r_squared']:.3f}, sigma_w={sigma_w:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 2. Run filter + RUL extraction + metrics for every test engine (resumable)
# ══════════════════════════════════════════════════════════════════════════
CHUNK_LIMIT = int(os.environ.get("CHUNK_LIMIT", "0")) or None

engine_records = {}
if os.path.exists(JSON_OUT) and os.path.getsize(JSON_OUT) > 0:
    try:
        with open(JSON_OUT) as f:
            prior = json.load(f)
        engine_records = {int(k): v for k, v in prior.get('engines', {}).items()}
        print(f"[OK] Resuming: {len(engine_records)} engines already done in {JSON_OUT}")
    except json.JSONDecodeError:
        print(f"[WARN] {JSON_OUT} exists but isn't valid JSON -- starting fresh.")

todo = [uid for uid in all_engines if uid not in engine_records]
if CHUNK_LIMIT:
    todo = todo[:CHUNK_LIMIT]
print(f"[OK] Processing {len(todo)} engines this run "
      f"({len(all_engines) - len(engine_records) - len(todo)} left after)")

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

    final_entry = trajectory[-1]
    ev['phm08_score'] = float(m.phm08_score(np.array([final_entry['rul_median']]), np.array([true_val]))[0])
    ev['ci_covered'] = bool(final_entry['rul_p5'] <= true_val <= final_entry['rul_p95'])

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
              f"phm08={ev['phm08_score']:>10.2f}  covered={ev['ci_covered']}  elapsed={elapsed:>6.1f}s", flush=True)

if skipped:
    print(f"[WARN] {len(skipped)} engines had no RUL_{DATASET}.txt entry, skipped: {skipped}")

fleet_evals = [rec['metrics'] for rec in engine_records.values()]
fleet_summary = m.evaluate_fleet(fleet_evals)
fleet_summary['phm08_total_score'] = float(np.sum([e['phm08_score'] for e in fleet_evals]))
fleet_summary['phm08_median_score'] = float(np.median([e['phm08_score'] for e in fleet_evals]))
fleet_summary['ci_coverage'] = float(np.mean([e['ci_covered'] for e in fleet_evals]))
print("\n[OK] Fleet summary (linear degradation model):", json.dumps(fleet_summary, indent=2))
print(f"[compare] exp_id=24 (exponential): fleet_rmse=244.0, phm08_total~4.7e17, ci_coverage=0.000")

# ══════════════════════════════════════════════════════════════════════════
# 3. Always write the portable JSON dump (sandbox-safe)
# ══════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
json_payload = {
    'config': {
        'dataset': DATASET, 'n_particles': N_PARTICLES, 'sigma_v': SIGMA_V, 'sigma_w': sigma_w,
        'failure_threshold': FAILURE_THRESHOLD, 'max_horizon': MAX_HORIZON,
        'hi_method': model.hi_method, 'degradation_model': model.degradation_model,
        'measurement_method': model.measurement_method,
        'degradation_r_squared': model.degradation_learner.fit_quality['r_squared'],
        'note': 'Same config as FD001 exp_id=24 (run_full_fd001.py) except degradation_model=linear.',
    },
    'fleet_summary': fleet_summary,
    'engines': engine_records,
}
with open(JSON_OUT, "w") as f:
    json.dump(json_payload, f, indent=2, default=_json_default)
print(f"[OK] Wrote portable results -> {JSON_OUT}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Try results.db (best-effort; run locally, same as the other Day 9/5 scripts)
# ══════════════════════════════════════════════════════════════════════════
if len(engine_records) < len(all_engines):
    print(f"[OK] {len(engine_records)}/{len(all_engines)} engines done -- "
          f"skipping results.db write until the full run is complete.")
    sys.exit(0)

try:
    db = ResultsDB("results.db")
    exp_id = db.create_experiment(
        dataset=DATASET, n_particles=N_PARTICLES, sigma_v=SIGMA_V, sigma_w=sigma_w,
        extra_params={**json_payload['config'], 'day9_fleet_metrics': {
            'phm08_total_score': fleet_summary['phm08_total_score'],
            'phm08_median_score': fleet_summary['phm08_median_score'],
            'ci_coverage': fleet_summary['ci_coverage'],
        }},
        description="Day 9 -- FD001 full run, LINEAR degradation model (vs. exp_id=24's exponential), "
                     "same config otherwise -- testing whether the exponential shape drives the over-prediction bias",
    )
    for uid, rec in engine_records.items():
        db.register_engine(engine_id=uid, dataset=DATASET, split="test",
                            n_cycles=rec['n_cycles'], true_rul=rec['true_rul'])
        for estimate in rec['trajectory']:
            db.store_rul_timeseries(exp_id=exp_id, engine_id=uid, dataset=DATASET, **estimate)
        db.commit()
        ev = rec['metrics']
        db.store_run_result(exp_id=exp_id, engine_id=uid, dataset=DATASET,
                             rmse=ev['final_abs_error'], mape=ev['final_pct_error'],
                             mean_ess=ev['mean_ess'], runtime_s=ev['runtime_s'])
    db.close()
    print(f"[OK] Stored linear-degradation FD001 run in results.db (exp_id={exp_id})")
except Exception as exc:
    print(f"[WARN] results.db write failed ({exc!r}). {JSON_OUT} has everything.")

print(f"\n[DONE] Total wall-clock time: {time.time() - t_start:.1f}s")
