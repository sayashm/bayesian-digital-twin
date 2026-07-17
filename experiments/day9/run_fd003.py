"""
run_fd003.py — Day 9: full FD003 test-set run (fault-mode robustness test)
=============================================================================
FD003 robustness check for the thesis's §4.1/§4.3 experimental setup: same
operating condition as FD001 (Sea Level, ONE condition) but TWO fault
modes instead of one (HPC degradation AND Fan degradation -- see
data/cmapss/readme.txt). This corrects an earlier mislabeling in
Thesis_claude.md / Thesis_timeline.md, which called FD003 a "multiple
operating conditions" test -- that description actually applies to
FD002/FD004 (six operating conditions each), which this pipeline does not
yet support (no operating-condition-aware normalisation in
health_index.py / degradation_learner.py). See Thesis_claude.md's Day 9
correction note for the full context.

Uses the EXACT SAME model configuration as the FD001 fleet run (exp_id=24
/ experiments/fd001_full_results.json) -- same hi_method, degradation
model shape, measurement noise model, N, sigma_v -- but re-FIT on FD003's
own training split (the fleet-pooled degradation/measurement models are
data-fit, not hardcoded, so they must be re-fit per dataset; see
chapt3.tex §3.1's fit-quality discussion). Keeping every other setting
identical to FD001 is what makes this a genuine robustness comparison
rather than a different experiment with confounded settings.

Same resumable-chunk design as run_full_fd001.py (CHUNK_LIMIT env var,
useful if this ever needs to run inside the Cowork sandbox in pieces),
same two output paths (results.db, best-effort; JSON dump, always
written), same three metric families: standard (rmse/mape/ess), PHM08
score, CI coverage -- computed inline this time (not a separate script
like run_fd001_metrics.py), since this is a fresh run rather than
reusing an existing one.

Run from the repo root:
    python experiments/day9/run_fd003.py
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

# ---- Identical to the FD001 config (run_full_fd001.py / chapt3.tex)
# ---- except the dataset itself -- see module docstring for why. ----
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD003"
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
MAX_HORIZON = 300

JSON_OUT = "experiments/day9/fd003_full_results.json"


def _json_default(o):
    """Same numpy-scalar coercion as run_full_fd001.py."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════════
# 1. Load data, fit the fleet-pooled model on FD003's own training split
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
print(f"[OK] DegradationModel.fit() complete on {DATASET} "
      f"(degradation R^2={model.degradation_learner.fit_quality['r_squared']:.3f}, sigma_w={sigma_w:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 2. Run filter + RUL extraction + metrics for every test engine
#    (resumable -- see run_full_fd001.py for the rationale)
# ══════════════════════════════════════════════════════════════════════════
CHUNK_LIMIT = int(os.environ.get("CHUNK_LIMIT", "0")) or None  # None = no cap, do all

engine_records = {}
if os.path.exists(JSON_OUT) and os.path.getsize(JSON_OUT) > 0:
    try:
        with open(JSON_OUT) as f:
            prior = json.load(f)
        engine_records = {int(k): v for k, v in prior.get('engines', {}).items()}
        print(f"[OK] Resuming: {len(engine_records)} engines already done in {JSON_OUT}")
    except json.JSONDecodeError:
        print(f"[WARN] {JSON_OUT} exists but isn't valid JSON (likely a crashed prior run) -- "
              f"starting fresh; it will be overwritten at the end of this run.")

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

    # Day 9 metrics, computed at the final observed cycle (same convention
    # as run_fd001_metrics.py, so FD001 vs. FD003 numbers are comparable).
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
              f"phm08={ev['phm08_score']:>6.2f}  covered={ev['ci_covered']}  elapsed={elapsed:>6.1f}s", flush=True)

if skipped:
    print(f"[WARN] {len(skipped)} engines had no RUL_{DATASET}.txt entry, skipped: {skipped}")

fleet_evals = [rec['metrics'] for rec in engine_records.values()]
fleet_summary = m.evaluate_fleet(fleet_evals)
fleet_summary['phm08_total_score'] = float(np.sum([e['phm08_score'] for e in fleet_evals]))
fleet_summary['ci_coverage'] = float(np.mean([e['ci_covered'] for e in fleet_evals]))
print("\n[OK] Fleet summary:", json.dumps(fleet_summary, indent=2))

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
        'note': 'Same config as FD001 exp_id=24 (see run_full_fd001.py), re-fit on FD003 train split.',
    },
    'fleet_summary': fleet_summary,
    'engines': engine_records,
}
with open(JSON_OUT, "w") as f:
    json.dump(json_payload, f, indent=2, default=_json_default)
print(f"[OK] Wrote portable results -> {JSON_OUT}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Try results.db (works on Sajjad's machine; sandbox SQLite is
#    known-broken -- see Day 4 note in test_rul.py -- so this is allowed
#    to fail here). Only attempted once every engine is done, to avoid
#    partial experiment rows when run in resumable chunks.
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
            'ci_coverage': fleet_summary['ci_coverage'],
        }},
        description="Day 9 -- full FD003 test-set run (fault-mode robustness test), same config as FD001 exp_id=24",
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
    print(f"[OK] Stored full FD003 run in results.db (exp_id={exp_id})")
except Exception as exc:
    print(f"[WARN] results.db write failed ({exc!r}). {JSON_OUT} has everything; "
          f"an import script analogous to import_fd001_results.py can be written if needed.")

print(f"\n[DONE] Total wall-clock time: {time.time() - t_start:.1f}s")
