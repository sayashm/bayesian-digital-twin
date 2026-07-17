"""
run_fd001_metrics.py — Day 9: PHM08 score + CI coverage for the FD001 fleet
=============================================================================
Adds the two Day-9 metrics (phm08_score/phm08_total_score, ci_coverage --
particle_twin/analysis/metrics.py) on top of the Day-5 full FD001 test-set
run, WITHOUT re-running the particle filter: the expensive part (PF + RUL
forward simulation over all 100 engines) already happened on Day 5 and is
saved in experiments/fd001_full_results.json (config: N=500, sigma_v=0.5,
weighted/exponential/gaussian -- the same config chapt3.tex documents).
Reusing it keeps this run's numbers identical to what the methodology
chapter already describes, and avoids redoing a 100-engine compute that's
already validated.

If experiments/fd001_full_results.json is missing or incomplete (e.g. a
fresh checkout with no prior run), this script falls back to computing
the missing engines itself, using the exact same model configuration, so
it is self-contained.

Two metrics computed at the FINAL observed cycle per engine (matching how
fleet_rmse / fleet_mape are already reported):
  - phm08_score / phm08_total_score : asymmetric PHM08 competition score
    (rul_median vs. true_rul)
  - ci_covered / ci_coverage : does the already-stored 90% credible
    interval [rul_p5, rul_p95] actually contain true_rul? (calibration
    check -- should land near 0.90 if well-calibrated)

Outputs
-------
  experiments/day9/fd001_day9_metrics.json  -- per-engine + fleet summary
  results.db, exp_id=24's extra_params      -- fleet-level day9 metrics
    merged in (best-effort; same sandbox-write caveat as Day 4/5 scripts).
    Deliberately NOT added as a new experiments/run_results row: exp_id=24
    already IS this exact 100-engine run, and creating a second row for
    the same compute is exactly the duplicate-experiment problem cleaned
    up on Day 7 (exp_id 7/23/24). Per-engine phm08/ci_covered are not
    added to run_results (its schema has no columns for them, and a
    schema migration wasn't part of today's scope) -- the full per-engine
    breakdown lives in the JSON output instead.

Run from the repo root:
    python experiments/day9/run_fd001_metrics.py
"""

import json
import os
import sqlite3

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.analysis.rul import extract_rul_trajectory
from particle_twin.analysis import metrics as m

np.random.seed(42)

# ---- Must match run_full_fd001.py exactly (Day 5), so any fallback
# ---- compute below reproduces results identical to what's in the DB. ----
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
MAX_HORIZON = 300

SOURCE_JSON = "experiments/fd001_full_results.json"    # Day 5 output (reused if present)
OUT_JSON = "experiments/day9/fd001_day9_metrics.json"   # this script's output
DB_PATH = "results.db"
EXISTING_EXP_ID = 24   # Day 5's full FD001 run, per Thesis_completing_Progress.md


def _json_default(o):
    """Same numpy-scalar coercion as run_full_fd001.py -- rul.py's dicts
    carry np.int64/np.float64, which json.dump doesn't know about."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════════
# 1. Load engine trajectories -- reuse Day 5's JSON if it has all engines,
#    otherwise compute whatever is missing (same model config as Day 5).
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir="data/cmapss")
test = loader.load_test(DATASET)
rul_true = loader.load_rul(DATASET)
all_engines = sorted(test['unit_id'].unique().tolist())

engine_records = {}
if os.path.exists(SOURCE_JSON):
    with open(SOURCE_JSON) as f:
        prior = json.load(f)
    engine_records = {int(k): v for k, v in prior.get('engines', {}).items()}
    print(f"[OK] Loaded {len(engine_records)} engines from {SOURCE_JSON}")

missing = [uid for uid in all_engines if uid not in engine_records]
if missing:
    print(f"[OK] {len(missing)} engines missing from {SOURCE_JSON} -- computing them now "
          f"(same config as Day 5: N={N_PARTICLES}, sigma_v={SIGMA_V}).")
    train = loader.load_train(DATASET)
    model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                              measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=0.05,
                              sensor_cols=SENSOR_COLS)
    model.fit(train)
    for uid in missing:
        engine_df = test[test['unit_id'] == uid]
        true_row = rul_true.loc[rul_true['unit_id'] == uid, 'true_rul'].values
        if len(true_row) == 0:
            continue
        true_val = float(true_row[0])
        n_cycles = len(engine_df)

        pf = BootstrapPF(model, n_particles=N_PARTICLES, ess_threshold=0.5)
        pf.run(engine_df)
        trajectory = extract_rul_trajectory(pf, failure_threshold=FAILURE_THRESHOLD, max_horizon=MAX_HORIZON)
        ev = m.evaluate_engine(trajectory, true_rul=true_val, n_cycles=n_cycles, n_particles=N_PARTICLES)
        engine_records[uid] = {'true_rul': true_val, 'n_cycles': n_cycles, 'metrics': ev, 'trajectory': trajectory}
        print(f"  computed engine {uid}")

assert len(engine_records) == len(all_engines), (
    f"only {len(engine_records)}/{len(all_engines)} engines available -- "
    f"check data/cmapss/ contains FD001, and {SOURCE_JSON} isn't corrupted."
)

# ══════════════════════════════════════════════════════════════════════════
# 2. Day 9 metrics: PHM08 score + CI coverage, at each engine's final cycle
# ══════════════════════════════════════════════════════════════════════════
final_preds, final_trues, final_lowers, final_uppers = [], [], [], []
per_engine_day9 = {}

for uid in all_engines:
    rec = engine_records[uid]
    final_entry = rec['trajectory'][-1]
    pred = final_entry['rul_median']
    true = rec['true_rul']
    lower = final_entry['rul_p5']
    upper = final_entry['rul_p95']

    score = float(m.phm08_score(np.array([pred]), np.array([true]))[0])
    covered = bool(lower <= true <= upper)

    per_engine_day9[uid] = {
        'final_rul_median': pred, 'true_rul': true,
        'rul_p5': lower, 'rul_p95': upper,
        'phm08_score': score, 'ci_covered': covered,
    }
    final_preds.append(pred)
    final_trues.append(true)
    final_lowers.append(lower)
    final_uppers.append(upper)

fleet_phm08_total = m.phm08_total_score(np.array(final_preds), np.array(final_trues))
fleet_ci_coverage = m.ci_coverage(np.array(final_trues), np.array(final_lowers), np.array(final_uppers))

print(f"\n[OK] FD001 fleet -- n_engines={len(all_engines)}")
print(f"     phm08_total_score = {fleet_phm08_total:.2f}")
print(f"     ci_coverage (90% interval, target ~0.90) = {fleet_ci_coverage:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Write output JSON
# ══════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
payload = {
    'config': {
        'dataset': DATASET, 'n_particles': N_PARTICLES, 'sigma_v': SIGMA_V,
        'failure_threshold': FAILURE_THRESHOLD, 'max_horizon': MAX_HORIZON,
        'source_json': SOURCE_JSON, 'existing_exp_id': EXISTING_EXP_ID,
    },
    'fleet_summary': {
        'n_engines': len(all_engines),
        'phm08_total_score': fleet_phm08_total,
        'ci_coverage': fleet_ci_coverage,
    },
    'per_engine': per_engine_day9,
}
with open(OUT_JSON, "w") as f:
    json.dump(payload, f, indent=2, default=_json_default)
print(f"[OK] Wrote {OUT_JSON}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Best-effort: merge fleet-level Day 9 metrics into results.db's
#    existing exp_id=24 row (extra_params JSON blob) rather than creating
#    a duplicate experiment row for the same 100-engine run.
# ══════════════════════════════════════════════════════════════════════════
try:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    row = conn.execute("SELECT extra_params FROM experiments WHERE exp_id = ?", (EXISTING_EXP_ID,)).fetchone()
    if row is None:
        print(f"[WARN] exp_id={EXISTING_EXP_ID} not found in {DB_PATH} -- skipping DB update. "
              f"(Results are still in {OUT_JSON}.)")
    else:
        extra = json.loads(row[0]) if row[0] else {}
        extra['day9_fleet_metrics'] = {
            'phm08_total_score': fleet_phm08_total,
            'ci_coverage': fleet_ci_coverage,
            'source': OUT_JSON,
        }
        conn.execute("UPDATE experiments SET extra_params = ? WHERE exp_id = ?",
                     (json.dumps(extra), EXISTING_EXP_ID))
        conn.commit()
        print(f"[OK] Merged day9_fleet_metrics into results.db exp_id={EXISTING_EXP_ID}.extra_params")
    conn.close()
except Exception as exc:
    print(f"[WARN] results.db update failed ({exc!r}) -- non-fatal, {OUT_JSON} has everything.")
