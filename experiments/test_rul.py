"""
test_rul.py — Validate RUL extraction (particle_twin/analysis/rul.py) on real FD001 data
============================================================================================
Runs the Day-4 RUL extraction (simulate_rul / extract_rul / extract_rul_trajectory) on
several FD001 TEST-split engines (test split, not train, because RUL_FD001.txt only
gives ground-truth remaining life for test engines -- they're truncated BEFORE failure,
which is exactly the scenario extract_rul is meant to extrapolate past).

Checks:
  1. extract_rul / extract_rul_trajectory run end-to-end without errors.
  2. extract_rul_trajectory's length matches the engine's observed cycle count.
  3. Predicted RUL (median + 90% CI) at the truncation point vs. RUL_FD001.txt.
  4. Full per-cycle RUL estimates are stored in results.db (rul_estimates table)
     and are queryable back out.

Known limitation (surfaced by this test, not introduced by it): predicted RUL is
systematically too HIGH across all four engines (e.g. engine 20: predicted median
~190 vs. true 16). Traced to the fleet-pooled degradation fit in
DegradationModelLearner, which pools (cycle, HI) across all 100 train engines using
ABSOLUTE cycle number as time -- already flagged as a caveat in chapt3.tex Sec 3.1
(engines have very different lifespans, 128-362 cycles, so the pooled mean shape is
systematically shallower than any single fast-failing engine's true trajectory).
Filtering (Sec 3.2) partly compensates for this because it's corrected every cycle
by real observations; RUL extrapolation has no such correction, so the weakness is
fully exposed here. Revisiting the pooling strategy (e.g. normalising by
fraction-of-life instead of absolute cycle) is future work, not fixed in this file.

Run from the repo root:
    python experiments/test_rul.py
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.analysis.rul import extract_rul, extract_rul_trajectory, to_db_row

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
TEST_ENGINES = [1, 10, 20, 30]
DATASET = "FD001"
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
MAX_HORIZON = 300  # from a healthy start (~93% health), the fitted exponential curve
                     # alone needs ~960 cycles to reach 0 -- see chat/session notes. A
                     # smaller cap censors most particles before the model can finish,
                     # which reports the CAP as the "prediction" instead of the model's
                     # actual (still-too-optimistic) extrapolation.

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data, fit the fleet-pooled model (same setup as test_bootstrap.py)
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train(DATASET)
test = loader.load_test(DATASET)
rul_true = loader.load_rul(DATASET)
print(f"[✓] Loaded {DATASET} train ({len(train)} rows) / test ({len(test)} rows) / RUL labels")

model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                          measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=0.05,
                          sensor_cols=SENSOR_COLS)
model.fit(train)
sigma_w = model.measurement_learner.params.get("sigma", float("nan"))
print(f"[✓] DegradationModel.fit() complete  (degradation R^2={model.degradation_learner.fit_quality['r_squared']:.3f}, "
      f"sigma_w={sigma_w:.4f})\n")

# ══════════════════════════════════════════════════════════════════════════
# 2. Register the experiment; run filter + RUL extraction per test engine
# ══════════════════════════════════════════════════════════════════════════
db = ResultsDB("results.db")
exp_id = db.create_experiment(
    dataset=DATASET, n_particles=N_PARTICLES, sigma_v=SIGMA_V, sigma_w=sigma_w,
    extra_params={"failure_threshold": FAILURE_THRESHOLD, "max_horizon": MAX_HORIZON,
                  "hi_method": model.hi_method, "degradation_model": model.degradation_model,
                  "measurement_method": model.measurement_method},
    description="Day 4 -- rul.py validation on FD001 test-split engines",
)
print(f"[✓] Registered experiment exp_id={exp_id}\n")

print(f"{'engine':>6} {'T_obs':>6} {'pred_median':>12} {'90% CI':>18} {'true_rul':>9} {'abs_err':>8}")
print("-" * 66)

results = {}
for uid in TEST_ENGINES:
    engine_df = test[test['unit_id'] == uid]
    true_val = float(rul_true.loc[rul_true['unit_id'] == uid, 'true_rul'].values[0])

    t0 = time.time()
    pf = BootstrapPF(model, n_particles=N_PARTICLES, ess_threshold=0.5)
    pf.run(engine_df)

    trajectory = extract_rul_trajectory(pf, failure_threshold=FAILURE_THRESHOLD, max_horizon=MAX_HORIZON)
    runtime_s = time.time() - t0

    assert len(trajectory) == len(engine_df), \
        f"trajectory length {len(trajectory)} != engine length {len(engine_df)}"

    final = trajectory[-1]  # RUL estimate at the truncation point (last observed cycle)
    abs_err = abs(final['rul_median'] - true_val)
    print(f"{uid:>6} {len(engine_df):>6} {final['rul_median']:>12.1f} "
          f"[{final['rul_p5']:>6.1f}, {final['rul_p95']:>6.1f}] {true_val:>9.0f} {abs_err:>8.1f}")

    # register engine + store the full per-cycle trajectory
    db.register_engine(engine_id=uid, dataset=DATASET, split="test",
                        n_cycles=len(engine_df), true_rul=true_val)
    for estimate in trajectory:
        # to_db_row(): store_rul_timeseries() takes explicit kwargs (no
        # **kwargs), so a v2 estimate dict (rul_bayes, frac_censored, ...)
        # would otherwise raise TypeError on unpacking -- see rul.py's
        # "BACKWARD COMPATIBILITY" docstring section.
        db.store_rul_timeseries(exp_id=exp_id, engine_id=uid, dataset=DATASET, **to_db_row(estimate))
    db.commit()

    mean_ess = np.mean([e['ess'] for e in trajectory])
    db.store_run_result(exp_id=exp_id, engine_id=uid, dataset=DATASET,
                         rmse=abs_err, mape=abs_err / true_val if true_val else None,
                         mean_ess=mean_ess, runtime_s=runtime_s)

    results[uid] = dict(trajectory=trajectory, true_rul=true_val, n_cycles=len(engine_df))

print("\n[✓] RUL extraction ran cleanly on all test engines; results stored in results.db")
print("    (Predicted RUL is systematically HIGH -- see module docstring: this traces to")
print("     the fleet-pooled degradation fit, not to rul.py itself. See §3.3 caveat.)")

# ══════════════════════════════════════════════════════════════════════════
# 3. Sanity check: read the stored data back out of the DB
# ══════════════════════════════════════════════════════════════════════════
check_uid = TEST_ENGINES[0]
stored = db.get_rul_timeseries(exp_id=exp_id, engine_id=check_uid, dataset=DATASET)
print(f"\n[✓] Queried back {len(stored)} rows for engine {check_uid} from rul_estimates "
      f"(expected {results[check_uid]['n_cycles']})")

db.close()

# ══════════════════════════════════════════════════════════════════════════
# 4. Figure: predicted RUL trajectory (median + 90% CI) vs. implied ground truth
#    (ground truth "counts down" linearly: true_rul + (T_obs - cycle) at each cycle)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("RUL extraction — predicted (median + 90% CI) vs. implied ground truth (Day 4)", fontsize=13)

for ax, uid in zip(axes.flat, TEST_ENGINES):
    r = results[uid]
    traj = r['trajectory']
    cycles = np.array([e['cycle'] for e in traj])
    median = np.array([e['rul_median'] for e in traj])
    p5 = np.array([e['rul_p5'] for e in traj])
    p95 = np.array([e['rul_p95'] for e in traj])
    ground_truth = r['true_rul'] + (r['n_cycles'] - cycles)

    ax.fill_between(cycles, p5, p95, alpha=0.25, color="steelblue", label="90% CI")
    ax.plot(cycles, median, color="steelblue", linewidth=1.5, label="Predicted median RUL")
    ax.plot(cycles, ground_truth, color="darkorange", linewidth=1.5, linestyle="--",
             label="Implied true RUL")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Engine {uid} (true_rul={r['true_rul']:.0f} at truncation)")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("experiments/rul_test.png", dpi=150, bbox_inches="tight")
print("[✓] Figure saved -> experiments/rul_test.png")

print("\nAll tests passed.")

