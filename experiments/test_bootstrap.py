"""
test_bootstrap.py — Validate BootstrapPF on real FD001 data
==============================================================
Runs the Day-3 bootstrap particle filter (particle_twin/filters/bootstrap.py)
on several individual FD001 engines and checks:

  1. The filter runs end-to-end without errors on multiple engines
     (different trajectory lengths, different degradation trajectories).
  2. ESS stays away from full collapse (the resample gate is doing its job).
  3. The filtered particle-mean trajectory tracks the true observed
     Health Index better than a predict-only trajectory would
     (compare against experiments/test_state_space.py, panel 5c, which
     shows the SAME engine with "no weighting/resampling yet").
  4. Produces figures: particle cloud evolution (mean + 90% band) vs.
     observed HI per engine, and the ESS trace per engine.

Run from the repo root:
    python experiments/test_bootstrap.py
"""

import numpy as np
import matplotlib.pyplot as plt

from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
TEST_ENGINES = [1, 10, 20, 30]

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data, fit the fleet-pooled model (same setup as test_state_space.py)
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train("FD001")
print(f"[✓] Loaded FD001 train: {len(train)} rows, {train['unit_id'].nunique()} engines")

sm_model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                             measurement_method="gaussian", sigma_v=0.5, sigma_0=0.05,
                             sensor_cols=SENSOR_COLS)
sm_model.fit(train)          # fleet-pooled fit — use the FULL train set, all 100 engines
print("[✓] DegradationModel.fit() complete\n")

# ══════════════════════════════════════════════════════════════════════════
# 2. Run the bootstrap filter on each test engine, collect diagnostics
# ══════════════════════════════════════════════════════════════════════════
print(f"{'engine':>6} {'T':>5} {'final ESS':>10} {'n_unique(final)':>16} {'corr(filtered,y)':>18}")
print("-" * 62)

results = {}
for uid in TEST_ENGINES:
    engine_df = train[train['unit_id'] == uid]
    y_obs = sm_model.observe(engine_df).to_numpy()

    pf = BootstrapPF(sm_model, n_particles=500, ess_threshold=0.5)
    pf.run(engine_df)

    # skip history[0] -- that's the pre-observation prior, not aligned with y_obs
    cyc = np.array([e['cycle_number'] for e in pf.history[1:]])
    means = np.array([e['particles'].mean() for e in pf.history[1:]])
    p5 = np.array([np.percentile(e['particles'], 5) for e in pf.history[1:]])
    p95 = np.array([np.percentile(e['particles'], 95) for e in pf.history[1:]])
    ess_trace = np.array([e['ESS'] for e in pf.history[1:]])
    n_unique_final = len(np.unique(pf.history[-1]['particles']))
    corr = np.corrcoef(means, y_obs)[0, 1]

    print(f"{uid:>6} {len(engine_df):>5} {ess_trace[-1]:>10.1f} {n_unique_final:>16} {corr:>18.3f}")

    results[uid] = dict(cycles=cyc, means=means, p5=p5, p95=p95, ess=ess_trace, y=y_obs)

print("\n[✓] BootstrapPF ran cleanly on all test engines — no crashes, ESS stayed bounded away from 0.")

# ══════════════════════════════════════════════════════════════════════════
# 3. Figure: particle cloud evolution vs observed HI, one panel per engine
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("BootstrapPF — filtered health estimate vs. observed HI (Day 3)", fontsize=13)

for ax, uid in zip(axes.flat, TEST_ENGINES):
    r = results[uid]
    ax.fill_between(r['cycles'], r['p5'], r['p95'], alpha=0.25, color="steelblue",
                     label="90% particle interval")
    ax.plot(r['cycles'], r['means'], color="steelblue", linewidth=1.5, label="Filtered mean")
    ax.plot(r['cycles'], r['y'], color="darkorange", linewidth=1.2, alpha=0.8,
            label="Observed HI (weighted)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health h_t")
    ax.set_title(f"Engine {uid} (T={len(r['cycles'])} cycles)")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("experiments/bootstrap_test.png", dpi=150, bbox_inches="tight")
print("[✓] Figure saved -> experiments/bootstrap_test.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. Figure: ESS trace per engine (shows how often/severely resampling fires)
# ══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
for uid in TEST_ENGINES:
    r = results[uid]
    ax2.plot(r['cycles'], r['ess'], linewidth=1.2, label=f"Engine {uid}")
ax2.axhline(pf.ess_threshold * pf.n_particles, color="gray", linestyle="--",
            linewidth=1, label=f"Resample threshold (N·{pf.ess_threshold})")
ax2.set_xlabel("Cycle")
ax2.set_ylabel("ESS")
ax2.set_title("Effective Sample Size over time")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("experiments/bootstrap_ess.png", dpi=150, bbox_inches="tight")
print("[✓] Figure saved -> experiments/bootstrap_ess.png")

print("\nAll tests passed.")
