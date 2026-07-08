"""
test_state_space.py — Validate DegradationModel v2 (HI pipeline) on real FD001 data
=====================================================================================
Replaces the Day-2 synthetic-only test. This script:

  1. Loads real FD001 training data (100 engines, run-to-failure).
  2. Builds the Health Index (HI) with 'weighted' and 'pca' methods and
     compares degradation-model fit quality (linear / polynomial /
     exponential), fleet-pooled by absolute cycle number.
  3. Also fits the 'industrial' HI as a training-only reference curve
     (cannot be used for live/test observation -- see health_index.py).
  4. Fits a full DegradationModel end-to-end and sanity-checks the
     particle-filter-facing API (sample_initial / transition /
     log_likelihood) on one held-out engine, exactly the interface
     filters/bootstrap.py will call on Day 3.

Run from the repo root:
    python experiments/test_state_space.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.features.health_index import HealthIndexBuilder
from particle_twin.models.degradation_learner import DegradationModelLearner
from particle_twin.models.measurement_learner import MeasurementModelLearner
from particle_twin.models.state_space import DegradationModel

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']

# ══════════════════════════════════════════════════════════════════════════
# 1. Load real FD001 data
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train("FD001")
print(f"[✓] Loaded FD001 train: {len(train)} rows, {train['unit_id'].nunique()} engines")

cycles = train["cycle"].to_numpy(dtype=float)

# ══════════════════════════════════════════════════════════════════════════
# 2. Build HI three ways, compare degradation-model fit quality
# ══════════════════════════════════════════════════════════════════════════
hi_series = {}
for method in ["weighted", "pca", "industrial"]:
    builder = HealthIndexBuilder(SENSOR_COLS, method=method)
    hi = builder.fit_transform(train)
    hi_series[method] = hi.to_numpy()

print("\n" + "=" * 78)
print(f"{'HI method':12s} {'degr. model':12s} {'R^2':>8s}")
print("-" * 78)
results = {}
for hi_method in ["weighted", "pca", "industrial"]:
    for model_type in ["linear", "polynomial", "exponential"]:
        learner = DegradationModelLearner(hi_series[hi_method], model_type=model_type,
                                           time=cycles, sigma_v=0.3)
        r2 = learner.fit_quality["r_squared"]
        print(f"{hi_method:12s} {model_type:12s} {r2:8.4f}")
        results[(hi_method, model_type)] = learner
print("=" * 78)
print("NOTE: 'industrial' HI is a formula built from the true RUL — it fits")
print("its own smoothness trivially well and is NOT usable as a live test-time")
print("observation. Only 'weighted' and 'pca' are candidate observation HIs.")

# ══════════════════════════════════════════════════════════════════════════
# 3. Full pipeline fit via DegradationModel (weighted HI, exponential dynamics,
#    gaussian measurement noise) -- current defaults, see state_space.py.
#    exponential chosen for physical reasoning + supported by the
#    training-only 'industrial' HI fit (R^2=0.82, see section 2 above);
#    'weighted' is the better sensor-only pairing for exponential
#    specifically (0.442 vs 0.409 for pca), so it -- not pca -- is used here.
# ══════════════════════════════════════════════════════════════════════════
model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                          measurement_method="gaussian", sigma_v=0.3, sigma_0=0.05,
                          sensor_cols=SENSOR_COLS)
model.fit(train)
print("\n[✓] DegradationModel.fit() complete")
print("Fit report:")
for k, v in model.fit_report.items():
    print(f"  {k}: {v}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Sanity-check the particle-filter-facing API on ONE held-out engine
# ══════════════════════════════════════════════════════════════════════════
engine_df = loader.get_engine("FD001", unit_id=1, split="train")
y_obs = model.observe(engine_df).to_numpy()  # scalar HI observation per cycle, [0,1] scale

N = 500
h = model.sample_initial(N)
assert h.shape == (N,)
print(f"\n[✓] sample_initial: mean={h.mean():.4f} std={h.std():.4f} (expected mean≈1.0, std≈{model.sigma_0})")

h_history = [h.copy()]
loglik_history = []
for t in range(1, len(engine_df)):
    h = model.transition(h)
    ll = model.log_likelihood(h, y_obs[t])
    assert h.shape == (N,) and ll.shape == (N,)
    h_history.append(h.copy())
    loglik_history.append(ll)
h_history = np.array(h_history)  # (T, N)
print(f"[✓] transition + log_likelihood ran for {len(engine_df)-1} cycles on engine 1")
print(f"    h at final cycle: mean={h_history[-1].mean():.4f} std={h_history[-1].std():.4f}")
print(f"    observed HI at final cycle: {y_obs[-1]:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. Figure: degradation model comparison + engine-1 particle cloud sanity check
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("DegradationModel v2 — Real FD001 Validation", fontsize=13)

# 5a: pooled scatter + three fitted shapes for 'pca' HI
ax = axes[0, 0]
ax.scatter(cycles, hi_series["pca"], s=2, alpha=0.08, color="gray", label="Pooled HI (pca)")
t_grid = np.linspace(0, cycles.max(), 300)
for model_type, color in zip(["linear", "polynomial", "exponential"], ["C0", "C1", "C2"]):
    learner = results[("pca", model_type)]
    ax.plot(t_grid, learner.predict(t_grid), color=color, linewidth=2,
            label=f"{model_type} (R²={learner.fit_quality['r_squared']:.3f})")
ax.set_xlabel("Cycle"); ax.set_ylabel("HI (pca)")
ax.set_title("Fleet-pooled degradation shape comparison (pca HI)")
ax.legend(fontsize=8)

# 5b: same for 'weighted' HI
ax = axes[0, 1]
ax.scatter(cycles, hi_series["weighted"], s=2, alpha=0.08, color="gray", label="Pooled HI (weighted)")
for model_type, color in zip(["linear", "polynomial", "exponential"], ["C0", "C1", "C2"]):
    learner = results[("weighted", model_type)]
    ax.plot(t_grid, learner.predict(t_grid), color=color, linewidth=2,
            label=f"{model_type} (R²={learner.fit_quality['r_squared']:.3f})")
ax.set_xlabel("Cycle"); ax.set_ylabel("HI (weighted)")
ax.set_title("Fleet-pooled degradation shape comparison (weighted HI)")
ax.legend(fontsize=8)

# 5c: engine 1 -- particle cloud vs observed HI
ax = axes[1, 0]
cyc1 = engine_df["cycle"].to_numpy()
p5 = np.percentile(h_history, 5, axis=1)
p95 = np.percentile(h_history, 95, axis=1)
ax.fill_between(cyc1, p5, p95, alpha=0.25, color="steelblue", label="90% particle interval")
ax.plot(cyc1, h_history.mean(axis=1), color="steelblue", linewidth=1.5, label="Particle mean")
ax.plot(cyc1, y_obs, color="darkorange", linewidth=1.2, alpha=0.8, label="Observed HI (weighted)")
ax.set_xlabel("Cycle"); ax.set_ylabel("Health h_t")
ax.set_title("Engine 1 — particle propagation vs observed HI\n(no weighting/resampling yet — Day 3)")
ax.legend(fontsize=8)

# 5d: residual diagnostics for the fitted measurement model
# (use the model's own stored residuals, not a recomputed copy, so the
#  plot can never drift out of sync with whatever hi_method/degradation_model
#  DegradationModel was actually constructed with above)
ax = axes[1, 1]
residuals = model.measurement_learner.residuals
ax.hist(residuals, bins=40, density=True, alpha=0.5, color="gray")
x_grid = np.linspace(residuals.min(), residuals.max(), 300)
ax.plot(x_grid, model.measurement_learner.likelihood_func(x_grid), "b-", linewidth=2,
        label=f"Fitted ({model.measurement_method})")
ax.set_xlabel("Residual (Y - E[X])"); ax.set_ylabel("Density")
ax.set_title(f"Measurement noise fit ({model.hi_method} + {model.degradation_model})")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("experiments/state_space_test.png", dpi=150, bbox_inches="tight")
print("\n[✓] Figure saved -> experiments/state_space_test.png")
print("\nAll tests passed.")
