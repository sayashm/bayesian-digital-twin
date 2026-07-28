"""
05_health_vs_rul.py — FD003 quick diagnostic: how does Health Index
relate to RUL, for TRAIN vs. TEST (both ground truth -- no model
predictions read back from results.db here)? Same script as
experiments/final_experiment/05_health_vs_rul.py, repointed at FD003.

10 engines: unit_id 1, 10, 20, ..., 90.
  - Train engines run to failure, so the RUL at any cycle is just
    (max_cycle - cycle) -- no model needed.
  - Test engines are truncated before failure. RUL_FD001.txt gives the
    true RUL at the truncation (last observed) cycle, so the RUL at any
    earlier cycle c is true_rul_at_truncation + (max_cycle - c) -- the
    same convention used everywhere else in this project (see
    particle_twin.analysis.metrics / 04_make_tables.py).

Health Index (both panels) is the observed/sensor-derived HI from the
fitted pooled DegradationModel -- deterministic, no particle filter
needed.

Both figures plot RUL on the x-axis and Health Index on the y-axis.

Run from the repo root:
    python experiments/final_experiment/library_method/FD003/05_health_vs_rul.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel

SENSOR_COLS = ['T24', 'T30', 'T50', 'Ps30', 'BPR', 'htBleed', 'W31', 'W32']
DATASET = "FD003"
ENGINES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
TRAIN_OUT_PATH = "experiments/final_experiment/library_method/FD003/figures/health_vs_rul_train.png"
TEST_OUT_PATH = "experiments/final_experiment/library_method/FD003/figures/health_vs_rul_test.png"

loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train(DATASET)
test = loader.load_test(DATASET)
rul_true = loader.load_rul(DATASET)

pooled_model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                                 measurement_method="gaussian", sigma_v=0.5, sigma_0=0.05,
                                 sensor_cols=SENSOR_COLS)
pooled_model.fit(train_df=train)

# -- Figure 1: train (actual) --
fig_train, axes_train = plt.subplots(2, 5, figsize=(22, 8))
for ax, engine_id in zip(axes_train.flat, ENGINES):
    # Train: actual Health Index vs. actual RUL (engine runs to failure, so
    # RUL at any cycle is just how many cycles remain until the last one).
    edf = train[train["unit_id"] == engine_id].sort_values("cycle")
    train_hi = pooled_model.observe(edf).to_numpy()
    train_rul = edf["cycle"].max() - edf["cycle"].to_numpy()

    ax.plot(train_rul, train_hi, "o-", color="steelblue", markersize=3)
    ax.set_title(f"Engine {engine_id}")
    ax.set_xlabel("RUL (cycles)")

axes_train[0, 0].set_ylabel("Health Index")
axes_train[1, 0].set_ylabel("Health Index")
fig_train.suptitle("Health Index vs. RUL — FD003 train (actual)", fontsize=14)
plt.tight_layout()
fig_train.savefig(TRAIN_OUT_PATH, dpi=150)
plt.close(fig_train)
print(f"[OK] saved -> {TRAIN_OUT_PATH}")

# -- Figure 2: test (actual / ground truth) --
fig_test, axes_test = plt.subplots(2, 5, figsize=(22, 8))
for ax, engine_id in zip(axes_test.flat, ENGINES):
    # Test: actual Health Index (observed from real sensor readings) vs.
    # actual RUL, reconstructed from RUL_FD001.txt's truncation-point value.
    edf = test[test["unit_id"] == engine_id].sort_values("cycle")
    test_hi = pooled_model.observe(edf).to_numpy()
    true_rul_at_truncation = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])
    test_rul = true_rul_at_truncation + (edf["cycle"].max() - edf["cycle"].to_numpy())

    ax.plot(test_rul, test_hi, "o-", color="darkorange", markersize=3)
    ax.set_title(f"Engine {engine_id}")
    ax.set_xlabel("RUL (cycles)")

axes_test[0, 0].set_ylabel("Health Index")
axes_test[1, 0].set_ylabel("Health Index")
fig_test.suptitle("Health Index vs. RUL — FD003 test (actual)", fontsize=14)
plt.tight_layout()
fig_test.savefig(TEST_OUT_PATH, dpi=150)
plt.close(fig_test)
print(f"[OK] saved -> {TEST_OUT_PATH}")