"""
make_figure_engine77.py — Day 5: first thesis-quality figure
================================================================
Two-panel figure for ONE representative test engine (engine 77 -- chosen
because its trajectory_rmse (314.9) sits almost exactly at the fleet
median (316.4) over all 100 FD001 test engines; see
experiments/fd001_full_results.json / the Day-5 hardest-engine analysis.
Not a best case, not a worst case -- what "typical" looks like.):

  Top panel    -- health tracking: observed Health Index (raw sensor
                  signal, noisy) vs the particle filter's posterior mean
                  health, with the 5th-95th percentile particle spread
                  shaded (the filter's own uncertainty about the CURRENT
                  state).
  Bottom panel -- RUL distribution: predicted median RUL + 90% credible
                  interval at every observed cycle, vs the implied
                  ground-truth countdown (only one point, true_rul, is
                  ever actually measured -- RUL_FD001.txt gives it at the
                  truncation cycle; earlier points are reconstructed by
                  counting forward, see metrics.trajectory_ground_truth).

Run from the repo root:
    python experiments/make_figure_engine77.py
"""

import numpy as np
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.analysis.rul import extract_rul_trajectory
from particle_twin.visualization.plots import engine_summary_figure

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
ENGINE_ID = 77
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
MAX_HORIZON = 1500

loader = CMAPSSLoader(data_dir="data/cmapss")
train = loader.load_train(DATASET)
test = loader.load_test(DATASET)
rul_true = loader.load_rul(DATASET)

model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                          measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=0.05,
                          sensor_cols=SENSOR_COLS)
model.fit(train)

engine_df = test[test['unit_id'] == ENGINE_ID].sort_values('cycle')
true_rul = float(rul_true.loc[rul_true['unit_id'] == ENGINE_ID, 'true_rul'].values[0])
n_cycles = len(engine_df)
observed_hi = model.observe(engine_df).to_numpy()  # [0,1] scale, same as particles

pf = BootstrapPF(model, n_particles=N_PARTICLES, ess_threshold=0.5)
pf.run(engine_df)
trajectory = extract_rul_trajectory(pf, failure_threshold=FAILURE_THRESHOLD, max_horizon=MAX_HORIZON)

# -- panel 1 data: particle percentiles at every cycle (skip history[0], the prior) --
cycles = np.array([h['cycle_number'] for h in pf.history[1:]])
p5 = np.array([np.percentile(h['particles'], 5) for h in pf.history[1:]])
p50 = np.array([np.percentile(h['particles'], 50) for h in pf.history[1:]])
p95 = np.array([np.percentile(h['particles'], 95) for h in pf.history[1:]])

# -- panel 2 data: RUL trajectory vs implied ground truth --
rul_cycles = np.array([e['cycle'] for e in trajectory])
rul_median = np.array([e['rul_median'] for e in trajectory])
rul_p5 = np.array([e['rul_p5'] for e in trajectory])
rul_p95 = np.array([e['rul_p95'] for e in trajectory])

fig = engine_summary_figure(
    engine_id=ENGINE_ID, cycles=cycles, observed_hi=observed_hi,
    particle_p5=p5, particle_p50=p50, particle_p95=p95,
    obs_cycles=engine_df['cycle'].to_numpy(),
    rul_cycles=rul_cycles, rul_median=rul_median, rul_p5=rul_p5, rul_p95=rul_p95,
    true_rul=true_rul, n_cycles=n_cycles,
    extra_title=(f"n_cycles={n_cycles}, true RUL at truncation={true_rul:.0f} cycles "
                 f"(fleet-median-typical run, trajectory RMSE≈315)"),
)
for out_path in ["experiments/fig_engine77_health_rul.png", "thesis/Fig/fig_engine77_health_rul.png"]:
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] saved -> {out_path}")
