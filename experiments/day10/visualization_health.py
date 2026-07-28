"""
visualization_health.py — per-engine observed Health Index vs. filtered
(predicted) Health Index, with the particle filter's 5th-95th percentile
spread as a band.

exp_id=24 (Day 5's full 100-engine FD001 run) only stored a scalar
health_mean per cycle in rul_estimates, not percentiles, so there's no
pre-computed CI band to read back. Health percentiles are cheap to get
directly from the particle cloud though (no PMMH involved, just the
bootstrap filter), so this refits the exact same pooled model
exp_id=24 used (config taken from db.get_experiment(24) / the original
experiments/run_full_fd001.py: hi_method="weighted",
degradation_model="exponential", measurement_method="gaussian",
sigma_v=0.5, sigma_0=0.05, n_particles=500, same 10 sensors) and reruns
BootstrapPF per engine -- same pattern experiments/make_figure_engine77.py
already used for one engine, extended here to all 100.

Writes one PNG per engine into particle_twin/visualization/ (e.g.
health_engine_001.png ... health_engine_100.png), using the existing
plot_health_tracking() building block from particle_twin/visualization/plots.py.

Run from the repo root:
    python experiments/day10/visualization_health.py
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.visualization.plots import plot_health_tracking

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
N_PARTICLES = 500       # matches exp_id=24's stored config exactly
SIGMA_V = 0.5
SIGMA_0 = 0.05
SEED = 42

OUT_DIR = os.path.join("particle_twin", "visualization")


def main():
    np.random.seed(SEED)

    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    all_engines = sorted(test['unit_id'].unique().tolist())
    print(f"[OK] Loaded {DATASET} train/test, {len(all_engines)} test engines")

    model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                              measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=SIGMA_0,
                              sensor_cols=SENSOR_COLS)
    model.fit(train_df=train)
    print(f"[OK] Pooled model fit — R^2={model.degradation_learner.fit_quality['r_squared']:.3f} "
          f"(same config as exp_id=24)")

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()

    for i, engine_id in enumerate(all_engines, start=1):
        engine_df = test[test['unit_id'] == engine_id].sort_values('cycle')
        observed_hi = model.observe(engine_df).to_numpy()  # [0,1] scale, same as particles

        pf = BootstrapPF(model, n_particles=N_PARTICLES, ess_threshold=0.5)
        pf.run(engine_df)

        cycles = np.array([h['cycle_number'] for h in pf.history[1:]])
        p5 = np.array([np.percentile(h['particles'], 5) for h in pf.history[1:]])
        p50 = np.array([np.percentile(h['particles'], 50) for h in pf.history[1:]])
        p95 = np.array([np.percentile(h['particles'], 95) for h in pf.history[1:]])

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_health_tracking(ax, cycles, observed_hi, p5, p50, p95,
                              obs_cycles=engine_df['cycle'].to_numpy())
        fig.suptitle(f"Engine {engine_id} — observed vs. filtered Health Index (exp_id=24 config)",
                     fontsize=12)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"health_engine_{engine_id:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        if i % 10 == 0 or i == len(all_engines):
            elapsed = time.perf_counter() - t_start
            print(f"[{i:>3}/{len(all_engines)}] engine {engine_id:>3} -> {out_path}  "
                  f"(elapsed {elapsed:.1f}s)")

    print(f"\n[OK] Wrote {len(all_engines)} figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()