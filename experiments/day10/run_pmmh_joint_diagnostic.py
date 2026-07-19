"""
run_pmmh_joint_diagnostic.py — Day 12: run the joint (growth_rate, sigma_v)
PMMH extension (particle_twin/inference/pmmh.py::pmmh_sample_joint) on the
Day-8 diagnostic set (training engines 1/10/20/30) before committing to a
full 100-engine library rebuild.

Why this set, why train engines, why these settings
-----------------------------------------------------------------------
- The library (experiments/day10/similarity_library_prototype.py) fits one
  exponential growth_rate per TRAINING engine via OLS regression. This
  script fits the same 4 training engines via full PF-PMMH instead, to
  see whether that closes whiteboard-gap #1 from the Day 10 design review
  (regression fit vs. a real PF-PMMH fit per engine).
- Engines 1/10/20/30 and n_particles=3200/n_iterations=1000/seed=42 match
  Day 8's sigma_v-only diagnostic exactly (experiments/test_pmmh.py) so
  accept rates are directly comparable to the already-documented 2/4-mixed
  result, not confounded by a different budget.
- regression_growth_rate (per-engine OLS fit, same library the one-shot
  matching already uses) is passed as the fallback for any chain that
  doesn't mix (accept_rate < FALLBACK_ACCEPT_THRESHOLD = 0.10).

Run from the repo root:
    python experiments/day10/run_pmmh_joint_diagnostic.py
"""

import json
import os
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel
from particle_twin.models.degradation_learner import DegradationModelLearner
from particle_twin.data.database import ResultsDB
from particle_twin.inference.pmmh import pmmh_sample_joint, GROWTH_RATE_PRIOR_MU

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
DIAGNOSTIC_ENGINES = [1, 10, 20, 30]   # Day 8 precedent, train-split engines
POOLED_SIGMA_V = 0.5                    # matches similarity_library_prototype.py's pooled fit
N_PARTICLES = 3200                      # matches Day 8's sigma_v-only diagnostic
N_ITERATIONS = 1000
SIGMA_V_INIT = 0.3
RW_STEP_GROWTH_RATE = 0.1
RW_STEP_SIGMA_V = 0.2
SEED = 42

OUT_DIR = "experiments/day10"
OUT_JSON = os.path.join(OUT_DIR, "pmmh_joint_diagnostic_results.json")
DB_PATH = "results.db"


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def build_regression_library(train_df, hi_builder, engine_ids, sigma_v=POOLED_SIGMA_V):
    """Per-engine OLS growth_rate for just the engines we need as a PMMH
    fallback (same fit DegradationModelLearner does inside build_library in
    similarity_library_prototype.py — duplicated here rather than imported,
    since experiments/day10 isn't a package; same convention pmmh.py itself
    already follows for _matched_model_from)."""
    library = {}
    for uid in engine_ids:
        edf = train_df[train_df['unit_id'] == uid].sort_values('cycle')
        hi = hi_builder.transform(edf).to_numpy()
        cycles = edf['cycle'].to_numpy(dtype=float)
        learner = DegradationModelLearner(
            health_series=hi, time=cycles, model_type="exponential", sigma_v=sigma_v,
        )
        library[uid] = learner.params['growth_rate']
    return library


def main():
    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    print(f"[OK] Loaded {DATASET} train: {len(train)} rows, {train['unit_id'].nunique()} engines")

    pooled_model = DegradationModel(
        hi_method="weighted", degradation_model="exponential",
        measurement_method="gaussian", sigma_v=POOLED_SIGMA_V, sigma_0=0.05,
        sensor_cols=SENSOR_COLS,
    )
    pooled_model.fit(train_df=train)
    print(f"[OK] Pooled model fit — R^2="
          f"{pooled_model.fit_report['degradation_model']['r_squared']:.3f}, "
          f"pooled growth_rate={pooled_model.degradation_learner.params['growth_rate']:.6f}")

    regression_rates = build_regression_library(train, pooled_model.hi_builder, DIAGNOSTIC_ENGINES)
    print(f"[OK] Regression fallback growth_rates: "
          f"{ {k: round(v, 6) for k, v in regression_rates.items()} }")

    db = ResultsDB(DB_PATH)
    exp_id = db.create_experiment(
        dataset=DATASET, n_particles=N_PARTICLES, sigma_v=SIGMA_V_INIT, sigma_w=0.05,
        extra_params={
            "method": "pmmh_sample_joint", "engines": DIAGNOSTIC_ENGINES,
            "n_iterations": N_ITERATIONS, "rw_step_growth_rate": RW_STEP_GROWTH_RATE,
            "rw_step_sigma_v": RW_STEP_SIGMA_V, "seed": SEED,
        },
        description="Day 12: joint (growth_rate, sigma_v) PMMH diagnostic, train engines 1/10/20/30 "
                     "(Day-8 precedent set), before full 100-engine library rebuild.",
    )
    print(f"[OK] Registered experiment exp_id={exp_id}")

    results = {}
    total_start = time.perf_counter()
    for engine in DIAGNOSTIC_ENGINES:
        engine_df = train[train['unit_id'] == engine]
        reg_rate = regression_rates[engine]

        engine_start = time.perf_counter()
        pmmh = pmmh_sample_joint(
            pooled_model=pooled_model, engine_df=engine_df,
            n_iterations=N_ITERATIONS, n_particles=N_PARTICLES,
            growth_rate_init=GROWTH_RATE_PRIOR_MU, sigma_v_init=SIGMA_V_INIT,
            rw_step_growth_rate=RW_STEP_GROWTH_RATE, rw_step_sigma_v=RW_STEP_SIGMA_V,
            regression_growth_rate=reg_rate, seed=SEED,
        )
        engine_elapsed = time.perf_counter() - engine_start

        db.store_engine_parameters(
            exp_id=exp_id, engine_id=engine, dataset=DATASET,
            growth_rate=pmmh["growth_rate_mean"], sigma_v=pmmh["sigma_v_mean"],
            source=pmmh["source"], accept_rate=pmmh["accept_rate"],
            n_iterations=N_ITERATIONS,
        )

        results[engine] = {
            "elapsed_s": engine_elapsed,
            "accept_rate": pmmh["accept_rate"],
            "growth_rate_mean": pmmh["growth_rate_mean"],
            "sigma_v_mean": pmmh["sigma_v_mean"],
            "source": pmmh["source"],
            "regression_growth_rate": reg_rate,
            "growth_rate_chain_last10": pmmh["growth_rate_chain"][-10:],
            "sigma_v_chain_last10": pmmh["sigma_v_chain"][-10:],
        }

        print("=" * 20)
        print(f"ENGINE = {engine}")
        print(f"elapsed = {engine_elapsed:.1f}s ({engine_elapsed / N_ITERATIONS:.2f}s/iteration)")
        print(f"accept_rate = {pmmh['accept_rate']:.3f}  -> source = {pmmh['source']}")
        print(f"growth_rate_mean (PMMH) = {np.mean(pmmh['growth_rate_chain']):.6f}   "
              f"regression fallback = {reg_rate:.6f}")
        print(f"reported growth_rate_mean = {pmmh['growth_rate_mean']:.6f}")
        print(f"sigma_v_mean = {pmmh['sigma_v_mean']:.4f}")

    total_elapsed = time.perf_counter() - total_start
    db.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "config": {
                "engines": DIAGNOSTIC_ENGINES, "n_particles": N_PARTICLES,
                "n_iterations": N_ITERATIONS, "rw_step_growth_rate": RW_STEP_GROWTH_RATE,
                "rw_step_sigma_v": RW_STEP_SIGMA_V, "seed": SEED, "exp_id": exp_id,
            },
            "results": results,
            "total_elapsed_s": total_elapsed,
        }, f, indent=2, default=_json_default)

    print("=" * 20)
    print(f"TOTAL elapsed for {len(DIAGNOSTIC_ENGINES)} engines x {N_ITERATIONS} iterations = "
          f"{total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"[OK] Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()