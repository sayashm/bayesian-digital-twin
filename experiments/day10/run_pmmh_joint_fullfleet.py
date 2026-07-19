"""
run_pmmh_joint_fullfleet.py — Day 12: full 100-engine PMMH library rebuild.

Extends the 4-engine diagnostic (run_pmmh_joint_diagnostic.py) to every
FD001 training engine, using the joint (growth_rate, sigma_v) PMMH sampler
(particle_twin/inference/pmmh.py::pmmh_sample_joint) instead of the OLS
regression fit similarity_library_prototype.py uses -- closing whiteboard
gap #1 from the Day 10 design review ("prototype used a fast per-engine
regression fit, diagram calls for a full PF-PMMH fit per engine").

Same config as the diagnostic (n_particles=3200, n_iterations=1000, seed=42)
so accept rates stay comparable. Estimated ~2.5-3 hours for all 100 engines
at ~90s/engine. Resumable: writes OUT_JSON and engine_parameters rows
incrementally (one engine at a time), so a re-run after any interruption
picks up where it left off rather than redoing completed engines, and
reuses the same exp_id (persisted in OUT_JSON's config on first run).

Run from the repo root:
    python experiments/day10/run_pmmh_joint_fullfleet.py
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
POOLED_SIGMA_V = 0.5
N_PARTICLES = 3200
N_ITERATIONS = 1000
SIGMA_V_INIT = 0.3
RW_STEP_GROWTH_RATE = 0.1
RW_STEP_SIGMA_V = 0.2
SEED = 42

OUT_DIR = "experiments/day10"
OUT_JSON = os.path.join(OUT_DIR, "pmmh_joint_fullfleet_results.json")
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
    all_engines = sorted(train['unit_id'].unique().tolist())
    print(f"[OK] Loaded {DATASET} train: {len(train)} rows, {len(all_engines)} engines")

    pooled_model = DegradationModel(
        hi_method="weighted", degradation_model="exponential",
        measurement_method="gaussian", sigma_v=POOLED_SIGMA_V, sigma_0=0.05,
        sensor_cols=SENSOR_COLS,
    )
    pooled_model.fit(train_df=train)
    print(f"[OK] Pooled model fit — R^2="
          f"{pooled_model.fit_report['degradation_model']['r_squared']:.3f}")

    regression_rates = build_regression_library(train, pooled_model.hi_builder, all_engines)

    # ---- Resume support: reuse exp_id + already-completed engines if OUT_JSON exists ----
    results = {}
    exp_id = None
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            prior = json.load(f)
        results = {int(k): v for k, v in prior.get("results", {}).items()}
        exp_id = prior.get("config", {}).get("exp_id")
        print(f"[OK] Resuming: {len(results)}/{len(all_engines)} engines already done, exp_id={exp_id}")

    db = ResultsDB(DB_PATH)
    if exp_id is None:
        exp_id = db.create_experiment(
            dataset=DATASET, n_particles=N_PARTICLES, sigma_v=SIGMA_V_INIT, sigma_w=0.05,
            extra_params={
                "method": "pmmh_sample_joint", "scope": "full_100_engine_library",
                "n_iterations": N_ITERATIONS, "rw_step_growth_rate": RW_STEP_GROWTH_RATE,
                "rw_step_sigma_v": RW_STEP_SIGMA_V, "seed": SEED,
            },
            description="Day 12: joint (growth_rate, sigma_v) PMMH full 100-engine library rebuild "
                         "(follow-up to the 4-engine diagnostic, exp_id=29).",
        )
        # Persist immediately so a resume after an early crash still reuses this exp_id.
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({"config": {"exp_id": exp_id, "n_particles": N_PARTICLES,
                                   "n_iterations": N_ITERATIONS, "seed": SEED},
                       "results": {}}, f, indent=2)
        print(f"[OK] Registered experiment exp_id={exp_id}")

    todo = [e for e in all_engines if e not in results]
    print(f"[OK] {len(todo)} engines remaining this run")

    total_start = time.perf_counter()
    n_pmmh = sum(1 for r in results.values() if r["source"] == "pmmh")
    n_fallback = len(results) - n_pmmh

    for i, engine in enumerate(todo):
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
        }
        if pmmh["source"] == "pmmh":
            n_pmmh += 1
        else:
            n_fallback += 1

        # Incremental write after every engine -- resumable if interrupted.
        with open(OUT_JSON, "w") as f:
            json.dump({
                "config": {"exp_id": exp_id, "n_particles": N_PARTICLES,
                           "n_iterations": N_ITERATIONS, "rw_step_growth_rate": RW_STEP_GROWTH_RATE,
                           "rw_step_sigma_v": RW_STEP_SIGMA_V, "seed": SEED},
                "results": results,
            }, f, indent=2, default=_json_default)

        done = len(results)
        elapsed_total = time.perf_counter() - total_start
        avg = elapsed_total / (i + 1)
        eta_min = avg * (len(todo) - i - 1) / 60
        print(f"[{done:>3}/{len(all_engines)}] engine={engine:>3}  "
              f"accept_rate={pmmh['accept_rate']:.3f}  source={pmmh['source']:<19}  "
              f"({engine_elapsed:.1f}s)  pmmh={n_pmmh} fallback={n_fallback}  "
              f"ETA={eta_min:.1f}min")

    total_elapsed = time.perf_counter() - total_start
    db.close()

    print("=" * 20)
    print(f"DONE. {len(results)}/{len(all_engines)} engines. "
          f"pmmh={n_pmmh} regression_fallback={n_fallback} "
          f"({100*n_pmmh/len(results):.0f}% converged)")
    print(f"This run's wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[OK] Wrote {OUT_JSON}, exp_id={exp_id}")


if __name__ == "__main__":
    main()