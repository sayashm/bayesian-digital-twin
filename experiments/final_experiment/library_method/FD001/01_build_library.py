"""
01_build_library.py — Final experiment, step 1: build the per-engine
training-fleet reference library and persist it to results.db.

============================================================================
WHAT THIS SCRIPT DOES
============================================================================
For every one of the 100 FD001 TRAINING engines, this fits a per-engine
exponential degradation model -- (growth_rate, sigma_v) -- and stores
those parameters in results.db's `engine_parameters` table, under one
fresh `exp_id` dedicated to this experiment. This is "the library" the
similarity-matching step (02_run_experiment.py) matches TEST engines
against: given a test engine's observed Health Index so far, find the
most similar training engines and borrow their fitted decay parameters,
instead of using one diluted fleet-pooled rate (the Day 9 problem this
whole Day 10-13 line of work traces back to).

WHERE THE PER-ENGINE PARAMETERS COME FROM
============================================================================
Two sources, blended:
  1. Regression (Day 10/11): one exponential curve fit directly to each
     training engine's own (cycle, Health Index) series via ordinary
     least squares (DegradationModelLearner) -- fast (~0.3s for all 100
     engines), always available, but a point estimate with no
     uncertainty quantification of its own.
  2. PMMH (Day 12): a full particle-marginal Metropolis-Hastings fit,
     jointly sampling (growth_rate, sigma_v) per engine against the
     particle filter's own likelihood -- principled, but expensive
     (~90s/engine at n_particles=3200, n_iterations=1000) and does not
     always mix (only 31/100 chains converged when this was run in full
     on 2026-07-18/19, exp_id=30 in results.db).

REUSE_EXISTING_PMMH (below) defaults to True: this script overlays the
ALREADY-COMPUTED PMMH results from exp_id=30 onto a freshly-built
regression library, rather than re-running ~3 hours of MCMC for numbers
that -- same model, same data, same seed -- would come out bit-identical
anyway (verified during the particle_twin/library/ refactor: rerunning the
4-engine diagnostic after refactoring pmmh.py reproduced exp_id=29's
accept rates/parameters exactly). Set REUSE_EXISTING_PMMH = False to
recalibrate every engine's (growth_rate, sigma_v) via PMMH from scratch
instead (e.g. if you change the PMMH config below) -- budget ~3 hours for
all 100 engines at these settings.

Either way, per Day 12's fallback rule, any engine whose PMMH chain does
not mix (accept_rate < FALLBACK_ACCEPT_THRESHOLD, in
particle_twin/inference/pmmh.py) keeps its regression-fit growth_rate
instead, with `source='regression_fallback'` recorded so this is never
silently hidden.

OUTPUT
============================================================================
  - results.db: `engine_parameters` rows for this experiment's exp_id
    (100 rows, one per training engine) via ReferenceLibrary.to_db().
  - experiments/final_experiment/config.json: records the exp_id this
    step created, so 02_run_experiment.py (and later steps) can find it
    without re-querying the database by description string.

Run from the repo root:
    python experiments/final_experiment/01_build_library.py
"""

import json
import os

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.library import ReferenceLibrary

# ---------------------------------------------------------------------------
# Configuration -- kept identical to experiments/day10/similarity_library_prototype.py
# and experiments/day10/run_pmmh_joint_fullfleet.py so this library is directly
# comparable to every prior Day 10/11/12 result.
# ---------------------------------------------------------------------------
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
POOLED_SIGMA_V = 0.5     # pooled-model construction value; overwritten per-engine by the library
SIGMA_0 = 0.05

# Existing PMMH results to reuse (see docstring above). Set to None (and
# REUSE_EXISTING_PMMH = False) to recalibrate from scratch instead.
EXISTING_PMMH_EXP_ID = 30
REUSE_EXISTING_PMMH = True

# Only used if REUSE_EXISTING_PMMH is False -- same defaults as
# experiments/day10/run_pmmh_joint_fullfleet.py, for comparability.
PMMH_N_PARTICLES = 3200
PMMH_N_ITERATIONS = 1000
PMMH_SEED = 42

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/config.json"


def _load_config() -> dict:
    """Read the shared pipeline config (exp_ids from earlier steps), or
    start empty if this is the first step ever run."""
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def main():
    # ------------------------------------------------------------------
    # 1. Load data, fit the pooled model (shared HI builder / measurement
    #    noise for every matched model this library will ever produce).
    # ------------------------------------------------------------------
    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    print(f"[OK] Loaded {DATASET} train: {len(train)} rows, {train['unit_id'].nunique()} engines")

    pooled_model = DegradationModel(
        hi_method="weighted", degradation_model="exponential",
        measurement_method="gaussian", sigma_v=POOLED_SIGMA_V, sigma_0=SIGMA_0,
        sensor_cols=SENSOR_COLS,
    )
    pooled_model.fit(train_df=train)
    print(f"[OK] Pooled model fit — R^2="
          f"{pooled_model.fit_report['degradation_model']['r_squared']:.3f}")

    # ------------------------------------------------------------------
    # 2. Build the regression library (all 100 training engines) -- this
    #    is the library's HI/cycles data AND the fallback growth_rate for
    #    every engine, converged PMMH or not.
    # ------------------------------------------------------------------
    library = ReferenceLibrary.build_from_regression(pooled_model, train, sigma_v=POOLED_SIGMA_V,
                                                       dataset=DATASET)
    print(f"[OK] Built regression library: {len(library)} engines")

    # ------------------------------------------------------------------
    # 3. Bring in PMMH-calibrated (growth_rate, sigma_v) -- either reused
    #    from an already-completed run, or freshly calibrated here.
    # ------------------------------------------------------------------
    db = ResultsDB(DB_PATH)

    if REUSE_EXISTING_PMMH:
        n_patched = library.overlay_from_db(db, exp_id=EXISTING_PMMH_EXP_ID)
        print(f"[OK] Overlaid PMMH results from exp_id={EXISTING_PMMH_EXP_ID}: "
              f"{n_patched} engines patched (reused, not recomputed)")
    else:
        print(f"[..] Calibrating all {len(library)} engines via PMMH from scratch "
              f"(n_particles={PMMH_N_PARTICLES}, n_iterations={PMMH_N_ITERATIONS}) -- "
              f"this takes hours, progress below:")

        def _log_progress(engine_id, result):
            print(f"  engine {engine_id:>3}: accept_rate={result['accept_rate']:.3f}  "
                  f"source={result['source']:<19}  ({result['elapsed_s']:.1f}s)")

        library.calibrate_with_pmmh(
            train, n_particles=PMMH_N_PARTICLES, n_iterations=PMMH_N_ITERATIONS,
            seed=PMMH_SEED, on_engine_done=_log_progress,
        )

    summary = library.summary()
    print(f"[OK] Final library summary: {summary}")

    # ------------------------------------------------------------------
    # 4. Persist this experiment's library to results.db under its own
    #    fresh exp_id, so the whole final_experiment pipeline is
    #    traceable end-to-end without depending on exp_id=30 by name.
    # ------------------------------------------------------------------
    exp_id = db.create_experiment(
        dataset=DATASET, n_particles=PMMH_N_PARTICLES, sigma_v=POOLED_SIGMA_V, sigma_w=SIGMA_0,
        extra_params={
            "step": "final_experiment/01_build_library",
            "reused_pmmh_exp_id": EXISTING_PMMH_EXP_ID if REUSE_EXISTING_PMMH else None,
            "library_summary": summary,
        },
        description="Final experiment — training library (100 FD001 engines, "
                     "regression fit + PMMH overlay, per-engine growth_rate/sigma_v).",
    )
    library.to_db(db, exp_id=exp_id)
    db.close()
    print(f"[OK] Stored library to results.db, exp_id={exp_id}")

    # ------------------------------------------------------------------
    # 5. Record this step's exp_id for the rest of the pipeline.
    # ------------------------------------------------------------------
    config = _load_config()
    config["library_exp_id"] = exp_id
    config["library_summary"] = summary
    _save_config(config)
    print(f"[OK] Wrote {CONFIG_JSON}")


if __name__ == "__main__":
    main()