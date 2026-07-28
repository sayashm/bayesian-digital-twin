"""
01_build_library.py — FD003 final experiment, step 1: build the per-engine
training-fleet reference library and persist it to results.db.

============================================================================
WHAT THIS SCRIPT DOES
============================================================================
Exactly the FD001 pipeline (experiments/final_experiment/01_build_library.py),
re-pointed at FD003: for every one of the 100 FD003 TRAINING engines, this
fits a per-engine exponential degradation model -- (growth_rate, sigma_v) --
and stores those parameters in results.db's `engine_parameters` table,
under one fresh `exp_id` dedicated to this experiment. This is "the
library" the similarity-matching step (02_run_experiment.py) matches TEST
engines against.

WHERE THE PER-ENGINE PARAMETERS COME FROM
============================================================================
Two sources, blended:
  1. Regression: one exponential curve fit directly to each training
     engine's own (cycle, Health Index) series via ordinary least squares
     (DegradationModelLearner) -- fast (~0.3s for all 100 engines), always
     available, but a point estimate with no uncertainty quantification
     of its own.
  2. PMMH: a full particle-marginal Metropolis-Hastings fit, jointly
     sampling (growth_rate, sigma_v) per engine against the particle
     filter's own likelihood -- principled, but expensive
     (~90s/engine at n_particles=3200, n_iterations=1000).

UNLIKE the FD001 script, there is no already-computed PMMH result for
FD003 to reuse (FD001's exp_id=30 is FD001-specific data) -- so
REUSE_EXISTING_PMMH is set to False here, meaning this run ALWAYS
recalibrates all 100 engines via PMMH from scratch. Budget ~3 hours for
the full fleet at these settings. Per the Day 12 fallback rule
(particle_twin/inference/pmmh.py), any chain that does not mix
(accept_rate < FALLBACK_ACCEPT_THRESHOLD) keeps its regression-fit
growth_rate instead, with source='regression_fallback' recorded so this
is never silently hidden. Once this has been run once, a later rerun can
set REUSE_EXISTING_PMMH=True with EXISTING_PMMH_EXP_ID pointed at the
exp_id this run creates, to reuse it instead of recalibrating (same
pattern the FD001 script uses with its exp_id=30).

OUTPUT
============================================================================
  - results.db: `engine_parameters` rows for this experiment's exp_id
    (100 rows, one per training engine) via ReferenceLibrary.to_db().
  - experiments/final_experiment/library_method/FD003/config.json: records the exp_id
    this step created, so 02_run_experiment.py (and later steps) can find
    it without re-querying the database by description string. Kept
    SEPARATE from the FD001 pipeline's config.json so the two datasets'
    pipelines never clobber each other's exp_ids.

Run from the repo root (budget ~3 hours -- this recalibrates PMMH from
scratch, see above):
    python experiments/final_experiment/library_method/FD003/01_build_library.py
"""

import json
import os

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.library import ReferenceLibrary

# ---------------------------------------------------------------------------
# Configuration -- identical to experiments/final_experiment/01_build_library.py
# except DATASET and the PMMH reuse flags (see module docstring: no existing
# FD003 PMMH result exists yet to reuse).
# ---------------------------------------------------------------------------
SENSOR_COLS = ['T24', 'T30', 'T50', 'Ps30', 'BPR', 'htBleed', 'W31', 'W32']
DATASET = "FD003"
POOLED_SIGMA_V = 0.5     # pooled-model construction value; overwritten per-engine by the library
SIGMA_0 = 0.05

# No existing FD003 PMMH run to reuse -- this run always (re)calibrates.
# Once this has completed once, point EXISTING_PMMH_EXP_ID at the resulting
# exp_id and set REUSE_EXISTING_PMMH=True to skip recalibrating on reruns,
# exactly as the FD001 script does with its exp_id=30.
EXISTING_PMMH_EXP_ID = None
REUSE_EXISTING_PMMH = False

# Same defaults as experiments/final_experiment/01_build_library.py /
# experiments/day10/run_pmmh_joint_fullfleet.py, for comparability.
PMMH_N_PARTICLES = 3200
PMMH_N_ITERATIONS = 1000
PMMH_SEED = 42

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/library_method/FD003/config.json"


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
    #    traceable end-to-end.
    # ------------------------------------------------------------------
    exp_id = db.create_experiment(
        dataset=DATASET, n_particles=PMMH_N_PARTICLES, sigma_v=POOLED_SIGMA_V, sigma_w=SIGMA_0,
        extra_params={
            "step": "final_experiment/FD003/01_build_library",
            "reused_pmmh_exp_id": EXISTING_PMMH_EXP_ID if REUSE_EXISTING_PMMH else None,
            "library_summary": summary,
        },
        description="Final experiment (FD003) — training library (100 FD003 engines, "
                     "regression fit + fresh PMMH calibration, per-engine growth_rate/sigma_v).",
    )
    library.to_db(db, exp_id=exp_id)
    db.close()
    print(f"[OK] Stored library to results.db, exp_id={exp_id}")

    # ------------------------------------------------------------------
    # 5. Record this step's exp_id for the rest of the FD003 pipeline.
    # ------------------------------------------------------------------
    config = _load_config()
    config["library_exp_id"] = exp_id
    config["library_summary"] = summary
    _save_config(config)
    print(f"[OK] Wrote {CONFIG_JSON}")


if __name__ == "__main__":
    main()
