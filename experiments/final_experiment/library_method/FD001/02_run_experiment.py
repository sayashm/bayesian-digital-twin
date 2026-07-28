"""
02_run_experiment.py — Final experiment, step 2: run BOTH similarity
strategies (one-shot and periodic) over every FD001 TEST engine, and
store the full per-cycle health-index AND RUL predictions in results.db.

============================================================================
WHAT THIS SCRIPT DOES
============================================================================
Loads the training library built by 01_build_library.py (config.json's
"library_exp_id"), then for EVERY one of the 100 FD001 test engines, runs
particle_twin.library.SimilarityStreamingFilter TWICE:

  one-shot : match once, after a WARMUP_CYCLES warm-up window, then keep
             that matched (growth_rate, sigma_v) fixed for the rest of the
             engine's life. The thesis's chosen default (see
             experiments/day10/day11_fullfleet_decision.md) -- wins on
             every fleet-level point-accuracy metric.
  periodic : re-match every REMATCH_EVERY cycles using all data observed
             so far, letting the matched parameters change over time.

For each (engine, strategy) pair, this extracts BOTH:
  - the health-index trajectory (predicted median + 5th/95th percentile,
    from SimilarityStreamingFilter.extract_health_trajectory), and
  - the RUL trajectory (predicted median + 90% credible interval, from
    particle_twin.analysis.rul.extract_rul_trajectory)
and stores every cycle of both into results.db's `rul_estimates` table
(which, since Day 13, also carries health_p5/health_p95 -- see
particle_twin/data/database.py), under two dedicated exp_ids (one per
strategy), so the two similarity methods stay cleanly separated and
independently queryable.

Resumable by (exp_id, engine_id): if this script is interrupted and rerun,
it reuses the SAME two exp_ids (persisted in config.json) and skips any
engine that already has rows under that exp_id, rather than starting over
or creating duplicate experiment rows.

RUL v2 (particle_twin/analysis/rul.py): the RUL threshold/horizon are no
longer the v1 hardcoded (failure_threshold=0.0, max_horizon=1500) -- see
rul.py's module docstring for why that was wrong. mu_fail/sigma_fail are
now calibrated ONCE on the training fleet (calibrate_failure_threshold,
mode="posterior") before the per-engine loop below, and RUL_CAP (set
below; see its own comment for the CURRENT value and how it compares to
rul.py's literature-recommended DEFAULT_RUL_CAP=125) caps the
piecewise-linear RUL target. IMPORTANT: because this script skips any
engine already stored under its exp_id, if one_shot_exp_id/periodic_exp_id
in config.json were populated by a run under a DIFFERENT RUL_CAP/threshold
config than the one below, rerunning will silently keep serving those
stale rows under the old config -- delete those two keys from config.json
(NOT library_exp_id) before rerunning so fresh exp_ids get created.

Run from the repo root (~5-10 minutes for all 100 engines x 2 strategies,
n_particles=500):
    python experiments/final_experiment/02_run_experiment.py
"""

import json
import os
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.library import ReferenceLibrary, SimilarityStreamingFilter
from particle_twin.analysis.rul import calibrate_failure_threshold, extract_rul_trajectory

# ---------------------------------------------------------------------------
# Configuration -- matches experiments/day10/similarity_library_prototype.py /
# full_fleet_comparison.py exactly, so these numbers are directly comparable
# to every prior Day 10/11 result.
# ---------------------------------------------------------------------------
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
POOLED_SIGMA_V = 0.5
SIGMA_0 = 0.05

N_PARTICLES = 500
WARMUP_CYCLES = 20
REMATCH_EVERY = 20
K_TOP = 5
MMD_LENGTH_SCALE = 1.0

# --- v2 RUL calibration (particle_twin/analysis/rul.py) -------------------
# FAILURE_THRESHOLD/MAX_HORIZON used to be hardcoded to the v1 values
# (0.0 / 1500), which forward-simulated every particle through a health
# range no real FD001 engine ever visits -- see rul.py's module docstring,
# causes (1)-(2). mu_fail/sigma_fail are now measured once below via
# calibrate_failure_threshold(mode="posterior") instead of asserted.
#
# RUL_CAP implements the C-MAPSS piecewise-linear RUL convention (rul.py
# docstring (4) / DEFAULT_RUL_CAP=125), but is set to 350 here rather than
# the literature-recommended 125 -- a deliberate experiment-specific choice
# to see the effect of a much looser cap, NOT the thesis's recommended
# default. Since RUL_CAP>=MAX_HORIZON, the cap never actually binds here
# (MAX_HORIZON is the tighter constraint and fires first); both are set to
# the same value so this run's true ceiling is simply MAX_HORIZON cycles.
CALIBRATION_N_ENGINES = 30      # rul.py: between-engine std already ~0.026 at n=30
CALIBRATION_N_PARTICLES = 300   # separate (smaller/faster) from the fleet's N_PARTICLES
RUL_CAP = 350
MAX_HORIZON = 350        # whichever of RUL_CAP/MAX_HORIZON is smaller binds first
SEED = 42

STRATEGIES = ["one_shot", "periodic"]

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/config.json"


def _load_config() -> dict:
    with open(CONFIG_JSON) as f:
        return json.load(f)


def _save_config(config: dict) -> None:
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def main():
    config = _load_config()
    if "library_exp_id" not in config:
        raise RuntimeError(
            "config.json has no library_exp_id -- run 01_build_library.py first."
        )
    library_exp_id = config["library_exp_id"]

    # ------------------------------------------------------------------
    # 1. Load data, rebuild the pooled model, load the persisted library.
    # ------------------------------------------------------------------
    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)
    all_test_engines = sorted(int(u) for u in test["unit_id"].unique())
    print(f"[OK] Loaded {DATASET}: {len(train)} train rows, {len(test)} test rows, "
          f"{len(all_test_engines)} test engines")

    pooled_model = DegradationModel(
        hi_method="weighted", degradation_model="exponential",
        measurement_method="gaussian", sigma_v=POOLED_SIGMA_V, sigma_0=SIGMA_0,
        sensor_cols=SENSOR_COLS,
    )
    pooled_model.fit(train_df=train)

    library = ReferenceLibrary.build_from_regression(pooled_model, train, sigma_v=POOLED_SIGMA_V,
                                                       dataset=DATASET)
    db = ResultsDB(DB_PATH)
    n_patched = library.overlay_from_db(db, exp_id=library_exp_id)
    print(f"[OK] Loaded library from exp_id={library_exp_id}: {n_patched} engines patched "
          f"({library.summary()})")

    sim_filter = SimilarityStreamingFilter(library, k_top=K_TOP, length_scale=MMD_LENGTH_SCALE)

    # ------------------------------------------------------------------
    # 1b. Calibrate the v2 failure threshold ONCE, on the training fleet
    #     (mode="posterior": absorbs the filter's own bias -- see rul.py's
    #     module docstring, causes (1)-(2) -- at ~50x mode="observed"'s
    #     cost, which is still only ~30s for 30 engines at these defaults).
    # ------------------------------------------------------------------
    mu_fail, sigma_fail = calibrate_failure_threshold(
        pooled_model, train, mode="posterior", sim_filter=sim_filter,
        n_engines=CALIBRATION_N_ENGINES, n_particles=CALIBRATION_N_PARTICLES,
        strategy="one_shot", warmup_cycles=WARMUP_CYCLES,
    )
    print(f"[OK] Calibrated failure threshold (posterior space): "
          f"mu_fail={mu_fail:.4f}, sigma_fail={sigma_fail:.4f} "
          f"(reproduces particle_twin.analysis.rul.FD001_FAILURE_THRESHOLD_POSTERIOR)")

    # ------------------------------------------------------------------
    # 2. One dedicated exp_id per strategy -- created once, reused across
    #    resumed runs (persisted in config.json).
    # ------------------------------------------------------------------
    strategy_exp_ids = {}
    for strategy in STRATEGIES:
        key = f"{strategy}_exp_id"
        if key in config:
            strategy_exp_ids[strategy] = config[key]
            continue
        exp_id = db.create_experiment(
            dataset=DATASET, n_particles=N_PARTICLES, sigma_v=POOLED_SIGMA_V, sigma_w=SIGMA_0,
            extra_params={
                "step": "final_experiment/02_run_experiment", "strategy": strategy,
                "library_exp_id": library_exp_id, "warmup_cycles": WARMUP_CYCLES,
                "rematch_every": REMATCH_EVERY, "k_top": K_TOP, "max_horizon": MAX_HORIZON,
                "rul_cap": RUL_CAP, "mu_fail": mu_fail, "sigma_fail": sigma_fail,
                "threshold_mode": "posterior",
            },
            description=f"Final experiment — {strategy} similarity matching "
                        f"(test fleet, health+RUL trajectories, library exp_id={library_exp_id}).",
        )
        strategy_exp_ids[strategy] = exp_id
        config[key] = exp_id
        _save_config(config)
        print(f"[OK] Registered experiment for strategy={strategy!r}: exp_id={exp_id}")

    # ------------------------------------------------------------------
    # 3. Run every (engine, strategy) pair, storing both trajectories.
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    n_done = 0
    n_total = len(all_test_engines) * len(STRATEGIES)

    for strategy in STRATEGIES:
        exp_id = strategy_exp_ids[strategy]
        already_done = {
            row["engine_id"] for row in db._conn.execute(
                "SELECT DISTINCT engine_id FROM rul_estimates WHERE exp_id = ? AND dataset = ?",
                (exp_id, DATASET),
            )
        }
        if already_done:
            print(f"[OK] strategy={strategy!r}: resuming, {len(already_done)}/"
                  f"{len(all_test_engines)} engines already stored under exp_id={exp_id}")

        for engine_id in all_test_engines:
            n_done += 1
            if engine_id in already_done:
                continue

            engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
            n_cycles = len(engine_df)
            true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])

            t0 = time.perf_counter()
            np.random.seed(SEED)  # identical seed per (engine, strategy) -> differences are model-driven
            pf, rematch_log = sim_filter.run(
                engine_df, strategy=strategy, n_particles=N_PARTICLES,
                warmup_cycles=WARMUP_CYCLES, rematch_every=REMATCH_EVERY,
            )

            rul_trajectory = extract_rul_trajectory(
                pf, failure_threshold=mu_fail, threshold_std=sigma_fail,
                rul_cap=RUL_CAP, max_horizon=MAX_HORIZON,
            )
            health_cycles, health_p5, health_p50, health_p95 = (
                SimilarityStreamingFilter.extract_health_trajectory(pf)
            )
            assert len(rul_trajectory) == len(health_cycles), (
                f"engine {engine_id}: RUL trajectory ({len(rul_trajectory)} cycles) and health "
                f"trajectory ({len(health_cycles)} cycles) misaligned -- both should skip "
                f"pf.history[0] identically."
            )

            db.register_engine(engine_id=engine_id, dataset=DATASET, split="test",
                                n_cycles=n_cycles, true_rul=true_val)

            for i, rul_row in enumerate(rul_trajectory):
                assert rul_row["cycle"] == health_cycles[i], (
                    f"engine {engine_id}, row {i}: RUL cycle {rul_row['cycle']} != "
                    f"health cycle {health_cycles[i]}"
                )
                # NOTE: rul_row["cycle"] is numpy.int64 (from engine_df["cycle"].to_numpy()
                # propagated through pf.history), not a native Python int. Unlike
                # numpy.float64 (a real subclass of float, so sqlite3 stores it as a
                # REAL automatically), numpy.int64 is NOT a subclass of int -- sqlite3
                # silently stores it via the buffer protocol as an 8-byte BLOB instead
                # of raising, so this cast is required, not just defensive style.
                db.store_rul_timeseries(
                    exp_id=exp_id, engine_id=engine_id, dataset=DATASET, cycle=int(rul_row["cycle"]),
                    rul_median=rul_row["rul_median"], rul_p5=rul_row["rul_p5"],
                    rul_p95=rul_row["rul_p95"], rul_mean=rul_row["rul_mean"],
                    rul_std=rul_row["rul_std"], health_mean=rul_row["health_mean"],
                    health_median=float(health_p50[i]), health_p5=float(health_p5[i]),
                    health_p95=float(health_p95[i]), ess=rul_row["ess"],
                )
            db.commit()

            elapsed = time.perf_counter() - t0
            total_elapsed = time.perf_counter() - t_start
            print(f"[{n_done:>3}/{n_total}] strategy={strategy:<9} engine={engine_id:>3}  "
                  f"n_cycles={n_cycles:>4}  n_rematches={len(rematch_log)}  ({elapsed:.1f}s, "
                  f"total {total_elapsed:.1f}s)")

    db.close()
    print(f"\n[OK] Done. Total wall-clock: {time.perf_counter() - t_start:.1f}s")
    print(f"[OK] one_shot exp_id={strategy_exp_ids['one_shot']}, "
          f"periodic exp_id={strategy_exp_ids['periodic']}")


if __name__ == "__main__":
    main()