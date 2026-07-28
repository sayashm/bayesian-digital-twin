"""
01_run_experiment.py — PF Fixed-Parameters experiment, FD003: run the
bootstrap particle filter with ONE pooled ("fixed") set of degradation
parameters over every FD003 test engine, using the v2 RUL extraction
pipeline (calibrated failure threshold, per-particle threshold
uncertainty, rul_cap) from particle_twin/analysis/rul.py.

============================================================================
WHAT "FIXED PARAMETERS" MEANS, AND WHY THIS EXISTS
============================================================================
Unlike library_method/ (one growth_rate/sigma_v PER ENGINE, borrowed from
the most similar training engines via SimilarityStreamingFilter), this
experiment fits ONE DegradationModel on the pooled training fleet and
applies it, completely unchanged, to every test engine -- no per-engine
matching, no one-shot/periodic rematching, no PMMH. This is the "before"
baseline the library method was built to beat (exp_id=24 for FD001 /
exp_id=26 for FD003 -- see experiments/run_full_fd001.py -- both run under
the OLD v1 RUL method).

Those old exp_ids are stale for a fair comparison: they used v1's RUL
extraction (hardcoded failure_threshold=0.0, max_horizon=1500 -- see
particle_twin/analysis/rul.py's module docstring for why that was wrong).
This script re-runs the identical fixed-parameters PF under the CURRENT v2
method, using the exact same config (N_PARTICLES, RUL_CAP, MAX_HORIZON,
calibration settings) as library_method/FD003/02_run_experiment.py, so the
library-vs-fixed-parameters comparison isolates the one variable that
actually differs between the two experiments (per-engine matching vs one
pooled fit), rather than being confounded by different RUL-extraction
settings.

OUTPUT
============================================================================
  - results.db: rul_estimates rows under one fresh exp_id
    (particle_twin.data.database.ResultsDB), one row per (engine, cycle).
  - experiments/final_experiment/PF_Fixed_Parameters/FD003/config.json:
    records the exp_id + calibration constants for the later steps.

Run from the repo root (~2-3 minutes for all 100 engines, n_particles=500):
    python experiments/final_experiment/PF_Fixed_Parameters/FD003/01_run_experiment.py
"""

import json
import os
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.data.database import ResultsDB
from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.analysis.rul import calibrate_failure_threshold, extract_rul_trajectory

# ---------------------------------------------------------------------------
# Configuration -- pooled-model settings match every prior fixed-parameter
# run in this project (experiments/run_full_fd001.py); RUL settings match
# library_method/FD003/02_run_experiment.py so the two experiments are
# directly comparable.
# ---------------------------------------------------------------------------
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD003"
POOLED_SIGMA_V = 0.5
SIGMA_0 = 0.05

N_PARTICLES = 500

# --- v2 RUL calibration (particle_twin/analysis/rul.py) -------------------
CALIBRATION_N_ENGINES = 30      # rul.py: between-engine std already ~0.026 at n=30
CALIBRATION_N_PARTICLES = 300   # separate (smaller/faster) from the fleet's N_PARTICLES
RUL_CAP = 350
MAX_HORIZON = 350        # whichever of RUL_CAP/MAX_HORIZON is smaller binds first
SEED = 42

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/PF_Fixed_Parameters/FD003/config.json"


def _load_config() -> dict:
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def main():
    config = _load_config()

    # ------------------------------------------------------------------
    # 1. Load data, fit the ONE pooled model used for every test engine.
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
    sigma_w = pooled_model.measurement_learner.params.get("sigma", float("nan"))
    print(f"[OK] Pooled model fit — R^2="
          f"{pooled_model.fit_report['degradation_model']['r_squared']:.3f}, sigma_w={sigma_w:.4f}")

    # ------------------------------------------------------------------
    # 1b. Calibrate the v2 failure threshold ONCE (posterior space), using
    #     the SAME plain BootstrapPF this experiment runs per test engine
    #     (sim_filter=None -> calibrate_failure_threshold's own
    #     BootstrapPF-on-pooled_model fallback, so the calibration model
    #     is identical to the one actually deployed below).
    # ------------------------------------------------------------------
    mu_fail, sigma_fail = calibrate_failure_threshold(
        pooled_model, train, mode="posterior", sim_filter=None,
        n_engines=CALIBRATION_N_ENGINES, n_particles=CALIBRATION_N_PARTICLES,
        seed=SEED,
    )
    print(f"[OK] Calibrated failure threshold (posterior space): "
          f"mu_fail={mu_fail:.4f}, sigma_fail={sigma_fail:.4f}")

    # ------------------------------------------------------------------
    # 2. One dedicated exp_id -- created once, reused across resumed runs.
    # ------------------------------------------------------------------
    db = ResultsDB(DB_PATH)

    if "exp_id" in config:
        exp_id = config["exp_id"]
        print(f"[OK] Resuming existing experiment, exp_id={exp_id}")
    else:
        exp_id = db.create_experiment(
            dataset=DATASET, n_particles=N_PARTICLES, sigma_v=POOLED_SIGMA_V, sigma_w=sigma_w,
            extra_params={
                "step": "final_experiment/PF_Fixed_Parameters/FD003/01_run_experiment",
                "max_horizon": MAX_HORIZON, "rul_cap": RUL_CAP,
                "mu_fail": mu_fail, "sigma_fail": sigma_fail, "threshold_mode": "posterior",
            },
            description="Final experiment (PF fixed parameters, FD003) — one pooled "
                        "DegradationModel run over every test engine, no per-engine "
                        "matching, v2 RUL extraction.",
        )
        config["exp_id"] = exp_id
        config["mu_fail"] = mu_fail
        config["sigma_fail"] = sigma_fail
        _save_config(config)
        print(f"[OK] Registered experiment: exp_id={exp_id}")

    already_done = {
        row["engine_id"] for row in db._conn.execute(
            "SELECT DISTINCT engine_id FROM rul_estimates WHERE exp_id = ? AND dataset = ?",
            (exp_id, DATASET),
        )
    }
    if already_done:
        print(f"[OK] Resuming: {len(already_done)}/{len(all_test_engines)} engines already stored "
              f"under exp_id={exp_id}")

    # ------------------------------------------------------------------
    # 3. Run every test engine through the plain BootstrapPF (fixed,
    #    pooled parameters -- no similarity matching of any kind).
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    n_done = 0
    n_total = len(all_test_engines)

    for engine_id in all_test_engines:
        n_done += 1
        if engine_id in already_done:
            continue

        engine_df = test[test["unit_id"] == engine_id].sort_values("cycle")
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true["unit_id"] == engine_id, "true_rul"].values[0])

        t0 = time.perf_counter()
        np.random.seed(SEED)  # identical seed per engine -> differences are model-driven
        pf = BootstrapPF(model=pooled_model, n_particles=N_PARTICLES)
        pf.run(engine_df)

        rul_trajectory = extract_rul_trajectory(
            pf, failure_threshold=mu_fail, threshold_std=sigma_fail,
            rul_cap=RUL_CAP, max_horizon=MAX_HORIZON,
        )

        # Health-index percentiles straight from the particle cloud, same
        # unweighted-percentile convention as
        # particle_twin.library.SimilarityStreamingFilter.extract_health_trajectory
        # (pf.history[0] is the pre-observation prior, skipped here too).
        health_cycles = np.array([entry["cycle_number"] for entry in pf.history[1:]])
        health_p5 = np.array([np.percentile(entry["particles"], 5) for entry in pf.history[1:]])
        health_p50 = np.array([np.percentile(entry["particles"], 50) for entry in pf.history[1:]])
        health_p95 = np.array([np.percentile(entry["particles"], 95) for entry in pf.history[1:]])
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
        print(f"[{n_done:>3}/{n_total}] engine={engine_id:>3}  n_cycles={n_cycles:>4}  "
              f"({elapsed:.1f}s, total {total_elapsed:.1f}s)")

    db.close()
    print(f"\n[OK] Done. Total wall-clock: {time.perf_counter() - t_start:.1f}s")
    print(f"[OK] exp_id={exp_id}")


if __name__ == "__main__":
    main()
