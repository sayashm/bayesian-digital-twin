"""
01_run_experiment.py — PF PMMH experiment, FD001: run the bootstrap
particle filter with ONE pooled set of degradation parameters -- exactly
PF_Fixed_Parameters' architecture -- but with (growth_rate, sigma_v) taken
from PMMH instead of the OLS/hand-set pooled fit, using the v2 RUL
extraction pipeline (calibrated failure threshold, per-particle threshold
uncertainty, rul_cap) from particle_twin/analysis/rul.py.

============================================================================
WHAT "PF PMMH" MEANS, AND HOW IT DIFFERS FROM ITS TWO SIBLINGS
============================================================================
PF_Fixed_Parameters/FD001/01_run_experiment.py fits ONE DegradationModel on
the pooled training fleet via OLS (growth_rate) with a hand-set
sigma_v=0.5, and applies it, completely unchanged, to every test engine --
the "before" baseline.

library_method/FD001/02_run_experiment.py goes to the opposite extreme:
one (growth_rate, sigma_v) pair PER ENGINE, borrowed from the most similar
training engines via SimilarityStreamingFilter, with one-shot/periodic
rematching.

This script sits in between: still ONE pooled parameter set applied to
every test engine (no per-engine matching, no rematching) -- but that one
set is the FLEET-AVERAGED, PMMH-calibrated (growth_rate, sigma_v) already
computed and persisted by library_method/FD001/01_build_library.py
(results.db, library_exp_id read from that folder's config.json). That
library blends a converged joint-PMMH fit (particle_twin/inference/pmmh.py
::pmmh_sample_joint) per training engine with an OLS regression fallback
for chains that didn't mix (Day 12's fallback rule) -- averaging over all
100 training engines' growth_rate/sigma_v gives one MCMC-informed pooled
estimate, reusing already-completed MCMC output rather than re-running
PMMH from scratch. This isolates ONE variable relative to
PF_Fixed_Parameters -- "PMMH-informed pooled parameters" vs. "OLS pooled
parameters" -- while holding the "one set for the whole fleet, no
per-engine specialisation" architecture fixed, so the three-way comparison
(PF_Fixed_Parameters vs. PF_PMMH vs. library_method) isolates exactly one
design choice at a time.

RUL settings (N_PARTICLES, RUL_CAP, MAX_HORIZON, calibration constants)
are identical to PF_Fixed_Parameters/FD001/01_run_experiment.py, so the
comparison is not confounded by different RUL-extraction settings.

OUTPUT
============================================================================
  - results.db: rul_estimates rows under one fresh exp_id
    (particle_twin.data.database.ResultsDB), one row per (engine, cycle).
  - experiments/final_experiment/PF_PMMH/FD001/config.json:
    records the exp_id + PMMH-derived (growth_rate, sigma_v) + calibration
    constants for the later steps.

Run from the repo root (~2-3 minutes for all 100 engines, n_particles=500;
excludes the PMMH library build, which is already done):
    python experiments/final_experiment/PF_PMMH/FD001/01_run_experiment.py
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
# run in this project; RUL settings match PF_Fixed_Parameters/FD001's
# 01_run_experiment.py so the two experiments are directly comparable.
# ---------------------------------------------------------------------------
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
POOLED_SIGMA_V = 0.5     # pooled-model construction value; overwritten below by the PMMH estimate
SIGMA_0 = 0.05

N_PARTICLES = 500

# --- v2 RUL calibration (particle_twin/analysis/rul.py) -------------------
CALIBRATION_N_ENGINES = 30      # rul.py: between-engine std already ~0.026 at n=30
CALIBRATION_N_PARTICLES = 300   # separate (smaller/faster) from the fleet's N_PARTICLES
RUL_CAP = 350
MAX_HORIZON = 350        # whichever of RUL_CAP/MAX_HORIZON is smaller binds first
SEED = 42

DB_PATH = "results.db"
CONFIG_JSON = "experiments/final_experiment/PF_PMMH/FD001/config.json"
LIBRARY_CONFIG_JSON = "experiments/final_experiment/library_method/FD001/config.json"


def _load_config() -> dict:
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def _pmmh_pooled_params(db: ResultsDB) -> tuple[float, float, int, dict]:
    """
    Fleet-averaged (growth_rate, sigma_v) from library_method/FD001's
    already-built PMMH-overlaid library (results.db's engine_parameters
    table, exp_id read from library_method/FD001/config.json). Averaging
    over all 100 training engines' (blended PMMH + regression-fallback)
    growth_rate/sigma_v gives one MCMC-informed pooled estimate -- see
    module docstring.
    """
    with open(LIBRARY_CONFIG_JSON) as f:
        library_config = json.load(f)
    library_exp_id = library_config["library_exp_id"]

    records = db.get_engine_parameters(library_exp_id)
    if not records:
        raise RuntimeError(f"No engine_parameters rows for library_exp_id={library_exp_id} -- "
                            f"run library_method/FD001/01_build_library.py first.")

    growth_rates = np.array([r["growth_rate"] for r in records])
    sigma_vs = np.array([r["sigma_v"] for r in records])
    n_pmmh = sum(1 for r in records if r["source"] == "pmmh")

    summary = {
        "library_exp_id": library_exp_id, "n_engines": len(records), "n_pmmh_converged": n_pmmh,
        "n_regression_fallback": len(records) - n_pmmh,
        "growth_rate_mean": float(growth_rates.mean()), "growth_rate_std": float(growth_rates.std()),
        "sigma_v_mean": float(sigma_vs.mean()), "sigma_v_std": float(sigma_vs.std()),
    }
    return float(growth_rates.mean()), float(sigma_vs.mean()), library_exp_id, summary


def main():
    config = _load_config()

    # ------------------------------------------------------------------
    # 1. Load data, fit the pooled model (shared HI builder / measurement
    #    noise -- only the transition dynamics get PMMH-swapped below).
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
    print(f"[OK] Pooled model fit — R^2="
          f"{pooled_model.fit_report['degradation_model']['r_squared']:.3f}")

    # ------------------------------------------------------------------
    # 1b. Fleet-averaged PMMH (growth_rate, sigma_v), reused from the
    #     already-built library_method library -- see module docstring.
    # ------------------------------------------------------------------
    db = ResultsDB(DB_PATH)
    growth_rate_pmmh, sigma_v_pmmh, library_exp_id, pmmh_summary = _pmmh_pooled_params(db)
    print(f"[OK] PMMH-informed pooled params (from library_exp_id={library_exp_id}, "
          f"{pmmh_summary['n_pmmh_converged']}/{pmmh_summary['n_engines']} converged chains): "
          f"growth_rate={growth_rate_pmmh:.6f}, sigma_v={sigma_v_pmmh:.4f}")

    pmmh_model = pooled_model.with_matched_dynamics(growth_rate_pmmh, sigma_v_pmmh)

    # ------------------------------------------------------------------
    # 1c. Calibrate the v2 failure threshold ONCE (posterior space), using
    #     the SAME PMMH-calibrated model this experiment deploys per test
    #     engine (sim_filter=None -> calibrate_failure_threshold's own
    #     BootstrapPF-on-`model` fallback, so the calibration model is
    #     identical to the one actually deployed below).
    # ------------------------------------------------------------------
    mu_fail, sigma_fail = calibrate_failure_threshold(
        pmmh_model, train, mode="posterior", sim_filter=None,
        n_engines=CALIBRATION_N_ENGINES, n_particles=CALIBRATION_N_PARTICLES,
        seed=SEED,
    )
    print(f"[OK] Calibrated failure threshold (posterior space): "
          f"mu_fail={mu_fail:.4f}, sigma_fail={sigma_fail:.4f}")

    # ------------------------------------------------------------------
    # 2. One dedicated exp_id -- created once, reused across resumed runs.
    # ------------------------------------------------------------------
    if "exp_id" in config:
        exp_id = config["exp_id"]
        print(f"[OK] Resuming existing experiment, exp_id={exp_id}")
    else:
        exp_id = db.create_experiment(
            dataset=DATASET, n_particles=N_PARTICLES, sigma_v=sigma_v_pmmh, sigma_w=SIGMA_0,
            extra_params={
                "step": "final_experiment/PF_PMMH/FD001/01_run_experiment",
                "max_horizon": MAX_HORIZON, "rul_cap": RUL_CAP,
                "mu_fail": mu_fail, "sigma_fail": sigma_fail, "threshold_mode": "posterior",
                "growth_rate_pmmh": growth_rate_pmmh, "sigma_v_pmmh": sigma_v_pmmh,
                "library_exp_id": library_exp_id,
            },
            description="Final experiment (PF PMMH, FD001) — one pooled DegradationModel "
                        "run over every test engine, dynamics from the fleet-averaged PMMH "
                        "library, no per-engine matching, v2 RUL extraction.",
        )
        config["exp_id"] = exp_id
        config["mu_fail"] = mu_fail
        config["sigma_fail"] = sigma_fail
        config["growth_rate_pmmh"] = growth_rate_pmmh
        config["sigma_v_pmmh"] = sigma_v_pmmh
        config["library_exp_id"] = library_exp_id
        config["pmmh_summary"] = pmmh_summary
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
    # 3. Run every test engine through the plain BootstrapPF, using the
    #    PMMH-calibrated pooled model (fixed across the fleet -- no
    #    similarity matching of any kind).
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
        pf = BootstrapPF(model=pmmh_model, n_particles=N_PARTICLES)
        pf.run(engine_df)

        rul_trajectory = extract_rul_trajectory(
            pf, failure_threshold=mu_fail, threshold_std=sigma_fail,
            rul_cap=RUL_CAP, max_horizon=MAX_HORIZON,
        )

        # Health-index percentiles straight from the particle cloud, same
        # unweighted-percentile convention as PF_Fixed_Parameters and
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
