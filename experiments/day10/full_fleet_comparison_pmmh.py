"""
full_fleet_comparison_pmmh.py — Day 12: rerun the full 100-engine
pooled/one-shot/periodic comparison (full_fleet_comparison.py) with the
PMMH-calibrated library instead of the pure-regression one.

Per the handoff's own "Report back" instruction: same script, same
matching mechanism (MMD, top-5, one-shot/periodic timing), same
n_particles/sigma_v/seed as full_fleet_comparison.py -- the ONLY change is
library[uid]['growth_rate'], which is swapped from the OLS regression fit
to run_pmmh_joint_fullfleet.py's per-engine growth_rate_mean (a converged
PMMH estimate for 31/100 engines, the same regression value for the other
69 that fell back). This isolates the effect of the growth_rate change
from every other design choice, so the comparison to
day11_fullfleet_decision.md's numbers is apples-to-apples.

Run from the repo root:
    python experiments/day10/full_fleet_comparison_pmmh.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from experiments.day10.similarity_library_prototype import (  # noqa: E402
    run_streaming_pf, _load_or_build_pooled_and_library,
    _json_default, N_PARTICLES, SIGMA_V, MAX_HORIZON, WARMUP_CYCLES,
    REMATCH_EVERY, K_TOP,
)
from particle_twin.analysis.rul import extract_rul_trajectory  # noqa: E402
from particle_twin.analysis import metrics as m  # noqa: E402

PMMH_JSON = "experiments/day10/pmmh_joint_fullfleet_results.json"
OUT_JSON = "experiments/day10/full_fleet_results_pmmh.json"


def main():
    t_setup = time.time()
    pooled_model, library, test, rul_true = _load_or_build_pooled_and_library()
    print(f"[setup] {time.time()-t_setup:.1f}s")

    with open(PMMH_JSON) as f:
        pmmh_results = json.load(f)["results"]
    assert len(pmmh_results) == len(library), (
        f"PMMH results cover {len(pmmh_results)} engines, library has {len(library)} -- "
        "full_pmmh rebuild must be complete before this comparison runs."
    )

    n_swapped = 0
    for uid_str, rec in pmmh_results.items():
        uid = int(uid_str)
        old_rate = library[uid]['growth_rate']
        library[uid]['growth_rate'] = rec['growth_rate_mean']
        library[uid]['pmmh_source'] = rec['source']
        library[uid]['pmmh_accept_rate'] = rec['accept_rate']
        if rec['source'] == 'pmmh':
            n_swapped += 1
    print(f"[OK] Swapped growth_rate for all {len(library)} library engines "
          f"({n_swapped} from converged PMMH chains, {len(library)-n_swapped} unchanged regression fallback)")

    rates = np.array([r['growth_rate'] for r in library.values()])
    print(f"[OK] PMMH-calibrated library growth_rate: mean={rates.mean():.6f} std={rates.std():.6f} "
          f"min={rates.min():.6f} max={rates.max():.6f}")

    results = {}
    t0_all = time.time()
    for uid in sorted(int(u) for u in test['unit_id'].unique()):
        engine_df = test[test['unit_id'] == uid].sort_values('cycle')
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true['unit_id'] == uid, 'true_rul'].values[0])

        variants = {
            'pooled_only': [],
            'one_shot': [min(WARMUP_CYCLES, n_cycles - 1)] if n_cycles > 1 else [],
            'periodic': [i for i in range(min(WARMUP_CYCLES, n_cycles - 1), n_cycles, REMATCH_EVERY)] if n_cycles > 1 else [],
        }

        engine_result = {'n_cycles': n_cycles, 'true_rul': true_val}
        for variant_name, rematch_at in variants.items():
            t0 = time.time()
            np.random.seed(42)
            pf, rematch_log = run_streaming_pf(pooled_model, engine_df, library, rematch_at,
                                                n_particles=N_PARTICLES, sigma_v=SIGMA_V, k_top=K_TOP)
            trajectory = extract_rul_trajectory(pf, failure_threshold=0.0, max_horizon=MAX_HORIZON)
            ev = m.evaluate_engine(trajectory, true_rul=true_val, n_cycles=n_cycles, n_particles=N_PARTICLES)
            score = float(m.phm08_score(np.array([ev['final_rul_median']]), np.array([true_val]))[0])
            engine_result[variant_name] = {
                'final_rul_median': ev['final_rul_median'],
                'final_abs_error': ev['final_abs_error'],
                'trajectory_rmse': ev['trajectory_rmse'],
                'phm08_score': score,
                'n_rematches': len(rematch_log),
            }
            dt = time.time() - t0
            print(f"  engine {uid:>3} [{variant_name:>11}] pred={ev['final_rul_median']:>7.1f}  "
                  f"true={true_val:>6.1f}  abs_err={ev['final_abs_error']:>7.1f}  ({dt:.2f}s)")
        results[uid] = engine_result

    os.makedirs("experiments/day10", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            'config': {
                'warmup_cycles': WARMUP_CYCLES, 'rematch_every': REMATCH_EVERY, 'k_top': K_TOP,
                'n_particles': N_PARTICLES, 'sigma_v': SIGMA_V, 'max_horizon': MAX_HORIZON,
                'library_source': 'pmmh_joint (fallback to regression when accept_rate < 0.10)',
            },
            'library_summary': {
                'n_engines': len(library),
                'n_pmmh_converged': n_swapped,
                'n_regression_fallback': len(library) - n_swapped,
                'growth_rate_mean': float(rates.mean()), 'growth_rate_std': float(rates.std()),
                'pooled_growth_rate': pooled_model.degradation_learner.params['growth_rate'],
            },
            'results': results,
        }, f, indent=2, default=_json_default)

    print(f"\n[OK] processed {len(results)} engines in {time.time()-t0_all:.1f}s, wrote {OUT_JSON}")


if __name__ == "__main__":
    main()