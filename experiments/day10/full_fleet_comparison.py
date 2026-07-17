"""
full_fleet_comparison.py — Day 10/11: run the Day-10 similarity-library
prototype (pooled / one-shot / periodic) on the FULL 100-engine FD001 test
fleet, not just the 5 diagnostic engines in similarity_library_prototype.py.

Deliberately reuses every function from similarity_library_prototype.py
unchanged (build_library, match_library, run_streaming_pf, etc.) -- this
script only changes WHICH/HOW MANY engines are processed and where the
output goes, so the 5-engine prototype numbers stay reproducible and this
full run is directly comparable (same seed, same config).

Regression-based growth rates only (no PMMH calibration) -- PMMH joint
(growth_rate, sigma_v) estimation is a separate, heavier follow-up (per
Thesis_completing_Progress.md) that needs to run locally via Claude Code,
not in the Cowork sandbox. This script answers a narrower, still-useful
question first: does one-shot or periodic win at full-fleet scale, using
the same regression-based library the 5-engine prototype already used?

Resumable-chunk design (same pattern as Day 5's run_full_fd001.py) to work
around the sandbox's per-call time limit: pass --start/--end to process a
slice of the 100 engines per call; results merge into one JSON file.

Run from the repo root:
    python full_fleet_comparison.py --start 1 --end 20
    python full_fleet_comparison.py --start 21 --end 40
    ... etc, until all 100 are done.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "day10_lib"))
sys.path.insert(0, ".")

from experiments.day10.similarity_library_prototype import (  # noqa: E402
    build_library, run_streaming_pf, _load_or_build_pooled_and_library,
    _json_default, N_PARTICLES, SIGMA_V, MAX_HORIZON, WARMUP_CYCLES,
    REMATCH_EVERY, K_TOP,
)
from particle_twin.analysis.rul import extract_rul_trajectory  # noqa: E402
from particle_twin.analysis import metrics as m  # noqa: E402

OUT_JSON = "experiments/day10/full_fleet_results.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--variants", type=str, default="pooled_only,one_shot,periodic")
    args = parser.parse_args()
    engines_to_run = list(range(args.start, args.end + 1))
    variants_to_run = args.variants.split(",")

    t_setup = time.time()
    pooled_model, library, test, rul_true = _load_or_build_pooled_and_library()
    rates = np.array([r['growth_rate'] for r in library.values()])
    print(f"[setup] {time.time()-t_setup:.1f}s")

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            prior = json.load(f)
        results = {int(k): v for k, v in prior.get('results', {}).items()}

    def _save():
        os.makedirs("experiments/day10", exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({
                'config': {
                    'warmup_cycles': WARMUP_CYCLES, 'rematch_every': REMATCH_EVERY, 'k_top': K_TOP,
                    'n_particles': N_PARTICLES, 'sigma_v': SIGMA_V, 'max_horizon': MAX_HORIZON,
                },
                'library_summary': {
                    'n_engines': len(library),
                    'growth_rate_mean': float(rates.mean()), 'growth_rate_std': float(rates.std()),
                    'pooled_growth_rate': pooled_model.degradation_learner.params['growth_rate'],
                },
                'results': results,
            }, f, indent=2, default=_json_default)

    t0_all = time.time()
    for uid in engines_to_run:
        engine_df = test[test['unit_id'] == uid].sort_values('cycle')
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true['unit_id'] == uid, 'true_rul'].values[0])

        variants = {
            'pooled_only': [],
            'one_shot': [min(WARMUP_CYCLES, n_cycles - 1)] if n_cycles > 1 else [],
            'periodic': [i for i in range(min(WARMUP_CYCLES, n_cycles - 1), n_cycles, REMATCH_EVERY)] if n_cycles > 1 else [],
        }

        engine_result = results.get(str(uid), results.get(uid, {'n_cycles': n_cycles, 'true_rul': true_val}))
        engine_result['n_cycles'] = n_cycles
        engine_result['true_rul'] = true_val

        for variant_name, rematch_at in variants.items():
            if variant_name not in variants_to_run:
                continue
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
        _save()

    print(f"\n[OK] processed {len(engines_to_run)} engines in {time.time()-t0_all:.1f}s, "
          f"wrote {OUT_JSON} ({len(results)}/100 engines total so far)")


if __name__ == "__main__":
    main()
