"""
library_ci_coverage.py — closes a gap found while drafting Chapter 5:
the Day 10/11 full-fleet comparison (full_fleet_comparison.py) only
tracked final_rul_median / final_abs_error / trajectory_rmse / phm08_score
per engine -- it never extracted rul_p5/rul_p95, so ci_coverage
(Eq. eq:ci-coverage) was never actually checked for the one-shot /
periodic library variants. Day 9 found ci_coverage=0.000 for pooled
dynamics; this script checks whether the library fix improved that at
all, which the thesis's central UQ argument depends on knowing before
Chapter 5 can honestly say whether the calibration problem is fixed.

Chunked (--start/--end) for the sandbox's 45s-per-call limit, same
pattern as full_fleet_comparison.py. Only checks one-shot (thesis
default) and pooled (consistency check against Day 9's known 0.000).

Run from repo root: python experiments/day10/library_ci_coverage.py --start 1 --end N
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from experiments.day10.similarity_library_prototype import (  # noqa: E402
    run_streaming_pf, _load_or_build_pooled_and_library, _json_default,
    WARMUP_CYCLES, N_PARTICLES, SIGMA_V, K_TOP, MAX_HORIZON,
)
from particle_twin.analysis.rul import extract_rul_trajectory  # noqa: E402
from particle_twin.analysis.metrics import ci_coverage  # noqa: E402

OUT_JSON = "experiments/day10/library_ci_results.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()
    engines_to_run = list(range(args.start, args.end + 1))

    pooled_model, library, test, rul_true = _load_or_build_pooled_and_library()

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            results = {int(k): v for k, v in json.load(f).get("results", {}).items()}

    def _save():
        with open(OUT_JSON, "w") as f:
            json.dump({"results": results}, f, indent=2, default=_json_default)

    t0_all = time.time()
    for uid in engines_to_run:
        engine_df = test[test["unit_id"] == uid].sort_values("cycle")
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true["unit_id"] == uid, "true_rul"].values[0])

        engine_result = {"n_cycles": n_cycles, "true_rul": true_val}
        for variant_name, rematch_at in [
            ("pooled_only", []),
            ("one_shot", [min(WARMUP_CYCLES, n_cycles - 1)] if n_cycles > 1 else []),
        ]:
            t0 = time.time()
            np.random.seed(42)
            pf, _ = run_streaming_pf(pooled_model, engine_df, library, rematch_at,
                                      n_particles=N_PARTICLES, sigma_v=SIGMA_V, k_top=K_TOP)
            traj = extract_rul_trajectory(pf, failure_threshold=0.0, max_horizon=MAX_HORIZON)
            final = traj[-1]
            engine_result[variant_name] = {
                "rul_median": final["rul_median"], "rul_p5": final["rul_p5"], "rul_p95": final["rul_p95"],
            }
            print(f"  engine {uid:>3} [{variant_name:>11}] median={final['rul_median']:>7.1f} "
                  f"p5={final['rul_p5']:>7.1f} p95={final['rul_p95']:>7.1f} true={true_val:>6.1f} "
                  f"({time.time()-t0:.2f}s)")
        results[uid] = engine_result
        _save()

    print(f"\n[OK] processed {len(engines_to_run)} engines in {time.time()-t0_all:.1f}s "
          f"({len(results)}/100 total so far)")

    if len(results) == 100:
        for variant in ["pooled_only", "one_shot"]:
            trues = [results[u]["true_rul"] for u in results]
            lowers = [results[u][variant]["rul_p5"] for u in results]
            uppers = [results[u][variant]["rul_p95"] for u in results]
            cov = ci_coverage(trues, lowers, uppers)
            print(f"[COVERAGE] {variant}: {cov:.3f}")


if __name__ == "__main__":
    main()
