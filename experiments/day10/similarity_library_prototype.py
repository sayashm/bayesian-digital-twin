"""
similarity_library_prototype.py — Day 10 prototype: per-engine reference
library + similarity matching, as a fast/small-scale alternative to the
single fleet-pooled degradation curve exposed as the root cause of the
Day 9 PHM08-score blowup.

WHAT THIS PROTOTYPE DOES (small scope, on purpose)
----------------------------------------------------------------------
1. Builds a REFERENCE LIBRARY: instead of one exponential decay rate
   fit across all 100 FD001 training engines pooled together
   (particle_twin/models/degradation_learner.py's existing behaviour,
   used unchanged for every test engine), fit ONE exponential curve PER
   TRAINING ENGINE, using the engine's own (cycle, HI) series. This
   reuses DegradationModelLearner exactly as-is -- it already supports
   single-engine fitting, we just call it 100 times instead of once on
   pooled data.

2. Implements a SIMILARITY MATCH between a test engine's observed
   Health Index so far and every library engine's Health Index over the
   same number of cycles, using biased MMD with an RBF kernel (the same
   statistic the paper uses, Eq. (3) in Cai et al. 2020 -- simplified
   here to a scalar Health Index instead of the paper's full sensor
   feature matrix, and to top-k nearest neighbours by MMD rather than
   the paper's full KTST pass/fail hypothesis test -- both simplifications
   are called out explicitly in day10_problem_solution.md).

3. Implements TWO streaming variants of when the match is (re-)done,
   both driven by the same underlying step-by-step filter loop
   (run_streaming_pf below), which reuses BootstrapPF's own
   normalize/ESS/resample methods untouched -- no changes to
   filters/bootstrap.py or models/state_space.py:
     - ONE-SHOT:   match once after a warm-up window, then keep that
                   matched decay rate fixed for the rest of the engine's
                   life (closest to the paper's own method).
     - PERIODIC:   re-match every N cycles using all data observed so
                   far, letting the matched rate change over time (more
                   faithful to a live/streaming digital twin).
   Before the FIRST match, both variants fall back to the existing
   pooled fleet dynamics (nothing better is known yet) -- realistic for
   an actual streaming deployment, and a deliberate design choice, not
   an oversight.

4. Runs both variants (+ a rematch_at=[] "pooled-only" run, which should
   reproduce Day 9's per-engine numbers as an internal consistency
   check) on a handful of FD001 test engines already used in earlier
   diagnostics (1, 10, 20, 30, 77), and tabulates final RUL error,
   PHM08 score, and trajectory RMSE for each.

Run from the repo root:
    python experiments/day10/similarity_library_prototype.py
"""

import argparse
import copy
import json
import os
import time

import numpy as np

from particle_twin.data.loader import CMAPSSLoader
from particle_twin.models.state_space import DegradationModel
from particle_twin.models.degradation_learner import DegradationModelLearner
from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.analysis.rul import extract_rul_trajectory
from particle_twin.analysis import metrics as m

np.random.seed(42)

# ---- Must match Day 5 / Day 9 exactly, so the rematch_at=[] pooled-only
# ---- run reproduces the already-validated numbers. ----
SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
DATASET = "FD001"
N_PARTICLES = 500
SIGMA_V = 0.5
FAILURE_THRESHOLD = 0.0
# NOTE: experiments/run_full_fd001.py's *committed* MAX_HORIZON constant
# reads 300, but the actual numbers stored in experiments/fd001_full_results.json
# (which Day 9's fd001_day9_metrics.json reuses verbatim, and which this
# script's consistency check compares against) contain rul_median values
# well above 300 (e.g. engine 1: 495, with a peak of 954.5 mid-trajectory) --
# only reachable if that run actually used MAX_HORIZON=1500, per the Day 4
# status note ("raised to 1500 to fix this specifically"). The script and its
# own stored output disagree; 1500 is what was actually used to produce the
# baseline numbers, so that's what this script uses too, for a fair/matching
# comparison. Flagged for Sajjad separately -- not this prototype's job to fix.
MAX_HORIZON = 1500

# ---- Prototype-specific parameters (small scope, on purpose) ----
TEST_ENGINES = [1, 10, 20, 30, 77]     # reuse engines from earlier diagnostics
WARMUP_CYCLES = 20                     # cycles observed before the first match
REMATCH_EVERY = 20                     # periodic variant: re-match interval
K_TOP = 5                              # number of nearest library engines averaged
MMD_LENGTH_SCALE = 1.0                 # RBF kernel length scale, HI on [0,1] scale

DAY9_JSON = "experiments/day9/fd001_day9_metrics.json"   # for the consistency check
OUT_DIR = "experiments/day10"
OUT_JSON = os.path.join(OUT_DIR, "similarity_library_results.json")
# NOTE: an earlier version of this script cached the pooled model + library
# to a pickle file to survive the sandbox's 45s-per-call limit. Turned out
# unnecessary -- fitting + building the 100-engine library takes ~0.3s, not
# worth caching (and DegradationModelLearner's dynamics_func closures aren't
# picklable anyway). --engines/--variants below handle the 45s limit instead,
# by letting each call process one test engine at a time.


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════════
# 1. Biased MMD (RBF kernel) — Eq. (3) in Cai et al. 2020, scalar version
# ══════════════════════════════════════════════════════════════════════════

def biased_mmd(x, y, length_scale: float = 1.0) -> float:
    """Biased MMD between two 1-D sets of scalars (HI values, [0,1] scale)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def rbf(a, b):
        diff = a[:, None] - b[None, :]
        return np.exp(-(diff ** 2) / (2.0 * length_scale ** 2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return float(np.sqrt(max(kxx + kyy - 2.0 * kxy, 0.0)))


def match_library(test_hi_so_far: np.ndarray, library: dict, k_top: int = K_TOP,
                   length_scale: float = MMD_LENGTH_SCALE):
    """
    Rank every library engine by MMD against the test engine's HI observed
    so far (compared over the SAME number of cycles, from the start —
    "locate the test data at the beginning of Fm", per the paper's Fig. 2a).

    Returns (ranked, avg_growth_rate) where ranked is a list of
    (unit_id, mmd) sorted ascending (most similar first) and
    avg_growth_rate is the mean growth_rate of the top-k matches.
    """
    w = len(test_hi_so_far)
    scored = []
    for uid, rec in library.items():
        if len(rec['hi']) < w:
            continue  # library engine too short to compare over this window
        ref_window = rec['hi'][:w]
        mmd = biased_mmd(test_hi_so_far, ref_window, length_scale=length_scale)
        scored.append((uid, mmd))
    scored.sort(key=lambda t: t[1])
    top = scored[:k_top]
    avg_rate = float(np.mean([library[u]['growth_rate'] for u, _ in top]))
    return top, avg_rate


# ══════════════════════════════════════════════════════════════════════════
# 2. Reference library: one exponential fit PER training engine
# ══════════════════════════════════════════════════════════════════════════

def build_library(train_df, hi_builder, sigma_v: float = SIGMA_V) -> dict:
    """
    For every training engine, compute its own HI series (via the SAME
    shared hi_builder used by the pooled model, so scales match) and fit
    an individual exponential DegradationModelLearner on just that
    engine's (cycle, HI) series -- no pooling across engines.
    """
    library = {}
    for uid in sorted(train_df['unit_id'].unique()):
        edf = train_df[train_df['unit_id'] == uid].sort_values('cycle')
        hi = hi_builder.transform(edf).to_numpy()          # 0-100 scale
        cycles = edf['cycle'].to_numpy(dtype=float)

        learner = DegradationModelLearner(
            health_series=hi, time=cycles, model_type="exponential", sigma_v=sigma_v,
        )
        library[uid] = {
            'growth_rate': learner.params['growth_rate'],
            'D': learner.params['D'],
            'r_squared': learner.fit_quality['r_squared'],
            'hi': hi / 100.0,       # store on [0,1] scale, matches MMD comparison scale
            'cycles': cycles,
        }
    return library


def make_matched_dynamics(growth_rate: float, sigma_v: float):
    """Same functional form as DegradationModelLearner._fit_exponential's
    exponential_dynamics, but with an externally supplied growth_rate
    (the matched-library average) instead of the fleet-pooled one."""
    def dyn(health, dt=1):
        damage = 100 - health
        new_damage = damage * np.exp(growth_rate * dt)
        noise = np.random.normal(0, sigma_v, size=np.shape(health))
        return np.clip(100 - new_damage + noise, 0, 100)
    return dyn


class _StubLearner:
    """Minimal stand-in for DegradationModelLearner exposing only what
    DegradationModel.transition() actually reads (.dynamics_func)."""
    def __init__(self, dynamics_func, growth_rate):
        self.dynamics_func = dynamics_func
        self.growth_rate = growth_rate  # kept for logging/diagnostics only


def matched_model_from(pooled_model: DegradationModel, growth_rate: float, sigma_v: float) -> DegradationModel:
    """Shallow-copy the pooled model (sharing hi_builder / measurement_learner
    -- both stay fleet-fit, unchanged) and swap in a matched dynamics_func."""
    mm = copy.copy(pooled_model)
    mm.degradation_learner = _StubLearner(make_matched_dynamics(growth_rate, sigma_v), growth_rate)
    return mm


# ══════════════════════════════════════════════════════════════════════════
# 3. Streaming PF driver — same step logic as BootstrapPF.run(), but the
#    active model can be swapped mid-sequence at rematch points.
# ══════════════════════════════════════════════════════════════════════════

def run_streaming_pf(pooled_model: DegradationModel, engine_df, library: dict,
                      rematch_at: list[int], n_particles: int = N_PARTICLES,
                      ess_threshold: float = 0.5, sigma_v: float = SIGMA_V,
                      k_top: int = K_TOP):
    """
    Runs the bootstrap filter cycle-by-cycle. At each cycle index in
    rematch_at (0-indexed position into the sorted cycle list, NOT the
    raw cycle number), re-match against the library using all HI
    observed so far and switch the active dynamics_func going forward.
    Before the first rematch point, uses the pooled fleet model
    (nothing better is known yet).

    Returns (pf, rematch_log) where pf is a BootstrapPF instance with
    .history populated (ready for extract_rul_trajectory) and .model set
    to whichever dynamics was active last (used for RUL extrapolation),
    and rematch_log is a list of dicts logging each rematch event.
    """
    engine_df = engine_df.sort_values('cycle')
    cycles = engine_df['cycle'].to_numpy()
    y = pooled_model.observe(df=engine_df).to_numpy()   # HI on [0,1] scale, shared across variants

    pf = BootstrapPF(model=pooled_model, n_particles=n_particles, ess_threshold=ess_threshold)
    h = pooled_model.sample_initial(n_particles)
    log_w = np.full(n_particles, -np.log(n_particles))
    pf.history = [{'cycle_number': 0, 'particles': h, 'weights': log_w, 'ESS': pf.ESS(log_w)}]

    active_model = pooled_model
    rematch_log = []
    rematch_set = set(rematch_at)

    for i, t in enumerate(cycles):
        if i in rematch_set:
            top, avg_rate = match_library(y[:i] if i > 0 else y[:1], library, k_top=k_top)
            active_model = matched_model_from(pooled_model, avg_rate, sigma_v)
            rematch_log.append({
                'cycle_index': i, 'cycle_number': int(t),
                'matched_engines': [int(u) for u, _ in top],
                'matched_mmds': [float(d) for _, d in top],
                'avg_growth_rate': avg_rate,
            })

        h = active_model.transition(h)
        log_w = log_w + active_model.log_likelihood(h, y[i])
        if pf.ESS(log_w) < ess_threshold * n_particles:
            h, log_w = pf.resample(h, log_w)
        pf.history.append({'cycle_number': t, 'particles': h, 'weights': log_w, 'ESS': pf.ESS(log_w)})

    pf.model = active_model   # RUL extrapolation continues with the last-active dynamics
    return pf, rematch_log


# ══════════════════════════════════════════════════════════════════════════
# 4. Main: build library, run 3 variants (pooled-only / one-shot / periodic)
#    on the selected test engines, tabulate results.
# ══════════════════════════════════════════════════════════════════════════

def _load_or_build_pooled_and_library():
    """Fit the pooled model + build the 100-engine library. Turns out to be
    cheap (~0.3s total, it's just linregress over pre-loaded arrays, not
    particle filtering) so no disk caching is needed -- redone every call."""
    loader = CMAPSSLoader(data_dir="data/cmapss")
    train = loader.load_train(DATASET)
    test = loader.load_test(DATASET)
    rul_true = loader.load_rul(DATASET)

    t0 = time.time()
    pooled_model = DegradationModel(hi_method="weighted", degradation_model="exponential",
                                     measurement_method="gaussian", sigma_v=SIGMA_V, sigma_0=0.05,
                                     sensor_cols=SENSOR_COLS)
    pooled_model.fit(train)
    print(f"[OK] Pooled model fit — R^2={pooled_model.fit_report['degradation_model']['r_squared']:.3f}, "
          f"growth_rate={pooled_model.degradation_learner.params['growth_rate']:.6f}")

    library = build_library(train, pooled_model.hi_builder, sigma_v=SIGMA_V)
    rates = np.array([r['growth_rate'] for r in library.values()])
    print(f"[OK] Built library of {len(library)} training engines in {time.time()-t0:.1f}s. "
          f"Per-engine growth_rate: mean={rates.mean():.6f}, std={rates.std():.6f}, "
          f"min={rates.min():.6f}, max={rates.max():.6f} "
          f"(pooled fit used a single fixed value = {pooled_model.degradation_learner.params['growth_rate']:.6f})")

    return pooled_model, library, test, rul_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", type=str, default=None,
                         help="Comma-separated unit_ids to process this call (default: all TEST_ENGINES).")
    parser.add_argument("--variants", type=str, default="pooled_only,one_shot,periodic",
                         help="Comma-separated subset of {pooled_only,one_shot,periodic} to run this call.")
    args = parser.parse_args()
    engines_to_run = [int(x) for x in args.engines.split(",")] if args.engines else TEST_ENGINES
    variants_to_run = args.variants.split(",")

    pooled_model, library, test, rul_true = _load_or_build_pooled_and_library()
    rates = np.array([r['growth_rate'] for r in library.values()])

    with open(DAY9_JSON) as f:
        day9 = json.load(f)['per_engine']

    # Load existing results (from prior chunked calls) so we can merge in.
    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            prior = json.load(f)
        results = {int(k): v for k, v in prior.get('results', {}).items()}

    for uid in engines_to_run:
        engine_df = test[test['unit_id'] == uid].sort_values('cycle')
        n_cycles = len(engine_df)
        true_val = float(rul_true.loc[rul_true['unit_id'] == uid, 'true_rul'].values[0])

        variants = {
            'pooled_only': [],                                              # consistency check vs Day 9
            'one_shot': [min(WARMUP_CYCLES, n_cycles - 1)] if n_cycles > 1 else [],
            'periodic': [i for i in range(min(WARMUP_CYCLES, n_cycles - 1), n_cycles, REMATCH_EVERY)] if n_cycles > 1 else [],
        }

        engine_result = results.get(uid, {'n_cycles': n_cycles, 'true_rul': true_val,
                                           'day9_pooled': day9.get(str(uid))})
        for variant_name, rematch_at in variants.items():
            if variant_name not in variants_to_run:
                continue
            t0 = time.time()
            np.random.seed(42)  # same seed every variant -> differences are model-driven, not RNG-driven
            pf, rematch_log = run_streaming_pf(pooled_model, engine_df, library, rematch_at,
                                                n_particles=N_PARTICLES, sigma_v=SIGMA_V, k_top=K_TOP)
            trajectory = extract_rul_trajectory(pf, failure_threshold=FAILURE_THRESHOLD, max_horizon=MAX_HORIZON)
            ev = m.evaluate_engine(trajectory, true_rul=true_val, n_cycles=n_cycles, n_particles=N_PARTICLES)
            score = float(m.phm08_score(np.array([ev['final_rul_median']]), np.array([true_val]))[0])
            engine_result[variant_name] = {
                'final_rul_median': ev['final_rul_median'],
                'final_abs_error': ev['final_abs_error'],
                'trajectory_rmse': ev['trajectory_rmse'],
                'phm08_score': score,
                'rematch_log': rematch_log,
            }
            print(f"  engine {uid:>3} [{variant_name:>11}] pred={ev['final_rul_median']:>7.1f}  "
                  f"true={true_val:>6.1f}  abs_err={ev['final_abs_error']:>7.1f}  phm08={score:>14.4g}  "
                  f"({time.time()-t0:.1f}s)")
        results[uid] = engine_result

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            'config': {
                'test_engines': TEST_ENGINES, 'warmup_cycles': WARMUP_CYCLES,
                'rematch_every': REMATCH_EVERY, 'k_top': K_TOP,
                'mmd_length_scale': MMD_LENGTH_SCALE, 'n_particles': N_PARTICLES, 'sigma_v': SIGMA_V,
            },
            'library_summary': {
                'n_engines': len(library),
                'growth_rate_mean': float(rates.mean()), 'growth_rate_std': float(rates.std()),
                'growth_rate_min': float(rates.min()), 'growth_rate_max': float(rates.max()),
                'pooled_growth_rate': pooled_model.degradation_learner.params['growth_rate'],
            },
            'results': results,
        }, f, indent=2, default=_json_default)
    print(f"\n[OK] Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
