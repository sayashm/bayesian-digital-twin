from particle_twin.data.loader import CMAPSSLoader
from particle_twin.inference.pmmh import pmmh_sample_sigma_v, run_pf_for_sigma_v

import time
import numpy as np

np.random.seed(42)

SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
TEST_ENGINES = [1, 10, 20, 30]
ASSUMED_SIGMA_V = 0.3   # hardcoded default used everywhere else so far (Day 2.5-7)
BURN_IN_FRAC = 0.2      # discard the first 20% of each chain before summarizing
RUN_FULL_PMMH = True    # set False to only run the diagnostic below (much faster)
N_PARTICLES = 3200      # practical compromise -- see Day-8 diagnostic notes:
                        # std(logZ) plateaus well above the literature's ~1
                        # target long before this budget is reasonable, so
                        # more MCMC iterations compensate instead of more N

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data (same setup as test_bootstrap.py)
# ══════════════════════════════════════════════════════════════════════════
loader = CMAPSSLoader(data_dir='data/cmapss')
train = loader.load_train("FD001")
print(f"[✓] Loaded FD001 train: {len(train)} rows, {train['unit_id'].nunique()} engines")

# ══════════════════════════════════════════════════════════════════════════
# 2. Diagnostic: noise in the log-likelihood estimator at a FIXED sigma_v.
#    PMMH mixes well only if std(logZ) is roughly ~1 (Pitt et al. 2012;
#    Doucet et al. 2015) -- if it's much larger, n_particles is too small
#    and no amount of rw_step tuning will fix the "sticky chain" symptom
#    seen in the last run (long runs of repeated values, falling accept
#    rate as rw_step shrank).
# ══════════════════════════════════════════════════════════════════════════
diag_engine_df = train[train['unit_id'] == 1]
diag_logZ = np.array([
    run_pf_for_sigma_v(
        sigma_v=ASSUMED_SIGMA_V, train_df=train, engine_df=diag_engine_df,
        hi_method='weighted', degradation_model='exponential',
        measurement_method='gaussian', sigma_0=0.05, n_particles=N_PARTICLES,
    )
    for _ in range(20)
])
print("=" * 20)
print(f"DIAGNOSTIC: logZ noise at sigma_v={ASSUMED_SIGMA_V}, engine=1, n_particles={N_PARTICLES}, 20 repeats")
print(f"logZ values = {diag_logZ}")
print(f"mean(logZ) = {diag_logZ.mean():.3f}   std(logZ) = {diag_logZ.std():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. PMMH sanity-check run, one engine at a time (timed)
# ══════════════════════════════════════════════════════════════════════════
results = {}
if RUN_FULL_PMMH:
    total_start = time.perf_counter()
    for engine in TEST_ENGINES:
        engine_df = train[train['unit_id'] == engine]

        engine_start = time.perf_counter()
        pmmh = pmmh_sample_sigma_v(
            train_df=train, engine_df=engine_df, n_iterations=1000,
            hi_method='weighted', degradation_model='exponential',
            measurement_method='gaussian',
            sigma_0=0.05, n_particles=N_PARTICLES, sigma_v_init=0.3,
            rw_step=0.1, seed=42,
        )
        engine_elapsed = time.perf_counter() - engine_start
        results[engine] = pmmh

        chain = pmmh['sigma_v_chain']
        n_burn = int(BURN_IN_FRAC * len(chain))
        post_burn_in = chain[n_burn:]

        print("=" * 20)
        print(f"ENGINE = {engine}")
        print(f"elapsed = {engine_elapsed:.1f}s ({engine_elapsed / len(chain):.2f}s/iteration)")
        print(f"accept_rate = {pmmh['accept_rate']:.3f}")
        print(f"sigma_v_chain[:10] = {chain[:10]}")
        print(f"sigma_v_chain[-10:] = {chain[-10:]}")
        print(f"post-burn-in mean sigma_v   = {post_burn_in.mean():.4f}")
        print(f"post-burn-in median sigma_v = {np.median(post_burn_in):.4f}")
        print(f"assumed (hardcoded) sigma_v = {ASSUMED_SIGMA_V:.4f}")

    total_elapsed = time.perf_counter() - total_start
    print("=" * 20)
    print(f"TOTAL elapsed for {len(TEST_ENGINES)} engines x 100 iterations = {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.1f} min)")
