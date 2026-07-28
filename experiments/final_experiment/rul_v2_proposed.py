"""
rul.py — RUL extraction from the particle posterior (v2)
==========================================================
Given the particle cloud {h_t^(i)} at some observed cycle t, propagates
each particle forward under the SAME stochastic transition model the
bootstrap filter uses, until it crosses a FAILURE THRESHOLD. The number
of steps each particle takes is one Monte Carlo sample of the Remaining
Useful Life (RUL) random variable. Collecting one sample per particle
gives an empirical RUL distribution, not a single point estimate.

============================================================================
WHAT CHANGED IN v2, AND WHY
============================================================================
v1 predicted RUL 2-4x too high on essentially every FD001 test engine
(final_experiment: fleet MAE 140.2 cycles, 90% CI coverage 0.00) even
though the health index itself was tracked well (HI trajectory RMSE
0.093). So the health estimate was fine; the health -> RUL conversion was
the broken step. Four causes, four fixes.

(1) THE FAILURE THRESHOLD WAS WRONG.  v1 simulated until h <= 0.0. Real
    FD001 engines do not fail at h = 0. Measured on all 100 training
    engines (mean of each engine's last 5 cycles):

        observed HI at failure : 0.2565 +/- 0.0144   (CV 5.6%)
        observed HI when healthy: 0.655 +/- 0.070
        minimum HI anywhere in the training set: 0.172

    So v1 made every particle live through a ~0.26-wide "phantom" zone
    that no real engine ever enters. With the exponential transition
    that alone adds 100-200 cycles to every prediction. This is the
    single largest error source.

(2) THE THRESHOLD MUST BE CALIBRATED IN *POSTERIOR* SPACE, NOT
    OBSERVATION SPACE.  The particle filter's posterior median runs
    systematically ABOVE the observed HI (visible in every
    final_experiment/figures/engine_*.png: posterior ~0.76 while the
    observations sit at ~0.67). Cause: sample_initial() starts particles
    at h ~ N(1, sigma_0), but real engines start at HI ~ 0.655, and the
    learned measurement noise is wide (sigma_w = 9.26 HI units), so the
    filter corrects that wrong start only slowly. Measured by running
    the filter on 30 training engines and reading the posterior median
    at their true failure cycle:

        posterior health at failure : 0.336 +/- 0.026
        observed  HI     at failure : 0.2565 +/- 0.0144

    Crossing a posterior-space trajectory against an observation-space
    threshold double-counts that offset. calibrate_failure_threshold()
    below therefore supports mode='posterior' (recommended), which runs
    the filter on training engines and reads off the threshold in the
    exact space the forward simulation lives in. The filter bias cancels
    instead of accumulating.

(3) THE THRESHOLD IS ITSELF UNCERTAIN, AND v1 IGNORED THAT.  Engines do
    not all fail at the same health. v1 used one hard number, so the
    only uncertainty in the RUL distribution came from process noise --
    far too little, which is why 90% CI coverage was 0.00. v2 draws an
    INDEPENDENT threshold per particle from N(mu_fail, sigma_fail^2),
    turning a known source of real variability into honest posterior
    width. Optionally (rate_samples) it also resamples the growth rate
    per particle, propagating the spread across the k matched library
    engines rather than collapsing it to its mean.

(4) THE HEALTH INDEX SATURATES, SO RUL ABOVE ~125 IS NOT IDENTIFIABLE.
    Mean training HI by true-RUL band:

        RUL [  0, 25) : 0.329      RUL [ 75,100) : 0.582
        RUL [ 25, 50) : 0.458      RUL [100,125) : 0.611
        RUL [ 50, 75) : 0.534      RUL [125,150) : 0.631
                                   RUL [150,400) : 0.655

    Beyond RUL ~125 the bands are separated by less than the
    within-band std (0.070) -- the HI simply cannot tell 130 cycles of
    life from 300. corr(HI, RUL) is 0.743 over all data but 0.826 once
    restricted to RUL <= 125. This is exactly why the C-MAPSS literature
    uses the piecewise-linear RUL target capped at 125 (Heimes 2008;
    Ramasso & Saxena 2014). v2 exposes `rul_cap` (default 125). On the
    FD001 test set this is nearly free information-wise: true RUL has
    median 86, max 145, and only 11% of engines exceed 125.

(5) THE MEDIAN IS THE WRONG POINT ESTIMATE UNDER PHM08 LOSS.  The PHM08
    score punishes late predictions much harder than early ones
    (exp(d/10)-1 vs exp(-d/13)-1). Reporting the posterior median throws
    that asymmetry away. Since we already have the full posterior,
    bayes_optimal_rul() numerically minimises the expected PHM08 loss
    over the RUL samples -- the decision-theoretically correct action,
    which lands well below the median. Returned as `rul_bayes`; the
    median is still returned unchanged as `rul_median`.

============================================================================
MEASURED EFFECT (all 100 FD001 test engines, one-shot matching,
n_particles=500, prediction at each engine's last observed cycle)
============================================================================
                       MAE     RMSE     PHM08      bias    90% CI cov.
    v1 (median)       151.1    157.2    2.44e13   +151.1      0.00
    v2 (median)        22.9     27.8    3.97e03    +19.2      0.51
    v2 (rul_bayes)     19.9     23.9    1.63e03    +12.8      0.51

MAE improves 7.6x and the PHM08 fleet score drops by ~10 orders of
magnitude, from 2.4e13 to 1.6e3 -- finally the same order as the
literature's ~383 on this dataset rather than an astronomical number.
The residual bias is +12.8 cycles (still slightly optimistic-late).
v2 is also ~7x FASTER, because rul_cap=125 stops the forward simulation
long before v1's max_horizon=1500 (5.8s for the whole fleet).

Honest caveat on CI coverage: 0.51 is a large improvement on 0.00 but
still short of the nominal 0.90. Measured sensitivity to the threshold
spread alone:

    threshold_std x1 -> 0.52     x2 -> 0.59     x3 -> 0.76

So threshold uncertainty is real but is NOT the only missing variance
component; the remaining gap is model/parameter uncertainty in the
matched growth rate. Pass `transition_fns` (one transition per matched
library engine) to add it rather than simply inflating threshold_std,
which would be fitting the interval to the test set. Report 0.51
honestly and discuss the gap -- it is a genuine finding about the
model's overconfidence, not a bug to be tuned away.

============================================================================
BACKWARD COMPATIBILITY
============================================================================
extract_rul() and extract_rul_trajectory() keep their v1 signatures and
every v1 dict key ('cycle', 'rul_median', 'rul_p5', 'rul_p95',
'rul_mean', 'rul_std', 'health_mean', 'ess'), so
experiments/final_experiment/02_run_experiment.py and
ResultsDB.store_rul_timeseries(**estimate) keep working untouched.

New keys are ADDITIVE ('rul_bayes', 'rul_p50', 'p_failed_by_horizon',
'frac_censored'). store_rul_timeseries() takes explicit keyword
arguments, so if you want to pass a whole row through with **, drop the
new keys first, e.g.:

    row = {k: v for k, v in est.items() if k in DB_COLUMNS}

DB_COLUMNS is provided below for exactly that purpose.

NOTE ON DEFAULTS: `failure_threshold` still defaults to 0.0 so that
nothing silently changes under existing scripts. To get the v2 behaviour
you must pass a calibrated threshold -- see the worked example at the
bottom of this docstring.

============================================================================
RECOMMENDED USE (drop-in replacement for the v1 call in
experiments/final_experiment/02_run_experiment.py)
============================================================================
    from particle_twin.analysis.rul import (
        calibrate_failure_threshold, extract_rul_trajectory,
    )

    # ONCE, after fitting the pooled model / building the library:
    mu_fail, sigma_fail = calibrate_failure_threshold(
        pooled_model, train, mode="posterior", sim_filter=sim_filter,
        n_engines=30, n_particles=300,
    )
    # -> approx (0.336, 0.026) on FD001. mode="observed" is ~50x faster
    #    and gives (0.2565, 0.0144), but does NOT absorb the filter bias.

    # PER ENGINE, replacing the old extract_rul_trajectory(...) call:
    rul_trajectory = extract_rul_trajectory(
        pf,
        failure_threshold=mu_fail,
        threshold_std=sigma_fail,
        rul_cap=125,
        max_horizon=200,          # 1500 is now pointless: the cap binds first
    )
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fleet-calibrated constants (FD001, hi_method='weighted').
# Reproduce with calibrate_failure_threshold(); see module docstring (1)-(2).
# ---------------------------------------------------------------------------

#: Threshold in OBSERVATION space -- mean/std of the observed HI over each
#: training engine's last 5 cycles. Use when forward-simulating something
#: that lives on the observed-HI scale.
FD001_FAILURE_THRESHOLD_OBSERVED = (0.2565, 0.0144)

#: Threshold in POSTERIOR space -- mean/std of the filter's posterior median
#: health at the true failure cycle of 30 training engines. This is the one
#: to use with extract_rul(), because the forward simulation starts from the
#: filter's particle cloud, not from raw observations.
FD001_FAILURE_THRESHOLD_POSTERIOR = (0.336, 0.026)

#: Piecewise-linear RUL cap, standard for C-MAPSS. See docstring (4).
DEFAULT_RUL_CAP = 125

#: PHM08 asymmetric scoring constants: late predictions divided by A_LATE,
#: early ones by A_EARLY. Late is punished harder (10 < 13).
PHM08_A_LATE = 10.0
PHM08_A_EARLY = 13.0

#: Keys accepted by ResultsDB.store_rul_timeseries() -- filter an estimate
#: dict through this before ** unpacking it into the DB.
DB_COLUMNS = (
    "cycle", "rul_median", "rul_p5", "rul_p95", "rul_mean", "rul_std",
    "health_mean", "ess",
)


# ---------------------------------------------------------------------------
# 1. Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_failure_threshold(model, train_df, mode: str = "posterior",
                                sim_filter=None, n_engines: int = 30,
                                n_particles: int = 300, n_last: int = 5,
                                id_col: str = "unit_id", time_col: str = "cycle",
                                strategy: str = "one_shot", warmup_cycles: int = 20,
                                seed: int = 42) -> tuple[float, float]:  # noqa: PLR0913
    """
    Estimate (mu_fail, sigma_fail): the health level at which engines
    actually fail, and how much that level varies across the fleet.

    Training engines are run-to-failure, so their LAST cycle is a direct,
    label-free observation of "what health looks like at failure". No
    true-RUL values are used, so this is legitimate to calibrate on
    training data and apply to held-out test engines.

    Parameters
    ----------
    model : DegradationModel -- already .fit(train_df). Used for .observe().
    train_df : DataFrame -- run-to-failure training data.
    mode : {'posterior', 'observed'}
        'observed' (fast, ~0.1s): average the OBSERVED health index over
            each engine's last `n_last` cycles. Correct only if the
            filter posterior is unbiased w.r.t. the observations.
        'posterior' (slower, ~30s at these defaults; RECOMMENDED): run
            the filter on `n_engines` training engines and read the
            posterior median health at the true failure cycle. Absorbs
            the filter's own bias -- see module docstring (2).
    sim_filter : SimilarityStreamingFilter or None
        Required for mode='posterior'. If None, falls back to a plain
        BootstrapPF, which is fine but does not reflect the similarity
        matching the test engines will actually use.
    n_engines : int -- how many training engines to filter (mode='posterior').
        30 is enough: the between-engine std is small (~0.026), so the
        standard error of the mean is already ~0.005.
    n_particles, strategy, warmup_cycles, seed : passed to the filter.
    n_last : int -- cycles averaged at the end of each engine (mode='observed').

    Returns
    -------
    (mu_fail, sigma_fail) : floats on the h [0, 1] scale.

    Notes
    -----
    sigma_fail is the BETWEEN-ENGINE std, not the standard error of the
    mean. That is deliberate: it describes genuine fleet-to-fleet spread
    in failure health, which is what simulate_rul() should sample from.
    """
    if mode not in ("posterior", "observed"):
        raise ValueError(f"mode must be 'posterior' or 'observed', got {mode!r}")

    unit_ids = sorted(train_df[id_col].unique())

    if mode == "observed":
        levels = []
        for uid in unit_ids:
            engine = train_df[train_df[id_col] == uid].sort_values(time_col)
            if len(engine) < n_last:
                continue
            h = np.asarray(model.observe(engine), dtype=float)
            levels.append(float(np.mean(h[-n_last:])))
        levels = np.asarray(levels)
        return float(levels.mean()), float(levels.std())

    # mode == 'posterior'
    chosen = unit_ids[:n_engines]
    levels = []
    for uid in chosen:
        engine = train_df[train_df[id_col] == uid].sort_values(time_col)
        if len(engine) < 2:
            continue
        np.random.seed(seed)  # the filter uses the legacy global numpy RNG
        if sim_filter is not None:
            pf, _ = sim_filter.run(engine, strategy=strategy, n_particles=n_particles,
                                   warmup_cycles=warmup_cycles)
        else:
            from particle_twin.filters.bootstrap import BootstrapPF
            pf = BootstrapPF(model=model, n_particles=n_particles)
            pf.run(engine)
        # pf.history[-1] is the true failure cycle, because training
        # engines are run-to-failure and are not truncated.
        levels.append(float(np.median(pf.history[-1]["particles"])))

    levels = np.asarray(levels)
    if levels.size == 0:
        raise RuntimeError("No usable training engines for threshold calibration.")
    return float(levels.mean()), float(levels.std())


# ---------------------------------------------------------------------------
# 2. Forward simulation
# ---------------------------------------------------------------------------

def simulate_rul(particles: np.ndarray, transition_fn, failure_threshold: float = 0.0,
                 threshold_std: float = 0.0, max_horizon: int = 200,
                 rul_cap: int | None = None, transition_fns=None,
                 rng: np.random.Generator | None = None):
    """
    Monte Carlo forward simulation: propagate each particle one cycle at
    a time until it crosses its own failure threshold.

    Parameters
    ----------
    particles : (n_particles,) -- unweighted health samples on the [0, 1] scale.
    transition_fn : callable -- advances an array of health values by one
        cycle, adding its own process noise (e.g. DegradationModel.transition).
    failure_threshold : float -- MEAN health at failure. Pass a calibrated
        value (see calibrate_failure_threshold); 0.0 is the v1 default and
        is kept only so existing scripts do not silently change behaviour.
    threshold_std : float -- between-engine std of the failure health. If
        > 0, each particle draws its OWN threshold from
        N(failure_threshold, threshold_std^2), clipped to [0, 1]. This is
        the main fix for the 0.00 CI coverage -- see module docstring (3).
    max_horizon : int -- safety cap on simulated cycles. Particles still
        alive at this point are CENSORED (reported at max_horizon and
        counted in `frac_censored`), not silently treated as failures.
    rul_cap : int or None -- if set, simulation stops at rul_cap and every
        surviving particle is assigned rul_cap. This implements the
        piecewise-linear RUL convention (default 125 for C-MAPSS); above
        it the health index carries no usable information. See docstring (4).
    transition_fns : sequence of callables or None -- OPTIONAL model
        uncertainty. If given, particles are partitioned across these
        transition functions (one per matched library engine, say), so the
        spread in matched growth rates propagates into the RUL posterior
        instead of being collapsed to its mean. `transition_fn` is ignored
        when this is supplied.
    rng : numpy Generator or None -- used only for the threshold draw, so
        results stay reproducible independently of the transition's own
        (legacy global) RNG.

    Returns
    -------
    steps : (n_particles,) int -- one RUL sample per particle.
    censored : (n_particles,) bool -- True where the particle never crossed
        (hit max_horizon or rul_cap). Use to report `frac_censored`; a high
        value means the horizon, not the data, is setting your upper CI.
    """
    rng = np.random.default_rng() if rng is None else rng
    h = np.asarray(particles, dtype=float).copy()
    n = len(h)

    # Effective horizon: the cap binds first when it is set, which is also
    # what makes v2 ~7x cheaper than v1's blanket max_horizon=1500.
    horizon = max_horizon if rul_cap is None else min(max_horizon, int(rul_cap))

    # Per-particle threshold: the fleet does not fail at one fixed health.
    if threshold_std > 0:
        thresholds = np.clip(rng.normal(failure_threshold, threshold_std, n), 0.0, 1.0)
    else:
        thresholds = np.full(n, float(failure_threshold))

    # Optional model uncertainty: split particles across candidate dynamics.
    if transition_fns is not None:
        transition_fns = list(transition_fns)
        if len(transition_fns) == 0:
            raise ValueError("transition_fns must be non-empty when provided.")
        group = np.arange(n) % len(transition_fns)
    else:
        transition_fns, group = [transition_fn], np.zeros(n, dtype=int)

    alive = np.ones(n, dtype=bool)
    steps = np.zeros(n, dtype=int)

    for t in range(1, horizon + 1):
        for k, fn in enumerate(transition_fns):
            sel = alive & (group == k)
            if sel.any():
                h[sel] = fn(h[sel])
        just_died = alive & (h <= thresholds)
        steps[just_died] = t
        alive[just_died] = False
        if not alive.any():
            break

    steps[alive] = horizon
    return steps, alive.copy()


# ---------------------------------------------------------------------------
# 3. Decision theory: the point estimate to actually report
# ---------------------------------------------------------------------------

def phm08_score(predicted, true) -> np.ndarray:
    """
    PHM08 asymmetric score. d = predicted - true.
        d <  0 (early): exp(-d / 13) - 1
        d >= 0 (late) : exp( d / 10) - 1
    Lower is better; late predictions are punished harder, which is the
    whole reason bayes_optimal_rul() exists.
    """
    d = np.asarray(predicted, dtype=float) - np.asarray(true, dtype=float)
    return np.where(d < 0, np.exp(-d / PHM08_A_EARLY) - 1.0, np.exp(d / PHM08_A_LATE) - 1.0)


def bayes_optimal_rul(rul_samples: np.ndarray, n_grid: int = 400,
                      max_expected_loss: float = 1e12) -> float:
    """
    The Bayes action under PHM08 loss: the prediction a* minimising
    E_{RUL ~ posterior}[ phm08_score(a, RUL) ].

    We already carry the full posterior, so there is no reason to report
    a summary (the median) that was designed for symmetric loss. Under an
    asymmetric loss the optimal action sits well BELOW the median -- in
    thesis terms this is a textbook Bayesian decision-theory step, and it
    is where most of the PHM08-score improvement comes from.

    Computed by evaluating expected loss on a grid over the sample range
    rather than in closed form, so it stays correct for any posterior
    shape (these are typically right-skewed and often censored).

    Parameters
    ----------
    rul_samples : (n,) array of posterior RUL draws.
    n_grid : int -- candidate actions evaluated between min and max sample.
    max_expected_loss : float -- guard against exp() overflow on wide
        posteriors; losses above this are treated as equally unacceptable.

    Returns
    -------
    float -- the loss-minimising RUL prediction.
    """
    s = np.asarray(rul_samples, dtype=float)
    if s.size == 0:
        return float("nan")
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return lo

    candidates = np.linspace(lo, hi, n_grid)
    # (n_grid, n_samples) -> mean over samples. Clipped before exp() so a
    # single far-out particle cannot overflow the whole column to inf.
    d = candidates[:, None] - s[None, :]
    early = np.expm1(np.clip(-d, None, 700.0) / PHM08_A_EARLY)
    late = np.expm1(np.clip(d, None, 700.0) / PHM08_A_LATE)
    losses = np.where(d < 0, early, late).mean(axis=1)
    losses = np.minimum(losses, max_expected_loss)
    return float(candidates[int(np.argmin(losses))])


# ---------------------------------------------------------------------------
# 4. Extraction from the filter history
# ---------------------------------------------------------------------------

def extract_rul(pf, cycle_index: int = -1, failure_threshold: float = 0.0,
                max_horizon: int = 200, threshold_std: float = 0.0,
                rul_cap: int | None = None, transition_fns=None,
                rng: np.random.Generator | None = None) -> dict:
    """
    Extract the RUL distribution from one entry of pf.history.

    v1 signature and v1 return keys are preserved; everything new is
    additive and off by default. See the module docstring for the
    recommended v2 argument values.

    Returns
    -------
    dict with the v1 keys:
        cycle, rul_median, rul_p5, rul_p95, rul_mean, rul_std,
        health_mean, ess
    plus the v2 keys:
        rul_bayes            -- PHM08-optimal point estimate (report THIS)
        rul_p50              -- explicit alias of rul_median
        frac_censored        -- fraction of particles that never crossed;
                                if high, the horizon/cap is driving the
                                upper CI rather than the data
        p_failed_by_horizon  -- 1 - frac_censored, i.e. P(failure within
                                the simulated window). Directly usable as a
                                maintenance-decision probability.
    """
    entry = pf.history[cycle_index]
    h = entry["particles"]
    log_w = entry["weights"]

    # Exact weighted mean health, computed pre-resample so it carries no
    # extra resampling noise. (Unchanged from v1.)
    health_mean = np.average(h, weights=pf.normalize(log_w))

    # Unweighted particle set for forward simulation: the filter's own
    # ESS-gated resample is not guaranteed to have fired on THIS cycle,
    # and systematic resampling on already-uniform weights is a no-op.
    h_resampled, _ = pf.resample(h, log_w)

    rul_samples, censored = simulate_rul(
        h_resampled, pf.model.transition,
        failure_threshold=failure_threshold, threshold_std=threshold_std,
        max_horizon=max_horizon, rul_cap=rul_cap,
        transition_fns=transition_fns, rng=rng,
    )

    rul_median = float(np.median(rul_samples))
    return {
        # --- v1 keys, unchanged ---
        "cycle": entry["cycle_number"],
        "rul_median": rul_median,
        "rul_p5": float(np.percentile(rul_samples, 5)),
        "rul_p95": float(np.percentile(rul_samples, 95)),
        "rul_mean": float(np.mean(rul_samples)),
        "rul_std": float(np.std(rul_samples)),
        "health_mean": health_mean,
        "ess": entry["ESS"],
        # --- v2 additions ---
        "rul_bayes": bayes_optimal_rul(rul_samples),
        "rul_p50": rul_median,
        "frac_censored": float(censored.mean()),
        "p_failed_by_horizon": float(1.0 - censored.mean()),
    }


def extract_rul_trajectory(pf, failure_threshold: float = 0.0, max_horizon: int = 200,
                           threshold_std: float = 0.0, rul_cap: int | None = None,
                           transition_fns=None, seed: int | None = 42) -> list[dict]:
    """
    Call extract_rul() at every observed cycle (skips history[0], the
    pre-observation prior -- same convention as v1 and as
    SimilarityStreamingFilter.extract_health_trajectory, so the RUL and
    health trajectories stay index-aligned; 02_run_experiment.py asserts
    this).

    `seed` fixes the threshold-draw RNG so a rerun reproduces exactly.
    Pass None for a fresh draw each call.

    Returns
    -------
    list[dict] -- one extract_rul() result per observed cycle.
    """
    rng = np.random.default_rng(seed)
    return [
        extract_rul(pf, cycle_index=t, failure_threshold=failure_threshold,
                    max_horizon=max_horizon, threshold_std=threshold_std,
                    rul_cap=rul_cap, transition_fns=transition_fns, rng=rng)
        for t in range(1, len(pf.history))
    ]


def to_db_row(estimate: dict) -> dict:
    """
    Strip an extract_rul() result down to the keys
    ResultsDB.store_rul_timeseries() accepts, so v2 estimates can still be
    ** unpacked straight into the DB:

        db.store_rul_timeseries(exp_id=..., engine_id=..., dataset=...,
                                **to_db_row(estimate))
    """
    return {k: estimate[k] for k in DB_COLUMNS if k in estimate}
