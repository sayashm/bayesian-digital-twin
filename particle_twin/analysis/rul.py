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
(fleet MAE 151.1 cycles, 90% CI coverage 0.00) even though the health
index itself was tracked well (HI trajectory RMSE 0.093). So the health
estimate was fine; the health -> RUL conversion was the broken step. Five
causes, five fixes. This is the same "two-stage" HI-then-RUL structure
Ramasso & Saxena (2014, PHM Society Annual Conference, "Review and
Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS
Dataset", Sec. 4.2 "functional mapping between health index and RUL")
identify as one of the two dominant families of C-MAPSS prognostics
methods -- and their formalism is exactly RUL(t) = t_f - t, where t_f
solves HI(t_f) = theta. v1's bug was step 2 (choosing theta), not step 1
(building the HI).

(1) THE FAILURE THRESHOLD WAS WRONG.  v1 simulated until h <= 0.0. Real
    FD001 engines do not fail at h = 0 -- and the literature does not
    assume they do either: de Beaulieu et al. (2022, IFAC SAFEPROCESS,
    doi:10.1016/j.ifacol.2022.07.212) use an EMPIRICALLY chosen
    VHI_EOL = 0.86 on their own normalised virtual health index, precisely
    because their test-set VHI never reaches its own upper bound (it maxes
    out at 0.82, not 1.0 -- see their Table 1 and Sec. 3). Measured on all
    100 FD001 training engines (mean of each engine's last 5 cycles):

        observed HI at failure : 0.2565 +/- 0.0144   (CV 5.6%)
        observed HI when healthy: 0.655 +/- 0.070
        minimum HI anywhere in the training set: 0.172

    So v1 made every particle live through a ~0.26-wide "phantom" zone
    that no real engine ever enters. With the exponential transition that
    alone adds 100-200 cycles to every prediction -- the single largest
    error source. calibrate_failure_threshold() below is how theta gets
    tuned on the TRAINING set, per Ramasso & Saxena's own caveat that this
    threshold "must be tuned", not assumed.

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
    width. Optionally (transition_fns) it also varies the transition
    dynamics per particle, propagating the spread across the k matched
    library engines rather than collapsing it to its mean.

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
    uses the piecewise-linear RUL target capped at 125-130 cycles
    (Heimes, F. O., 2008, "Recurrent neural networks for remaining useful
    life estimation", International Conference on Prognostics and Health
    Management; also used by Ramasso & Saxena 2014 in their benchmarking
    guidelines). v2 exposes `rul_cap` (default 125). On the FD001 test
    set this is nearly free information-wise: true RUL has median 86,
    max 145, and only 11% of engines exceed 125.

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
n_particles=500, prediction at each engine's last observed cycle) --
see experiments/test_rul_v2.py for the reproduction script.
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
long before v1's max_horizon=1500 (a few seconds for the whole fleet).

Honest caveat on CI coverage: 0.51 is a large improvement on 0.00 but
still short of the nominal 0.90. Measured sensitivity to the threshold
spread alone:

    threshold_std x1 -> 0.52     x2 -> 0.59     x3 -> 0.76

So threshold uncertainty is real but is NOT the only missing variance
component; the remaining gap is model/parameter uncertainty in the
matched growth rate. Pass `transition_fns` (one transition per matched
library engine) to add it rather than simply inflating threshold_std,
which would be fitting the interval to the test set. See
experiments/test_rul_v2.py for an exploratory (not tuned) measurement of
how much transition_fns actually closes the gap -- report the honest
number either way; a residual gap is a genuine finding about the model's
overconfidence, not a bug to be tuned away.

============================================================================
WHAT v2 ADDS BEYOND POSTERIOR FORWARD-SIMULATION
============================================================================
The Bayesian forward-simulation approach above is not the only way the
prognostics literature converts a health index into a RUL. This module
also implements the classical, non-Bayesian baseline so the comparison
in the thesis is honest and reproducible in-package rather than asserted:

  fit_threshold_crossing_rul() -- fit HI(t) = a*exp(b*t) (b<0) or a
      straight line to the DEGRADATION SEGMENT of the observed HI history
      (not the whole life -- see detect_degradation_onset()), solve
      t_f from HI(t_f) = theta, report RUL = t_f - t. This is the
      textbook two-stage HI->RUL pipeline (Ramasso & Saxena 2014, Sec.
      4.2; the NASA/PHM Society formula sheet in RUL_PROBLEM/ works out
      the same t_f = ln(theta/a)/b closed form). It is the baseline the
      Bayesian approach above should beat, and having it in-package
      (rather than only in a one-off notebook) makes that comparison
      reproducible.

  detect_degradation_onset() -- both reference notes in RUL_PROBLEM/ flag
      that fitting the exponential/linear model over the FULL life
      (including the flat healthy plateau) biases b toward zero and
      inflates the predicted RUL. This is a simple changepoint heuristic
      (first sustained drop below the healthy baseline band) that picks
      the degradation onset automatically.

  hi_monotonicity() / isotonic_smooth() -- threshold crossing is only a
      well-posed question if HI(t) is monotone (Ramasso & Saxena 2014);
      a non-monotone trajectory can cross theta more than once, or not at
      all, while the engine is still clearly degrading. hi_monotonicity()
      reports how monotone our HI actually is (diagnostic, not a gate);
      isotonic_smooth() is an optional preprocessing step that enforces it.

  threshold_sensitivity() -- sweeps theta over a range and reports fleet
      MAE/RMSE/PHM08 at each value, so the choice of theta is defended by
      a curve (as both Ramasso & Saxena 2014 and de Beaulieu et al. 2022
      say it should be -- "the exact threshold should be tuned on the
      training set") rather than asserted as a single number.

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
arguments (no **kwargs), so any caller that unpacks a WHOLE estimate dict
straight into it (`db.store_rul_timeseries(..., **estimate)`) will now
raise TypeError on the new keys -- filter through to_db_row() first:

    row = to_db_row(estimate)   # == {k: v for k, v in estimate.items() if k in DB_COLUMNS}
    db.store_rul_timeseries(exp_id=..., engine_id=..., dataset=..., **row)

(Every in-repo caller that did this directly has been updated accordingly
-- see experiments/test_rul.py, import_fd001_results.py, run_full_fd001.py,
day9/run_fd003.py, day9/run_fd001_linear.py.)

NOTE ON DEFAULTS: `failure_threshold` still defaults to 0.0 so that
nothing silently changes under existing scripts that don't pass it. To
get the v2 behaviour you must pass a calibrated threshold -- see the
worked example below.

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
from scipy.stats import linregress

# phm08_score is not redefined here: particle_twin.analysis.metrics already
# owns the PHM08 formula (and mae/rmse/ci_coverage alongside it), so this
# module imports and re-exports it rather than keeping a second copy of the
# same five-line formula in sync by hand. bayes_optimal_rul() below still
# has its OWN internal loss computation, because it needs overflow-safe
# clipping while searching a candidate grid -- a different numerical need
# from phm08_score's job of scoring a single, already-finite prediction.
from particle_twin.analysis.metrics import mae as _mae
from particle_twin.analysis.metrics import phm08_score
from particle_twin.analysis.metrics import rmse as _rmse

# ---------------------------------------------------------------------------
# Fleet-calibrated constants (FD001, hi_method='weighted').
# Reproduce with calibrate_failure_threshold(); see module docstring (1)-(2).
# ---------------------------------------------------------------------------

#: Threshold in OBSERVATION space -- mean/std of the observed HI over each
#: training engine's last 5 cycles. Reproduce:
#:     calibrate_failure_threshold(model, train_df, mode="observed")
#: Use when forward-simulating (or curve-fitting -- see
#: fit_threshold_crossing_rul) something that lives on the observed-HI
#: scale, e.g. the deterministic baseline, which never touches the filter.
FD001_FAILURE_THRESHOLD_OBSERVED = (0.2565, 0.0144)

#: Threshold in POSTERIOR space -- mean/std of the filter's posterior median
#: health at the true failure cycle of 30 training engines. Reproduce:
#:     calibrate_failure_threshold(model, train_df, mode="posterior",
#:                                  sim_filter=sim_filter, n_engines=30,
#:                                  n_particles=300)
#: This is the one to use with extract_rul()/simulate_rul(), because the
#: forward simulation starts from the filter's particle cloud, not from
#: raw observations.
FD001_FAILURE_THRESHOLD_POSTERIOR = (0.336, 0.026)

#: Piecewise-linear RUL cap, standard for C-MAPSS. See module docstring (4).
DEFAULT_RUL_CAP = 125

#: Keys accepted by ResultsDB.store_rul_timeseries() -- filter an estimate
#: dict through to_db_row() before ** unpacking it into the DB.
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
    training data and apply to held-out test engines -- exactly the
    "tune theta on the training set" caveat in Ramasso & Saxena (2014,
    Sec. 4.2) and de Beaulieu et al. (2022, IFAC SAFEPROCESS), who choose
    their VHI_EOL=0.86 the same way.

    Parameters
    ----------
    model : DegradationModel -- already .fit(train_df). Used for .observe().
    train_df : DataFrame -- run-to-failure training data.
    mode : {'posterior', 'observed'}
        'observed' (fast, ~0.1s for 100 engines): average the OBSERVED
            health index over each engine's last `n_last` cycles. Correct
            only if the filter posterior is unbiased w.r.t. the
            observations -- see module docstring (2) for why it usually
            is not, on this dataset.
        'posterior' (~50x slower, ~30s at these defaults; RECOMMENDED):
            run the filter on `n_engines` training engines and read the
            posterior median health at the true failure cycle. Absorbs
            the filter's own bias.
    sim_filter : SimilarityStreamingFilter or None
        Used for mode='posterior'. If None, falls back to a plain
        BootstrapPF on the pooled model, which is fine but does not
        reflect the similarity matching the test engines will actually
        use.
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
    in failure health, which is what simulate_rul() should sample from
    (module docstring (3)).
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
#
# phm08_score(predicted, true) is imported above from
# particle_twin.analysis.metrics (same formula: d = predicted - true;
# exp(-d/13)-1 early, exp(d/10)-1 late) and re-exported here so
# `from particle_twin.analysis.rul import phm08_score` keeps working for
# callers of this module specifically.

def bayes_optimal_rul(rul_samples: np.ndarray, n_grid: int = 400,
                       max_expected_loss: float = 1e12) -> float:
    """
    The Bayes action under PHM08 loss: the prediction a* minimising
    E_{RUL ~ posterior}[ phm08_score(a, RUL) ].

    We already carry the full posterior, so there is no reason to report
    a summary (the median) that was designed for symmetric loss. Under an
    asymmetric loss the optimal action sits well BELOW the median -- in
    thesis terms this is a textbook Bayesian decision-theory step, and it
    is where most of the PHM08-score improvement comes from (module
    docstring (5)).

    Computed by evaluating expected loss on a grid over the sample range
    rather than in closed form, so it stays correct for any posterior
    shape (these are typically right-skewed and often censored). Uses its
    own clipped exp computation (rather than calling phm08_score directly)
    because the grid can span a wide range of candidate actions and a
    single extreme candidate/sample pair must not be allowed to overflow
    the whole column to inf -- phm08_score itself does not need this
    guard, since it only ever scores one already-finite prediction.

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
    early = np.expm1(np.clip(-d, None, 700.0) / 13.0)
    late = np.expm1(np.clip(d, None, 700.0) / 10.0)
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


# ---------------------------------------------------------------------------
# 5. Classical baseline: threshold crossing on a curve-fit HI trajectory
# ---------------------------------------------------------------------------
#
# Everything above treats RUL as a Bayesian posterior built from a particle
# filter. The C-MAPSS/PHM literature's more common baseline skips the
# filter entirely: fit a simple deterministic curve straight through the
# observed HI history, solve for when it crosses theta, and report
# RUL = t_f - t_now (Ramasso & Saxena 2014 Sec. 4.2; worked formulas in
# RUL_PROBLEM/how estimate RUL from Health index for NASA CMAPPS.md and
# RUL_PROBLEM/Python code for C-MAPSS RUL estimation using exponential
# model.md). This is the baseline the Bayesian approach above should beat
# -- see experiments/test_rul_v2.py for both measured side by side.

def hi_monotonicity(hi_hist) -> float:
    """
    Fraction of consecutive observed cycles where the HI is non-increasing
    (hi[t+1] <= hi[t]) -- the diagnostic Ramasso & Saxena (2014) point to
    when they note that threshold-crossing is only a well-posed question
    if the HI trajectory is monotone: a non-monotone curve can cross
    theta more than once (or not at all) while the engine keeps
    degrading, which silently corrupts fit_threshold_crossing_rul()'s
    t_f solve.

    1.0 = perfectly monotone non-increasing (ideal). 0.5 = no better than
    a coin flip at each step (pure noise, no discernible trend). Lower
    than that would mean the HI trends UP over time, which would itself
    be a red flag about the HI construction.

    Example: FD001 engine 20's 'weighted' observed HI has monotonicity
    ~0.78 -- mostly decreasing, with sensor-noise-driven upward blips
    roughly 1 cycle in 5. Not perfectly monotone, but decreasing on
    average, which is why isotonic_smooth() below is offered as an
    OPTIONAL preprocessing step rather than a mandatory one.
    """
    hi = np.asarray(hi_hist, dtype=float)
    if len(hi) < 2:
        return float("nan")
    return float(np.mean(np.diff(hi) <= 0))


def isotonic_smooth(t_hist, hi_hist) -> np.ndarray:
    """
    Enforce a monotone non-increasing HI trajectory via isotonic
    regression (sklearn.isotonic.IsotonicRegression), the standard fix
    for the monotonicity caveat in hi_monotonicity()'s docstring. Removes
    local noise-driven upward blips while leaving the overall degradation
    trend untouched, which stabilises fit_threshold_crossing_rul()'s
    slope estimate without hand-tuning a smoothing window.

    Parameters
    ----------
    t_hist, hi_hist : array-like, same length -- need not be pre-sorted.

    Returns
    -------
    np.ndarray, same length and order as hi_hist -- the isotonic fit,
    evaluated at each t_hist value.
    """
    from sklearn.isotonic import IsotonicRegression

    t = np.asarray(t_hist, dtype=float)
    hi = np.asarray(hi_hist, dtype=float)
    order = np.argsort(t)
    smoothed = np.empty_like(hi)
    smoothed[order] = IsotonicRegression(increasing=False).fit_transform(t[order], hi[order])
    return smoothed


def detect_degradation_onset(hi_hist, baseline_frac: float = 0.2, min_baseline: int = 5,
                              n_sigma: float = 1.5) -> int:
    """
    Changepoint heuristic: return the index of the first cycle where the
    HI drops below its own "healthy" band, so
    fit_threshold_crossing_rul() can fit only the degradation segment.

    Both RUL_PROBLEM/ reference notes flag the same failure mode: fitting
    HI(t) = a*exp(b*t) (or a line) over the FULL life -- including the
    long flat healthy plateau most C-MAPSS engines spend most of their
    life in -- biases the slope b toward zero, which inflates every
    downstream RUL prediction (a shallower fitted decay reaches theta
    later than the true one).

    Method: take the first `baseline_frac` fraction of the observed
    history (at least `min_baseline` cycles) as the "healthy baseline",
    compute its mean/std, then scan forward for the first cycle whose HI
    falls more than `n_sigma` standard deviations below that baseline
    mean. Simple on purpose -- a full Bayesian changepoint model is out of
    scope for what is meant to be the CLASSICAL baseline's own
    preprocessing step, not a second inference problem.

    Example: healthy baseline mean 0.66, std 0.02, n_sigma=1.5 -> band
    floor 0.63. If HI is 0.65, 0.66, 0.64, 0.61, 0.55, ... the onset index
    is 3 (the first point below 0.63).

    Parameters
    ----------
    hi_hist : array-like, in OBSERVATION order (i.e. already sorted by
        time -- this function does not re-sort, unlike
        fit_threshold_crossing_rul which does).
    baseline_frac, min_baseline : how much of the start of the history
        counts as "healthy" for computing the band.
    n_sigma : how many baseline standard deviations below the mean counts
        as "degrading".

    Returns
    -------
    int -- index into hi_hist where the degradation segment starts. 0
        (fit the whole history) if the series never leaves its own
        healthy band, or if there are too few points to define one --
        both are honest "nothing better is knowable yet" fallbacks, not
        errors.
    """
    hi = np.asarray(hi_hist, dtype=float)
    n = len(hi)
    n_baseline = max(min_baseline, int(round(baseline_frac * n)))
    if n_baseline >= n:
        return 0  # too short a history to separate a baseline from the rest

    baseline = hi[:n_baseline]
    mu, sigma = float(baseline.mean()), float(baseline.std())
    if sigma == 0.0:
        sigma = 1e-6  # guard a perfectly flat baseline (would otherwise never trigger)
    band_floor = mu - n_sigma * sigma

    below = np.where(hi[n_baseline:] < band_floor)[0]
    if below.size == 0:
        return 0  # never leaves the healthy band (yet) -- fit on the full history
    return int(n_baseline + below[0])


def fit_threshold_crossing_rul(t_hist, hi_hist, t_now: float, theta: float,
                                model: str = "exponential", onset: str | int = "auto",
                                min_segment_len: int = 5) -> dict:
    """
    Classical deterministic HI->RUL baseline: fit a curve through the
    (degradation segment of the) observed HI history, solve for the cycle
    t_f at which it crosses theta, and report RUL = max(t_f - t_now, 0).

    Two curve shapes (RUL_PROBLEM/how estimate RUL from Health index for
    NASA CMAPPS.md):

        'exponential'  HI(t) = a * exp(b*t),  b < 0
                       t_f = ln(theta / a) / b
        'linear'       HI(t) = a + b*t,        b < 0
                       t_f = (theta - a) / b

    Both are fit ONLY on the degradation segment (see
    detect_degradation_onset), because fitting over the whole life --
    including the flat healthy plateau -- biases b toward zero and
    inflates the predicted RUL (both RUL_PROBLEM/ reference notes flag
    this explicitly).

    Example: engine with theta=0.2565 (FD001_FAILURE_THRESHOLD_OBSERVED),
    observed HI decaying from ~0.66 at cycle 20 to ~0.40 at cycle 150
    (t_now=150). An exponential fit to the post-onset segment might give
    a~0.66, b~-0.0034, so t_f = ln(0.2565/0.66) / -0.0034 ~= 277, and
    RUL = 277 - 150 = 127 cycles.

    Parameters
    ----------
    t_hist, hi_hist : array-like, same length -- need not be pre-sorted
        (this function sorts by t_hist internally).
    t_now : float -- the cycle at which RUL is wanted (usually the last
        observed cycle == t_hist[-1]).
    theta : float -- failure threshold, on the SAME scale as hi_hist
        (pass FD001_FAILURE_THRESHOLD_OBSERVED[0] for the h [0,1] scale
        used everywhere else in this module).
    model : {'exponential', 'linear'}
    onset : 'auto' (use detect_degradation_onset), 'full' (fit the whole
        history, the naive approach both reference notes warn against),
        or an explicit integer index.
    min_segment_len : if the detected/requested segment is shorter than
        this, fall back to fitting the full history rather than
        overfitting a 2-parameter model to a handful of points.

    Returns
    -------
    dict:
        rul            -- point RUL estimate (float, clipped at 0; nan if
                          the fit did not converge -- see `converged`)
        t_f            -- predicted failure cycle (nan if not converged)
        a, b           -- fitted curve parameters
        r_squared      -- fit quality on the fitted segment
        onset_index    -- index into the (sorted) history the fit started at
        onset_cycle    -- the cycle number at that index
        n_fit_points   -- how many points the curve was fit on
        converged      -- False if the fit found b >= 0 (health not
                          decreasing -- solving for a crossing is
                          meaningless) or theta/a <= 0 (exponential model
                          only); `rul`/`t_f` are nan in that case rather
                          than a silently wrong number.
    """
    if model not in ("exponential", "linear"):
        raise ValueError(f"model must be 'exponential' or 'linear', got {model!r}")

    t = np.asarray(t_hist, dtype=float)
    hi = np.asarray(hi_hist, dtype=float)
    if len(t) != len(hi):
        raise ValueError("t_hist and hi_hist must have the same length.")
    order = np.argsort(t)
    t, hi = t[order], hi[order]

    if onset == "auto":
        onset_idx = detect_degradation_onset(hi)
    elif onset == "full":
        onset_idx = 0
    else:
        onset_idx = int(onset)

    if len(t) - onset_idx < min_segment_len:
        onset_idx = 0  # segment too short to fit reliably -- fall back to full history

    t_fit, hi_fit = t[onset_idx:], hi[onset_idx:]

    if model == "exponential":
        # log(HI) = log(a) + b*t requires HI > 0; floor at a small epsilon
        # rather than dropping points (keeps n_fit_points meaningful).
        log_hi = np.log(np.maximum(hi_fit, 1e-6))
        slope, intercept, r_value, _, _ = linregress(t_fit, log_hi)
        b, a = float(slope), float(np.exp(intercept))
        converged = bool(b < 0 and theta > 0 and a > 0)
        t_f = float(np.log(theta / a) / b) if converged else float("nan")
    else:  # 'linear'
        slope, intercept, r_value, _, _ = linregress(t_fit, hi_fit)
        b, a = float(slope), float(intercept)
        converged = bool(b < 0)
        t_f = float((theta - a) / b) if converged else float("nan")

    rul = float(max(t_f - t_now, 0.0)) if converged else float("nan")

    return {
        "rul": rul,
        "t_f": t_f,
        "a": a,
        "b": b,
        "r_squared": float(r_value ** 2),
        "onset_index": int(onset_idx),
        "onset_cycle": float(t[onset_idx]),
        "n_fit_points": int(len(t_fit)),
        "converged": converged,
    }


def threshold_sensitivity(predict_fn, true_ruls, theta_grid) -> dict:
    """
    Sweep candidate failure thresholds theta and report fleet MAE/RMSE/
    PHM08 at each one, so the choice of theta is defended by a curve
    rather than asserted -- both Ramasso & Saxena (2014) and de Beaulieu
    et al. (2022) state the threshold "should be tuned on the training
    set"; this is the tool that does the tuning/validation transparently
    (run it on TRAINING engines only, to respect the "don't tune on the
    test set" rule the rest of this module follows).

    Parameters
    ----------
    predict_fn : callable(theta) -> array of RUL point predictions, one
        per engine, using that failure threshold. Cache whatever is
        theta-INDEPENDENT (particle clouds, transition functions, observed
        HI histories) outside this call and only redo the (cheap)
        threshold-crossing step inside predict_fn -- see
        experiments/test_rul_v2.py for a worked example that reuses each
        engine's already-filtered particle cloud across the whole sweep,
        rather than re-running the particle filter per theta.
    true_ruls : array-like, one true RUL per engine, aligned with
        predict_fn's output order.
    theta_grid : array-like of candidate thresholds to try.

    Returns
    -------
    dict of arrays, one entry per theta: 'theta', 'mae', 'rmse', 'phm08'
    ('phm08' is the SUM across the fleet, matching the PHM08 competition's
    own reported convention).
    """
    true_ruls = np.asarray(true_ruls, dtype=float)
    thetas = np.asarray(theta_grid, dtype=float)
    out = {
        "theta": thetas,
        "mae": np.empty_like(thetas),
        "rmse": np.empty_like(thetas),
        "phm08": np.empty_like(thetas),
    }
    for i, theta in enumerate(thetas):
        preds = np.asarray(predict_fn(float(theta)), dtype=float)
        out["mae"][i] = _mae(preds, true_ruls)
        out["rmse"][i] = _rmse(preds, true_ruls)
        out["phm08"][i] = float(np.sum(phm08_score(preds, true_ruls)))
    return out
