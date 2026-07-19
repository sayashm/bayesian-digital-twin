"""
pmmh.py — Particle Marginal Metropolis-Hastings
================================================
Estimates sigma_v (process noise scale) from data by embedding the
bootstrap particle filter inside a Metropolis-Hastings sampler.

Target: p(sigma_v | y_1:T) via the pseudo-marginal likelihood estimate
produced by BootstrapPF (accumulated in BootstrapPF.log_likelihood_total —
see bootstrap.py). All other model components (HI method, degradation
shape, measurement noise family) are held fixed at the values already
selected in Chapter 3; only sigma_v is treated as uncertain here.

Sampling is done in u = log(sigma_v) space so the random-walk proposal
is symmetric and sigma_v = exp(u) is always positive by construction
(see Day-8 teaching notes, Segment 5). The prior log_prior_sigma_v is
stated in sigma_v-space (more interpretable for the thesis writeup), so
the acceptance ratio includes an explicit Jacobian term (+u) for the
change of variables sigma_v = exp(u), d(sigma_v) = sigma_v * du.

Joint calibration (Day 10/11 extension)
----------------------------------------
pmmh_sample_joint() extends the above to jointly sample (growth_rate,
sigma_v) per engine, using a Day-10-style model-swap (matched_model_from)
instead of re-fitting DegradationModel/DegradationModelLearner from
scratch on each MCMC step. The HI builder and measurement noise are shared
from a pre-fitted pooled model; only the dynamics_func changes each step.

growth_rate is also sampled in log-space (it is always positive and spans
~4x across the 100-engine library: 0.00175–0.00664). Independent step
sizes per dimension: growth_rate's log-space scale ~0.3 vs. sigma_v's
~0.1–1, so a shared step would freeze one dimension. If a chain's
accept_rate < FALLBACK_ACCEPT_THRESHOLD the growth_rate estimate is not
trustworthy; the caller can supply a regression_growth_rate fallback.
"""

import copy

import numpy as np
import pandas as pd
from scipy import stats

from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF

# Weakly-informative half-normal width for the sigma_v prior -- keeps the
# previously-hardcoded default (sigma_v=0.3) near the bulk of the prior
# mass without ruling out larger process noise. See thesis Sec. 3.4.
TAU_SIGMA_V = 0.3

# growth_rate prior: Normal anchored to the Day-10 per-engine library's
# observed spread across 100 FD001 training engines (mean=0.003727,
# std=0.001150). The mean is ~3.2 sigma above zero so negative values get
# negligible prior mass without needing explicit truncation.
GROWTH_RATE_PRIOR_MU = 0.003727
GROWTH_RATE_PRIOR_SIGMA = 0.001150

# Accept-rate threshold below which a joint PMMH chain is judged stuck and
# the growth_rate estimate falls back to the regression-fit value (same
# pathology seen for 2/4 engines in Day 8's sigma_v-only diagnostic).
FALLBACK_ACCEPT_THRESHOLD = 0.10


def log_prior_sigma_v(sigma_v: float) -> float:
    """
    Prior: sigma_v ~ HalfNormal(tau=TAU_SIGMA_V).
    Weakly informative -- keeps sigma_v positive, centred near the
    previously-assumed value, without ruling out larger process noise.
    """
    return stats.halfnorm(scale=TAU_SIGMA_V).logpdf(sigma_v)


def log_prior_growth_rate(growth_rate: float) -> float:
    """
    Prior: growth_rate ~ Normal(GROWTH_RATE_PRIOR_MU, GROWTH_RATE_PRIOR_SIGMA).
    Anchored to the Day-10 per-engine library's observed spread (100 training
    engines, mean=0.003727, std=0.001150). The mean lies ~3.2 sigma above
    zero so effectively no mass on negative values without explicit truncation.
    Stated in growth_rate-space (interpretable); the acceptance ratio in
    pmmh_sample_joint adds a Jacobian term +v (v = log(growth_rate)) for
    the log-space random-walk proposal.
    """
    return stats.norm(loc=GROWTH_RATE_PRIOR_MU, scale=GROWTH_RATE_PRIOR_SIGMA).logpdf(growth_rate)


# ---------------------------------------------------------------------------
# Model-swap helpers for the joint PMMH (mirrors the Day-10 prototype in
# experiments/day10/similarity_library_prototype.py — kept here so the
# package doesn't depend on the experiments/ tree).
# ---------------------------------------------------------------------------

def _make_matched_dynamics(growth_rate: float, sigma_v: float):
    """Exponential damage dynamics closure with specific growth_rate and sigma_v."""
    def dyn(health, dt=1):
        damage = 100 - health
        new_damage = damage * np.exp(growth_rate * dt)
        noise = np.random.normal(0, sigma_v, size=np.shape(health))
        return np.clip(100 - new_damage + noise, 0, 100)
    return dyn


class _StubLearner:
    """Minimal stand-in for DegradationModelLearner — only .dynamics_func
    is read by DegradationModel.transition(); growth_rate is kept for logging."""
    def __init__(self, dynamics_func, growth_rate: float):
        self.dynamics_func = dynamics_func
        self.growth_rate = growth_rate


def _matched_model_from(pooled_model: DegradationModel, growth_rate: float,
                         sigma_v: float) -> DegradationModel:
    """Shallow-copy pooled_model and swap in a (growth_rate, sigma_v)-specific
    dynamics_func. The HI builder and measurement_learner are shared (fleet-fit,
    unchanged across MCMC steps) — only the transition dynamics change."""
    mm = copy.copy(pooled_model)
    mm.degradation_learner = _StubLearner(_make_matched_dynamics(growth_rate, sigma_v), growth_rate)
    return mm


def run_pf_for_sigma_v(sigma_v: float, train_df: pd.DataFrame, engine_df: pd.DataFrame,
                        hi_method: str, degradation_model: str, measurement_method: str,
                        sigma_0: float, n_particles: int) -> float:
    """
    Fit a fresh DegradationModel with the given sigma_v, run BootstrapPF
    on engine_df, and return the accumulated log-likelihood estimate
    (BootstrapPF.log_likelihood_total).

    This re-fits the HI builder / degradation OLS / measurement noise
    every call -- deliberate, see Day-8 session notes on why this is
    cheap relative to the particle filter pass itself, and avoids
    touching any already-tested fitting code.
    """
    model = DegradationModel(
        hi_method=hi_method,
        degradation_model=degradation_model,
        measurement_method=measurement_method,
        sigma_0=sigma_0,
        sigma_v=sigma_v,
    )
    model.fit(train_df=train_df)

    pf = BootstrapPF(model=model, n_particles=n_particles)
    pf.run(engine_df=engine_df)

    return pf.log_likelihood_total


def pmmh_sample_sigma_v(
    train_df: pd.DataFrame,
    engine_df: pd.DataFrame,
    n_iterations: int,
    hi_method: str = "weighted",
    degradation_model: str = "exponential",
    measurement_method: str = "gaussian",
    sigma_0: float = 0.05,
    n_particles: int = 500,
    sigma_v_init: float = 0.3,
    rw_step: float = 0.2,
    seed: int | None = None,
) -> dict:
    """
    Run PMMH to sample from p(sigma_v | y_1:T).

    Returns a dict with at least:
        'sigma_v_chain'  : array, shape (n_iterations,) -- accepted sigma_v at every iteration
                            (including repeats when a proposal is rejected)
        'log_lik_chain'  : array, shape (n_iterations,) -- the log_likelihood_total kept at each iteration
        'accept_rate'    : float, fraction of proposals accepted

    Algorithm (Day-8 teaching notes, Segment 5, with Jacobian term since
    log_prior_sigma_v is stated in sigma_v-space, not u-space):
      1. u_0 = log(sigma_v_init); logZ_0 = run_pf_for_sigma_v(exp(u_0), ...)
      2. for m in 1..n_iterations:
           u_prop = u_{m-1} + Normal(0, rw_step)          # symmetric RW in log-space
           sigma_v_prop = exp(u_prop)
           logZ_prop = run_pf_for_sigma_v(sigma_v_prop, ...)
           log_alpha = (logZ_prop + log_prior_sigma_v(sigma_v_prop) + u_prop) \
                     - (logZ_{m-1} + log_prior_sigma_v(exp(u_{m-1})) + u_{m-1})
           accept if log(Uniform(0,1)) < log_alpha
           if accept: u_m, logZ_m = u_prop, logZ_prop
           else:      u_m, logZ_m = u_{m-1}, logZ_{m-1}   # MUST reuse old logZ -- see Segment 5 check
    """
    np.random.seed(seed)

    u_prev = np.log(sigma_v_init)
    logZ_prev = run_pf_for_sigma_v(
        sigma_v=np.exp(u_prev),
        train_df=train_df,
        engine_df=engine_df,
        hi_method=hi_method,
        degradation_model=degradation_model,
        measurement_method=measurement_method,
        sigma_0=sigma_0,
        n_particles=n_particles,
    )

    n_accepted = 0
    sigma_v_chain = np.empty(n_iterations)
    log_lik_chain = np.empty(n_iterations)

    for m in range(n_iterations):
        u_prop = u_prev + np.random.normal(0, rw_step)
        sigma_v_prop = np.exp(u_prop)

        logZ_prop = run_pf_for_sigma_v(
            sigma_v=sigma_v_prop,
            train_df=train_df,
            engine_df=engine_df,
            hi_method=hi_method,
            degradation_model=degradation_model,
            measurement_method=measurement_method,
            sigma_0=sigma_0,
            n_particles=n_particles,
        )

        sigma_v_prev = np.exp(u_prev)
        log_alpha = (
            (logZ_prop + log_prior_sigma_v(sigma_v_prop) + u_prop)
            - (logZ_prev + log_prior_sigma_v(sigma_v_prev) + u_prev)
        )

        if np.log(np.random.uniform(0, 1)) < log_alpha:
            u_prev, logZ_prev = u_prop, logZ_prop
            n_accepted += 1

        sigma_v_chain[m] = np.exp(u_prev)
        log_lik_chain[m] = logZ_prev

    return {
        "sigma_v_chain": sigma_v_chain,
        "log_lik_chain": log_lik_chain,
        "accept_rate": n_accepted / n_iterations,
    }


# ===========================================================================
# Joint (growth_rate, sigma_v) PMMH — Day 10/11 extension
# ===========================================================================

def run_pf_for_params(
    growth_rate: float,
    sigma_v: float,
    pooled_model: DegradationModel,
    engine_df: pd.DataFrame,
    n_particles: int,
) -> float:
    """
    Swap (growth_rate, sigma_v) into a shallow copy of pooled_model, run
    BootstrapPF on engine_df, and return log_likelihood_total.

    Unlike run_pf_for_sigma_v, this does NOT re-fit the HI builder or
    measurement noise on each call — both are shared from the pre-fitted
    pooled_model. Only the transition dynamics_func changes, which is the
    entire point of the Day-10 model-swap mechanism.

    Parameters
    ----------
    pooled_model : DegradationModel
        Must already have .fit(train_df) called. HI builder and measurement
        noise are reused as-is; only the dynamics are overridden.
    """
    model = _matched_model_from(pooled_model, growth_rate, sigma_v)
    pf = BootstrapPF(model=model, n_particles=n_particles)
    pf.run(engine_df=engine_df)
    return pf.log_likelihood_total


def pmmh_sample_joint(
    pooled_model: DegradationModel,
    engine_df: pd.DataFrame,
    n_iterations: int,
    n_particles: int = 500,
    growth_rate_init: float = GROWTH_RATE_PRIOR_MU,
    sigma_v_init: float = 0.3,
    rw_step_growth_rate: float = 0.1,
    rw_step_sigma_v: float = 0.2,
    regression_growth_rate: float | None = None,
    seed: int | None = None,
) -> dict:
    """
    2D PMMH: jointly sample (growth_rate, sigma_v) given one engine's data.

    Both parameters are sampled in log-space (always positive). Proposals
    are independent Gaussian random walks per dimension — rw_step_growth_rate
    applies to v = log(growth_rate), rw_step_sigma_v to u = log(sigma_v).

    Acceptance ratio:
        log_alpha = [logZ_prop + log_prior_growth_rate(gr_prop) + v_prop
                                + log_prior_sigma_v(sv_prop)   + u_prop]
                  - [logZ_prev + log_prior_growth_rate(gr_prev) + v_prev
                                + log_prior_sigma_v(sv_prev)   + u_prev]
    The +v and +u Jacobian terms account for the log-space change of variables
    (both priors are stated in original parameter space — same convention as
    pmmh_sample_sigma_v's +u term).

    Fallback: if accept_rate < FALLBACK_ACCEPT_THRESHOLD the growth_rate
    chain has not mixed and the PMMH estimate is untrustworthy.  When
    regression_growth_rate is supplied, growth_rate_mean in the returned
    dict is replaced by that value and source is set to 'regression_fallback'.

    Parameters
    ----------
    pooled_model : DegradationModel
        Pre-fitted on training data. Shared across all MCMC iterations.
    engine_df : pd.DataFrame
        Test-split rows for the single engine being calibrated.
    rw_step_growth_rate : float
        Random-walk step size in log(growth_rate) space. Default 0.1 is
        appropriate for the observed log-scale spread (~0.3 across the
        100-engine library).
    rw_step_sigma_v : float
        Random-walk step size in log(sigma_v) space. Default 0.2 (same
        as the 1D sigma_v-only pmmh_sample_sigma_v).
    regression_growth_rate : float or None
        Per-engine regression-fit growth_rate from the Day-10 library.
        Used as the fallback estimate when the chain is stuck.

    Returns
    -------
    dict with keys:
        'growth_rate_chain'      : array (n_iterations,)
        'sigma_v_chain'          : array (n_iterations,)
        'log_lik_chain'          : array (n_iterations,)
        'accept_rate'            : float
        'growth_rate_mean'       : float — posterior mean, or regression fallback
        'sigma_v_mean'           : float — posterior mean of sigma_v chain
        'source'                 : 'pmmh' or 'regression_fallback'
        'regression_growth_rate' : the value passed in (or None)
    """
    np.random.seed(seed)

    v_prev = np.log(growth_rate_init)
    u_prev = np.log(sigma_v_init)

    logZ_prev = run_pf_for_params(
        growth_rate=np.exp(v_prev),
        sigma_v=np.exp(u_prev),
        pooled_model=pooled_model,
        engine_df=engine_df,
        n_particles=n_particles,
    )

    n_accepted = 0
    growth_rate_chain = np.empty(n_iterations)
    sigma_v_chain = np.empty(n_iterations)
    log_lik_chain = np.empty(n_iterations)

    for m in range(n_iterations):
        v_prop = v_prev + np.random.normal(0, rw_step_growth_rate)
        u_prop = u_prev + np.random.normal(0, rw_step_sigma_v)
        gr_prop = np.exp(v_prop)
        sv_prop = np.exp(u_prop)

        logZ_prop = run_pf_for_params(
            growth_rate=gr_prop,
            sigma_v=sv_prop,
            pooled_model=pooled_model,
            engine_df=engine_df,
            n_particles=n_particles,
        )

        gr_prev = np.exp(v_prev)
        sv_prev = np.exp(u_prev)
        log_alpha = (
            (logZ_prop + log_prior_growth_rate(gr_prop) + v_prop
                       + log_prior_sigma_v(sv_prop)     + u_prop)
            - (logZ_prev + log_prior_growth_rate(gr_prev) + v_prev
                         + log_prior_sigma_v(sv_prev)     + u_prev)
        )

        if np.log(np.random.uniform(0, 1)) < log_alpha:
            v_prev, u_prev, logZ_prev = v_prop, u_prop, logZ_prop
            n_accepted += 1

        growth_rate_chain[m] = np.exp(v_prev)
        sigma_v_chain[m] = np.exp(u_prev)
        log_lik_chain[m] = logZ_prev

    accept_rate = n_accepted / n_iterations
    source = "pmmh" if accept_rate >= FALLBACK_ACCEPT_THRESHOLD else "regression_fallback"
    growth_rate_mean = float(np.mean(growth_rate_chain))
    sigma_v_mean = float(np.mean(sigma_v_chain))
    if source == "regression_fallback" and regression_growth_rate is not None:
        growth_rate_mean = regression_growth_rate

    return {
        "growth_rate_chain": growth_rate_chain,
        "sigma_v_chain": sigma_v_chain,
        "log_lik_chain": log_lik_chain,
        "accept_rate": accept_rate,
        "growth_rate_mean": growth_rate_mean,
        "sigma_v_mean": sigma_v_mean,
        "source": source,
        "regression_growth_rate": regression_growth_rate,
    }
