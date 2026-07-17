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
"""

import numpy as np
import pandas as pd
from scipy import stats

from particle_twin.models.state_space import DegradationModel
from particle_twin.filters.bootstrap import BootstrapPF

# Weakly-informative half-normal width for the sigma_v prior -- keeps the
# previously-hardcoded default (sigma_v=0.3) near the bulk of the prior
# mass without ruling out larger process noise. See thesis Sec. 3.4.
TAU_SIGMA_V = 0.3


def log_prior_sigma_v(sigma_v: float) -> float:
    """
    Prior: sigma_v ~ HalfNormal(tau=TAU_SIGMA_V).
    Weakly informative -- keeps sigma_v positive, centred near the
    previously-assumed value, without ruling out larger process noise.
    """
    return stats.halfnorm(scale=TAU_SIGMA_V).logpdf(sigma_v)


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
