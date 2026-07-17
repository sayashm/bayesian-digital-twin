"""
rul.py — RUL extraction from the particle posterior
=====================================================
Given the particle cloud {h_t^(i)} at some observed cycle t (usually the
last one), propagates each particle forward in time -- under the SAME
stochastic transition model the bootstrap filter uses -- until it crosses
the failure threshold (h = 0). The number of steps each particle takes to
cross is one Monte Carlo sample of the Remaining Useful Life (RUL) random
variable. Collecting one sample per particle gives an empirical RUL
distribution, not a single point estimate.

Why a distribution: RUL uncertainty has two stacked sources --
  1. uncertainty in the CURRENT health state (the spread of h_t^(i)
     itself, inherited from the filter's posterior), and
  2. uncertainty in FUTURE degradation (transition() adds independent
     process noise at every forward step).
Monte Carlo forward simulation captures both automatically, because it
starts from the full particle spread and re-applies the noisy transition
at every future step.

Algorithm (Day 4):
  1. simulate_rul(): vectorised forward simulation. Advances only the
     still-"alive" particles each step; records the step count the
     instant each particle crosses the failure threshold; censors
     (caps) any particle that survives past max_horizon.
  2. extract_rul(): pulls the particle cloud + weights from one entry of
     BootstrapPF.history, resamples to an unweighted set (the filter's
     own ESS-gated resample is not guaranteed to have fired on THIS
     exact cycle), forward-simulates via simulate_rul(), and summarises
     the resulting RUL samples (median, 90% CI, mean, std) alongside the
     exact weighted health_mean (computed pre-resample, so it carries no
     extra resampling noise).
  3. extract_rul_trajectory(): calls extract_rul() at every observed
     cycle (skipping history[0], the pre-observation prior), giving the
     full per-timestep RUL trajectory that the `rul_estimates` DB table
     expects.

Outputs, per cycle:
  - RUL median (point estimate)
  - 5th / 95th percentile (90% credible interval)
  - RUL mean, std
  - health_mean, ess (carried through for the DB row / diagnostics)
"""

import numpy as np


def simulate_rul(particles: np.ndarray, transition_fn, failure_threshold: float = 0.0,
                  max_horizon: int = 400) -> np.ndarray:
    """
    Monte Carlo forward simulation: propagate each particle forward one
    cycle at a time via transition_fn until it crosses failure_threshold.

    Parameters
    ----------
    particles : shape (n_particles,) -- unweighted health samples, [0, 1] scale.
    transition_fn : callable, e.g. DegradationModel.transition -- advances
        an array of health values by one cycle (adds its own process noise).
    failure_threshold : float -- h <= failure_threshold counts as failed.
        0.0 is exact and reachable because transition() clips h to [0, 1].
    max_horizon : int -- safety cap on forward cycles simulated; any
        particle still alive at this point is censored (RUL = max_horizon).

    Returns
    -------
    steps : shape (n_particles,), int -- RUL sample for each particle.
    """
    h = particles.copy()
    alive = np.ones(len(h), dtype=bool)
    steps = np.zeros(len(h), dtype=int)

    for t in range(1, max_horizon + 1):
        h[alive] = transition_fn(h[alive])
        just_died = alive & (h <= failure_threshold)
        steps[just_died] = t
        alive[just_died] = False

        if not alive.any():
            break

    steps[alive] = max_horizon
    return steps


def extract_rul(pf, cycle_index: int = -1, failure_threshold: float = 0.0,
                 max_horizon: int = 400) -> dict:
    """
    Extract the RUL distribution from one entry of pf.history.

    Parameters
    ----------
    pf : BootstrapPF -- already .run() on one engine; pf.history holds the
        per-cycle particle clouds and (log-)weights.
    cycle_index : int -- index into pf.history (default -1, the last
        observed cycle). Use a positive index (skipping 0, the
        pre-observation prior) to build a full trajectory -- see
        extract_rul_trajectory().
    failure_threshold, max_horizon : passed through to simulate_rul().

    Returns
    -------
    dict with keys matching ResultsDB.store_rul_timeseries()'s parameters:
        cycle, rul_median, rul_p5, rul_p95, rul_mean, rul_std,
        health_mean, ess
    """
    entry = pf.history[cycle_index]
    h = entry['particles']
    log_w = entry['weights']

    # exact weighted mean health, for reporting -- no resampling noise
    health_mean = np.average(h, weights=pf.normalize(log_w))

    # unweighted particle set for forward simulation -- the filter's own
    # ESS-gated resample is not guaranteed to have fired on this exact
    # cycle, so we resample here regardless (a no-op if weights were
    # already uniform, since systematic resampling on uniform weights
    # reproduces the same particles).
    h_resampled, _ = pf.resample(h, log_w)

    rul_samples = simulate_rul(h_resampled, pf.model.transition,
                                failure_threshold=failure_threshold, max_horizon=max_horizon)

    return {
        'cycle': entry['cycle_number'],
        'rul_median': np.median(rul_samples),
        'rul_p5': np.percentile(rul_samples, 5),
        'rul_p95': np.percentile(rul_samples, 95),
        'rul_mean': np.mean(rul_samples),
        'rul_std': np.std(rul_samples),
        'health_mean': health_mean,
        'ess': entry['ESS'],
    }


def extract_rul_trajectory(pf, failure_threshold: float = 0.0, max_horizon: int = 400) -> list[dict]:
    """
    Call extract_rul() at every observed cycle (skips history[0], the
    pre-observation prior -- same convention as test_bootstrap.py).

    Returns
    -------
    list[dict] -- one extract_rul() result per observed cycle, ready to
    be unpacked straight into ResultsDB.store_rul_timeseries(**estimate).
    """
    rul_trajectory = []
    for t in range(1, len(pf.history)):
        rul_trajectory.append(extract_rul(pf, cycle_index=t, failure_threshold=failure_threshold,
                                           max_horizon=max_horizon))
    return rul_trajectory
