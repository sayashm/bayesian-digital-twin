"""
metrics.py — Evaluation metrics
=================================
Turns raw RUL predictions (from rul.py) and filter diagnostics (from
bootstrap.py's ESS trace) into the numbers the results chapter reports:
RMSE, MAPE, and an ESS summary. No database or plotting code lives here —
this module only computes numbers from arrays/dicts already in memory.

Two error scopes, both needed for different jobs
--------------------------------------------------
1. FINAL-CYCLE error (one number per engine): compares the RUL predicted
   at the engine's last observed cycle against the single ground-truth
   value in RUL_FD001.txt. This is what the C-MAPSS / PHM08 literature
   normally reports, and it is what run_results.rmse / .mape already
   expect (one row per engine).

2. TRAJECTORY error (one number per engine, but built from every observed
   cycle): compares the FULL predicted RUL curve against the implied
   ground-truth countdown (true_rul + cycles remaining) at every cycle,
   not just the last one. A single final-cycle error is one sample and
   can be flattered or unlucky; the trajectory RMSE uses the whole curve
   test_rul.py already plots, so it is more stable and is what item 4
   (which engines are hardest, and why) is actually built on.

Fleet-level RMSE/MAPE (used for the headline number in the results
chapter) are then the final-cycle errors aggregated with sqrt(mean(.))
/ mean(.) across all engines — NOT the per-engine trajectory error,
which is a diagnostic, not the reported metric.

Day 9 addition
------------------------------------------------
Two more metrics needed for the baseline comparison (§4.6) and the
calibration argument (§4.2): phm08_score/phm08_total_score (the
asymmetric PHM08 competition score) and ci_coverage (credible interval
calibration check). Written by Sajjad step by step, reviewed each step
-- two real bugs caught: phm08_total_score initially returned a bare
np.sum() result (np.float64, not float, breaking the file's own
convention and the same numpy-scalar/JSON issue documented in
run_full_fd001.py's _json_default); ci_coverage's boolean mask was
first written as an unparenthesised chained comparison with `&`
(`lower <= true & true <= upper`), which is a real Python operator-
precedence trap -- `&` binds tighter than `<=`, so without parentheses
around each comparison it does not evaluate to what it looks like.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Pointwise error functions (element-wise; work on scalars or arrays)
# ----------------------------------------------------------------------

def absolute_error(pred, true):
    """|pred - true|, element-wise."""
    return np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float))


def squared_error(pred, true):
    """(pred - true)^2, element-wise."""
    return (np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)) ** 2


def percentage_error(pred, true, eps: float = 1.0):
    """
    |pred - true| / max(true, eps), element-wise.

    eps guards the division when true RUL is 0 or near 0 (an engine
    observed right up to failure). Default eps=1.0 cycle -- small enough
    not to distort normal-sized errors, large enough that a true_rul of 0
    doesn't produce inf/nan.
    """
    true = np.asarray(true, dtype=float)
    denom = np.maximum(true, eps)
    return absolute_error(pred, true) / denom


# ----------------------------------------------------------------------
# Aggregators (reduce an array of pointwise errors to one number)
# ----------------------------------------------------------------------

def rmse(preds, trues) -> float:
    """sqrt(mean squared error). Use across engines (fleet RMSE) or across
    cycles within one engine (trajectory RMSE) -- same formula either way."""
    return float(np.sqrt(np.mean(squared_error(preds, trues))))


def mape(preds, trues, eps: float = 1.0) -> float:
    """Mean absolute percentage error (fraction, not %). Same dual use as rmse()."""
    return float(np.mean(percentage_error(preds, trues, eps=eps)))


def mae(preds, trues) -> float:
    """Mean absolute error. Same dual use as rmse()/mape() -- across
    engines (fleet MAE) or across cycles within one engine (trajectory
    MAE) -- same formula either way. Added Day 13 for the
    final_experiment health-index/RUL summary tables."""
    return float(np.mean(absolute_error(preds, trues)))


# ----------------------------------------------------------------------
# ESS summary (filter health diagnostic, not a prediction-error metric)
# ----------------------------------------------------------------------

def ess_summary(ess_values, n_particles: int | None = None) -> dict:
    """
    Summarise an engine's ESS trace (one entry per observed cycle).

    Parameters
    ----------
    ess_values : array-like -- ESS at each cycle (from pf.history / the
        RUL trajectory's 'ess' field).
    n_particles : int or None -- if given, also reports ESS as a fraction
        of n_particles (0-1 scale), which is what's actually comparable
        across runs using different particle counts.

    Returns
    -------
    dict: mean_ess, min_ess, max_ess, std_ess, and (if n_particles given)
    mean_ess_frac, min_ess_frac.
    """
    ess = np.asarray(ess_values, dtype=float)
    out = {
        'mean_ess': float(np.mean(ess)),
        'min_ess': float(np.min(ess)),
        'max_ess': float(np.max(ess)),
        'std_ess': float(np.std(ess)),
    }
    if n_particles:
        out['mean_ess_frac'] = out['mean_ess'] / n_particles
        out['min_ess_frac'] = out['min_ess'] / n_particles
    return out


# ----------------------------------------------------------------------
# Ground truth reconstruction (shared with test_rul.py's figure code)
# ----------------------------------------------------------------------

def trajectory_ground_truth(cycles, true_rul: float, n_cycles: int) -> np.ndarray:
    """
    Implied true RUL at every observed cycle, given only the ONE ground
    truth value RUL_FD001.txt provides (at the truncation point, i.e. the
    engine's last observed cycle).

    C-MAPSS test engines are truncated before failure; true_rul is how
    many cycles remain AFTER the last observed cycle. Going backward from
    there, RUL simply counts down linearly with cycle number:
        ground_truth(cycle) = true_rul + (n_cycles - cycle)
    e.g. true_rul=16, n_cycles=184: at cycle 184 -> 16, at cycle 174 -> 26.
    """
    cycles = np.asarray(cycles, dtype=float)
    return true_rul + (n_cycles - cycles)


# ----------------------------------------------------------------------
# Per-engine and fleet-level summaries
# ----------------------------------------------------------------------

def evaluate_engine(trajectory: list[dict], true_rul: float, n_cycles: int,
                     n_particles: int | None = None) -> dict:
    """
    Summarise one engine's full RUL trajectory (output of
    rul.extract_rul_trajectory) against its ground truth.

    Parameters
    ----------
    trajectory : list[dict] -- extract_rul_trajectory() output; each dict
        has 'cycle', 'rul_median', 'ess', etc.
    true_rul : float -- ground truth from RUL_FD001.txt.
    n_cycles : int -- number of observed cycles for this engine (length
        of the test-split sequence before truncation).
    n_particles : int or None -- passed through to ess_summary().

    Returns
    -------
    dict with: final_cycle, final_rul_median, final_abs_error,
    final_pct_error, final_squared_error, trajectory_rmse,
    trajectory_mape, plus every key from ess_summary().
    """
    cycles = np.array([e['cycle'] for e in trajectory])
    medians = np.array([e['rul_median'] for e in trajectory])
    ess_values = np.array([e['ess'] for e in trajectory])

    ground_truth = trajectory_ground_truth(cycles, true_rul, n_cycles)

    final_pred = float(medians[-1])
    final_true = float(ground_truth[-1])  # == true_rul, kept for clarity

    out = {
        'final_cycle': int(cycles[-1]),
        'final_rul_median': final_pred,
        'final_abs_error': float(absolute_error(final_pred, final_true)),
        'final_pct_error': float(percentage_error(final_pred, final_true)),
        'final_squared_error': float(squared_error(final_pred, final_true)),
        'trajectory_rmse': rmse(medians, ground_truth),
        'trajectory_mape': mape(medians, ground_truth),
        'n_cycles': int(n_cycles),
        'true_rul': float(true_rul),
    }
    out.update(ess_summary(ess_values, n_particles=n_particles))
    return out


# ----------------------------------------------------------------------
# PHM08 asymmetric score (Day 9)
# ----------------------------------------------------------------------

def phm08_score(preds, trues) -> np.ndarray:
    """
    PHM08 competition scoring function (Saxena et al. 2008), element-wise.

    d = pred - true (signed error, e.g. at the final observed cycle)
      d < 0  (predicted RUL is LESS than true  -- early/conservative):
              score = exp(-d / 13) - 1
      d >= 0 (predicted RUL is MORE than true  -- late/dangerous):
              score = exp(d / 10) - 1

    Asymmetric on purpose: overestimating RUL is penalised faster
    (denominator 10) than underestimating it (denominator 13), because
    the engine can fail before a late prediction says it will.

    Parameters
    ----------
    preds, trues : array-like, same shape.

    Returns
    -------
    np.ndarray of per-element scores (same shape as input). Every score
    is >= 0; 0 is a perfect prediction.
    """
    d = np.asarray(preds, dtype=float) - np.asarray(trues, dtype=float)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def phm08_total_score(preds, trues) -> float:
    """
    Sum of phm08_score() across all engines -- the single number the
    PHM08 competition itself reported (lower is better, 0 is perfect).
    """
    return float(np.sum(phm08_score(preds=preds, trues=trues)))


# ----------------------------------------------------------------------
# Credible interval coverage / calibration (Day 9)
# ----------------------------------------------------------------------

def ci_coverage(trues, lowers, uppers) -> float:
    """
    Fraction of engines whose true RUL falls inside [lower, upper]
    (e.g. rul_p5 / rul_p95 at the final observed cycle -- the 90% CI
    already stored per cycle by rul.py / database.py).

    A well-calibrated 90% interval should cover close to 0.90 of
    engines. Much lower -> intervals too narrow (overconfident). Much
    higher -> intervals too wide (overly conservative, uninformative).

    Parameters
    ----------
    trues, lowers, uppers : array-like, same length -- one entry per
        engine (or per cycle, depending what you're checking
        calibration over).

    Returns
    -------
    float in [0, 1] -- the empirical coverage fraction.
    """
    trues_arr = np.asarray(trues, dtype=float)
    lowers_arr = np.asarray(lowers, dtype=float)
    uppers_arr = np.asarray(uppers, dtype=float)
    mask = (lowers_arr <= trues_arr) & (trues_arr <= uppers_arr)
    return float(np.mean(mask))


def evaluate_fleet(engine_evaluations: list[dict]) -> dict:
    """
    Aggregate per-engine evaluate_engine() dicts into fleet-level numbers.

    Fleet RMSE/MAPE use the FINAL-CYCLE errors (one point per engine),
    matching how run_results / the C-MAPSS literature report it. Trajectory
    RMSE is also aggregated (mean/median across engines) since it's the
    basis for the hardest-engine analysis.

    Returns
    -------
    dict: n_engines, fleet_rmse, fleet_mape, mean_trajectory_rmse,
    median_trajectory_rmse, mean_ess_frac (if present in the inputs).
    """
    final_preds = np.array([e['final_rul_median'] for e in engine_evaluations])
    final_trues = np.array([e['true_rul'] for e in engine_evaluations])
    traj_rmses = np.array([e['trajectory_rmse'] for e in engine_evaluations])

    out = {
        'n_engines': len(engine_evaluations),
        'fleet_rmse': rmse(final_preds, final_trues),
        'fleet_mape': mape(final_preds, final_trues),
        'mean_trajectory_rmse': float(np.mean(traj_rmses)),
        'median_trajectory_rmse': float(np.median(traj_rmses)),
    }
    if all('mean_ess_frac' in e for e in engine_evaluations):
        out['mean_ess_frac'] = float(np.mean([e['mean_ess_frac'] for e in engine_evaluations]))
    return out
