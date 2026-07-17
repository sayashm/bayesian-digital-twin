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

Deliberately NOT implemented here (scope note)
------------------------------------------------
The PHM08 asymmetric scoring function (penalises late predictions more
than early ones) is standard in the C-MAPSS literature but was not asked
for on Day 5 and isn't needed until the baseline comparison (§4.6,
Day 9), so it is left out for now rather than guessed at.
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
