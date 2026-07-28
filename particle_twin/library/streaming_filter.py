"""
streaming_filter.py — one-shot / periodic similarity-matched health-index
prediction, class version of experiments/day10/similarity_library_prototype.py's
run_streaming_pf().

SimilarityStreamingFilter runs the bootstrap particle filter cycle-by-cycle
against a ReferenceLibrary, re-matching the test engine's observed Health
Index so far against the library either once after a warm-up window
(one-shot, the thesis's chosen default — see
experiments/day10/day11_fullfleet_decision.md) or on a fixed interval
(periodic). Before the first match, both strategies use the library's
pooled model (nothing better is known yet) — a deliberate choice, realistic
for an actual streaming digital twin, not an oversight.

pf.history (returned by .run()) IS the health-index prediction: the
filtered posterior at every observed cycle. Feed it to
particle_twin.analysis.rul.extract_rul_trajectory() for RUL, or use
extract_health_trajectory() below / particle_twin.visualization.plots.
plot_health_tracking() directly for the health-index figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from particle_twin.filters.bootstrap import BootstrapPF
from particle_twin.library.reference_library import ReferenceLibrary

VALID_STRATEGIES = ("pooled", "one_shot", "periodic")


class SimilarityStreamingFilter:
    """
    Parameters
    ----------
    library : ReferenceLibrary
        Already built (build_from_regression and/or calibrate_with_pmmh).
    k_top : int
        Number of nearest library engines averaged at each (re)match.
    length_scale : float
        RBF kernel length scale for the MMD similarity metric (HI on
        [0,1] scale).
    """

    def __init__(self, library: ReferenceLibrary, k_top: int = 5, length_scale: float = 1.0):
        self.library = library
        self.k_top = k_top
        self.length_scale = length_scale

    def _rematch_cycles(self, strategy: str, n_cycles: int, warmup_cycles: int,
                         rematch_every: int) -> list[int]:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")
        if n_cycles <= 1 or strategy == "pooled":
            return []
        if strategy == "one_shot":
            return [min(warmup_cycles, n_cycles - 1)]
        # periodic
        return list(range(min(warmup_cycles, n_cycles - 1), n_cycles, rematch_every))

    def run(self, engine_df: pd.DataFrame, strategy: str = "one_shot", n_particles: int = 500,
            warmup_cycles: int = 20, rematch_every: int = 20, ess_threshold: float = 0.5):
        """
        Run the filter cycle-by-cycle, (re)matching per `strategy`.

        Returns
        -------
        pf : BootstrapPF-like object with .history populated (ready for
            extract_rul_trajectory / plot_health_tracking) and .model set
            to whichever dynamics was active last (for RUL extrapolation).
        rematch_log : list[dict] — one entry per (re)match event, logging
            which library engines matched and the resulting blended
            (growth_rate, sigma_v).
        """
        engine_df = engine_df.sort_values("cycle")
        cycles = engine_df["cycle"].to_numpy()
        pooled_model = self.library.pooled_model
        y = pooled_model.observe(df=engine_df).to_numpy()

        rematch_at = set(self._rematch_cycles(strategy, len(cycles), warmup_cycles, rematch_every))

        pf = BootstrapPF(model=pooled_model, n_particles=n_particles, ess_threshold=ess_threshold)
        h = pooled_model.sample_initial(n_particles)
        log_w = np.full(n_particles, -np.log(n_particles))
        pf.history = [{"cycle_number": 0, "particles": h, "weights": log_w, "ESS": pf.ESS(log_w)}]

        active_model = pooled_model
        rematch_log = []

        for i, t in enumerate(cycles):
            if i in rematch_at:
                top, avg_growth_rate, avg_sigma_v = self.library.match(
                    y[:i] if i > 0 else y[:1], k_top=self.k_top, length_scale=self.length_scale
                )
                active_model = self.library.matched_model(avg_growth_rate, avg_sigma_v)
                rematch_log.append({
                    "cycle_index": i, "cycle_number": int(t),
                    "matched_engines": [int(eid) for eid, _ in top],
                    "matched_mmds": [float(d) for _, d in top],
                    "avg_growth_rate": avg_growth_rate, "avg_sigma_v": avg_sigma_v,
                })

            h = active_model.transition(h)
            log_w = log_w + active_model.log_likelihood(h, y[i])
            if pf.ESS(log_w) < ess_threshold * n_particles:
                h, log_w = pf.resample(h, log_w)
            pf.history.append({"cycle_number": t, "particles": h, "weights": log_w, "ESS": pf.ESS(log_w)})

        pf.model = active_model
        return pf, rematch_log

    @staticmethod
    def extract_health_trajectory(pf):
        """
        Convenience extraction of the filtered (predicted) health-index
        trajectory straight from pf.history — percentiles taken directly
        from the particle cloud at each cycle (unweighted; same convention
        experiments/make_figure_engine77.py already used, since by the
        time a cycle is read here the filter's own ESS-gated resampling
        has typically already equalized weights).

        Returns
        -------
        cycles, p5, p50, p95 : arrays, one entry per OBSERVED cycle
            (skips history[0], the pre-observation prior).
        """
        cycles = np.array([entry["cycle_number"] for entry in pf.history[1:]])
        p5 = np.array([np.percentile(entry["particles"], 5) for entry in pf.history[1:]])
        p50 = np.array([np.percentile(entry["particles"], 50) for entry in pf.history[1:]])
        p95 = np.array([np.percentile(entry["particles"], 95) for entry in pf.history[1:]])
        return cycles, p5, p50, p95