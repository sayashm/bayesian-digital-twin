"""
reference_library.py — per-engine reference library + similarity matching
============================================================================
Consolidates the Day 10/11/12 similarity-library work into a single,
reusable, class-based implementation, replacing three copy-pasted
versions of the same logic that had accumulated across
experiments/day10/similarity_library_prototype.py (build_library,
match_library, the model-swap helpers) and their re-derivation in
particle_twin/inference/pmmh.py's PMMH follow-up.

Two classes:
  EngineRecord      -- one library entry: growth_rate, sigma_v, the
                        training engine's own Health Index series, and
                        where the parameters came from (a per-engine OLS
                        regression fit, or a converged PMMH chain, or the
                        regression fallback when a PMMH chain didn't mix).
  ReferenceLibrary   -- builds/holds all EngineRecords for a training
                        fleet, does MMD-based similarity matching against
                        a test engine's observed HI so far, and builds a
                        "matched" DegradationModel via
                        DegradationModel.with_matched_dynamics() (shared
                        HI builder/measurement noise from a pooled fit,
                        swapped-in per-engine dynamics).

See particle_twin/library/streaming_filter.py for the one-shot/periodic
particle-filter driver that actually uses a ReferenceLibrary to produce
per-engine health-index predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from particle_twin.models.state_space import DegradationModel
from particle_twin.models.degradation_learner import DegradationModelLearner


@dataclass
class EngineRecord:
    """One reference-library entry for a single training engine."""

    engine_id: int
    growth_rate: float
    sigma_v: float
    hi: np.ndarray            # this engine's own HI series, [0,1] scale
    cycles: np.ndarray        # this engine's own cycle numbers
    source: str = "regression"     # 'regression' | 'pmmh' | 'regression_fallback'
    r_squared: float | None = None
    accept_rate: float | None = None   # PMMH accept rate; None unless source involves PMMH
    n_iterations: int | None = None    # PMMH iteration count; None for pure regression


def _biased_mmd(x, y, length_scale: float = 1.0) -> float:
    """Biased MMD between two 1-D sets of scalars (HI values, [0,1] scale)
    with an RBF kernel — Eq. (3) in Cai et al. 2020, scalar version (the
    paper's own statistic uses a full sensor feature matrix; this thesis
    simplifies to a scalar Health Index, per day10_problem_solution.md)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def rbf(a, b):
        diff = a[:, None] - b[None, :]
        return np.exp(-(diff ** 2) / (2.0 * length_scale ** 2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return float(np.sqrt(max(kxx + kyy - 2.0 * kxy, 0.0)))


class ReferenceLibrary:
    """
    A per-engine reference library: one exponential (growth_rate, sigma_v)
    pair per training engine, matched against a test engine's observed
    Health Index via MMD similarity.

    Parameters
    ----------
    pooled_model : DegradationModel
        Must already have .fit(train_df) called. Its hi_builder and
        measurement_learner are shared, unchanged, by every matched model
        this library produces — only the transition dynamics differ
        per engine.
    dataset : str
        Dataset label used when persisting to/from ResultsDB.
    """

    def __init__(self, pooled_model: DegradationModel, dataset: str = "FD001"):
        if pooled_model.hi_builder is None:
            raise RuntimeError("pooled_model must already be .fit(train_df) before building a library.")
        self.pooled_model = pooled_model
        self.dataset = dataset
        self.records: dict[int, EngineRecord] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records.values())

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build_from_regression(cls, pooled_model: DegradationModel, train_df: pd.DataFrame,
                               sigma_v: float = 0.5, dataset: str = "FD001") -> "ReferenceLibrary":
        """
        Fit ONE exponential growth_rate per training engine via OLS
        (DegradationModelLearner, reused unchanged — just called once per
        engine instead of once on pooled data), using the pooled model's
        own hi_builder so HI scales match exactly. This is the Day 10/11
        library-building step — fast (~0.3s for 100 engines), and the
        default fallback source whenever a PMMH chain (see
        calibrate_with_pmmh below) doesn't converge.
        """
        lib = cls(pooled_model, dataset=dataset)
        # int(): train_df["unit_id"].unique() yields numpy.int64, not a Python
        # int -- left uncast, it silently gets BLOB-encoded by sqlite3 when
        # later written via to_db() (numpy.int64 supports the buffer protocol,
        # so sqlite3 stores it as 8 raw bytes instead of raising). Cast here
        # at the source, not just at the database.py write boundary.
        for engine_id in sorted(int(u) for u in train_df["unit_id"].unique()):
            edf = train_df[train_df["unit_id"] == engine_id].sort_values("cycle")
            hi = pooled_model.hi_builder.transform(edf).to_numpy()
            cycles = edf["cycle"].to_numpy(dtype=float)

            learner = DegradationModelLearner(
                health_series=hi, time=cycles, model_type="exponential", sigma_v=sigma_v,
            )
            lib.records[engine_id] = EngineRecord(
                engine_id=engine_id,
                growth_rate=learner.params["growth_rate"],
                sigma_v=sigma_v,
                hi=hi / 100.0,      # store on [0,1] scale, matches MMD comparison scale
                cycles=cycles,
                source="regression",
                r_squared=learner.fit_quality["r_squared"],
            )
        return lib

    def calibrate_with_pmmh(self, train_df: pd.DataFrame, n_particles: int = 3200,
                             n_iterations: int = 1000, rw_step_growth_rate: float = 0.1,
                             rw_step_sigma_v: float = 0.2, seed: int = 42,
                             engine_ids: list[int] | None = None,
                             on_engine_done=None) -> dict[int, dict]:
        """
        Refine growth_rate/sigma_v for some or all library engines via the
        joint PMMH sampler (particle_twin.inference.pmmh.pmmh_sample_joint)
        instead of the OLS regression fit — the Day 12 extension. Each
        engine's own existing regression growth_rate (from
        build_from_regression) is passed as the fallback, so a chain that
        doesn't mix (accept_rate < FALLBACK_ACCEPT_THRESHOLD) leaves that
        record's growth_rate unchanged and just marks source accordingly.

        This mutates self.records in place and also returns a plain dict
        of per-engine results (accept_rate, elapsed_s, etc.) for logging —
        callers that need resumability across a long run (this is
        expensive: ~90s/engine at the defaults) should persist after each
        `on_engine_done` callback rather than waiting for this method to
        return, e.g. via ReferenceLibrary.to_db().

        Parameters
        ----------
        on_engine_done : callable(engine_id, result_dict) or None
            Invoked after each engine finishes — e.g. to checkpoint
            progress to disk/DB for a long, resumable run.
        """
        # Imported here, not at module load, so a library that never
        # calibrates via PMMH doesn't pay for importing scipy.stats etc.
        from particle_twin.inference.pmmh import pmmh_sample_joint, GROWTH_RATE_PRIOR_MU

        import time

        targets = engine_ids if engine_ids is not None else sorted(self.records)
        results: dict[int, dict] = {}

        for engine_id in targets:
            rec = self.records[engine_id]
            engine_df = train_df[train_df["unit_id"] == engine_id]

            t0 = time.perf_counter()
            pmmh_result = pmmh_sample_joint(
                pooled_model=self.pooled_model, engine_df=engine_df,
                n_iterations=n_iterations, n_particles=n_particles,
                growth_rate_init=GROWTH_RATE_PRIOR_MU, sigma_v_init=0.3,
                rw_step_growth_rate=rw_step_growth_rate, rw_step_sigma_v=rw_step_sigma_v,
                regression_growth_rate=rec.growth_rate, seed=seed,
            )
            elapsed = time.perf_counter() - t0

            rec.growth_rate = pmmh_result["growth_rate_mean"]
            rec.sigma_v = pmmh_result["sigma_v_mean"]
            rec.source = pmmh_result["source"]
            rec.accept_rate = pmmh_result["accept_rate"]
            rec.n_iterations = n_iterations

            engine_result = {
                "accept_rate": pmmh_result["accept_rate"],
                "growth_rate_mean": pmmh_result["growth_rate_mean"],
                "sigma_v_mean": pmmh_result["sigma_v_mean"],
                "source": pmmh_result["source"],
                "elapsed_s": elapsed,
            }
            results[engine_id] = engine_result
            if on_engine_done is not None:
                on_engine_done(engine_id, engine_result)

        return results

    # ------------------------------------------------------------------
    # Persistence (particle_twin/data/database.py's engine_parameters table)
    # ------------------------------------------------------------------

    def to_db(self, db, exp_id: int) -> None:
        """Store every record's calibrated parameters via
        ResultsDB.store_engine_parameters (INSERT OR REPLACE, so reruns
        under the same exp_id are idempotent)."""
        for rec in self.records.values():
            db.store_engine_parameters(
                exp_id=exp_id, engine_id=rec.engine_id, dataset=self.dataset,
                growth_rate=rec.growth_rate, sigma_v=rec.sigma_v,
                source=rec.source, accept_rate=rec.accept_rate,
                n_iterations=rec.n_iterations,
            )

    def overlay_from_db(self, db, exp_id: int) -> int:
        """
        Patch growth_rate/sigma_v/source/accept_rate from a previously
        run calibrate_with_pmmh (e.g. results.db's exp_id=30) without
        re-running any PMMH — this library must already have HI/cycles
        for these engines (build_from_regression first). Returns the
        number of records patched.
        """
        n_patched = 0
        for row in db.get_engine_parameters(exp_id):
            rec = self.records.get(row["engine_id"])
            if rec is None:
                continue
            rec.growth_rate = row["growth_rate"]
            rec.sigma_v = row["sigma_v"]
            rec.source = row["source"]
            rec.accept_rate = row["accept_rate"]
            rec.n_iterations = row["n_iterations"]
            n_patched += 1
        return n_patched

    # ------------------------------------------------------------------
    # Similarity matching + model swap
    # ------------------------------------------------------------------

    def match(self, test_hi_so_far, k_top: int = 5, length_scale: float = 1.0):
        """
        Rank every library engine by MMD against the test engine's HI
        observed so far (compared over the SAME number of cycles, from
        the start — "locate the test data at the beginning of Fm", per
        Cai et al. 2020's Fig. 2a).

        Returns
        -------
        ranked : list[(engine_id, mmd)], sorted ascending (most similar first)
        avg_growth_rate : float — mean growth_rate of the top-k matches
        avg_sigma_v : float — mean sigma_v of the top-k matches (NOTE: the
            original Day 10/11 prototype used one hardcoded global sigma_v
            for every matched model; this returns a per-match blend
            instead, so a PMMH-calibrated per-engine sigma_v actually
            has an effect — pass a fixed sigma_v to matched_model()
            yourself if you need the old fixed-sigma_v behaviour.)
        """
        w = len(test_hi_so_far)
        scored = []
        for engine_id, rec in self.records.items():
            if len(rec.hi) < w:
                continue  # library engine too short to compare over this window
            ref_window = rec.hi[:w]
            mmd = _biased_mmd(test_hi_so_far, ref_window, length_scale=length_scale)
            scored.append((engine_id, mmd))
        scored.sort(key=lambda t: t[1])
        top = scored[:k_top]
        avg_growth_rate = float(np.mean([self.records[eid].growth_rate for eid, _ in top]))
        avg_sigma_v = float(np.mean([self.records[eid].sigma_v for eid, _ in top]))
        return top, avg_growth_rate, avg_sigma_v

    def matched_model(self, growth_rate: float, sigma_v: float) -> DegradationModel:
        """Build a DegradationModel with this pair swapped in, sharing the
        pooled model's HI builder/measurement noise. Thin wrapper around
        DegradationModel.with_matched_dynamics() so callers don't need to
        reach into pooled_model themselves."""
        return self.pooled_model.with_matched_dynamics(growth_rate, sigma_v)

    def summary(self) -> dict:
        """Fleet-level summary of the library's current growth_rate/sigma_v,
        and how many records came from each source — same numbers this
        project's session logs have reported since Day 10/11."""
        growth_rates = np.array([r.growth_rate for r in self.records.values()])
        sigma_vs = np.array([r.sigma_v for r in self.records.values()])
        sources = [r.source for r in self.records.values()]
        return {
            "n_engines": len(self.records),
            "growth_rate_mean": float(growth_rates.mean()),
            "growth_rate_std": float(growth_rates.std()),
            "growth_rate_min": float(growth_rates.min()),
            "growth_rate_max": float(growth_rates.max()),
            "sigma_v_mean": float(sigma_vs.mean()),
            "sigma_v_std": float(sigma_vs.std()),
            "n_pmmh": sources.count("pmmh"),
            "n_regression": sources.count("regression"),
            "n_regression_fallback": sources.count("regression_fallback"),
        }
