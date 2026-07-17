# Day 11 — Calibration check on the similarity-library fix

Found while preparing to write Chapter 5 (Discussion): the full-fleet
comparison (`full_fleet_comparison.py`, feeding `day11_fullfleet_decision.md`)
only tracked `final_rul_median`/`final_abs_error`/`trajectory_rmse`/`phm08_score`
per engine — it never extracted `rul_p5`/`rul_p95`, so credible-interval
coverage (Eq. `eq:ci-coverage`) was never actually checked for the
similarity-library variants. §4.6 makes the calibration argument the
thesis's central claim for the particle filter's advantage over the
baseline; §4.7 needed to state honestly whether the library fix resolved
it, not just whether point accuracy improved.

## Result

New script: `experiments/day10/library_ci_coverage.py`. Full 100-engine
FD001 fleet, final-cycle 90% credible interval, pooled vs. one-shot
(thesis default):

| variant | ci_coverage |
|---|---|
| pooled (baseline) | 0.000 |
| one-shot (thesis default) | **0.000** |

Identical to Day 9's pooled-baseline number. Every one of the 100
engines' one-shot intervals still sits entirely above its true RUL —
e.g. engine 1: true RUL 112, one-shot interval [259.9, 444.1]; engine
100: true RUL 20, one-shot interval [121.0, 191.0]. The library fix
narrows the gap between the interval and the truth substantially (engine
78: pooled interval [392.9, 780.0] → one-shot [229.0, 402.1], against a
true RUL of 107) but does not close it, and coverage as a binary
in/out-of-interval check is unmoved by that narrowing unless the
interval actually reaches down to the truth.

## What this means for the thesis argument

The similarity-library fix (§3.5/§4.7) is a real, substantial
improvement — cuts mean abs. error ~35-40%, PHM08 total by 4-5 orders of
magnitude — but it is a **point-accuracy** fix, not a **calibration**
fix. The two are separate problems in this pipeline: point accuracy was
driven by the wrong \emph{mean} growth rate (fixed by matching); interval
width/placement is governed by $\sigma_v$ and the forward-simulation
noise model, neither of which the library touches. §4.6's central
argument — "PF's real advantage is calibrated UQ, which isn't yet
realised" — still holds exactly as stated on Day 9, even after today's
fix. This is now the honest, load-bearing framing for Chapter 5: the
library closes the "is the PF at least in the right neighborhood" gap;
it does not close the "can you trust its stated uncertainty" gap, which
was always the higher bar this thesis set for itself.

## Where this got added

- `chapt4.tex` §4.7: new paragraph stating coverage=0.000, inserted
  before the one-shot-vs-periodic decision (the decision was already
  based on point-accuracy metrics only, since coverage doesn't
  distinguish variants at 0.000-vs-0.000 either way).
- `chapt5.tex`: gained a chapter-level `\label{ch:discussion}` so §4.7
  could forward-reference the Discussion honestly rather than pointing
  at an unwritten placeholder.
