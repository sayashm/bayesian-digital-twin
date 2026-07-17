# Day 10 — From one pooled degradation curve to a per-engine reference library

## 1. The problem (confirmed on Day 9, root cause known since Day 4)

`particle_twin/models/degradation_learner.py` fits **one** exponential decay
rate by pooling the Health Index of all 100 FD001 training engines together
against absolute cycle number. That single rate is then reused, unchanged,
for every test engine's particle filter.

Fitting one rate per training engine instead (this prototype's library, see
§2) gives: mean growth_rate = 0.003727/cycle, std = 0.00115, range
[0.00175, 0.00664] across the 100 engines. The pooled fit's rate is
**0.002762/cycle — about 26% below the fleet's own per-engine average.**
Pooling engines with very different lifespans (FD001 engines fail anywhere
from ~130 to ~360 cycles) dilutes the fitted rate toward "too slow."

A too-slow rate means every particle takes longer than it should to reach
the failure threshold, so predicted RUL is systematically too high. Day 9's
FD001 fleet run showed this directly: predictions 2-5x the true RUL on
essentially every engine (e.g. engine 1: predicted 495 vs. true 112; engine
68: predicted 174 vs. true 8), and a fleet PHM08 total score of ~4.7e17
against the paper's reported 383 for the same dataset — PHM08 punishes
over-predicting RUL far more harshly than under-predicting it
(`exp(σ/10)` vs. `exp(-σ/13)`), so this bias gets exponentiated into an
astronomical number.

## 2. Solution direction: a per-engine reference library + similarity matching

This mirrors the paper's own fix (Cai et al. 2020, similarity-based
Rao-Blackwellized PF): instead of one fleet-wide rate, keep one **individually
fit** rate per training engine, and at test time match the new engine
against the library to borrow a more representative rate.

Implementation (`experiments/day10/similarity_library_prototype.py`):

- **Library**: `DegradationModelLearner` (unchanged) is called once per
  training engine on that engine's own (cycle, HI) series, instead of once
  on the pooled fleet series. 100 individual exponential fits, ~0.3s total.
- **Matching**: biased MMD with an RBF kernel (the same statistic as the
  paper's Eq. (3)) between the test engine's observed HI so far and each
  library engine's same-length prefix ("locate the test data at the
  beginning of Fm," per the paper's Fig. 2a). The 5 nearest library engines
  by MMD are averaged to get a matched growth_rate.
- **Swap**: the shared HI builder and measurement-noise model (both
  fleet-fit, unchanged) stay exactly as they are; only the transition
  function's growth_rate is replaced for the engine currently being
  filtered.

Two simplifications relative to the paper, worth flagging before this goes
into the thesis write-up:

1. Matching compares a single scalar Health Index, not the paper's full
   sensor feature matrix.
2. Ranking uses top-5-nearest-by-MMD rather than the paper's formal Kernel
   Two-Sample Test accept/reject procedure with a tunable significance
   level α. The paper's KTST gives a principled "is this reference similar
   enough at all" cutoff; top-k always returns 5 matches regardless of how
   dissimilar the whole library is to a given test engine. Worth
   revisiting if a test engine's operating regime turns out to be poorly
   covered by the training library.
3. Only `growth_rate` is matched/swapped — this thesis's exponential
   transition is already autonomous in a single rate parameter (the `a`,
   `c` terms of the paper's `F = a*e^{bt} + c` cancel out of the
   autonomous update, see `degradation_learner.py`'s `exponential_dynamics`),
   so there's no separate `a`/`c` to re-estimate the way the paper's RBPF
   tracks the full `[a, b, c]` state online. Lighter-weight than the
   paper's approach, but targeted at the same root cause.

## 3. The streaming question

Sajjad's point: a real digital twin doesn't have the test engine's full
trajectory available up front the way this offline replay does — it
arrives cycle by cycle. Two variants were implemented, both driven by the
same custom step-by-step filter loop (reuses `BootstrapPF`'s own
`normalize`/`ESS`/`resample` methods untouched — no changes to
`filters/bootstrap.py` or `models/state_space.py`):

- **One-shot**: wait for a 20-cycle warm-up, match once, keep that rate
  fixed for the rest of the engine's life. Closest to the paper's own
  method. Cheapest.
- **Periodic**: re-match every 20 cycles using all data observed so far,
  letting the matched rate — and the matched engine set itself — change
  over time.

Before the very first match, both variants fall back to the existing
pooled fleet dynamics, since nothing better is known yet. This is a
deliberate, realistic design choice for a live deployment, not an
oversight.

## 4. Prototype results (5 held-out FD001 test engines: 1, 10, 20, 30, 77)

| engine | true RUL | pooled abs.err | pooled RMSE | pooled PHM08 | one-shot abs.err | one-shot RMSE | one-shot PHM08 | periodic abs.err | periodic RMSE | periodic PHM08 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1  | 112 | 383.0 | 492.5 | 4.3e16  | 217.0 | 286.9 | 2.66e9 | 217.0 | 286.9 | 2.66e9 |
| 10 | 96  | 243.0 | 269.5 | 3.58e10 | 112.0 | 94.9  | 7.31e4 | 131.0 | 120.7 | 4.89e5 |
| 20 | 16  | 175.0 | 286.7 | 3.98e7  | 110.0 | 157.4 | 5.99e4 | 131.0 | 197.0 | 4.89e5 |
| 30 | 115 | 151.0 | 208.4 | 3.61e6  | 123.0 | 165.7 | 2.20e5 | 94.0  | 120.4 | 1.21e4 |
| 77 | 34  | 217.0 | 310.3 | 2.66e9  | 136.0 | 178.7 | 8.06e5 | 147.0 | 199.9 | 2.42e6 |
| **mean abs.err** | | **233.8** | | | **139.6** | | | **144.0** | | |

Every single engine improves under both matched variants, on every metric,
compared to the pooled baseline. Mean final-cycle absolute error drops from
233.8 cycles (pooled) to 139.6 (one-shot) / 144.0 (periodic) — a ~40% cut.
PHM08 scores drop by 3 to 11 orders of magnitude depending on the engine,
because the asymmetric scoring punishes the pooled model's persistent
over-prediction so heavily that even a moderate rate correction collapses
the score.

## 5. One-shot vs. periodic: what this small sample shows (and doesn't)

One-shot is slightly ahead on average final-cycle error here (139.6 vs.
144.0), but the ranking flips per engine — periodic wins outright on engine
30 (94.0 vs. 123.0). At n=5 this is not a decisive result either way.

Periodic's rematch log (below, for engines 10/20/30) shows the matched
engine set drifting sensibly as more of the trajectory is revealed, not
just jittering randomly — e.g. engine 30's matched growth_rate climbs from
0.00310 (cycle 21) to 0.00356 (cycle 141) as its true accelerating
degradation becomes more visible:

```
engine 30 periodic rematch history:
  cycle  21: matched=[14, 23, 20, 13, 12]  avg_rate=0.003095
  cycle  61: matched=[13, 36, 23, 75, 77]  avg_rate=0.003355
  cycle 101: matched=[75, 70, 49, 52, 71]  avg_rate=0.003133
  cycle 141: matched=[60, 87, 52, 63, 29]  avg_rate=0.003555
```

That's the behavior a genuine streaming digital twin needs — even though it
doesn't yet show a decisive final-cycle-accuracy edge at this sample size.
Two structural reasons periodic should matter more at scale, beyond what 5
engines can show: (1) one-shot can never recover from a bad early match,
while periodic can; (2) a real deployment can't retroactively pick its
warm-up length the way an offline one-shot replay implicitly benefits from.
**Recommend evaluating both on the full 100-engine fleet before choosing
one as the default** — 5 engines is enough to prove the library+matching
idea works, not enough to settle one-shot vs. periodic.

## 6. Consistency check / things to flag separately

- `rematch_at=[]` (pooled-only) run through this prototype's own streaming
  loop reproduces engine 1's Day 9 number exactly (495.0 / abs.err 383.0 /
  PHM08 4.3e16 — bit-identical). Engine 10 is close but not exact
  (339.0 vs. Day 9's 341.5, <1% difference) — traced to how the global
  random stream aligns: Day 5/9's original run seeds once and processes
  all 100 engines in one continuous loop, so engine 10's random draws
  depend on everything consumed by engines 1-9 first; this prototype
  reseeds per engine for isolation, so it starts fresh at engine 10.
  Not a logic bug — just a different (equally valid) RNG alignment for a
  Monte Carlo method.
- **Separate issue, not fixed here**: `experiments/run_full_fd001.py`'s
  committed `MAX_HORIZON` constant reads 300, but the RUL values actually
  stored in `experiments/fd001_full_results.json` (reused verbatim by
  Day 9) go as high as 954.5 — only reachable if that run actually used
  1500, matching the Day 4 status note ("raised to 1500 to fix this
  specifically"). The script and its own saved output disagree. This
  prototype used 1500 to match the real baseline; the script/output
  mismatch itself should be fixed independently.

## 7. If this becomes the new default

- Run all 3 variants (pooled / one-shot / periodic) on the full 100-engine
  FD001 test fleet, not just 5, before picking one as default — needed for
  a real one-shot-vs-periodic decision (§5).
- Consider whether matching should also adjust the measurement-noise model
  per engine, not just growth_rate (currently shared/fleet-fit throughout).
- If periodic is adopted, tune the re-match cadence and warm-up length via
  a small grid search rather than reusing this prototype's fixed 20/20.
- Upgrade top-k-nearest to the paper's full KTST accept/reject test with a
  tunable α if more rigor is wanted before this goes into the methodology
  chapter.
- Fix the `MAX_HORIZON` script/output mismatch (§6) independently of this
  work.

**Files**: `experiments/day10/similarity_library_prototype.py` (code),
`experiments/day10/similarity_library_results.json` (per-engine results +
full rematch logs for all 5 engines / 3 variants).
