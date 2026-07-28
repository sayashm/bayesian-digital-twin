# Prompt for Claude Code — upgrade `particle_twin/analysis/rul.py`

Copy everything below the line into Claude Code.

---

## Task

Upgrade `particle_twin/analysis/rul.py` (currently the v1 implementation) into a
literature-grounded v2. A draft of the new version already exists at
`experiments/final_experiment/rul_v2_proposed.py` — **read it first**, it contains
measured FD001 numbers and a full rationale. Your job is to promote it into the
package, reconcile it with the prognostics literature I have put in `RUL_PROBLEM/`,
and add the pieces it is still missing.

Before writing code, read:

1. `particle_twin/analysis/rul.py` (v1 — what we have)
2. `experiments/final_experiment/rul_v2_proposed.py` (v2 draft — 551 lines)
3. `particle_twin/models/state_space.py` (transition model, HI is on a 0–100 scale,
   `h = x/100`)
4. `particle_twin/features/health_index.py` (how the HI is built; note the
   `'industrial'` method is oracle-only)
5. `particle_twin/filters/bootstrap.py` and `particle_twin/library/streaming_filter.py`
6. `experiments/final_experiment/02_run_experiment.py` (the caller)
7. `particle_twin/data/database.py` → `store_rul_timeseries()` signature

## Why the current v1 is wrong (this is the core of the change)

v1 forward-simulates every particle until `h <= 0.0` and reports the median.
On FD001 that gives fleet MAE ≈ 151 cycles and 90% CI coverage of 0.00. The health
tracking itself is fine (HI trajectory RMSE 0.093) — the **health → RUL conversion**
is the broken step. Five separate causes:

**(1) Wrong failure threshold.** The literature formalism is
`RUL(t) = t_f − t` where `t_f` solves `HI(t_f) = θ`, and **θ must be tuned on the
training set**, not assumed to be zero. (Ramasso & Saxena 2014, PHM Society,
`RUL_PROBLEM/20150007677.pdf`, §4.2 "functional mapping between health index and
RUL"; and de Beaulieu et al., IFAC SAFEPROCESS 2022,
`RUL_PROBLEM/SafeProcess_Article___Short_Version(2)(2).pdf`, which uses an
empirically chosen `VHI_EOL = 0.86` precisely because their normalised HI never
reaches its own bound — their test-set VHI maxes out at 0.82, not 1.0.)
Our measurement on all 100 FD001 training engines (mean of each engine's last 5
cycles): observed HI at failure = **0.2565 ± 0.0144**, healthy HI = 0.655 ± 0.070,
global minimum 0.172. So v1 made every particle traverse a ~0.26-wide zone no real
engine ever enters.

**(2) The threshold must be calibrated in posterior space, not observation space.**
The filter's posterior median sits systematically above the observed HI (posterior
≈0.76 vs observations ≈0.67 in every `final_experiment/figures/engine_*.png`),
because `sample_initial()` starts particles at `h ~ N(1, sigma_0)` while real
engines start at HI ≈ 0.655, and the learned measurement noise (`sigma_w = 9.26` HI
units) corrects that only slowly. Crossing a posterior-space trajectory against an
observation-space threshold double-counts that offset. Measured posterior health at
the true failure cycle over 30 training engines: **0.336 ± 0.026**.

**(3) The threshold is itself uncertain.** Engines do not all fail at the same
health. Draw an independent threshold per particle from `N(mu_fail, sigma_fail²)`
so this real variability becomes honest posterior width instead of false confidence.

**(4) The HI saturates — RUL above ~125 is not identifiable.** Mean training HI by
true-RUL band: [0,25)=0.329, [25,50)=0.458, [50,75)=0.534, [75,100)=0.582,
[100,125)=0.611, [125,150)=0.631, [150,400)=0.655. Beyond RUL ≈125 the bands are
separated by less than the within-band std (0.070). `corr(HI, RUL)` = 0.743 over all
data but 0.826 restricted to RUL ≤ 125. This is exactly why the C-MAPSS literature
uses the piecewise-linear RUL target capped at 125 (Heimes 2008; Ramasso & Saxena
2014). Expose `rul_cap` (default 125).

**(5) The median is the wrong point estimate under PHM08 loss.** The PHM08 score is
asymmetric — late predictions cost `exp(d/10)−1`, early ones `exp(−d/13)−1`. Since
we already have the full posterior, numerically minimise expected PHM08 loss over
the RUL samples. That is the decision-theoretically correct action and it lands
well below the median.

## What to implement

Rewrite `particle_twin/analysis/rul.py` with:

- `calibrate_failure_threshold(model, train_df, mode="posterior"|"observed", ...)`
  returning `(mu_fail, sigma_fail)`. `"observed"` reads the last-k-cycle HI on the
  training fleet (fast); `"posterior"` runs the filter on N training engines and
  reads the posterior median at the true failure cycle (recommended). Document the
  ~50x cost difference.
- `simulate_rul(particles, transition_fn, failure_threshold, threshold_std=0.0,
  rul_cap=125, max_horizon=200, transition_fns=None)` — per-particle threshold
  draws; optional per-particle transition function (one per matched library engine)
  so growth-rate/model uncertainty propagates instead of collapsing to its mean.
  Return censoring info, not just the step counts.
- `phm08_score(predicted, true)` and `bayes_optimal_rul(rul_samples, ...)`.
- `extract_rul()` / `extract_rul_trajectory()` keeping **all v1 dict keys**
  (`cycle`, `rul_median`, `rul_p5`, `rul_p95`, `rul_mean`, `rul_std`, `health_mean`,
  `ess`) so `02_run_experiment.py` and `store_rul_timeseries(**estimate)` keep
  working. New keys are additive (`rul_bayes`, `p_failed_by_horizon`,
  `frac_censored`). Provide `DB_COLUMNS` + `to_db_row()` so callers can filter
  before `**`-unpacking into the DB.
- Module-level calibrated constants `FD001_FAILURE_THRESHOLD_OBSERVED = (0.2565,
  0.0144)` and `FD001_FAILURE_THRESHOLD_POSTERIOR = (0.336, 0.026)`, each with a
  comment saying which space it lives in and how to reproduce it.

### Things the v2 draft does NOT yet have — add them

- **A deterministic curve-fit baseline for comparison.** The classical HI→RUL
  method is: fit `HI(t) = a·exp(b·t)` (b<0) or a linear fit to the *degradation
  segment only* (not the whole life), solve `t_f = ln(θ/a)/b`, report `RUL = t_f − t`.
  Implement this as `fit_threshold_crossing_rul(t_hist, hi_hist, t_now, theta,
  model="exponential"|"linear")`. This is the standard baseline the thesis should
  beat, and having it in-package makes the comparison honest. See
  `RUL_PROBLEM/how estimate RUL from Health index for NASA CMAPPS.md` and
  `RUL_PROBLEM/Python code for C-MAPSS RUL estimation using exponential model.md`
  for the exact formulas.
- **Fit only on the degradation segment.** Both reference notes flag this: fitting
  the exponential over the full life (including the flat healthy phase) biases `b`
  toward zero and inflates RUL. Add a change-point / first-crossing heuristic to
  select the degradation onset.
- **Monotonicity handling.** The literature stresses the HI should be monotone for
  threshold crossing to be meaningful. Add an optional smoothing/isotonic step and
  report the measured monotonicity of our HI as a diagnostic.
- **Threshold sensitivity as a first-class output**, not a footnote. Provide a
  helper that sweeps θ over a range and returns fleet MAE/RMSE/PHM08, so the
  choice of θ is defended by a curve rather than asserted.

## Validation (required before you call this done)

Reproduce the v2 draft's measured numbers on all 100 FD001 test engines
(one-shot matching, `n_particles=500`, prediction at each engine's last observed
cycle) and report a table:

|                   | MAE   | RMSE  | PHM08   | bias   | 90% CI cov. |
|-------------------|-------|-------|---------|--------|-------------|
| v1 (median)       | 151.1 | 157.2 | 2.44e13 | +151.1 | 0.00        |
| v2 (median)       | 22.9  | 27.8  | 3.97e03 | +19.2  | 0.51        |
| v2 (`rul_bayes`)  | 19.9  | 23.9  | 1.63e03 | +12.8  | 0.51        |

If your numbers differ from these, **tell me — do not silently adjust defaults to
match.** Also add the new deterministic-baseline row.

Write the validation as `experiments/test_rul_v2.py` following the style of the
existing `experiments/test_rul.py`.

## Constraints — read carefully

- **Do not tune anything on the test set.** The 90% CI coverage of 0.51 is short of
  nominal. The draft measured that inflating `threshold_std` ×2 → 0.59 and ×3 → 0.76.
  Do **not** do that. Report 0.51 honestly; the fix is propagating growth-rate
  uncertainty via `transition_fns`, and if that still leaves a gap, that gap is a
  genuine finding about model overconfidence and belongs in the thesis discussion,
  not in a tuned constant.
- Keep backward compatibility. Nothing in `experiments/` should break.
- Every non-obvious number in the code must carry a comment saying where it was
  measured and how to reproduce it.
- Docstrings in the same style as the current `rul.py` — explain the *why*, in plain
  language, with concrete numeric examples. This code is read by thesis examiners.
- Cite in the docstrings: Ramasso & Saxena (2014) for the HI→RUL mapping taxonomy
  and the C-MAPSS benchmarking caveats; de Beaulieu et al. (2022, IFAC SAFEPROCESS,
  doi:10.1016/j.ifacol.2022.07.212) for threshold-crossing on a normalised VHI with
  an empirically chosen θ; Heimes (2008) for the piecewise-linear RUL cap. Both PDFs
  are in `RUL_PROBLEM/`.

## Plan first

Before editing, give me a short plan: what functions change, what is added, what the
new call site in `02_run_experiment.py` looks like, and anything in the v2 draft you
disagree with. Wait for my go-ahead.
