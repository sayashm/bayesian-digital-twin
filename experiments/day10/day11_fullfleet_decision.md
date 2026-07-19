# Day 11 — Full 100-engine fleet result: one-shot vs. periodic decided

Follow-up to `day10_problem_solution.md` §5, which left one-shot vs.
periodic undecided at n=5 ("not enough to settle one-shot vs. periodic").
This uses the same regression-based library (no PMMH calibration yet —
see the separate PMMH extension item) scaled to all 100 FD001 test
engines. Script: `experiments/day10/full_fleet_comparison.py`. Raw
per-engine results: `experiments/day10/full_fleet_results.json`.

## Result

| variant | mean abs. error | median abs. error | fleet RMSE | mean trajectory RMSE | PHM08 total |
|---|---|---|---|---|---|
| pooled (baseline) | 236.5 | 235.2 | 244.3 | 321.8 | 6.90e17 |
| **one-shot** | **149.9** | **140.0** | **156.0** | **199.6** | **9.19e12** |
| periodic | 162.6 | 155.0 | 166.8 | 219.5 | 9.25e12 |

Head-to-head, per engine: one-shot beats periodic on 63/100 engines,
periodic beats one-shot on 28/100, tied on 9. Both variants improve over
the pooled baseline on ~90/100 engines.

**Decision: one-shot is the thesis default.** It wins on every fleet-level
point-accuracy metric and on the majority of individual engines. This
settles the question the 5-engine prototype couldn't.

## One honest nuance (report this, don't hide it)

Periodic has a *lower* error standard deviation across the fleet (37.4
vs. one-shot's 43.1, vs. pooled's 60.9) — it's less accurate on average
but slightly more consistent engine-to-engine. Checked where each wins:
periodic pulls ahead mostly on engines with more cycles observed (more
chances to re-match and correct an early error); one-shot pulls ahead
mostly on shorter-lived engines (less time for periodic's later rematches
to help, and each rematch adds a chance to drift onto a worse-matched
rate). This matches the day10 write-up's structural reasoning (one-shot
can't recover from a bad early match; periodic can) — it just isn't the
dominant effect at this MMD/top-5/20-cycle configuration. Worth mentioning
in §4.7/§5 as a real, secondary finding rather than declaring periodic
strictly worse.

## What this does NOT settle

This is still the **regression-based** library (one exponential fit per
training engine via OLS-style `DegradationModelLearner`, same as the
5-engine prototype) — not the PMMH-calibrated joint (growth_rate,
sigma_v) extension planned separately (needs to run locally via Claude
Code; see the handoff prompt). That extension could still change the
absolute numbers, though it's unlikely to flip the one-shot-vs-periodic
ranking, since both variants share the same matching mechanism and would
be affected similarly by a better-calibrated sigma_v.

## Update (Day 12) — PMMH-calibrated rerun, prediction confirmed

Ran via Claude Code: `experiments/day10/run_pmmh_joint_fullfleet.py` fit
(growth_rate, sigma_v) jointly per training engine via PMMH
(n_particles=3200, n_iterations=1000; 176.6 min wall-clock) — **only
31/100 chains converged** (accept_rate ≥ 0.10), the rest fell back to the
regression-fit growth_rate already in this library. Mean accept_rate
0.107 (range 0.004–0.518); this convergence rate is notably lower than
the 4-engine diagnostic's 50% (2/4), so the diagnostic set was not
representative of the full fleet's mixing difficulty — a real finding in
its own right, not just diagnostic noise (see `Thesis_completing_Progress.md`,
Day 12 entry, for the per-engine detail).

`experiments/day10/full_fleet_comparison_pmmh.py` then reran the exact
same comparison (same script, same matching mechanism, same seed) with
only the library's `growth_rate` swapped to this PMMH/fallback blend:

| variant | mean abs. error | median abs. error | fleet RMSE | error std | mean traj. RMSE | PHM08 total |
|---|---|---|---|---|---|---|
| pooled (baseline, unchanged) | 236.5 | 235.2 | 244.3 | 60.9 | 321.8 | 6.90e17 |
| **one-shot (PMMH)** | **144.7** (was 149.9) | **136.5** (was 140.0) | **152.0** (was 156.0) | 46.7 (was 43.1) | **192.7** (was 199.6) | **9.19e12** |
| periodic (PMMH) | 154.9 (was 162.6) | 149.0 (was 155.0) | 160.9 (was 166.8) | 43.5 (was 37.4) | 209.6 (was 219.5) | 9.25e12 |

Head-to-head: one-shot beats periodic on 61/100 engines (was 63/100),
periodic on 29/100 (was 28/100), 9 ties → 10.

**The prediction above holds exactly.** PMMH calibration gives a small,
consistent accuracy improvement on every point-accuracy metric for both
variants (~3-5% lower mean/median/RMSE error) — and **does not flip the
one-shot-vs-periodic decision**: one-shot still wins by essentially the
same margin (61 vs. 63 out of 100). One new nuance, not previously
visible: PMMH calibration *increases* the fleet's error standard
deviation for both variants (one-shot 43.1→46.7, periodic 37.4→43.5) even
as the mean improves — a handful of engines get a materially different
matched rate than their regression fit and end up worse off individually,
even though the fleet average improves. Worth a sentence in §4.7 if this
rerun is folded into the thesis text, alongside the honest note that
69/100 of this "PMMH-calibrated" library is actually still the regression
fit (mixing didn't converge for those engines).

## Figures produced

- `thesis/Fig/fig_library_fullfleet_comparison.png` — bar chart, mean
  abs. error (±1 std) and PHM08 total, pooled vs. one-shot vs. periodic.
- `thesis/Fig/fig_engine78_library_before_after.png` — qualitative
  example on engine 78 (already used in §4.2/§4.6 discussion of
  ci_coverage=0), health tracking + RUL trajectory, pooled vs. one-shot
  side by side. Pooled abs. error 410 cycles → one-shot 188 cycles on
  this specific engine.
