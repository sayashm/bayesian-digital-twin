# Thesis update plan — after the RUL v2 experiment (exp_id 47/48)

**Written:** 20 July 2026
**Trigger:** RUL extraction rewritten (calibrated failure threshold, threshold
uncertainty, RUL cap 350). Final numbers now come from `exp_id=47` (one-shot) and
`exp_id=48` (periodic), library `exp_id=35`.

---

## The headline: your thesis's main conclusion has flipped

This is the most important thing to understand before editing anything.

The thesis as written tells a **negative story**. Its central claim, repeated in
Chapter 4, Chapter 5's discussion, and the conclusions, is:

> "Point accuracy improved, but credible-interval coverage is **0.000** — the
> particle filter has the *structure* for calibrated uncertainty but does not
> deliver it. On calibration, the honest answer is **no, not yet**."

That is no longer true. Coverage is now **0.75**, and the residual gap is modest.
So the thesis now tells a **positive story with an honest caveat**:

> "Both axes improved. Point error fell from 244 to 44 cycles RMSE, and coverage
> rose from 0.000 to 0.75 against a nominal 0.90. The remaining gap is
> attributable to unpropagated growth-rate uncertainty."

Chapter 5 cannot be patched number-by-number — its argument is built on
coverage = 0.000. It needs rewriting. Chapters 3 and 4 need number updates plus
one genuinely new methodological section.

---

## What changed numerically

| Quantity | Thesis says now | Should say | Source |
|---|---|---|---|
| Fleet RMSE (PF, pooled) | 244.0 | unchanged (still the "before" case) | exp 24 |
| Fleet RMSE (one-shot library) | 156.0 | **44.35** | exp 47 |
| Fleet MAE (one-shot library) | 149.9 | **35.53** | exp 47 |
| Fleet MAE (periodic) | 162.6 | **39.03** | exp 48 |
| PHM08 total (one-shot) | 9.19e12 | **3.58e5** | exp 47 |
| CI coverage (one-shot) | 0.000 | **0.75** | exp 47 |
| CI coverage (periodic) | 0.000 | **0.70** | exp 48 |
| Trajectory RMSE (one-shot) | — | **77.9** (mean) / 71.9 (median) | exp 47 |
| Health-index tracking | unchanged | unchanged (RMSE 0.093, coverage 0.63) | exp 47 |
| Primary exp_ids | 24 / 26 / 27 | **47 / 48**, library 35 | — |

Health-index tracking is **identical** before and after — worth stating
explicitly, because it proves the improvement came from the health→RUL
conversion, not from better filtering.

---

## Chapter 3 — Methodology (`chapt3.tex`)

### 3.1 §3.3 "RUL Extraction from the Particle Posterior" — substantive rewrite

The method described here is the old one and is now wrong in three places.

**(a) The failure threshold. §3.3.1, Eq. (3.x) and surrounding text.**
Currently: "until it first crosses the failure threshold $x_\text{fail} = 0$".
That is exactly the bug the new experiment fixed. Replace with a calibrated,
*uncertain* threshold:

- Real FD001 engines do not fail at HI = 0. Observed HI over each training
  engine's last 5 cycles is **0.2565 ± 0.0144**; the global minimum anywhere in
  training is 0.172.
- The threshold must be calibrated in **posterior space**, not observation space,
  because the filter's posterior median runs above the observed HI. Measured on
  30 training engines at their true failure cycle: **0.336 ± 0.026**
  (the run used mu_fail = 0.3269, sigma_fail = 0.0264).
- Cite de Beaulieu et al. (2022, IFAC SAFEPROCESS) — they hit the identical
  problem and chose $\text{VHI}_\text{EOL} = 0.86$ empirically because their
  normalised VHI tops out at 0.82, not 1.0. Cite Ramasso & Saxena (2014) for
  the HI→RUL mapping taxonomy. Both PDFs are in `RUL_PROBLEM/`.

**(b) Threshold uncertainty as a modelled quantity — this is new and is a
contribution.** Add a short subsection. Each particle now draws its own
threshold $\theta^{(i)} \sim \mathcal{N}(\mu_\text{fail}, \sigma_\text{fail}^2)$,
so Eq. (3.x) becomes

$$R^{(i)} = \min\{\, r \ge 1 : x_{T+r}^{(i)} \le \theta^{(i)} \,\}$$

Argue *why*: engines genuinely do not all fail at the same health, so a single
hard threshold understates real variability. This is the single change most
responsible for coverage moving off 0.000, and it is a defensible modelling
choice, not a tuning knob.

**(c) The horizon cap. §3.3.3.** Currently framed only as a numerical safety
device ("a practical necessity ... not a modelling assumption"). It is now a
substantive choice at 350 cycles, justified from the data: the longest-lived
FD001 training engine runs 362 cycles, so 350 is an empirical upper bound on
plausible life, not an imported convention.

**Add a paragraph defending 350 against the literature's 125.** An examiner
familiar with C-MAPSS will ask why you did not use the standard piecewise-linear
cap of 125 (Heimes 2008). Your answer is good and should be in the text: the
125 cap is a convention for *supervised label construction*, where the target is
capped too. This thesis reports a *trajectory*, and at 125 the predicted RUL
curve is pinned flat at the cap for most of an engine's life, carrying no
information — see the engine 1 / 14 / 58 figures. Report both numbers so the
choice is transparent: at cap 125, final-cycle MAE is ≈21 but the trajectory is
degenerate; at 350 MAE is 35.5 and the trajectory is informative.

**(d) Algorithm 3.x pseudocode** — update: threshold draw per particle, cap
renamed, `while x > θ^(i)`.

### 3.2 §3.3.5 "Limitation: Dependence on the Fleet-Pooled Degradation Fit"

Still valid, but reframe. It currently predicts over-prediction as the dominant
failure mode. That is still directionally right (bias is +33 cycles) but the
magnitude claim ("hundreds of forward cycles") is stale.

---

## Chapter 4 — Results (`chapt4.tex`)

### 4.1 §4.3 "Experimental Setup" — reproducibility paragraph
Line 318: exp_ids 24 / 26 / 27 → **47 (one-shot), 48 (periodic), 35 (library)**.
Add the RUL-extraction config: mu_fail, sigma_fail, rul_cap = 350,
max_horizon = 350, threshold_mode = posterior.

Also: the §"Parameter estimation: hand-set vs PMMH" subsection says PMMH was run
on 4 engines only. That is stale — the joint (growth_rate, σ_v) PMMH was run on
the full 100-engine fleet on Day 12, with **31/100 chains converging** and the
rest falling back to regression fits. This is what `exp_id=35`'s library is built
from. Update it.

### 4.2 §4.5 "Results: RUL Prediction" — rewrite with new numbers
Replace RMSE 244.0 / coverage 0.000 throughout. Keep the pooled-dynamics result
as the documented "before" case — it is the motivation for the library — but
make clear the final configuration is the library + calibrated threshold.

### 4.3 §4.6 "Comparison with Baseline" — the comparison is now close, not lopsided
Table 4.4 currently reads:

| Model | RMSE | MAPE | PHM08 |
|---|---|---|---|
| Linear baseline | **32.0** | **0.53** | **16,121** |
| Particle filter | 244.0 | 5.44 | 4.73e17 |

New reality: PF RMSE **44.35**, MAPE **0.63**, PHM08 **3.58e5**. The baseline
still wins on point accuracy, but by ~40%, not by an order of magnitude — and
the PF now additionally supplies an interval that covers 75% of engines, which
the baseline cannot supply at all.

This means the whole framing paragraph ("The baseline wins decisively ... a
result this one-sided cannot be waved away") must be rewritten. **Keep the
honesty, drop the defeatism.** The argument is now genuinely available to you:
comparable point accuracy *plus* usable uncertainty. Note that the baseline row
is unaffected by the RUL change (it is a point predictor), so 32.0 stands and
does not need re-running.

### 4.4 §4.7 "Similarity-Based Library vs. Pooled Dynamics"
Table 4.5 numbers → exp 47/48. The paragraph beginning "90% credible interval
coverage under one-shot matching ... is **0.000** — identical to the pooled
baseline. Every engine's interval still sits entirely above its true RUL" is now
simply false and must go.

Replace with the honest new version: coverage 0.75 against nominal 0.90; the
residual gap is real; interval width is wide (mean 133 cycles at the final
cycle) so coverage is bought partly with width, not only with better centring.
Say that plainly — it is the kind of caveat that earns credibility.

### 4.5 **Gap: FD003 has not been re-run**
`exp_id=26` (FD003) still uses the old RUL method, so its reported coverage of
0.000 is not comparable to FD001's 0.75. Either re-run it or explicitly scope it
out. See "Decisions needed" below.

### 4.6 Figures
`thesis/Fig/` still holds the old figures (`fig_engine77_health_rul.png`,
`fig_engine78_library_before_after.png`, `fig_library_fullfleet_comparison.png`).
All three were generated from the old runs and show the old behaviour. Regenerate
from exp 47/48 and copy across. The 100 fresh `experiments/final_experiment/figures/engine_*.png`
are already correct — pick the representative engines from those.

Consider adding one **new** figure that earns its place: the cap comparison
(125 vs 350 trajectory for one engine), which visually justifies §3.3's cap
argument and pre-empts the examiner's question.

---

## Chapter 5 — Discussion and Conclusions (`chapt5.tex`) — rewrite, not patch

Almost every subsection is built on coverage = 0.000.

| Subsection | Action |
|---|---|
| §5.1.1 "What the Particle Filter Delivers" | Keep structure; update the library result numbers (35–40% → ~80% error reduction). |
| §5.1.2 "The Central Limitation: Point Accuracy Is Not Calibration" | **Premise gone.** Retitle and rewrite, e.g. "Calibration: Substantially Improved, Not Yet Complete". The interesting finding is now *which* change bought the calibration — modelling threshold uncertainty — not that calibration failed. |
| §5.1.3 "Likely Sources of the Calibration Gap" | Rewrite. The old text blames hand-set σ_v. The new residual gap (0.75 vs 0.90) is better explained by unpropagated growth-rate/model uncertainty across the k matched library engines. Do **not** claim this is proven — the honest statement is that inflating threshold_std alone raises coverage (×2 → 0.59, ×3 → 0.76 measured on the earlier configuration), which shows threshold spread is not the only missing variance component. |
| §5.1.4 "Other Limitations" | Mostly survives. Add: interval width (mean 133 cycles) is large relative to typical true RUL (median 86); the cap choice at 350 is defensible but does affect reported metrics. |
| §5.1.5 "Connection to the Digital Twin Framing" | Rewrite the ending. "A checkable claim that fails the check" is no longer the situation. |
| §5.2 "Conclusions" | Rewrite the third and fourth paragraphs. "On calibration ... the honest answer is no, not yet" → substantially yes. The maintenance-engineer paragraph currently tells the reader **not to trust the interval**; that advice should now be "the interval is usable but somewhat wide and mildly optimistic — treat 0.75 coverage as the operative number, not the nominal 0.90." |
| §5.3 "Future Work" | Bullets 1 and 2 (fleet-wide PMMH σ_v; joint growth-rate + σ_v estimation) are **done** — that is exp_id 30/35. Replace with the genuinely open items: propagate per-engine growth-rate uncertainty via `transition_fns`; reduce interval width while holding coverage; threshold sensitivity as a function of HI construction method; the deterministic exponential curve-fit baseline as a third comparator. |

---

## Also update

- **`abstract_english.tex`** — 9 lines, will state the old conclusion.
- **`chapt1.tex`** — check the objective/contribution framing for coverage claims.
- **`Thesis_completing_Progress.md`** — add the Day-13 entry.
- **Bibliography** — add `debeaulieu2022unsupervised` (doi:10.1016/j.ifacol.2022.07.212)
  and `ramasso2014review`; `heimes2008recurrent` if you cite the 125 convention.

---

## Suggested order of work

1. Re-run FD003 (decision below) — it is the only thing that blocks Chapter 4.
2. Regenerate the three thesis figures from exp 47/48.
3. Chapter 3 §3.3 rewrite (threshold calibration + threshold uncertainty + cap).
   Do this first among the writing, because Chapters 4 and 5 reference it.
4. Chapter 4 number updates + §4.6 reframing.
5. Chapter 5 rewrite.
6. Abstract, Chapter 1 check, bibliography, progress file.

---

## Decisions taken (20 July)

1. **FD003 — re-run with the new RUL method.** Sajjad will build this experiment
   via Claude Code. Needs: a reference library fit on FD003 *training* engines
   (FD001's library must not be reused — different fault modes), its own
   `calibrate_failure_threshold(mode="posterior")` call, and the same
   `rul_cap`/`max_horizon`. Note FD003's longest training life is 525 cycles, so
   the cap may need to differ from FD001's 350 — decide it from FD003's own data
   and say so in the text.
2. **`rul_bayes` — include fully.** Store it in `rul_estimates`, add a column to
   the results table, and give it a short paragraph in §3.3 (the asymmetric PHM08
   loss makes the median the wrong action; the loss-minimising point estimate is
   only computable because a full posterior is available — this is a direct
   argument for the probabilistic approach that the linear baseline cannot make).
   Requires a re-run of exp 47/48 with the extra field stored.
3. **Cap defence — prose plus a sensitivity table.** Report final-cycle MAE/RMSE/
   PHM08/coverage *and* trajectory RMSE at cap ∈ {125, 350, 400}. The trajectory
   column is the one that makes the argument: 125 wins on final-cycle MAE but is
   degenerate on trajectory. Data for 125/400 already exists (exp 45/46 and
   43/44) — no re-run needed, just a table script.

### Re-run checklist implied by these decisions

- [ ] FD003 library + threshold calibration + one-shot/periodic run (new exp_ids)
- [ ] Re-run FD001 exp 47/48 storing `rul_bayes` (or backfill from stored samples)
- [ ] Cap sensitivity table from exp 43/44 (400), 45/46 (125), 47/48 (350)
- [ ] Regenerate the three `thesis/Fig/` figures + optional cap-comparison figure
