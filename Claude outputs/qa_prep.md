# سوالات احتمالی و پاسخ‌ها

بر اساس یادداشت‌های استاد و تام رو annotated PDF و پاسخ‌های تاییدشده‌ی قبلی.

---

## بخش ۱ — Introduction

منبع: `revise/revision_tracker.md` (کامنت‌های تام روی چاپت۱ — قبلاً همه‌شون تو متن فعلی chapt1.tex اصلاح شدن، ولی ممکنه سر جلسه شفاهی دوباره بپرسن)

**Q1. What exactly do you mean by "principled" in your objective statement?**
A: It means both the per-cycle update of the health state and the downstream RUL extraction follow directly from Bayes' rule applied to the state-space model — no heuristic corrections, and no uncertainty band glued onto a point estimate after the fact. Everything the model reports about uncertainty comes from the same probabilistic machinery that produced the estimate.

**Q2. Isn't a particle filter just an algorithm — how is that the "digital twin"?**
A: Correct distinction — the digital twin is the state-space model itself: the Health Index state, the degradation dynamics, and the observation model, built in Chapter 3. The particle filter is the inference algorithm that updates that twin's posterior distribution every time a new sensor reading arrives. The twin is the model; the particle filter is how we do inference in it.

**Q3. What about run-to-failure *tests* — those are deliberate too, aren't they?**
A: Yes — a controlled run-to-failure test bench is a deliberate experiment where the asset is expendable by design. That's different from an unplanned in-service failure, which is unsafe and costly. This thesis is about avoiding the second kind, not the first.

**Q4. What would it mean for calibration to be "realized in principle" versus empirically?**
A: Any Bayesian model produces a credible interval "in principle" just by having a posterior distribution — that's automatic from the math. That alone doesn't mean the interval is trustworthy. Whether it actually contains the true RUL close to 90% of the time has to be checked empirically against held-out test engines — which is exactly what Chapter 4 does, and it's the thesis's central result.

**Q5. Why does uncertainty matter so much — why not just optimize point accuracy like everyone else?**
A: Because a point number alone can't support a maintenance decision. "50 cycles left" means something very different if it's "give or take 3" versus "give or take 40" — same number, opposite action. Point-accuracy-only models were never asked to represent that, so they can't distinguish the two cases. That's the practical gap this thesis targets.

---
## بخش ۲ — The Data

**Q1. Why is the test set truncated — isn't that unusual? Did you do that yourself?**
A: No — that's the original design of the C-MAPSS / PHM08 benchmark, not something done in this thesis. It's meant to mimic a real deployment scenario: you never get to observe an asset running all the way to failure before you have to make a maintenance decision. NASA truncates each test engine's sequence at an arbitrary point and holds back the true remaining life separately, only for evaluation.

**Q2. If the test set is truncated, how do you even know the true RUL to evaluate against?**
A: NASA ran the full simulation to failure internally and released only the truncated sensor sequence to the user; the true remaining cycles at the truncation point are provided as a separate ground-truth file, used only for scoring — never visible to the model at prediction time.

**Q3. Why FD001 and FD003, and not FD002/FD004?**
A: FD002/FD004 add six operating conditions instead of one. Because FD003 keeps the same single operating condition as FD001 and only adds a second fault mode, it isolates robustness to fault-mode complexity specifically. The health index/normalisation pipeline (Chapter 3) uses a single fleet-wide sensor range and has no mechanism to condition on operating regime, so extending to FD002/FD004 would need a different normalisation scheme — left as future work.

**Q4. Why do test sequences vary so much in length compared to train?**
A: Because the truncation point is different for every test engine — some are cut early, some almost at failure — while every training engine runs the full trajectory to failure. That's exactly what produces the spread you see in the test true-RUL column (6 to 145).

---
## بخش ۳ — Data Analysis: Sensor Selection

**Q1. A constant sensor isn't necessarily useless — why exclude it outright?**
A: These sensors are constant across the *entire fleet* in both FD001 and FD003 — literally zero variance in the data actually used. With no variance there's nothing for a monotonic-trend or correlation-based criterion to detect, so they carry no usable signal for this pipeline's normalisation and tracking approach.

**Q2. Why five different criteria instead of just correlation with RUL?**
A: Each criterion catches something the others miss. Mann-Kendall/monotonicity/trendability catch consistent directional drift, robust to noise, without assuming linearity. Mutual information catches nonlinear relevance a linear correlation would completely miss. VIF is different in kind — it doesn't measure relevance to RUL at all, it flags sensors that are redundant with sensors already selected, which a relevance-only criterion would never catch. Combining them gives a fuller picture than any single statistic.

**Q3. Where does the VIF threshold come from — is >10 a standard rule?**
A: Yes, VIF > 10 is a commonly used rule of thumb for problematic multicollinearity in regression. Here it's used to flag sensors highly redundant with others already in the selected set — the two removed (Nc, NRc) had VIF around 16-17 in FD001, well above that threshold.

**Q4. You compute the composite score across all four C-MAPSS sub-datasets, but only use FD001/FD003 in your experiments — why?**
A: To make the sensor selection itself more robust rather than tuned narrowly to just the two datasets used later. Averaging across all four, including the six-operating-condition datasets FD002/FD004, checks that a sensor's diagnostic value holds more generally, not just in the specific slice of data this thesis happens to evaluate on.

**Q5. Is the composite-ranking figure only for FD001? What about FD003?**
A: The figure itself is FD001 only — there's no separate FD003 version of that chart. But the number the selection decision is actually based on isn't FD001-only: the composite score for each sensor is computed separately per sub-dataset and then averaged across all four (FD001-FD004), so it already reflects FD003's behaviour too. And the outcome — the 10 selected sensors — is a single list used identically for both FD001 and FD003, not a separate list per dataset. The figure is just one illustrative breakdown of that already-averaged process.

---
## بخش ۴ — Methodology: State Space Model

منبع: `revise/revision_tracker.md` — کامنت‌های ORANGE (سوالات احتمالی دفاع، مستقیماً همینا رو تام تگ کرده بود) و BLUE مربوط به Ch3

**Q1. You mention a "non-trivial" AIC improvement from a skew-normal, but still chose Gaussian. What makes the skew-normal harder to use, given a particle filter doesn't require Gaussian noise?**
A: It's a convenience choice, not an algorithmic constraint — the bootstrap filter's weight update works with any noise density. The Gaussian was kept because it gives a closed-form likelihood and matches the additive-noise fitting pipeline used elsewhere (least-squares transition fitting). The AIC gain, while statistically non-trivial, is modest in absolute terms, so the skew-normal is reported as a robustness check rather than adopted as the default.

**Q2. Did you do any outlier filtering before min-max normalising the sensors? Outliers directly distort a min-max range.**
A: No explicit outlier filtering — the min-max range is the raw observed training-fleet range. That's a real simplification risk I acknowledge: one extreme reading can compress the effective range for everyone. It's also why the same range, fitted on FD001, is reused unchanged on FD003 rather than refitted — FD003 values that fall outside that range are deliberately *not* clipped, so this kind of range sensitivity shows up honestly rather than being hidden.

**Q3. You use uniform weights across sensors, but wouldn't some sensors be more diagnostic than others?**
A: Yes, plausibly — but at this stage no sensor was judged a priori more diagnostic, so uniform weights were the simplest, most defensible default. The sensor-selection funnel from the previous section already filters out weak/redundant sensors first, so the 10 that remain are comparably informative by construction. A diagnosability-weighted average is a natural extension, left as future work.

**Q4. Why exponential dynamics specifically, and not linear or polynomial?**
A: Two independent reasons, not the R² table. First, NASA's own C-MAPSS simulator documentation states its damage propagation model is exponential in form. Second, exponential growth is the standard shape for wear-out failure processes in the reliability literature. The R² comparison across shapes on the sensor-based Health Index was actually inconclusive — pooling engines with very different lifetimes at a fixed cycle number suppresses R² for every shape roughly equally, so that table isn't read as deciding evidence.

**Q5. The measurement noise (σ≈9.26) is about 20x larger than the process noise (σ_v=0.5) — where does that come from, and isn't that a big gap?**
A: They come from different sources and aren't meant to be compared directly at this stage. σ_w=9.26 is fitted from data — the spread of training residuals around the fitted degradation trajectory. σ_v=0.5 is a hand-set starting value, not yet fitted; refining it properly, from data, is exactly what Particle Marginal Metropolis-Hastings does in the next section.

**Q6. The initial distribution h_0 ~ N(1, σ_0²) technically puts probability mass above h=1 (an "over-healthy" engine) — is that a problem?**
A: In principle yes, about half the initial mass sits above 1, which is slightly outside the valid health range. In practice it isn't a real problem: that tail is heavily downweighted or resampled out within the first few cycles, because the very first real observation (training-fleet HI starts around 0.65) already pulls the particle cloud toward the data. A bounded alternative like a uniform or beta distribution would avoid this technically but wasn't adopted, for the same closed-form-simplicity reason as the Gaussian noise model.

---
## بخش ۵ — Particle Filter Algorithm

منبع: `revise/revision_tracker.md` — ORANGE #3, #4 (سوالات احتمالی دفاع، دقیقاً رو همین بخش) + BLUE #12, #13, #14

**Q1. What's the downside of the bootstrap filter (proposing from the transition prior)? Any upside beyond simplicity?**
A: Upside: no need to design or evaluate a custom proposal density — the weight update reduces directly to the observation likelihood, simple and cheap per step. Downside: if the observation likelihood is much more peaked than the transition prior, most particles land where the likelihood is low and only a few carry meaningful weight — that's the degeneracy risk. Here the measurement noise (σ≈9.26) is much wider than the process noise (σ_v≈0.5), so that risk stays limited on any single step — which is exactly why the plain bootstrap filter is sufficient here rather than a more elaborate proposal.

**Q2. What is resampling actually doing — why is it needed?**
A: It discards particles carrying negligible weight and duplicates the ones carrying most of the posterior mass, so subsequent computation concentrates where the posterior actually is. Without it, effective sample size collapses toward 1 within 5–10 cycles — almost all particles end up wasting computation while carrying essentially zero weight.

**Q3. How exactly are the log-space weights normalised back to sum to 1?**
A: Via the log-sum-exp identity: subtract the maximum log-weight from every particle's log-weight before exponentiating — that keeps every exponent ≤ 0, so it can't overflow — then exponentiate. The result already sums to 1, no separate normalising step needed afterward.

**Q4. You say degeneracy risk is "limited" per step, but then say ESS collapses to ~1 within 5-10 cycles without resampling — isn't that a contradiction?**
A: No — different timescales. "Limited per step" is about how sharply a *single* cycle's likelihood reweights the prior, which is mild here because the observation noise is wide relative to the process noise. The 5–10-cycle collapse is what happens when many such mild reweightings compound across *consecutive* cycles with no resampling in between. Adaptive resampling exists precisely to keep that compounding effect bounded.

**Q5. Why τ=0.5 for the ESS threshold, and why not resample every cycle?**
A: Resampling every cycle needlessly discards particle diversity and adds computational cost at up to 350 cycles per engine; waiting until ESS is already very low leaves too few effective particles for resampling to meaningfully restore the cloud. τ=0.5 is a standard middle-ground threshold used in the sequential Monte Carlo literature.

**Q6. Why systematic resampling rather than simple multinomial resampling?**
A: Systematic resampling draws all N new particle indices from a single shared random offset rather than N independent draws, which gives a lower-variance set of resampled particle counts for the same computational cost — less noise added by the resampling step itself.

---
## بخش ۶ — RUL Extraction from the Particle Posterior

منبع: `revise/revision_tracker.md` — BLUE #15, #16, #17, #18 (دقیقاً همین بخش، سوالات تام)؛ اعداد censoring هم با `thesis-tom-final-revision` (حافظه‌ی پروژه) چک شد و مطابقته

**Q1. What exactly is the 0.336 ± 0.026 number?**
A: It's the fleet-average posterior-median Health Index measured at each training engine's own true failure cycle: take all 100 training engines, look at what the filter's posterior median says right at that engine's last cycle, and average across engines. The ±0.026 is the standard deviation of that value across engines, not a measurement-noise error bar.

**Q2. Why would comparing a posterior-space trajectory against an observation-space threshold "double-count" the offset — can you explain?**
A: The filter's posterior systematically sits a bit above the raw observed Health Index — particles start near full health while real engines already show some wear, and the fairly wide measurement noise only slowly corrects that. Since forward simulation propagates particles living in that same posterior space, comparing them against a threshold measured on raw observations would make every particle fall not just to the true failure level, but an extra amount equal to that posterior-vs-observation gap on top — over-correcting. Calibrating the threshold directly in posterior space cancels that gap by construction.

**Q3. You quote 0.352 here but 0.336 earlier — where does 0.352 come from?**
A: Two different, related quantities. 0.336 is the general fleet-average posterior-median health at failure. 0.352 is μ_fail specifically — the calibrated threshold obtained when the calibration procedure is re-run against the FD001 training fleet under the fixed-parameters configuration. They differ slightly because that configuration's posterior spreads a bit differently around the same underlying failure level, not due to any error.

**Q4. How is censoring actually handled? A particle that hasn't crossed by r_max doesn't truly have RUL = r_max — does this happen often, and does it affect coverage?**
A: Correct, it's a known simplification — a particle that hasn't crossed by r_max has a true RUL somewhere in [r_max, ∞), and recording it as exactly r_max piles mass at the cap, shifting the reported posterior slightly left and narrowing its upper tail. Frequency: on FD001 it's small for a typical engine (0.2% for the median engine) but averages 5.3% fleet-wide, exceeding 10% on 16 of 100 engines. On FD003 it's much larger — a 29.7% fleet average, above 10% on 80 of 100 engines. Yes, it does affect coverage — the reported coverage numbers in the results include this censoring effect implicitly rather than being separately corrected for, so it's part of the honestly-reported picture, not hidden.

**Q5. Why does the threshold need to be random per particle rather than one fixed calibrated number?**
A: Because engines genuinely differ in the health level at which they actually fail. Treating that as one hard number is exactly what left the earlier configuration's credible intervals with essentially no coverage — process noise alone was far too small a source of spread. Drawing each particle's own threshold from a fitted distribution turns that real, measurable engine-to-engine variability into honest posterior width instead of leaving it out.

---
## بخش ۷ — Parameter Estimation via PMMH

منبع: `revise/revision_tracker.md` — ORANGE #8 (سوال احتمالی دفاع) + BLUE مربوط به §3.4

**Q1. Did you investigate easier methods to estimate σ_v, instead of full PMMH?**
A: The other parameters all have closed-form or simple-regression estimators because they're computed directly from observed data (OLS/log-linear regression for k, residual statistics for the noise parameters). σ_v is different — it controls the spread of an unobserved latent state that never appears directly in the data, so there's no equivalent closed-form estimator; a proper likelihood-based estimate has to marginalise over every possible hidden trajectory, which is the intractable integral PMMH exists to handle. Since the bootstrap filter already computes an unbiased likelihood estimate as a byproduct of ordinary filtering, PMMH was the natural extension rather than a separate ad hoc method.

**Q2. Is the state-space model itself not also an input, alongside the data, when computing this posterior?**
A: Yes, explicitly — the posterior is written $p(\sigma_v \mid y_{1:T}, \mathcal{M})$ specifically to make that visible: PMMH takes the full state-space model (transition function, noise family, initial distribution) as a fixed given alongside the observations, not as something being estimated itself.

**Q3. If the posterior concentrates far from the hand-set value, couldn't that just mean the model is wrong, not that the hand-set value was wrong?**
A: Yes, both readings are live, and the thesis is explicit about that — a posterior far from the hand-set value is evidence either that the original hand-set choice needs revising, or that the state-space model itself is somewhat misspecified in a way that pushes the process-noise parameter into an unusual region to compensate. The σ_v≈0.66 vs hand-set 0.5 result on its own can't fully separate these two explanations; both are worth keeping in mind when interpreting it.

**Q4. I find the fixed vs. PMMH comparison confusing — is PMMH estimating both the process noise and the growth rate k?**
A: Yes — the derivation is written for σ_v alone to keep the notation manageable, but the runs actually used throughout the results are the two-dimensional version: PMMH jointly estimates the pair (k, σ_v) together, since both enter the likelihood only through the same particle-filter forward pass. So "PMMH configuration" means both parameters were estimated jointly from data, not just the noise scale.

**Q5. What exactly is the "regression-fitted estimate" used as a fallback for non-converged chains?**
A: It's the same growth-rate estimate from the degradation-dynamics fitting step — log-linear regression on the pooled (cycle, Health Index) training data, the same one used in the fixed-parameters configuration. When a particular engine's PMMH chain doesn't reach the convergence threshold, that engine simply falls back to this regression-based rate instead of contributing an unreliable PMMH estimate.

**Q6. Only 31/100 (FD001) and 8/100 (FD003) chains converged — isn't that a serious problem?**
A: It's reported directly rather than hidden or glossed over. The likelihood surface is genuinely harder to sample well with a single global random-walk step size applied across a diverse fleet, and FD003's second fault mode makes it noisier still, hence the even lower convergence there. Engines whose chains don't converge fall back to the regression rate rather than being dropped or given an unreliable estimate, so the pipeline stays honest about which engines its own MCMC can actually be trusted for.

---
## بخش ۸ — Similarity-Based Reference Library

منبع: `revise/revision_tracker.md` — ORANGE #9 (سوال احتمالی دفاع، دقیقاً رو همین انتخاب MMD) + BLUE #22, #23

**Q1. Why MMD rather than just fitting a slope to both Health Index windows and comparing those directly? Doesn't MMD discard the time-ordering / trend information?**
A: MMD compares the full empirical distribution of Health Index values over the shared window, not just a single summary slope, so it's sensitive to shape differences (curvature, spread) that two engines with a similar linear slope could still differ on. That said, it's a fair point that comparing values via a kernel doesn't directly use the *time-ordering* within the window the way a slope comparison would. MMD was adopted primarily because it's the exact method used by Cai et al. (2020), whose similarity-based prognostics approach this thesis adapts — a direct slope-comparison baseline wasn't tested head-to-head against it here, and would be a reasonable robustness check for future work.

**Q2. What do you do when the test engine has more observed cycles than a given library engine's own trajectory?**
A: The comparison window is truncated to whichever is shorter — the overlapping first min(t, T_u) cycles for that specific library engine — rather than extrapolating or padding the shorter trajectory. This matters especially on FD003, whose training engines have a much wider lifespan spread than FD001's.

**Q3. What does the kernel length scale ℓ do, and what value did you use?**
A: ℓ controls how quickly the similarity kernel decays as two Health Index values diverge — a small ℓ counts only very close values as similar, a large ℓ blurs the comparison. ℓ=10 HI units is used throughout, chosen to be about the same order as the fitted measurement-noise standard deviation (σ_w≈9.26), so that Health Index differences within roughly one noise standard deviation are treated as "compatible" and larger differences are progressively downweighted. Results weren't sensitive to ℓ across the range 5–20 tested during prototyping.

**Q4. What did you simplify relative to Cai et al.'s original method?**
A: Three deliberate simplifications for tractability: (1) matching uses a single scalar Health Index trajectory rather than Cai et al.'s full sensor feature matrix; (2) the 5 nearest library engines are simply averaged rather than passed through their formal Kernel Two-Sample Test accept/reject procedure at a tunable significance level, which would formally decide whether a test engine is similar enough to any library engine to trust a match at all; (3) only the growth rate is matched and swapped, since this thesis's exponential dynamics already reduce to one rate parameter, unlike Cai et al.'s Rao-Blackwellized filter that tracks a full dynamics-parameter vector online.

**Q5. Why fall back to pooled fleet dynamics before the first match, instead of using per-engine data from cycle 1?**
A: Because at that point in the stream, nothing specific to that particular test engine is known yet — there isn't enough of its own trajectory to make a meaningful similarity match. Falling back to the pooled dynamics is the safest default until the first genuine match can be made, after the 20-cycle warm-up.

---
## بخش ۹ — Results

منبع: `revise/revision_tracker.md` — ORANGE #1, #10, #11 + BLUE #24, #25, #26, #27, #29, #30, #31, #32, #33, #34 + RED #9, #10 (این‌ها الان تو متن نهایی فیکس شدن، ولی مفهوم پشتشون هنوز سوال خوبیه)

**Q1. (ORANGE #1) Do you have an explanation for why the linear regression baseline wins on point accuracy?**
A: Two things happen at once between the baseline and the filter, not one. First, an architecture difference — point regression vs. a full sequential Bayesian filter. Second, an *information* difference — the baseline consumes all ten sensors as separate features, while my filter first compresses them into a single scalar Health Index with fixed uniform weights, and only filters that scalar. So the filter enters the comparison having already thrown away information about how the ten channels differ from each other. This comparison doesn't cleanly isolate which of those two axes causes the gap — a fully fair test would either give the baseline the same scalar Health Index, or extend the filter to a vector state. Both are noted as future work. What the comparison does show cleanly is that more input features beats a compressed scalar on point accuracy, and that only the filter reports a credible interval at all.

**Q2. (ORANGE #10, BLUE variant) Could the baseline's win be an information effect rather than an architecture effect?**
A: Yes — see Q1. That's exactly the ambiguity: ten raw features vs. one compressed scalar, and architecture (regression vs. filter) both change at the same time. I flag this directly in the thesis rather than claiming the comparison isolates architecture cleanly.

**Q3. (ORANGE #11) FD001's coverage (0.68) is somewhat overconfident — how would you fix that?**
A: The direct route is moving from a single fleet-wide $\sigma_v$ (process-noise scale) to a genuinely per-engine $\sigma_v$ estimate, so interval width reflects each engine's own noise characteristics rather than a fleet average. Right now PMMH estimates one $\sigma_v$ shared across the whole fleet — the same pooling weakness that motivated the reference library for the growth rate $k$, just not yet applied to $\sigma_v$. That's flagged as the priority future-work item.

**Q4. (BLUE #24/#25) Why does FD003 have *longer* cycle lengths despite having two fault modes — didn't you expect the opposite?**
A: I expected the opposite too at first, but it comes down to how the datasets were generated (Saxena et al.), not the number of fault modes directly. The two fault modes in FD003 don't mean engines fail faster — they mean *different* engines fail through *different* degradation mechanisms, some of which simply take longer to reach failure than FD001's single mode. The longer and more variable run-to-failure horizon is the signature of that mechanism diversity, not of severity.

**Q5. (BLUE #26) I find the fixed-vs-PMMH setup confusing — is PMMH estimating both the process noise and the degradation rate?**
A: Yes, jointly. PMMH estimates both $\sigma_v$ (process-noise scale) and $k$ (degradation growth rate) together per engine, replacing the fixed-parameters configuration's hand-set $\sigma_v=0.5$ and pooled-regression $k$. Because they're estimated jointly, the RMSE/coverage improvement in the table reflects their *combined* effect — it doesn't tell you how much each one contributed on its own.

**Q6. (BLUE #31) Since PMMH estimates both k and σᵥ together, how do you know which one is actually responsible for the improvement?**
A: Honestly, I can't fully separate them from this ablation alone — that's a real limitation I state directly. What I can say is indirect evidence: the pooled-$k$ weakness is the more likely dominant cause, because the RMSE bias is largest specifically on short-lived engines, exactly where a mismatched $k$ compounds fastest — and Section 4.7 (the library) confirms this indirectly, because swapping *only* $k$ (leaving $\sigma_v$ untouched) delivers a further gain of similar magnitude on its own. A clean decomposition — estimate $\sigma_v$ alone, then $k$ alone — would separate the two contributions cleanly, but wasn't run within this thesis's time budget.

**Q7. (BLUE #29) What evidence do you have that the particle cloud stays diverse rather than collapsing?**
A: The effective sample size, $\widehat{\mathrm{ESS}}_t$ — it's a direct, standard diagnostic for exactly this. Pooled across the whole FD001 test fleet it averages 0.79 of the full particle count $N$ (SD 0.02), and adaptive resampling triggers whenever it dips below half of $N$, so the cloud is never running on a handful of duplicated particles.

**Q8. (BLUE #30) Aren't you mixing "short observed cycles" with "short-lived engines"? Engine 1 has only 31 observed cycles but a true RUL of 112 — a lifetime of 143.**
A: That's a fair and precise distinction, and I make it explicitly in the text. The $r=-0.58$ correlation mixes two effects that happen to point the same direction: some briefly-observed engines are early in an otherwise long life (weak signal, not much evidence yet), and some are genuinely short-lived (where the pooled rate under-estimates their true rate). Engine 1 is the first kind — truncated early, not short-lived. Both effects make short-observation-window engines harder, just for different underlying reasons.

**Q9. (RED #9-adjacent, still a good conceptual question) A well-calibrated interval should cover ~90% of engines — does a coverage far below that always mean the interval is too narrow?**
A: No, and that's worth being precise about. Coverage is a joint test of *location and width* — an interval of exactly the right width still covers nothing if it's centered in the wrong place. FD001's 0.68 could in principle reflect either an interval that's too narrow, or one that's well-sized but systematically mis-centered (which matches what I see — persistent over-prediction on short-observed engines, i.e. a location problem more than a pure width problem).

**Q10. (BLUE #33) The library's coverage (0.75 FD001) is still below the 0.90 nominal target — so is this really an improvement?**
A: Yes, on two counts. First, in absolute terms 0.75 is the closest any configuration in this thesis gets to nominal — closer than fixed parameters (0.49) or PMMH alone (0.68). Second, and more importantly, it gets there *without* trading away point accuracy — RMSE is simultaneously at its lowest ($44.3$). A fleet-wide rate could in principle be "recalibrated" just by widening the interval until it covers 90%, but that would come at the cost of a less informative (wider) interval; the library instead corrects *where* the RUL trajectory is centered, so both dimensions improve together rather than trading off against each other.

**Q11. (BLUE #34) Isn't the real cause of the residual weakness a shortage of observed evidence, rather than the pooled rate — as you argue elsewhere?**
A: Both are true and not in conflict — they compound. A shortage of evidence (few observed cycles) makes any degradation-rate estimate less reliable regardless of source; a pooled fleet-wide rate additionally biases the *center* of that estimate for engines whose true rate differs from the fleet average. The library specifically fixes the second problem (matching to similar engines rather than pooling), which is why it helps — but it can't manufacture observed cycles that were never collected, which is why the shortest-observation engines remain the hardest case even for the library, as I state directly in the results.

**Q12. Why does one-shot library matching beat periodic re-matching, if periodic can correct a bad initial match?**
A: One-shot does win empirically — on FD001 it beats periodic 59 engines to 30 (11 tied), and it also wins on FD003's aggregate RMSE. My interpretation: periodic's ability to recover from one bad match is real, but at the matching configuration used here, each re-match is also a fresh opportunity to drift onto a *less* well-matched rate than the one it already had — and empirically that risk outweighs the recovery benefit. I didn't assume this in advance; I tested both variants on the full fleet and let the results decide, which is exactly why one-shot became the adopted final configuration rather than a default choice.

---
## بخش ۱۰ — Limitations

منبع: `revise/revision_tracker.md` — BLUE #35, #36 (RED #11 مرتبط), #37, #38, #41 + RED #11 (تناقض ظاهری، الان حل‌شده تو متن نهایی)

**Q1. (RED #11, حل‌شده) Section 4 attributes the short-lived/long-lived tracking gap to the pooled degradation fit, but Section 4.7 attributes the same engines' residual difficulty to a shortage of observed evidence. Which is it?**
A: Both, at different stages — this isn't a contradiction, it's two different causes under two different configurations, and I say so explicitly. Health *tracking* itself is not systematically worse on short-lived engines — the filter tracks the observed Health Index well even on the shortest-observed engines. Where the gap shows up is downstream, in RUL prediction, and the cause depends on configuration: under fixed parameters and PMMH, it's the pooled/fleet-aggregate growth rate under-estimating the fastest-degrading engines' true rate; under the adopted library configuration, which already replaces the pooled rate with a per-engine matched one, the residual difficulty is a genuine shortage of observed evidence — the shortest engines simply don't have enough cycles for *any* matched rate to lock onto. Neither is a failure of the filter mechanics themselves.

**Q2. (BLUE #35) How do you know particle diversity is "consistently healthy" — what's the evidence?**
A: The effective sample size, pooled across the whole fleet and every cycle: $\widehat{\mathrm{ESS}}_t/N = 0.79\pm0.02$ for FD001. That's not cherry-picked from favorable engines — it's the fleet-wide mean — and adaptive resampling with $\tau=0.5$ guarantees the cloud never drops below half-populated regardless.

**Q3. (BLUE #37) What about PMMH failing to converge on 69/100 FD001 engines (and 92/100 FD003)? Isn't that itself a failure?**
A: It's a real limitation, but of PMMH specifically, not of the particle filter's core machinery. PMMH is a Metropolis-Hastings sampler sitting *on top of* the filter — its chain-mixing difficulty concerns the curvature of the parameter posterior surface, not the correctness of the particle filter's likelihood evaluation, which is the quantity PMMH consumes and which behaved as expected on every single engine. Where a chain didn't converge, a regression-fitted rate is used as fallback, so the pipeline doesn't silently fail — but I agree the convergence rate itself (31/100, 8/100) is a real weakness worth improving, and it's named directly as a limitation and as the top future-work priority.

**Q4. (BLUE #38) In the end, isn't a fleet-wide aggregate still being used, even though the estimates came from individual engines?**
A: Yes, for $\sigma_v$ specifically — that's exactly right, and it's why FD001/FD003 calibration isn't fully solved yet. PMMH estimates $\sigma_v$ per engine, but the *converged* per-engine posteriors are then aggregated into one fleet-wide $\sigma_v$ applied to every test engine — so it's an improvement over a hand-set value, but it's still one number for the whole fleet. The growth rate $k$ is the one parameter that gets a genuinely per-engine treatment, via the reference library. Moving $\sigma_v$ to the same per-engine treatment is the direct next step, named in Future Work.

**Q5. (BLUE #41) You say the library uses "a single matched growth-rate parameter... rather than tracked jointly with the state as an online-updated dynamics component" — but couldn't a single parameter also be tracked online?**
A: Yes, and I make that distinction explicit rather than conflating it: the limitation isn't "one parameter vs. many," it's whether the matched rate is refined by every incoming observation or held constant between matches. A single parameter could equally well be tracked online — e.g. by augmenting the state with $k$ and letting the filter update both jointly in a Rao-Blackwellised or self-tuning scheme. That's not what this thesis does (the rate is matched once, or refreshed periodically, then held fixed in between) — it's a real simplification, just not the one a first read might assume.

**Q6. Why not just fix all of this — better PMMH convergence, per-engine σᵥ — within this thesis?**
A: Time budget, stated honestly. A per-engine adaptive step size for PMMH, or a formal decomposition of $k$ vs. $\sigma_v$'s separate contributions, or extending to FD002/FD004 with operating-condition normalization — all of these are real, identified next steps, not things I ran out of ideas for. I'd rather report the current, honest numbers — including where they fall short of nominal — than claim a level of calibration the current pipeline doesn't actually reach.

---
