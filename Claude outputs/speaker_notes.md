# متن نوت‌های سخنرانی (بخش Notes هر اسلاید)

---

## بخش ۱ — Introduction

### نوت اسلاید ۱

"Every industrial system degrades with use — a turbofan engine wears a little more with every flight cycle. Traditionally there are two ways to handle this: run the part until it fails, or replace it on a fixed schedule. Both are bad — run-to-failure is dangerous and expensive when it happens in service, and calendar-based replacement throws away usable life. Predictive maintenance sits in between: we use sensor data to estimate the Remaining Useful Life, or RUL — how many cycles are left — and schedule maintenance around the actual condition of the machine. But most existing methods for RUL just give you one number, like '47 cycles left,' with no sense of how confident that number is. And that's not really enough to make a safe decision — is 50 cycles 'plus or minus 3' or 'plus or minus 40'? Same number, very different action."

### نوت اسلاید ۲

"So the objective of this thesis is to build a probabilistic digital twin of engine degradation, using a particle filter as the inference engine — that's a Bayesian sequential Monte Carlo method. The hidden state we track is a scalar Health Index, h_t, between 0 and 1, where 1 is healthy and 0 is failed. At every cycle, instead of tracking one number, the filter keeps a full posterior distribution over the health state. And when we need a RUL estimate, we don't get one number either — we get a distribution: a median and a 90% credible interval. I evaluate this on the NASA C-MAPSS benchmark, on two sub-datasets, FD001 and FD003, and — importantly — I check not just point accuracy, but whether those credible intervals are actually calibrated, meaning they contain the true RUL as often as they claim to."

---
## بخش ۲ — The Data

### نوت اسلاید ۳

"Before I get into the methodology, let me quickly show what the data actually looks like. This is NASA's C-MAPSS turbofan dataset, simulated run-to-failure data from the PHM08 challenge. I use two sub-datasets: FD001, which has one fault mode, and FD003, which has two — an extra fan degradation fault on top of the same compressor fault. Each row is a snapshot of one engine at one cycle: an engine ID, the cycle number, three operating settings, and 21 raw sensor readings.

Here's the important part: in the training set, every engine runs all the way to failure, so I know the exact RUL at every single cycle. But the test set is different — each test engine's sequence is cut off at some point before it actually fails. That's not something I did — it's the design of the benchmark itself, meant to mimic a real deployment, where you never get to observe an asset all the way to failure before you have to make a prediction. The true remaining life at that cutoff point is given separately, only for evaluation, and you can see in this table it's spread all over the place — from 6 cycles left up to 145 — so it's not just 'stopped right before failure,' the cutoffs are genuinely all over the range."

---
## بخش ۳ — Data Analysis: Sensor Selection

### نوت اسلاید ۴

"Since this is a statistical analysis thesis, I want to show briefly how the sensors were actually chosen, not just used as given. We start from 21 raw sensors. Seven of them are constant across the entire fleet — zero variance, no signal — so those are dropped immediately, leaving 14. Each of those 14 is scored on five statistical criteria: a Mann-Kendall test for a significant monotonic trend, a monotonicity score for how consistently it drifts within one engine's life, a trendability score for how consistently it drifts across different engines, mutual information with RUL to catch nonlinear relationships a simple correlation would miss, and the variance inflation factor to flag sensors that are basically redundant with others. I combine these five into one composite score, and I actually compute it across all four C-MAPSS sub-datasets, not just the two I use later, so the selection itself isn't overfit to FD001 and FD003. Sensors scoring below 0.40 get dropped, and two more get removed for being too collinear with sensors already selected. That leaves the 10 sensors that feed into the Health Index you'll see in the next section."

---
## بخش ۴ — Methodology: State Space Model

### نوت اسلاید ۵ (Health Index)

"Now let's get into the methodology. The hidden state I track is a scalar Health Index, x_t, on a 0-to-100 scale — 100 is healthy, 0 is failed. I build it from the 10 selected sensors: each one is min-max normalised to 0-100 using the training fleet's range, oriented so that all of them decrease as the engine degrades — some raw sensors actually increase with degradation, so those get flipped — and then averaged with equal weights, since no sensor was judged more diagnostic than another going in. I also tried a PCA-based version, projecting onto the first principal component, but the weighted average ended up fitting better once paired with the degradation model I'll show next, so that's what the filter actually observes."

### نوت اسلاید ۶ (Dynamics + Observation + Initial Distribution — فقط Dynamics/Observation رو اسلاید هست، بقیه شفاهی)

"Two more pieces complete the state-space model. First, the degradation dynamics — how the health index evolves from one cycle to the next. I use an exponential decay: the health index moves toward failure a little faster the more damaged the engine already is. I didn't pick this shape by just fitting and comparing R-squared values — that comparison was actually inconclusive because of how I pooled engines with very different lifetimes. I picked exponential because it's what NASA's own simulator uses internally to generate this data, and it's also the standard shape for wear-out failures in the reliability literature. The rate parameter k is fitted by regression on the training data; the process noise, sigma_v, I leave free for now — I'll refine that later with PMMH.

Second, the observation model — how the noisy sensor reading relates to the true health index. I fit the noise distribution on the residuals from training data. A Gaussian fits reasonably well, with a standard deviation around 9.26. A skew-normal technically fits a bit better by AIC, but I kept the Gaussian as the default because it gives a closed-form likelihood and keeps the fitting simple — the skew-normal version is reported later just as a robustness check.

One last piece, quickly, not on the slide: I initialise the filter at cycle zero with a small spread rather than a single fixed value — a Gaussian centered at 1 with a standard deviation of 0.05 — because engines don't all start in exactly the same condition. Everything else in the model is fitted once from training data and held fixed; the only parameter still left free at this point is the process noise, sigma_v, which is exactly what I tune next with Particle Marginal Metropolis-Hastings."

---
## بخش ۵ — Particle Filter Algorithm

### نوت اسلاید ۸

"Now the actual filtering algorithm. I use a bootstrap particle filter — the simplest variant, where you propose new particles directly from the transition model, so the particle weight collapses to just the observation likelihood, no custom proposal density needed. At every cycle, for every particle, I do two things: predict — draw a new state from the transition model — and weight — multiply in the observation likelihood. I do this in log-space, because over a full engine trajectory of up to 350 cycles, the likelihoods in linear space would just underflow to zero.

Without resampling, the particle weights degenerate fast — the effective sample size collapses to about 1 within 5 to 10 cycles, meaning almost all particles become numerically irrelevant. So I use adaptive resampling: I only resample when the effective sample size drops below half of N, using systematic resampling. That's a standard trade-off — resampling every cycle wastes diversity and computation, waiting too long leaves too few effective particles to fix. And to be clear, each of the 100 engines is filtered completely independently — they share the same fitted dynamics model, but each engine gets its own posterior from its own sensor readings."

---
## بخش ۶ — RUL Extraction from the Particle Posterior

### نوت اسلاید ۹ (Forward Simulation)

"Once I've filtered up to the last observed cycle, I still need to turn that into a Remaining Useful Life estimate. I do this by forward simulation: I take every particle and keep propagating it forward under the exact same degradation model, drawing new random noise at every future step, until it crosses a failure threshold. The number of steps it took is one sample of RUL. I do this for every particle, so I end up with a full empirical distribution of RUL — not one number, a median and a 90% credible interval. There's no new sensor data involved here — it's a pure extrapolation from the model, past the point I actually observed anything."

### نوت اسلاید ۱۰ (Calibrating the Failure Threshold)

"Now, here's the part that matters most in this whole methodology. Where do I stop the forward simulation — what counts as 'failed'? The naive answer is health equals zero. That's wrong, for three reasons, and I measured all three directly from the training data rather than assuming them.

First, real engines never actually reach zero — averaging the last five cycles of all 100 training engines, they fail around a health index of 0.256, not 0. Simulating all the way down to zero adds on the order of a hundred spurious extra cycles to every prediction.

Second, the threshold has to be measured in the same space the forward simulation actually lives in — the filter's posterior — not the raw observed sensor data. The posterior systematically sits a bit above the raw observations, so if I calibrated against the raw observation-space number instead, I'd be asking every particle to fall further than it actually needs to — double-counting that gap.

Third — and this is the big one — the threshold itself isn't one fixed number. Real engines genuinely differ in the health level at which they actually fail. So instead of one hard threshold, I draw a separate threshold per particle, from a distribution fitted on training data.

And I want to say this clearly, because it's really the central result of the methodology: this threshold calibration — getting these three things right — is the single change that makes the extracted RUL both accurate and calibrated. It's the biggest single reason the credible interval coverage goes from essentially zero to something that actually holds up empirically, which you'll see in the results."

---
## بخش ۷ — Parameter Estimation via PMMH

### نوت اسلاید ۱۱ (PMMH: The Idea)

"Every parameter so far was fitted once and held fixed — except one: the process noise, sigma_v. Unlike the others, it has no closed-form estimator, because it controls the spread of a hidden state you never directly observe. So instead of hand-setting it, I want to treat it as a random variable with its own posterior, given the data. The problem is that posterior has no closed form — computing it exactly would mean marginalising over every possible hidden trajectory, the same intractable integral that motivated using a particle filter in the first place, just one level up, now over a parameter instead of a state.

The trick is called Pseudo-Marginal Metropolis-Hastings. The key result is that you don't need the exact likelihood inside a Metropolis-Hastings step — a noisy but unbiased estimate of it works just as well, and the resulting chain still converges to exactly the correct posterior, not an approximation. And here's the nice part: the bootstrap filter I already built computes exactly this likelihood estimate for free, as a byproduct of ordinary filtering. So PMMH isn't new machinery — it's just wrapping the filter I already have inside a Metropolis-Hastings loop."

### نوت اسلاید ۱۲ (Fleet-Wide Results)

"I ran this per engine across the whole fleet, jointly estimating both the growth rate k and sigma_v together, with 3200 particles and 1000 iterations per chain. A chain only counts as converged if its acceptance rate reaches 0.10; where it doesn't, that engine falls back to the regression-fitted rate from earlier, so every engine still gets a usable estimate either way.

Convergence wasn't universal, and I report that honestly rather than hiding it: only 31 of 100 chains converged on FD001, and just 8 of 100 on FD003, where the second fault mode makes the likelihood surface noisier and harder to sample. Aggregating what did converge gives a fleet-wide process noise of about 0.66 on both datasets — notably higher than the hand-set default of 0.5, which confirms, now with real data, that the original hand-set value understated the true process noise. The growth rate comes out around 0.0039 on FD001 and 0.0018 on FD003 — the lower FD003 rate matches its longer, more variable time to failure."

---
## بخش ۸ — Similarity-Based Reference Library

### نوت اسلاید ۱۳ (Per-Engine Library + Matching)

"This next part I actually think is one of the strongest pieces of this thesis. Remember the degradation rate I fitted earlier — that was one single rate, pooled across the whole fleet. That's a population average, and it's biased: it gets pulled simultaneously by short-lived engines degrading fast and long-lived engines degrading slowly, so it doesn't represent any real engine.

The fix is conceptually simple: instead of one pooled rate, I build a reference library — I fit a separate growth rate for every single training engine, individually, using its own trajectory. On FD001 the per-engine average comes out to about 0.0037, noticeably higher than the pooled fit's 0.0028 — which is direct evidence the pooling really was biasing things downward.

Then, at test time, for the engine I'm actually filtering, I compare its observed Health Index so far against every engine in that library, using a similarity measure called Maximum Mean Discrepancy — it's a kernel-based way of comparing two sets of values. I take the 5 most similar library engines and average their rates, and that becomes this specific engine's own matched growth rate, replacing the generic pooled one.

And I think this is really the point where the model becomes a genuine digital twin, not just a fleet-average model — each engine, going forward, is compared against and matched to engines that actually behaved like it, not tracked against one generic curve for the whole fleet."

### نوت اسلاید ۱۴ (Streaming Deployment)

"One more practical piece: a real digital twin doesn't get an engine's whole life story handed to it up front — the Health Index arrives one cycle at a time, live. So I consider two ways to actually deploy this matching in that streaming setting. One-shot: wait for a 20-cycle warm-up, match once, and keep that matched rate fixed for the rest of the engine's observed life — this is cheap, and it's the version closest to the original method this is based on. Periodic: re-match every 20 cycles as more data comes in, so both the matched rate and which engines it's matched to can update over time. Before the very first match in either case, the filter just falls back to the pooled fleet dynamics, since nothing more specific is known yet. Which of the two actually performs better isn't something I assumed — I test both on the full test fleet and let the results decide, which comes up in the next section."

---
## بخش ۹ — Results

### نوت اسلاید ۱۵ (Health State Tracking)

"Before I get to the headline RUL numbers, I want to spend one slide checking a prerequisite: does the filter actually track health well in the first place? Across the whole FD001 test fleet — over thirteen thousand cycles — the filtered Health Index stays within about a tenth of a unit of the observed, sensor-derived Health Index. And the particle cloud itself stays healthy: the effective sample size averages 79% of the full particle count, so it's never collapsing onto a handful of duplicates.

Even on the worst-tracked engine — one that's only 31 cycles into its 143-cycle life when the data cuts off — the health tracking itself is still good. Its big error shows up later, in the RUL projection, not here. The one real pattern I found is that tracking error correlates with how many cycles are observed — engines seen only briefly are harder, both because the signal is weak early on and because the pooled degradation rate under-serves them. That's exactly the weakness the rest of this section addresses."

### نوت اسلاید ۱۶ (RUL Prediction: Fixed vs. PMMH)

"Now the main question: how accurate is the RUL estimate, and how well calibrated is its uncertainty? Moving from a hand-set process noise to PMMH's data-driven estimate roughly halves the error on both datasets. But calibration tells two different stories. On FD001, coverage climbs from 0.49 to 0.68 — still under the 90% target, so the intervals are a bit overconfident, but a real improvement over the essentially zero coverage of the earlier, uncalibrated configuration. On FD003, both configurations already cover 100% of engines — the opposite problem, intervals that are wide enough to always contain the truth but less informative than they could be, because that dataset's second fault mode injects more noise.

There's still a residual gap on the shortest-observed engines — PMMH improves engine 1's forecast from 336 cycles down to 226, but the true value is 112, so it's still over-predicting. That gap is exactly what the next piece — the reference library — goes after."

### نوت اسلاید ۱۷ (Similarity Library: Final Results)

"This is where everything comes together. Going from fixed parameters, to PMMH, to the one-shot library, the improvement is monotonic on both datasets — RMSE keeps falling and coverage keeps climbing toward the 90% target, at the same time. On FD001 the library configuration reaches 0.75 coverage at the lowest RMSE of any configuration — that's the best-calibrated *and* the most accurate result in the whole thesis.

Between the two ways I considered deploying the library in a streaming setting — matching once after a warm-up, or re-matching periodically — one-shot wins on both fleets. Re-matching sounds like it should help by recovering from a bad initial match, but in practice each re-match is also a chance to drift onto a less well-matched rate, and that risk outweighs the benefit here. So one-shot library matching is what I adopted as this thesis's final configuration.

The honest remaining limit is the same one from two slides ago: engines with very short observation windows — where there just isn't much evidence yet — are still the hardest case for every configuration, not because of which degradation rate you pick, but because there isn't enough data yet to pick well."

### نوت اسلاید ۱۸ (Comparison with Baseline)

"And this last slide is really the summary of the whole argument. I compared the full pipeline against a plain linear regression baseline on the same ten sensors — and I want to be upfront about this: the baseline wins on point accuracy, on both datasets. A simple regression, given the same sensors, produces a sharper single-number forecast than my particle filter does. I'm not hiding that.

But the baseline can only ever give you one number. It can't tell you how confident to be in that number — the question 'does the 90% interval cover the truth' isn't even something you can ask of it. My filter gives up some of that point-accuracy edge, and in exchange delivers a calibrated interval — 75% coverage on FD001, 99% on FD003 — that a bare point estimate structurally cannot provide. And for a maintenance decision, where acting too late is much more costly than acting a bit too early, knowing *how much* to trust a single number is arguably more valuable than shaving a few cycles off the point estimate. That's the trade this thesis is making, and I think it's the right one for this problem."

---
## بخش ۱۰ — Limitations

### نوت اسلاید ۱۹ (Residual Miscalibration)

"I want to be honest about what's still not perfect here. Two fixes already close most of the calibration gap — PMMH fixed the fixed-parameters overconfidence, and the library fixed the systematic over-prediction. What's left is a scale problem that goes in opposite directions on the two datasets.

On FD001, coverage is 0.75 — still under the 90% target. That happens because a single matched rate per engine still can't capture an engine whose true rate simply has no close analogue anywhere in the training library. On FD003, coverage is 0.99 — over-conservative. And I want to be precise about why, because it's not that the noise estimate is bigger — sigma-v converges to about 0.66 on both datasets. It's two other things compounding: FD003 engines run longer, so the same per-cycle noise accumulates into a wider spread over more cycles; and matching under two coexisting fault modes is just noisier, because a test engine following one fault mode can end up matched to training engines that were following the other.

Underneath both of these is one more honest number: the PMMH chain only actually converged on 31 of 100 FD001 engines, and 8 of 100 on FD003 — the rest fall back to a regression-fitted estimate. So the fleet-wide noise scale each dataset uses is built from an uneven, and on FD003 quite small, set of converged posteriors. Both calibration problems point to the same fix: better per-engine PMMH convergence, and moving to a genuinely per-engine noise estimate instead of one shared across the fleet."

### نوت اسلاید ۲۰ (Other Limitations)

"Three more limitations, stated plainly rather than left implicit. First, the similarity matching I described earlier simplifies the method it's based on in three specific ways — I don't claim any of those simplifications are harmless in general, just that they were tractable within this thesis's scope, and the fleet-wide result shows they're good enough to deliver a real improvement, not that they're the best possible version of this idea.

Second, the same PMMH convergence issue from the last slide — a single global step size for the random-walk proposal isn't equally well suited to every engine's likelihood surface.

And third, a real scope limitation: this pipeline has no mechanism for operating-condition-aware normalization, so as it stands it doesn't extend to FD002 or FD004, which involve multiple operating conditions. Everything in this thesis is validated on the single-operating-condition case."

---
