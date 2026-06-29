# Thesis Learning Progress
**Learning period:** May–June 2026 (4 weeks)  
**Primary resource:** Chopin & Papaspiliopoulos, *An Introduction to Sequential Monte Carlo* (2020, Springer)  
**Last updated:** June 28, 2026

---

## What Was Studied

### Week 1 — Bayesian Foundations
**Topics covered:**
- Probability fundamentals and Bayes' Theorem
- Prior distributions, likelihood functions, posterior distributions
- Why exact Bayesian inference is intractable in most real problems (curse of dimensionality, non-conjugate models)
- Monte Carlo basics — sampling from distributions to approximate expectations
- Importance sampling — how to sample from a difficult distribution using a simpler proposal
- Sequential Monte Carlo — the idea of propagating particles through time

**Key understanding achieved:** WHY particle filters exist — they solve the problem that classical Bayesian inference cannot handle dynamically evolving hidden states with nonlinear/non-Gaussian models.

---

### Week 2–3 — Particle Filter Mechanics
**Topics covered:**
- Sequential Importance Sampling (SIS) — the basic particle filter algorithm
- Weight collapse / particle degeneracy — why SIS breaks down over time
- Resampling algorithms: multinomial, stratified, systematic
- The SIR (Sequential Importance Resampling) filter
- Adaptive resampling based on Effective Sample Size (ESS)
- Particle degeneracy stages — visual understanding of collapse
- ESS as a diagnostic: $\text{ESS} = 1 / \sum_i (w_i^{(t)})^2$

**Key understanding achieved:** The full SIR cycle — predict → weight → resample — and why each step is necessary. Understanding of ESS as a health metric for the particle cloud.

**Implementations built:**
- SIS filter (manual implementation)
- SIR filter with multinomial resampling
- Adaptive resampling with ESS threshold
- Particle degeneracy visualizer
- Animations of particle evolution over time

---

### Week 4 — Application to Digital Twins & Predictive Maintenance
**Topics covered:**
- Bootstrap particle filter (special case of SIR where proposal = transition prior)
- State space model for equipment degradation: `[health, degradation_rate, hidden_fault]`
- Transition model: how each state component evolves over time steps
- Observation model: how sensor readings relate to hidden health state
- RUL extraction: propagating particles forward past current time to estimate failure
- RUL as a distribution — median and credible intervals, not a point estimate
- Benchmarking: particle count $N$ vs. accuracy (RMSE) and speed
- Finding: variance decreases as $1/\sqrt{N}$ — diminishing returns for large $N$

**Parameter estimation explored:**
- Brief exposure to PMMH (Particle Marginal Metropolis-Hastings) for parameter estimation
- Brief exposure to Gibbs sampling within SMC
- These were tried before fully understanding PF — results were scattered and mixed. Will be revisited properly on thesis Day 8.

**Implementations built:**
- Bootstrap particle filter on 3-component equipment state model
- RUL extraction with uncertainty quantification
- Benchmarking experiment ($N \in \{50, 100, 500, 1000\}$)
- Week 4 Performance Report with recommended $N$

**Key files from Week 4:**
- `Week_4_Day1_Results.png`, `Week_4_Day2_RUL_Distribution.png`, `Week_4_Day3_Benchmarking.png`
- `Week_4_Performance_Report.md`
- `quick_guide_4_summary.md`, `quick_guide_day2_week4_rul.md`

---

## What Is Solid

| Concept | Confidence |
|---------|-----------|
| Bayes' Theorem and posterior inference | ★★★★★ |
| Why particle filters are needed | ★★★★★ |
| SIR filter algorithm (predict → weight → resample) | ★★★★★ |
| Effective Sample Size (ESS) | ★★★★☆ |
| Resampling strategies | ★★★★☆ |
| Bootstrap particle filter | ★★★★☆ |
| RUL extraction from posterior | ★★★★☆ |
| Particle degeneracy / weight collapse | ★★★★★ |
| Benchmarking $N$ vs. accuracy | ★★★★☆ |

---

## What Needs Attention Before/During Thesis

| Concept | Gap | Action |
|---------|-----|--------|
| PMMH for parameter estimation | Tried it early but not deeply understood | Dedicated Day 8 teaching session |
| State space model design for C-MAPSS specifically | Previous model was synthetic; C-MAPSS has 21 sensors | Need to design carefully on Day 2 |
| Operating condition handling (FD002, FD004) | Multiple operating conditions add complexity | Address after FD001 baseline works |
| Kalman filter as baseline | Used in Week 4.5 but not rigorous | Implement cleanly on Day 9 |
| Scientific literature on PF for RUL | Not yet surveyed systematically | Background chapter Day 6, literature day Day 8 |
| LaTeX for thesis writing | No experience | Learn from Day 1 with Overleaf |

---

## Key Formulas to Remember

**Bayes' Theorem:**
$$p(\theta | y) = \frac{p(y | \theta) \, p(\theta)}{p(y)}$$

**State space model:**
$$X_t = f(X_{t-1}) + \text{process noise}$$
$$Y_t = g(X_t) + \text{observation noise}$$

**Importance weight update:**
$$w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \frac{p(Y_t | X_t^{(i)}) \, p(X_t^{(i)} | X_{t-1}^{(i)})}{q(X_t^{(i)} | X_{t-1}^{(i)}, Y_t)}$$

**Bootstrap PF (proposal = prior):**
$$w_t^{(i)} \propto p(Y_t | X_t^{(i)})$$

**Effective Sample Size:**
$$\text{ESS} = \frac{1}{\sum_{i=1}^{N} (w_t^{(i)})^2}$$

**RMSE for RUL evaluation:**
$$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{j=1}^{M} (\hat{\text{RUL}}_j - \text{RUL}_j)^2}$$

---

## Previous Code Notes (Learning Project)
The learning project (`/Learning Particle Filter/`) contains working implementations but they should NOT be directly copied into the thesis package. Key issues:
1. **Single-model-for-all-engines mistake** — the old code tried to fit one particle filter to all engines simultaneously. This is wrong. Each engine runs independently.
2. **Scattered experiments** — multiple versions of the same idea with no consistent structure.
3. **No database** — results were saved as loose `.png` files and `.md` reports.

The thesis package (`particle_twin/`) is a clean rebuild with the correct architecture. Use the old code only as reference for mathematical logic, not structure.

---

## Recommended Chopin Chapters for Thesis Reference

| Chapter | Topic | Relevance |
|---------|-------|-----------|
| Ch. 2–3 | Monte Carlo, importance sampling | Background §2.1–2.2 |
| Ch. 7 | Sequential Monte Carlo basics | Background §2.3 |
| Ch. 10 | Bootstrap filter, state space models | Methodology §3.1–3.2 |
| Ch. 12 | Particle smoothing, forward-backward | Methodology §3.3 (optional) |
| Ch. 17–18 | PMMH, parameter estimation | Methodology §3.4 |
