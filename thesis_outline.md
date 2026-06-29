# Thesis Outline
**Title:** A Bayesian Approach to Digital Twins with Applications to Predictive Maintenance  
**Student:** Sajjad Ayashmand  
**Program:** MSc Statistical Data Analysis — Ghent University  
**Supervisors:** Prof. Koen De Turck (promoter), Prof. Dieter Fiems (co-promoter)

---

## Abstract *(written last)*
A concise summary of the research question, methodology, key result, and practical significance.  
Target: ~250 words. Written after the full draft is complete.

---

## Chapter 1 — Introduction

### §1.1 Problem Context: Predictive Maintenance and Digital Twins
Modern industrial systems, such as aircraft engines, operate under continuous stress and require timely maintenance to avoid costly failures. This section introduces the predictive maintenance problem: using sensor data to estimate when a component will fail, so maintenance can be scheduled before failure occurs. It motivates the concept of a *digital twin* — a computational model that mirrors a physical asset in real time — and argues that such a twin must represent *uncertainty*, not just a best guess.

### §1.2 The Limitation of Classical Approaches
Existing approaches to remaining useful life (RUL) estimation — including regression-based and neural network methods — produce point estimates that discard uncertainty. This section identifies this as a fundamental limitation: a maintenance engineer who receives "RUL = 50 cycles" with no confidence interval cannot make a principled decision about when to intervene. The section frames uncertainty quantification as the central scientific challenge this thesis addresses.

### §1.3 Research Objective
This thesis proposes and evaluates a *particle filter* (Sequential Monte Carlo method) as a principled Bayesian digital twin for turbofan engine degradation. The research objective is stated precisely: to track the latent health state of individual engines and estimate the distribution of RUL from sensor observations, using the NASA C-MAPSS benchmark dataset.

### §1.4 Thesis Outline
A one-paragraph map of the remaining chapters, telling the reader what each chapter contributes to answering the research question.

---

## Chapter 2 — Background

### §2.1 Bayesian Inference
This section introduces the Bayesian framework — prior belief, likelihood, and posterior — as the mathematical foundation for updating knowledge in light of data. It establishes the notation used throughout the thesis and explains why Bayesian inference is uniquely suited to the sequential, uncertainty-bearing nature of the degradation tracking problem.

### §2.2 State Space Models and Sequential Bayesian Filtering
Degradation processes unfold over time in a hidden state that must be inferred from noisy observations. This section formalises the state space model (transition equation + observation equation) and derives the recursive Bayesian filter equations: prediction via the Chapman–Kolmogorov equation and update via Bayes' theorem. It establishes why the exact posterior is intractable for nonlinear/non-Gaussian systems, motivating the approximate methods in §2.3.

### §2.3 Particle Filters — Sequential Monte Carlo
Particle filters approximate the filtering distribution using a weighted set of samples (particles), each representing a plausible state trajectory. This section covers the core SIR (Sequential Importance Resampling) algorithm: importance sampling, weight update, and systematic resampling. It discusses degeneracy, effective sample size (ESS), and the bootstrap filter (prior as proposal) used in this thesis. Key convergence properties are stated without proof; references are given.

### §2.4 Digital Twins in Predictive Maintenance
This section reviews the digital twin concept in the engineering literature and distinguishes *data-driven*, *physics-based*, and *hybrid* approaches. It argues that a Bayesian filter constitutes a principled probabilistic digital twin: the posterior distribution over the hidden state plays the role of the twin's internal model. The section connects this framing to the thesis application.

### §2.5 Related Work — RUL Estimation for Turbofan Engines
A structured review of published methods applied to the C-MAPSS benchmark: (a) classical regression and time-series methods, (b) deep learning approaches (LSTM, CNN), (c) prior Bayesian and particle filter approaches. The review identifies the gap this thesis fills: most existing work ignores uncertainty quantification, and those that address it do not provide a clean particle-filter-based digital twin framework with systematic parameter estimation.

---

## Chapter 3 — Methodology

### §3.1 State Space Model for Turbofan Degradation
This section defines the mathematical model at the heart of the thesis. The hidden state $h_t \in [0,1]$ is a scalar *health index*, declining monotonically from 1 (healthy) toward 0 (failure). The transition equation is a Gaussian random walk with drift: $h_t = h_{t-1} - \delta + v_t$, $v_t \sim \mathcal{N}(0, \sigma_v^2)$. The observation equation links $h_t$ to the informative C-MAPSS sensor readings via a linear map estimated from training data. The choice of model, its assumptions, and its limitations are discussed explicitly.

### §3.2 Bootstrap Particle Filter Algorithm
This section details the exact algorithm implemented in `particle_twin/filters/bootstrap.py`. Starting from $N$ particles sampled from the prior, each cycle involves: (i) propagation through the transition model, (ii) weight update using the observation likelihood, and (iii) systematic resampling when ESS falls below $N/2$. Pseudocode and complexity analysis ($\mathcal{O}(N)$ per step) are provided. The per-engine independence assumption — each engine runs its own independent filter — is justified.

### §3.3 RUL Extraction from the Particle Posterior
Given the particle cloud $\{h_T^{(i)}, w_T^{(i)}\}_{i=1}^N$ at the last observed cycle $T$, this section describes how to propagate particles forward until each crosses the failure threshold. The resulting empirical distribution over time-to-failure constitutes the RUL posterior. Point estimates (median) and credible intervals (5th–95th percentile) are extracted and stored. The section explains why the distribution, not just the median, is the correct output.

### §3.4 Parameter Estimation via PMMH
Model parameters ($\delta, \sigma_v, \sigma_w$) are initially set by hand but estimated from data using Particle Marginal Metropolis–Hastings (PMMH). This section outlines the PMMH algorithm: embedding the particle filter inside an MCMC sampler that targets the marginal likelihood of the parameters. The implementation in `particle_twin/inference/pmmh.py` is described. This section is more advanced and is developed on Day 8 after the filter is validated.

---

## Chapter 4 — Experiments and Results

### §4.1 Dataset Description: NASA C-MAPSS
This section describes the C-MAPSS simulation benchmark: four sub-datasets (FD001–FD004) varying by number of operating conditions and fault modes. The thesis focuses on FD001 (single operating condition, single fault) for primary evaluation, with FD003 (multiple operating conditions) as a robustness test. The pre-processing pipeline — sensor selection, normalisation, train/test split handling — is documented.

### §4.2 Evaluation Metrics
Results are evaluated using: RMSE (root mean squared error of the final RUL estimate); MAPE (mean absolute percentage error); the asymmetric scoring function from the C-MAPSS literature (penalises late predictions more than early); and ESS (to monitor particle degeneracy). Reporting uncertainty across engines (not just fleet-mean RMSE) is a deliberate choice aligned with the thesis's uncertainty-quantification objective.

### §4.3 Experimental Setup
This section describes the experimental protocol: number of particles ($N \in \{500, 1000, 2000\}$), parameter values (hand-set baseline vs. PMMH-estimated), random seeds, and the structure of the results database. Reproducibility: all runs are stored in `results.db` and the code is submitted alongside the thesis.

### §4.4 Results — Health State Estimation
Particle filter health tracking is shown for representative engines from FD001. Figures display the median health trajectory with 90% credible band overlaid on the true (training) trajectory. The section discusses how well the filter tracks degradation onset and late-stage rapid decline. Engines where the filter struggles are identified and discussed.

### §4.5 Results — RUL Prediction
The RUL posterior at the final observed cycle is compared against ground truth for all FD001 test engines. Fleet-level RMSE is reported with its distribution (box plot) across engines. The 90% credible interval coverage is computed to assess calibration. FD003 results are presented separately to assess generalisation.

### §4.6 Results — Comparison with Baseline
The particle filter is compared against a simple linear regression baseline trained on the same sensor data. The comparison uses RMSE and the asymmetric scoring function. The section argues that the particle filter's primary advantage is not necessarily a lower RMSE but its ability to quantify uncertainty — something the baseline cannot do.

---

## Chapter 5 — Discussion and Conclusions

### §5.1 Discussion
This section interprets the results in terms of the original research objective. It discusses: what the particle filter does well (online state tracking, uncertainty quantification); where it struggles (engines with atypical degradation profiles); the sensitivity to parameter choices; and the role of PMMH in improving performance. Limitations of the study — simplified observation model, single sensor-to-health mapping — are acknowledged. The section connects back to the digital twin framing: the particle filter genuinely constitutes a probabilistic digital twin.

### §5.2 Conclusions
The research question is answered directly: a bootstrap particle filter applied per-engine on C-MAPSS data provides principled, calibrated RUL distributions with competitive point-estimate accuracy. The uncertainty quantification is a concrete scientific contribution absent from the majority of published C-MAPSS work. Practical implications for maintenance engineers are stated in accessible terms.

### §5.3 Future Work
Three concrete directions: (1) more expressive observation models (non-linear, data-learned); (2) extension to FD002/FD004 with multiple operating conditions using a cluster-conditional model; (3) real-time adaptation of parameters via online PMMH or ensemble methods.

---

## References
APA style, managed in `Writing_Templates/Templates/Thesis_bib.bib`.  
Key entries to add from day 1: Chopin & Papaspiliopoulos (2020), Saxena et al. (2008) for C-MAPSS.

---

## Appendix A — Code Architecture *(optional)*
A concise description of the `particle_twin` package structure, key design decisions, and instructions for reproducing all experimental results. Included only if the page count requires it; otherwise a README in the GitHub repository serves this purpose.
