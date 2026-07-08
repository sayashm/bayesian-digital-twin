# Thesis Completion Progress
**Last updated:** July 8, 2026 (Day 3 — Bootstrap Particle Filter complete)  
**Status legend:** ⬜ Not started | 🔄 In progress | ✅ Done | ❌ Blocked

---

## Admin & Milestones

| Item | Status | Target Date | Notes |
|------|--------|-------------|-------|
| Overleaf project created | ✅ | Day 1 | Live at Overleaf (Bayesian_DT_Thesis_Ayashm) |
| GitHub repo created | ✅ | Day 1 | https://github.com/sayashm/bayesian-digital-twin |
| SQLite schema designed | ✅ | Day 1 | 4 tables in database.py; schema verified |
| Supervisor report sent | ⬜ | Day 12 | 1–2 pages, one figure, one question |
| First draft to supervisor | ⬜ | July 15 | Required ≥1 month before defense |
| Poster pre-defense | ⬜ | TBD | Coordinate with Koen De Turck |
| Final submission (Plato) | ⬜ | ~Aug 10 | Required ≥2 weeks before defense |
| Public defense | ⬜ | Mid-Aug 2026 | |

---

## Code Modules (`particle_twin` package)

| Module | Status | Day | Notes |
|--------|--------|-----|-------|
| Package skeleton (`__init__.py`, folder structure) | ✅ | Day 1 | installed with pip install -e |
| `data/loader.py` — C-MAPSS reader (all 4 datasets) | ✅ | Day 1 | full implementation done |
| `data/database.py` — SQLite interface | ✅ | Day 1 | 4-table schema, verified |
| `features/health_index.py` — HealthIndexBuilder | ✅ | Day 2.5 | NEW. 3 methods: weighted, pca (sensor-only, fit/transform, usable train+test), industrial (RUL-based, train-only reference — cannot be used as a live observation) |
| `models/degradation_learner.py` — DegradationModelLearner | ✅ | Day 2.5 (bug fixed Day 3) | NEW. Fits mean dynamics on fleet-pooled (cycle, HI): linear / polynomial / exponential, selectable. Added `time=` param (pooled fit) and `predict(t)`; replaced hardcoded noise constant with explicit `sigma_v`. **Day 3 bug fix:** all three `dynamics_func` implementations drew process noise as a single shared scalar (`np.random.normal(0, sigma_v)`) applied identically to every particle, so resampled duplicate particles could never re-diversify. Fixed to `size=np.shape(health)` (independent noise per particle) after Sajjad noticed particle intervals/ESS were suspiciously static in `test_bootstrap.py`. |
| `models/measurement_learner.py` — MeasurementModelLearner | ✅ | Day 2.5 | NEW. Fits observation noise on residuals Y_t - E[X_t]: gaussian / empirical KDE / best-AIC-of-4 parametric family. |
| `models/state_space.py` — `DegradationModel` (v2) | ✅ | Day 2.5 | REWRITTEN. Orchestrates HI construction + degradation dynamics + measurement noise. Observation is now a scalar HI (`y = x + w`), not 21 raw sensors (`y = Cx + d + w`). API unchanged for the filter: `sample_initial`, `transition`, `log_likelihood`. See `experiments/test_state_space.py` for real-FD001 validation + model-comparison numbers. |
| `filters/bootstrap.py` — bootstrap particle filter | ✅ | Day 3 | Per-engine, not per-fleet. `BootstrapPF` class: init → predict → weight → resample (systematic, ESS-gated), all self-written by Sajjad line-by-line with review each step. Two real bugs caught and fixed during review: numpy `+=` aliasing (`self.history` entries all pointing at the same mutating array) and `logsumexp` misuse in `resample` (returned a scalar instead of the normalized weight array, collapsing every particle to one duplicated value each cycle). `normalize()` extracted as a shared helper used by both `ESS` and `resample`. |
| `analysis/rul.py` — RUL extraction from posterior | ⬜ | Day 4 | Distribution, not point estimate |
| `analysis/metrics.py` — RMSE, ESS, MAPE, time | ⬜ | Day 5 | |
| `visualization/plots.py` — thesis-quality figures | ⬜ | Day 5 | |
| `inference/pmmh.py` — PMMH parameter estimation | ⬜ | Day 8 | Now also a natural place to refine `sigma_v` (process noise) |
| Baseline model — linear/Kalman comparison | ⬜ | Day 9 | |

**New dependency:** `scikit-learn` (for PCA in `health_index.py`). Add to `requirements.txt` / `pyproject.toml` and run `pip install scikit-learn` in `.venv` before next session.

---

## Experiments

| Experiment | Status | Day | Notes |
|------------|--------|-----|-------|
| Single engine test (FD001, debug) | ✅ | Day 3 | `experiments/test_bootstrap.py` — runs 4 engines (1, 10, 20, 30), no crashes. **Final numbers (post degradation_learner.py noise fix):** filtered particle mean correlates 0.82–0.95 with observed HI (engine 1: 0.909, 10: 0.951, 20: 0.816, 30: 0.949); final unique-particle counts 300–500 out of 500 (healthy diversity); ESS shows a proper sawtooth pattern (climbs to N after each resample, decays to the τ·N=250 threshold, repeats) for the full trajectory on all 4 engines. Earlier same-day numbers (corr. 0.59–0.90, unique particles as low as 2–9) were from before the shared-noise bug fix and are superseded. Figures: `bootstrap_test.png` (particle cloud vs. observed HI), `bootstrap_ess.png` (ESS trace, resample threshold marked) — both regenerated post-fix. |
| Full FD001 test set (all engines) | ⬜ | Day 5 | Get first RMSE |
| PMMH parameter estimation (FD001) | ⬜ | Day 8 | |
| Full FD001 systematic run | ⬜ | Day 9 | All metrics logged |
| FD003 run (multiple operating conditions) | ⬜ | Day 9 | Robustness test |
| Baseline comparison | ⬜ | Day 9 | PF vs. simple model |
| Final figures (thesis quality) | ⬜ | Day 10 | |

---

## Thesis Chapters

### Chapter 1: Introduction
| Section | Status | Day | Notes |
|---------|--------|-----|-------|
| §1.1 Problem context (digital twins, predictive maintenance) | ⬜ | Day 5 | |
| §1.2 Research objective | ⬜ | Day 5 | |
| §1.3 Thesis outline | ⬜ | Day 5 | |

### Chapter 2: Background
| Section | Status | Day | Notes |
|---------|--------|-----|-------|
| §2.1 Bayesian inference (prior, likelihood, posterior) | ⬜ | Day 6 | |
| §2.2 Sequential Bayesian filtering & state space models | ⬜ | Day 6 | |
| §2.3 Particle filters / SMC (SIR, bootstrap, resampling) | ⬜ | Day 6 | |
| §2.4 Digital twins in predictive maintenance | ⬜ | Day 6 | |
| §2.5 Related work — RUL estimation literature | ⬜ | Day 8 | |

### Chapter 3: Methodology
| Section | Status | Day | Notes |
|---------|--------|-----|-------|
| §3.1 State space model for turbofan degradation | 🔄 | Day 2.5 | Day-2 draft superseded by the 3-model architecture change; original prose kept as a comment in chapt3.tex. New structure + all equations (HI construction, generalised transition, revised observation) are in place as TODO scaffolding with real fit-quality numbers from FD001 — Sajjad to draft prose next |
| §3.2 Bootstrap particle filter algorithm | ✅ | Day 3 | 4 subsections + intro, all drafted by Sajjad (Q&A-to-prose method for §3.2.3/§3.2.4) and reviewed line-by-line: SIR/bootstrap choice, filtering recursion (log-space, real -2000 log-likelihood number), ESS/adaptive resampling (real ESS-collapse number, systematic vs. multinomial corrected — Sajjad had conflated systematic with stratified/residual), algorithm summary + per-engine independence. New citations verified and added: gordon1993bootstrap, kitagawa1996mcf, doccappe2005resampling. Algorithm~1 (algorithm2e) documents `BootstrapPF.run()` exactly. |
| §3.3 RUL extraction from posterior | ⬜ | Day 4 | |
| §3.4 Parameter estimation via PMMH | ⬜ | Day 8 | |

### Chapter 4: Experiments & Results
| Section | Status | Day | Notes |
|---------|--------|-----|-------|
| §4.1 Dataset description (C-MAPSS) | ⬜ | Day 9 | |
| §4.2 Evaluation metrics | ⬜ | Day 9 | |
| §4.3 Experimental setup | ⬜ | Day 9 | |
| §4.4 Results — health state estimation | ⬜ | Day 10 | |
| §4.5 Results — RUL prediction | ⬜ | Day 10 | |
| §4.6 Results — comparison with baseline | ⬜ | Day 10 | |

### Chapter 5: Discussion & Conclusions
| Section | Status | Day | Notes |
|---------|--------|-----|-------|
| §5.1 Discussion (interpretation, limitations) | ⬜ | Day 11 | |
| §5.2 Conclusions | ⬜ | Day 11 | |
| §5.3 Future work | ⬜ | Day 11 | |

### Front/Back Matter
| Item | Status | Day | Notes |
|------|--------|-----|-------|
| Abstract (English) | ⬜ | After full draft | Written last |
| Acknowledgements | ⬜ | Final week | |
| References / Bibliography | 🔄 Ongoing | — | Add citations as you write |
| Appendix — code architecture | ⬜ | Final week | Optional |

---

## Writing Quality Checklist (per section before marking ✅)

- [ ] Mathematical notation is precise and consistent throughout
- [ ] Every claim from a source has a citation (APA style)
- [ ] Uncertainty is quantified — no results stated as bare point estimates
- [ ] Figures are labeled (caption, axes, units)
- [ ] Section connects to thesis argument (how does this support "PF is a better digital twin"?)
- [ ] Plagiarism check: all paraphrased ideas cited (Ephorus will scan the submission)

---

## Session Log
*(Update at end of every working session)*

| Date | What was done | Next session starts with |
|------|--------------|--------------------------|
| 2026-06-28 | Planning session — all 4 guide files created | Day 1: infrastructure setup |
| 2026-06-29 | Day 1 complete: particle_twin package (loader.py, database.py, all stubs); SQLite schema verified; thesis_outline.md; thesis/ LaTeX on Overleaf; GitHub repo live at sayashm/bayesian-digital-twin; .venv created | Day 2: design degradation state space model on paper first, then implement state_space.py |
| 2026-07-01 | Day 2 complete: state space math derived; DegradationModel implemented (fit, sample_initial, transition, log_likelihood); loader.py updated to physical sensor names + unit_id; synthetic test passed (experiments/test_state_space.py); §3.1 all 5 paragraphs written and in chapt3.tex | Day 3: implement bootstrap.py particle filter; run on one FD001 engine |
| 2026-07-06 | Day 2.5 (architecture revision, before starting Day 3): replaced the single linear/21-sensor model with a 3-model pipeline built from Sajjad's own prototype classes (HealthIndexBuilder, DegradationModelLearner, MeasurementModelLearner). New `features/health_index.py` (weighted/pca/industrial HI), `models/degradation_learner.py` (linear/polynomial/exponential dynamics, now fittable on pooled fleet data via `time=`), `models/measurement_learner.py` (gaussian/empirical/custom noise). `models/state_space.py` rewritten: observation is now `y=x+w` (scalar HI) instead of `y=Cx+d+w` (21 sensors). Validated end-to-end on real FD001 data (`experiments/test_state_space.py`) — R² comparison across 2 HI methods × 3 degradation shapes, AIC comparison for measurement noise; `state_space_test.png` regenerated. chapt3.tex §3.1 restructured with new equations + honest fit-quality discussion (pooling-by-absolute-cycle caveat); original Day-2 prose kept as a comment. **Still needs:** Sajjad to draft the prose for the new §3.1 structure; pick final default (hi_method, degradation_model, measurement_method) combination for the results chapter, informed by the comparison table already in chapt3.tex. | Day 3 (as planned): implement bootstrap.py — no changes needed to its intended interface, it consumes `DegradationModel.sample_initial/transition/log_likelihood` exactly as before |
| 2026-07-08 | **Day 3 complete.** `filters/bootstrap.py` (`BootstrapPF`) implemented by Sajjad step-by-step (init → predict → weight → resample), reviewed at each step: caught and fixed a numpy `+=` aliasing bug (all `self.history` weight entries pointing at the same mutating array) and a `logsumexp` misuse in `resample` (scalar instead of normalized array, collapsing every particle to one duplicated value per cycle). Ran on FD001 engines 1/10/20/30 (`experiments/test_bootstrap.py`); found and fixed a real bug in Day-2.5's `degradation_learner.py` (process noise was a single shared scalar applied to every particle, not independent per particle) — after the fix, filtered-vs-observed correlation improved from 0.59–0.90 to 0.82–0.95 and particle diversity from single digits to 300–500/500. Figures `bootstrap_test.png` / `bootstrap_ess.png` regenerated. Wrote thesis §3.2 (all 4 subsections + intro) via a Q&A-to-prose method for the last two subsections (Sajjad answered plain-language questions, Claude corrected/formalized); caught a real conceptual error (Sajjad had conflated systematic resampling with stratified/residual — they're distinct schemes). Added and verified 3 new citations: `gordon1993bootstrap`, `kitagawa1996mcf`, `doccappe2005resampling`. | Day 4: implement `analysis/rul.py` — propagate the final particle cloud (cycle T) forward until a failure threshold, extract RUL as a distribution (median + 90% CI), not a point estimate; then write thesis §3.3 |
