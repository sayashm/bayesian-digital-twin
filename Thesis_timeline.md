# Thesis Timeline
**Student:** Sajjad Ayashmand  
**Thesis Deadline:** Mid-August 2026  
**Focused Work Period:** Two free weeks starting ~June 29, 2026  
**Last updated:** June 28, 2026

---

## Macro Timeline (Full Period)

| Period | Focus | Key Deliverable |
|--------|-------|-----------------|
| Week 1–2 (June 29 – July 12) | Infrastructure + Core Implementation + Chapter drafts | Working PF on C-MAPSS, thesis skeleton |
| Week 3–4 (July 13 – July 26) | Full experiments + Results chapter + Discussion | Complete first draft |
| Week 5–6 (July 27 – Aug 9) | Revision + Supervisor feedback + Polish | Near-final thesis |
| Week 7 (Aug 10–15) | Final submission prep | Submitted thesis |

**Draft to supervisor:** No later than July 15 (to give promoter 1 month before likely defense)  
**Poster pre-defense:** Coordinate with supervisor — must happen before public defense  
**Final submission via Plato:** At least 2 weeks before defense date  

---

## Week 1 Micro Plan — Build the Machine
*~6 hours/day | Morning = Code | Afternoon = Write*

### Day 1 — Infrastructure (No particle filter yet)
**Goal:** Everything set up before any science happens.

| Time | Task |
|------|------|
| 9–11 | Set up `particle_twin` Python package structure + GitHub repo |
| 11–1 | Design SQLite schema (`results.db`): engines, experiments, results tables |
| 2–4 | Set up Overleaf thesis from UGent template; title page, chapter headings |
| 4–6 | Write thesis outline: every section heading + 2–3 sentences describing what each section will argue |

**Deliverables:** `particle_twin/` package skeleton, `results.db` schema, Overleaf project live, thesis outline

---

### Day 2 — State Space Model
**Goal:** Define the mathematical heart of the thesis before coding the filter.

| Time | Task |
|------|------|
| 9–11 | Design degradation model: health index $h_t$ as hidden state, C-MAPSS sensors as observations. Write down transition and observation equations on paper first. |
| 11–1 | Implement `models/state_space.py` — Python class for transition and observation |
| 2–4 | Test model with synthetic data; visualize degradation trajectories |
| 4–6 | *Write:* Methodology — State Space Model section (first draft) |

**Deliverables:** `state_space.py` working, Methodology section draft §3.1

---

### Day 3 — Bootstrap Particle Filter
**Goal:** Core algorithm implemented and understood line by line.

| Time | Task |
|------|------|
| 9–12 | Implement `filters/bootstrap.py` step by step (Sajjad writes, Claude guides): initialization → predict → weight → resample |
| 12–1 | Run on one FD001 engine; plot particle cloud evolution over time |
| 2–4 | Debug; understand what's happening physically with each particle |
| 4–6 | *Write:* Methodology — Particle Filter Algorithm section (first draft) |

**Deliverables:** `bootstrap.py` working on one engine, Methodology §3.2 draft

---

### Day 4 — RUL Extraction
**Goal:** Get a real RUL prediction and store it.

| Time | Task |
|------|------|
| 9–11 | Implement `analysis/rul.py` — propagate particles forward; extract distribution (median + 90% CI) |
| 11–1 | Test on FD001; compare predicted vs. true RUL for one engine |
| 2–4 | Store results in `results.db`; verify data is queryable |
| 4–6 | *Write:* Methodology — RUL Extraction section (first draft) |

**Deliverables:** `rul.py` working, first RMSE number in DB, Methodology §3.3 draft

---

### Day 5 — First Real Results
**Goal:** End-to-end run on all FD001 test engines; first thesis-quality figure.

| Time | Task |
|------|------|
| 9–11 | Run filter on all FD001 test engines; compute RMSE across fleet |
| 11–1 | Create first thesis-quality figure: health tracking + RUL distribution for one engine |
| 2–4 | Tune noise parameters; investigate why some engines are harder |
| 4–6 | *Write:* Introduction — first draft (2–3 pages) |

**Deliverables:** Full FD001 results in DB, first thesis figure, Introduction §1 draft

---

### Day 6 — Background Writing Day
**Goal:** Write the Background chapter while knowledge is fresh.

| Time | Task |
|------|------|
| All day | *Write:* Chapter 2 — Background |
| | §2.1 Bayesian inference (prior, likelihood, posterior) |
| | §2.2 Sequential Bayesian filtering (state space models, Chapman-Kolmogorov) |
| | §2.3 Particle filters / SMC (SIR, bootstrap, resampling) |
| | §2.4 Digital twins in predictive maintenance |
| | §2.5 Related work (existing RUL methods for C-MAPSS) |

**Process:** Sajjad drafts each subsection → Claude gives line-level feedback → Sajjad rewrites  
**Deliverables:** Chapter 2 first draft complete

---

### Day 7 — Review & Consolidate
**Goal:** Fix what's broken before Week 2 begins.

| Time | Task |
|------|------|
| Morning | Review all code from Week 1; fix any outstanding bugs |
| Afternoon | Read all written sections; mark gaps, weak arguments, missing citations |

**Deliverables:** Clean codebase, annotated thesis draft with revision notes

---

## Week 2 Micro Plan — Results & Writing

### Day 8 — PMMH Learning & Implementation
**Goal:** Learn parameter estimation; integrate into the pipeline.

| Time | Task |
|------|------|
| 9–12 | **Teaching session:** PMMH — what it is, why it works, how to implement (one segment at a time) |
| 12–2 | Implement `inference/pmmh.py` for estimating degradation parameters |
| 2–4 | Test: compare PMMH-estimated parameters vs. assumed parameters |
| 4–6 | *Write:* Related Work section (§2.5 refined + additional PF/RUL literature) |

**Deliverables:** `pmmh.py` working, parameters estimated for FD001

---

### Day 9 — Full Experiment Suite
**Goal:** Systematic experiments across datasets; build baseline comparison.

| Time | Task |
|------|------|
| 9–11 | Run PF on FD001 — all test engines, all metrics logged to DB |
| 11–1 | Run on FD003 (single condition, two fault modes — harder case; corrected Day 9 from the earlier "multiple operating conditions" mislabel, which actually describes FD002/FD004) |
| 2–4 | Implement simple baseline (e.g., linear regression on degradation sensors for RUL) |
| 4–6 | *Write:* Experiments chapter setup (§4.1 dataset, §4.2 evaluation metrics, §4.3 experimental setup) |

**Deliverables:** Full experiment results in DB, baseline implemented, Experiments §4.1–4.3 draft

---

### Day 10 — Results Chapter
**Goal:** All results written up with interpretation.

| Time | Task |
|------|------|
| 9–12 | Create all thesis-quality figures (health tracking, RUL distributions, RMSE comparison, ESS plots) |
| 12–2 | Statistical analysis: RMSE distribution across engines, variance, hard vs. easy engines |
| 2–6 | *Write:* Results chapter (§4.4) — every figure gets a paragraph of interpretation |

**Deliverables:** All thesis figures finalized, Results section draft

---

### Day 11 — Discussion & Conclusion
**Goal:** Interpret results; write the "so what" sections.

| Time | Task |
|------|------|
| 9–12 | *Write:* Discussion (§5.1) — what worked, what didn't, comparison with baseline, limitations |
| 12–2 | *Write:* Conclusions (§5.2) + Future Work (§5.3) |
| 2–4 | Revise Introduction now that you know exactly what you proved |
| 4–6 | Read entire thesis end to end; note every gap and inconsistency |

**Deliverables:** Discussion + Conclusion draft, revised Introduction

---

### Day 12 — Supervisor Report
**Goal:** Send supervisors a concrete, defensible progress report.

| Time | Task |
|------|------|
| 9–12 | Draft supervisor report (1–2 pages): methodology in 3 sentences + one figure + one technical question |
| 12–2 | Polish and review with Claude |
| 2–6 | Revise any thesis section exposed as weak by writing the report |

**Report strategy:** Short, precise, confident. One result (health-tracking plot on real engine). One specific question to supervisors (not "is this good?" but a concrete methodological choice). Shows you are thinking at the right level.

**Deliverables:** Supervisor report sent, thesis weaknesses addressed

---

### Day 13–14 — Buffer & Handoff to Remaining Weeks
**Goal:** Ensure the first draft is complete enough to continue in reduced-time weeks.

| Time | Task |
|------|------|
| Day 13 | Complete any unfinished writing sections; add missing citations to `.bib` file |
| Day 14 | Final review of all Week 1–2 deliverables; update `Thesis_completing_Progress.md` |

---

## Remaining Period (July 13 – August 15)

These weeks have less free time but the hardest work (full draft, revision) is done.

| Week | Priority |
|------|----------|
| July 13–19 | Revise Chapter 2 (Background) based on results — remove what you didn't use |
| July 20–26 | Full thesis revision pass; Abstract; fix LaTeX formatting |
| July 27 – Aug 2 | Incorporate supervisor feedback; polish figures; finalize bibliography |
| Aug 3–9 | Proof-read; check UGent formatting requirements; prepare Plato submission |
| Aug 10–15 | Final submission buffer; prepare for oral defense questions |
