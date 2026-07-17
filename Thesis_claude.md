# Thesis Project — Claude Instructions
**Project:** A Bayesian Approach to Digital Twins with Applications to Predictive Maintenance  
**Student:** Sajjad Ayashmand | s.ayashm@gmail.com  
**Program:** Master of Science in Statistical Data Analysis — Ghent University (UGent)  
**Supervisors:** Prof. Koen De Turck (promoter), Prof. Dieter Fiems (co-promoter) — Department TW07  
**Deadline:** Mid-August 2026

---

## 1. Who Sajjad Is

- **Background:** Data Analysis and Statistics. Strong in classical statistics; no prior Bayesian background before starting this thesis preparation.
- **Completed:** A structured 4-week self-study of particle filtering using Chopin & Papaspiliopoulos, *An Introduction to Sequential Monte Carlo*. See `Thesis_learning_progress.md` for full details.
- **Language:** Scientific writing is a significant challenge. Sajjad can think clearly but struggles to express ideas in formal academic English. He needs active writing coaching — not just feedback after the fact.
- **Coding style:** Understands Python well. Prefers to understand code deeply, not just copy-paste. Wants a mix of guided coding (Claude explains each part) and writing code himself.
- **Supervisor situation:** Supervisors are present but not directive. They give approval without detailed guidance. Sajjad needs to be self-sufficient in defining scope and direction, and should build a supervisor report that demonstrates concrete progress.

---

## 2. Thesis Overview

### Core Argument
A particle filter (Sequential Monte Carlo) serves as a principled digital twin model. Unlike classical approaches that represent system state as a single estimate, the particle filter maintains a *distribution* over possible states — a swarm of particles each representing a plausible scenario. Incoming sensor observations recalibrate particle weights continuously, providing both state tracking and uncertainty quantification. Applied to predictive maintenance, this allows RUL (Remaining Useful Life) to be expressed as a distribution, not a point estimate — enabling better-informed maintenance decisions.

### Dataset
**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**  
- 4 sub-datasets: FD001, FD002, FD003, FD004  
- Contains run-to-failure turbofan engine sensor data  
- Ground truth RUL provided for test engines  
- Start with FD001 (single operating condition, single fault mode — simplest case)  
- Extend to FD003 (single operating condition, **two fault modes**: HPC + Fan degradation) for robustness testing. **Correction (Day 9):** FD003 was previously mislabeled here as "multiple operating conditions" — that description actually belongs to FD002/FD004 (six operating conditions each), which the current pipeline does not yet support (no operating-condition-aware normalization in `health_index.py`/`degradation_learner.py`). FD003's robustness axis is an additional fault mode, not operating-condition variation.

### Thesis Structure (UGent MaStat)
Following the `Writing_Templates/Templates/` LaTeX template:
1. Abstract
2. Chapter 1: Introduction & Research Objectives
3. Chapter 2: Background (Bayesian inference, SMC/particle filters, digital twins, predictive maintenance)
4. Chapter 3: Methodology (state space model, particle filter design, RUL extraction, parameter estimation via PMMH)
5. Chapter 4: Experiments & Results (C-MAPSS, per-engine filtering, comparison with baseline)
6. Chapter 5: Discussion & Conclusions
7. References (APA style, `.bib` file)
8. Appendix (code architecture overview if needed)

---

## 3. Technical Architecture (Decided — Do Not Change Without Discussion)

### Python Package: `particle_twin`
A proper, expandable Python package — not notebooks. Notebooks are only used in `experiments/` to call the package.

```
particle_twin/
├── data/
│   ├── loader.py        # C-MAPSS reader for all 4 datasets
│   └── database.py      # SQLite interface for storing results
├── models/
│   └── state_space.py   # Degradation state space model (shared across engines)
├── filters/
│   └── bootstrap.py     # Bootstrap particle filter (runs per-engine independently)
├── inference/
│   └── pmmh.py          # PMMH for parameter estimation (added on Day 8)
├── analysis/
│   ├── rul.py           # RUL extraction from particle posterior
│   └── metrics.py       # RMSE, ESS, MAPE, computational time
└── visualization/
    └── plots.py         # Thesis-quality figures
```

**Critical design rule:** The state space model (equations) is shared across all engines. But each engine runs its own independent particle filter. This is the correct approach — one model was the previous mistake.

### Database: SQLite (`results.db`)
Single-file database. No server required. Stores:
- Raw and processed C-MAPSS data
- Every experiment run (parameters, dataset, engine ID)
- Results (RUL estimates, RMSE, ESS per timestep)

Purpose: nothing is lost between sessions; all experiments are reproducible and queryable.

### Writing: LaTeX via Overleaf
- Template: `Writing_Templates/Templates/CNSDAN_ThesisTemplateEN.tex`  
- Style class: `CNSDANthesisEN.cls`  
- Bibliography: APA style (apalike), managed in `Thesis_bib.bib`  
- Supervisors can view the Overleaf draft via share link

---

## 4. How Claude Should Work in This Project

### 4.1 Writing Coaching
**UPDATED 2026-07-17 (Day 8) — Sajjad explicitly changed this process due to limited remaining time.**
The original "Sajjad writes first" rule below is superseded for the rest of the
thesis. New process, chosen by Sajjad to stay involved despite Claude writing
full prose now:

**Process for every writing task (current):**
1. Claude drafts a plain-language, simple-English version of the section's content first — ideas only, not polished academic prose. This is the content checkpoint: easier for Sajjad to judge "is this the right substance" than to critique dense academic English.
2. Sajjad reads it and approves, or asks for changes to the content/ideas.
3. Once approved, Claude writes the actual academic-style `.tex` prose directly into the file — calibrated to Sajjad's own voice/register (not overly polished/native-level English; see [[sajjad-language-simplicity]] memory).
4. Sajjad reviews the final academic text, section by section, before moving to the next section (not one pass at the very end).

**Superseded process (kept for reference, do not follow):**
~~1. Claude explains what the section needs to say and why~~
~~2. Sajjad writes a draft (even rough)~~
~~3. Claude gives specific, line-level feedback — not rewrites~~
~~4. Sajjad rewrites~~
~~5. Claude approves or gives one more round of feedback~~
~~Never write a full section for Sajjad unprompted. Push him to write first.~~

**Tone to maintain:** Academic, precise, third-person. Avoid colloquial expressions. Every claim needs a citation or derivation.

**Plagiarism rules (UGent — strictly enforced via Ephorus software):**
- Every idea, fact, or formula from a source must be cited — even when paraphrased
- Only Sajjad's own derivations, observations, and experimental results are uncited
- Common knowledge in statistics does not need citation; interpretations always do
- See `Writing_Templates/Schrijven _ Plagiarism.pdf` for full policy

### 4.2 Coding Guidance
- Before writing any code together, Claude explains what the function does mathematically and why
- Sajjad writes the first attempt; Claude reviews and corrects
- For complex algorithms (PMMH, particle filter internals), Claude may provide a skeleton with blanks for Sajjad to fill
- All code must be documented with docstrings and comments explaining the math

### 4.3 Teaching Mode (for new concepts)
When a new concept is needed (e.g., PMMH on Day 8), use the established teaching method:
- One concept segment per message (definition → intuition → thesis connection → worked example → comprehension check)
- Wait for Sajjad's response before moving to the next segment
- Academic, formal tone — like a university lecture
- Always connect to the thesis application (turbofan degradation, digital twin)
- Create a quick reference guide after each teaching session

### 4.4 Session Start Protocol
At the start of every working session:
1. Read `Thesis_completing_Progress.md` to see current status
2. Read `Thesis_timeline.md` to know what today's tasks are
3. Briefly confirm with Sajjad what to work on
4. Update `Thesis_completing_Progress.md` at the end of each session

### 4.5 Supervisor Report (Day 12)
Target: 1–2 pages, one clear figure, one technical question.  
Strategy: show a concrete result (health tracking on one engine), state methodology in 3 sentences, ask a specific question that demonstrates deep thinking. Do NOT send a long report or a vague one.

---

## 5. UGent Thesis Requirements (from MaStat Guidelines 2022)

Claude must always be aware of these when giving writing advice:

- **Mathematical precision:** The statistical model must be formulated precisely with notation
- **Uncertainty quantification:** Results must include uncertainty — not just point estimates. This is a core strength of the particle filter approach.
- **Reproducibility:** All results must be reproducible. Code must be well-documented and submitted with the thesis (via Ufora/Plato).
- **Data analysis is mandatory:** The thesis must include a real data analysis — C-MAPSS satisfies this.
- **Policy implications:** The conclusion must discuss practical/policy implications for a broad audience (maintenance engineers, not just statisticians).
- **Assessment criteria:** Clarity of problem setting, accuracy of methodology, adequacy of data processing, quality of discussion, relevance of conclusions.
- **Draft deadline:** A draft must reach the promoter *at least 1 month before* the oral defense.
- **Final submission:** At least 2 weeks before the defense date, submitted via Plato.
- **Poster pre-defense:** Required before the public defense. Prepare a poster showing research question, objectives, and methods outline.

---

## 6. Key References

- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo.* Springer. — Primary textbook (in `sources/` folder of learning project)
- NASA C-MAPSS dataset — primary dataset
- `Writing_Templates/MaStat_GuidelinesThesis_2022.pdf` — UGent thesis regulations
- `Writing_Templates/Templates/` — LaTeX thesis template files
- `Writing_Templates/Schrijven _ Plagiarism.pdf` — Plagiarism policy

---

## 7. What NOT to Do

- Do not suggest rewriting code from scratch without a clear reason
- Do not write thesis sections in full without Sajjad attempting a draft first
- Do not move to the next day's tasks without completing today's deliverables
- Do not use a single particle filter for multiple engines simultaneously
- Do not skip the UGent formatting/citation requirements
- Do not let Sajjad go down a rabbit hole of new concepts — scope is fixed
- Do not forget to update `Thesis_completing_Progress.md` at end of session
