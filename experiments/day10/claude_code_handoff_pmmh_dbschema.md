# Handoff to Claude Code (run locally) — PMMH joint calibration + DB schema

**Why this can't run in Cowork:** the regression-based full-fleet run just
completed here (100 engines × 3 variants) needed ~14 minutes of chunked
execution even with the *cheap* per-engine fit (~0.3s for all 100 OLS-style
fits). PMMH is MCMC over the particle filter itself — Day 8 used
`n_particles=3200, n_iterations=1000` per single-parameter chain and that
already didn't fit one sandbox call. Jointly estimating two parameters
per engine, across some or all of the fleet, is well beyond what the
Cowork sandbox's 45-second-per-call limit can run. This needs to happen
on your machine via Claude Code.

**Context Claude Code needs (paste this whole file in, or point it at
`Thesis_completing_Progress.md` + this file):**

- The Day 10/11 similarity-library work (`experiments/day10/`) fits ONE
  exponential growth_rate per training engine (regression, not PMMH) and
  matches test engines against that library. Just decided, on the full
  100-engine fleet: **one-shot matching is the thesis default** (beats
  periodic and the old pooled-fleet-rate baseline on every metric — see
  `experiments/day10/day11_fullfleet_decision.md`).
- `particle_twin/inference/pmmh.py` currently only calibrates `sigma_v`
  (Day 8), always against the OLD pooled degradation fit — it has never
  touched `growth_rate` and was never run against the per-engine library.
- The library's regression-fit growth_rates (n=100 training engines) have
  mean=0.003727, std=0.001150, range=[0.001750, 0.006645] — this can
  anchor a prior for growth_rate below.

## Task 1: extend `pmmh.py` to jointly sample (growth_rate, sigma_v)

- Add a `log_prior_growth_rate` (suggest half-normal or normal, centered
  near the library mean 0.00373 with std ~0.00115 — matches the
  already-observed per-engine spread, don't just invent a new prior from
  scratch).
- Extend `run_pf_for_sigma_v` (rename or add a variant) to accept BOTH
  parameters and build the particle filter using Day 10's model-swap
  mechanism (`matched_model_from` / `make_matched_dynamics` in
  `experiments/day10/similarity_library_prototype.py`) instead of
  modifying `DegradationModel`/`DegradationModelLearner` directly — reuse
  what already works.
- `pmmh_sample_sigma_v` becomes a 2D random-walk MH: independent proposal
  step sizes per dimension (growth_rate's scale is ~1e-3, sigma_v's is
  ~0.1-1, so a shared step size will make one dimension barely move).
  Keep the existing log-space positivity handling for sigma_v; decide
  whether growth_rate also needs a log-space transform (it's always
  positive too, and Day 10's range spans ~4x — probably yes, worth
  discussing with Claude Code before just copying the sigma_v pattern).
- **Fallback rule, from the existing tracker note:** Day 8 found only
  2/4 engines mixed well (accept rates 0.336-0.478) while 2/4 got stuck
  (0.051-0.011). If a chain's accept_rate < ~0.10, treat it as
  unconverged and fall back to the regression-fit growth_rate (already
  in the library) rather than reporting a PMMH point estimate that isn't
  trustworthy. Log which source was used per engine — needed for the DB
  schema below and for an honest §3.4/§4.7 writeup either way.
- Decide scope with Claude Code before running: all 100 test engines, or
  a smaller diagnostic set first (Day 8's precedent was engines
  1/10/20/30 before committing to a full run) — given the mixing
  problems already seen once, a small diagnostic pass first is probably
  the safer default here too.

## Task 2: `particle_twin/data/database.py` schema addition

None of the 4 existing tables (engines/experiments/run_results/
rul_estimates) fit "per-engine model parameters." Add a new table, e.g.:

```
CREATE TABLE engine_parameters (
    id INTEGER PRIMARY KEY,
    exp_id INTEGER,
    engine_id INTEGER,
    dataset TEXT,
    growth_rate REAL,
    sigma_v REAL,
    source TEXT,          -- 'pmmh' or 'regression_fallback'
    accept_rate REAL,     -- NULL if source='regression_fallback'
    n_iterations INTEGER,
    FOREIGN KEY (exp_id) REFERENCES experiments(id)
)
```
Adjust column names/types to match the existing schema's conventions in
`database.py` (check `run_results`/`rul_estimates` for the actual style
used — id/exp_id naming, etc. — before assuming the above verbatim).
**Remember:** `ResultsDB.__init__` already sets `PRAGMA journal_mode=WAL`
(Day 7 fix) — don't remove that when touching this file.

## Report back (to sync `Thesis_completing_Progress.md`)

- Which engines converged (accept_rate) vs. fell back to regression.
- Final growth_rate/sigma_v per engine, and whether they materially
  change the one-shot-vs-periodic full-fleet numbers in
  `experiments/day10/day11_fullfleet_decision.md` (rerun the full-fleet
  comparison with PMMH-calibrated rates once this is done, if time
  allows — same script, just swap the library's growth_rates).
