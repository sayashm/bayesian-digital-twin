# Final Experiments — Step-by-Step Prompts

Workflow: hand each **[CODE]** prompt to Claude Code, bring results back here, then run
the matching **[COWORK]** prompt. Go one step at a time in order.

Experiment map:

| # | Method              | Dataset |
|---|---------------------|---------|
| 1 | PF Fixed Parameters | FD001   |
| 2 | PF Fixed Parameters | FD003   |
| 3 | PF PMMH             | FD001   |
| 4 | PF PMMH             | FD003   |
| 5 | library method      | FD001   |
| 6 | library method      | FD003   |

---

## Step 1 — [CLAUDE CODE] IDA & EDA for FD003 (+ FD001 notebooks)

```
Run the IDA (Initial Data Analysis) and EDA (Exploratory Data Analysis) for FD003,
and also produce the equivalent notebooks for FD001, inside the folder:
/Users/sajjad/MyProjects/Bayesian_DT/experiments/IDA

Match the structure, style, and outputs of the existing IDA/EDA work already in that
folder — same figures, tables, and notebook layout, just for these datasets. Save all
results (notebooks + figures + tables) in that same IDA folder. When you are done,
print a short summary of what was created and where, and confirm it is finished.
```

---

## Step 2 — [CLAUDE COWORK] Put FD003 IDA/EDA results in the thesis

```
Claude Code has finished the IDA and EDA for FD003 in
/Users/sajjad/MyProjects/Bayesian_DT/experiments/IDA.
Read those FD003 results (figures, tables, and any notes) and add them into the thesis
in the correct section, matching how the FD001 IDA/EDA is already presented. Keep the
writing consistent with the existing thesis style.
```

---

## Step 3 — [CLAUDE CODE] PF with Fixed Parameters — FD001 & FD003 (Exp 1 & 2)

```
We changed the RUL definition. Use the NEW RUL method (do NOT modify rul.py — it is
already correct). Run the Particle Filter with FIXED parameters experiment for both
FD001 and FD003, saving into:
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/PF_Fixed_Parameters/FD001
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/PF_Fixed_Parameters/FD003

Each dataset must produce, in its figures/ and tables/ subfolders:
- A per-engine figure for every engine showing RUL and health index (same style as the
  per-engine figures in the library_method).
- Health-vs-RUL figures (train and test), equivalent to
  final_experiment/library_method/FD001/05_health_vs_rul.py.
- Results tables equivalent to
  final_experiment/library_method/FD001/04_make_tables.py
  (rul_per_engine.csv, rul_summary.csv, health_index_per_engine.csv,
  health_index_summary.csv).

Reuse the library_method scripts as the template for figures and tables so the outputs
are directly comparable. When done, print a summary of files created and confirm.
```

---

## Step 4 — [CLAUDE CODE] PF with PMMH — FD001 & FD003 (Exp 3 & 4)

```
Same as before but for the Particle Filter with PMMH. Use the NEW RUL method (do NOT
modify rul.py). Run PF-PMMH for both FD001 and FD003, saving into:
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/PF_PMMH/FD001
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/PF_PMMH/FD003

Each dataset must produce, in its figures/ and tables/ subfolders:
- Per-engine RUL + health-index figures for every engine (library_method style).
- Health-vs-RUL figures (train and test), like library_method .../05_health_vs_rul.py.
- Results tables like library_method .../04_make_tables.py (rul_per_engine.csv,
  rul_summary.csv, health_index_per_engine.csv, health_index_summary.csv).

Keep outputs directly comparable to the library_method and PF_Fixed_Parameters runs.
When done, print a summary of files created and confirm.
```

---

## Step 5 — [CLAUDE COWORK] Put PF results in the thesis (final results)

```
Claude Code has finished the two Particle Filter experiments:
- PF Fixed Parameters (FD001 & FD003) in final_experiment/PF_Fixed_Parameters
- PF PMMH (FD001 & FD003) in final_experiment/PF_PMMH

Read the figures and tables from both, and update the thesis with these as our FINAL
results. Where previous/older results are now superseded, remove them so the thesis
reflects only the final numbers. Keep the presentation consistent with the existing
results sections.
```

---

## Step 6 — [CLAUDE CODE] Rerun library_method FD003 with updated params

```
Using the FD003 IDA/EDA we just produced, rerun the experiments in
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/library_method/FD003
and update the results. Do NOT change the code structure at all — only adjust the
parameters based on the FD003 IDA/EDA. In particular, set the Max-Horizon so that
Max-Horizon >= the maximum RUL in the FD003 training set. Regenerate all figures and
tables (per-engine figures, health_vs_rul, and the tables). When done, print a summary
and confirm.
```

---

## Step 7 — [CLAUDE COWORK] Put library_method FD003 results in the thesis

```
Claude Code has updated the library_method results for FD003 in
final_experiment/library_method/FD003. Read the updated figures and tables and add/
update the FD003 library-method results in the thesis, consistent with how the FD001
library-method results are presented.
```

---

## Step 8 — [CLAUDE CODE] Linear baseline re-run on the NEW RUL (FD001 & FD003)

```
The linear regression baseline needs to be re-run on the SAME (new) RUL target
used by the PF and library experiments, so it is directly comparable to the final
results. Use the new RUL method (do NOT modify rul.py). Run the baseline for both
FD001 and FD003 and save into:
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/Linear_baseline/FD001
/Users/sajjad/MyProjects/Bayesian_DT/experiments/final_experiment/Linear_baseline/FD003

Keep the same code/folder structure as the other final_experiment subfolders
(numbered scripts, config.json, figures/, tables/). The baseline is the existing
LinearRULBaseline: the same ten selected sensors, ordinary least squares, pooled
across the training fleet, producing one point RUL prediction per test engine from
its last observed cycle — no state space model, no filtering, no credible interval.

In each dataset's tables/ produce, matching the other experiments' schema where
applicable:
- rul_summary.csv: fleet_rmse, fleet_mae, fleet_mape, phm08_total, n_engines
  (there is NO ci_coverage column — a point estimate has no interval).
- rul_per_engine.csv: engine_id, n_cycles, true_rul, pred_rul, final_abs_error
  (and trajectory_rmse if the baseline is evaluated per cycle).

In each dataset's figures/ produce, matching the existing style:
- A fleet predicted-vs-true RUL scatter (with the y=x line).
- Per-engine predicted-RUL figures if the baseline is scored per cycle, otherwise
  skip the per-engine panels.

Use the same seed and the same train/test split as the PF/library runs so the
numbers line up. When done, print RMSE / MAE / MAPE / PHM08 for FD001 and FD003 and
confirm it is finished.
```

---

## Step 9 — [CLAUDE COWORK] Finalize the baseline (§4.7) and rewrite Chapter 5

```
Claude Code has re-run the linear baseline on the new RUL in
final_experiment/Linear_baseline/FD001 and .../FD003. Do two things:

1. §4.7 (Comparison with Baseline): replace the old-RUL baseline numbers with the new
   ones, remove the "old RUL target / not strictly comparable" caveat, and reframe the
   comparison as now valid — baseline vs. the final one-shot library method, on both
   FD001 and FD003, noting that the PF/library additionally provides a calibrated
   interval the baseline cannot.

2. Chapter 5 (Discussion): rewrite it to match the final results. Specifically remove
   the coverage 0.000 claims (chapt5 lines ~49 and ~181), reframe the similarity
   library as fixing BOTH point accuracy and calibration (not just the mean), present
   one-shot matching as the adopted final method, fold in the FD003 robustness finding
   (accuracy improves, but intervals are over-conservative under the second fault
   mode), and update any remaining old-RUL numbers so Chapter 5 is consistent with the
   final Chapter 4.
```
