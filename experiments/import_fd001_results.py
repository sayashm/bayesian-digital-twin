"""
import_fd001_results.py — load experiments/fd001_full_results.json into results.db
======================================================================================
Day 5's full FD001 run (100 engines) was executed in Cowork's sandbox, where
results.db writes fail with a SQLite "disk I/O error" (same limitation
Day 4 hit -- see test_rul.py's docstring). run_full_fd001.py therefore
always writes a portable JSON dump alongside the (failed) DB attempt.

This script does the cheap part -- reading that JSON and inserting rows --
on your machine, where results.db actually works. It does NOT redo any of
the particle filter / RUL computation, so it runs in seconds, not minutes.

Run from the repo root (after pulling/copying fd001_full_results.json):
    python experiments/import_fd001_results.py
"""

import json

from particle_twin.data.database import ResultsDB
from particle_twin.analysis.rul import to_db_row

JSON_IN = "experiments/fd001_full_results.json"

with open(JSON_IN) as f:
    payload = json.load(f)

config = payload['config']
engines = payload['engines']

db = ResultsDB("results.db")
exp_id = db.create_experiment(
    dataset=config['dataset'], n_particles=config['n_particles'],
    sigma_v=config['sigma_v'], sigma_w=config['sigma_w'],
    extra_params=config,
    description="Day 5 -- full FD001 test-set run (all engines), imported from sandbox JSON dump",
)
print(f"[OK] Registered experiment exp_id={exp_id}")

for uid_str, rec in engines.items():
    uid = int(uid_str)
    db.register_engine(engine_id=uid, dataset=config['dataset'], split="test",
                        n_cycles=rec['n_cycles'], true_rul=rec['true_rul'])
    for estimate in rec['trajectory']:
        # to_db_row(): store_rul_timeseries() takes explicit kwargs (no
        # **kwargs), so a v2-generated estimate dict (rul_bayes, frac_censored,
        # ...) would otherwise raise TypeError on unpacking -- see rul.py's
        # "BACKWARD COMPATIBILITY" docstring section.
        db.store_rul_timeseries(exp_id=exp_id, engine_id=uid, dataset=config['dataset'],
                                 **to_db_row(estimate))
    db.commit()

    ev = rec['metrics']
    db.store_run_result(exp_id=exp_id, engine_id=uid, dataset=config['dataset'],
                         rmse=ev['final_abs_error'], mape=ev['final_pct_error'],
                         mean_ess=ev['mean_ess'], runtime_s=ev.get('runtime_s'))

db.close()
print(f"[OK] Imported {len(engines)} engines into results.db (exp_id={exp_id})")

# sanity check: read back
db = ResultsDB("results.db")
check = db.get_run_results(exp_id)
print(f"[OK] Queried back {len(check)} run_results rows for exp_id={exp_id}")
db.close()
