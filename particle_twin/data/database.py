"""
database.py — SQLite interface for experiment results
======================================================
All experiment outputs are stored in a single SQLite file (results.db).
Nothing is lost between sessions; every run is queryable after the fact.

Schema overview (see also schema.sql):
  engines       -- static metadata per C-MAPSS engine
  experiments   -- one row per experimental configuration (parameters, dataset)
  run_results   -- per-engine summary metrics for one experiment
  rul_estimates -- full RUL posterior summary per timestep per engine

Usage
-----
>>> from particle_twin.data.database import ResultsDB
>>> db = ResultsDB("results.db")
>>> exp_id = db.create_experiment(dataset="FD001", n_particles=1000,
...                               sigma_v=0.01, sigma_w=0.05,
...                               description="baseline run")
>>> db.store_run_result(exp_id, engine_id=1, rmse=12.3, mape=0.08,
...                     mean_ess=450.0, runtime_s=3.2)
>>> db.close()
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
-- engines: static catalogue of every C-MAPSS engine we have processed.
CREATE TABLE IF NOT EXISTS engines (
    engine_id   INTEGER NOT NULL,  -- 1-indexed, matches C-MAPSS file
    dataset     TEXT    NOT NULL,  -- 'FD001' | 'FD002' | 'FD003' | 'FD004'
    split       TEXT    NOT NULL,  -- 'train' | 'test'
    n_cycles    INTEGER,           -- length of the observed sequence
    true_rul    REAL,              -- ground-truth RUL (test split only)
    PRIMARY KEY (engine_id, dataset, split)
);

-- experiments: one row per experimental configuration.
-- All hyper-parameters are stored so results are fully reproducible.
CREATE TABLE IF NOT EXISTS experiments (
    exp_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    dataset     TEXT    NOT NULL,
    n_particles INTEGER NOT NULL,
    sigma_v     REAL    NOT NULL,  -- process noise std (transition)
    sigma_w     REAL    NOT NULL,  -- observation noise std
    extra_params TEXT,             -- JSON blob for any additional parameters
    description TEXT               -- free-text label
);

-- run_results: per-engine summary for one experiment.
CREATE TABLE IF NOT EXISTS run_results (
    result_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    exp_id      INTEGER NOT NULL REFERENCES experiments(exp_id),
    engine_id   INTEGER NOT NULL,
    dataset     TEXT    NOT NULL,
    rmse        REAL,              -- RMSE of final RUL estimate vs. true RUL
    mape        REAL,              -- Mean Absolute Percentage Error
    mean_ess    REAL,              -- Mean Effective Sample Size across timesteps
    runtime_s   REAL,             -- wall-clock time in seconds
    UNIQUE (exp_id, engine_id, dataset)
);

-- rul_estimates: full posterior summary per timestep per engine.
-- Stores the RUL distribution (not just a point estimate) at each cycle.
CREATE TABLE IF NOT EXISTS rul_estimates (
    estimate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    exp_id      INTEGER NOT NULL REFERENCES experiments(exp_id),
    engine_id   INTEGER NOT NULL,
    dataset     TEXT    NOT NULL,
    cycle       INTEGER NOT NULL,
    rul_median  REAL    NOT NULL,  -- 50th percentile of particle RUL
    rul_p5      REAL    NOT NULL,  -- 5th percentile  (90% CI lower)
    rul_p95     REAL    NOT NULL,  -- 95th percentile (90% CI upper)
    rul_mean    REAL    NOT NULL,  -- particle mean
    rul_std     REAL    NOT NULL,  -- particle std
    health_mean REAL,              -- mean health index h_t across particles
    ess         REAL,              -- effective sample size at this timestep
    UNIQUE (exp_id, engine_id, dataset, cycle)
);

-- engine_parameters: per-engine calibrated model parameters from PMMH or
-- regression fallback. One row per (exp_id, engine_id, dataset) — records
-- which growth_rate and sigma_v were actually used, and whether they came
-- from a converged PMMH chain or the regression library fallback.
CREATE TABLE IF NOT EXISTS engine_parameters (
    param_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    exp_id       INTEGER NOT NULL REFERENCES experiments(exp_id),
    engine_id    INTEGER NOT NULL,
    dataset      TEXT    NOT NULL,
    growth_rate  REAL    NOT NULL,
    sigma_v      REAL    NOT NULL,
    source       TEXT    NOT NULL,   -- 'pmmh' or 'regression_fallback'
    accept_rate  REAL,               -- NULL when source='regression_fallback'
    n_iterations INTEGER,
    UNIQUE (exp_id, engine_id, dataset)
);
"""


class ResultsDB:
    """
    Thin wrapper around a SQLite connection for storing and querying
    experiment results.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite file.  Created (with schema) if it does not exist.
    """

    def __init__(self, db_path: str | Path = "results.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row   # dict-like row access
        self._conn.execute("PRAGMA foreign_keys = ON;")
        # WAL mode avoids the classic rollback-journal dance (a stale
        # "-journal" file left behind after an interrupted/failed write,
        # which then blocks every subsequent open until manually removed --
        # hit repeatedly on this project, both from the Cowork sandbox
        # mount and from DB Browser locally). WAL instead appends to a
        # "-wal" file and checkpoints it back into the main file
        # automatically; it is the standard fix for exactly this symptom.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._initialise_schema()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _initialise_schema(self) -> None:
        """Create tables if they do not already exist."""
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        dataset: str,
        n_particles: int,
        sigma_v: float,
        sigma_w: float,
        extra_params: dict[str, Any] | None = None,
        description: str = "",
    ) -> int:
        """
        Register a new experiment configuration.

        Returns the auto-assigned exp_id (use this in subsequent calls).
        """
        extra_json = json.dumps(extra_params) if extra_params else None
        cur = self._conn.execute(
            """
            INSERT INTO experiments
                (dataset, n_particles, sigma_v, sigma_w, extra_params, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset, n_particles, sigma_v, sigma_w, extra_json, description),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_experiment(self, exp_id: int) -> dict:
        """Fetch a single experiment row by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE exp_id = ?", (exp_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Experiment {exp_id} not found.")
        return dict(row)

    # ------------------------------------------------------------------
    # Engine catalogue
    # ------------------------------------------------------------------

    def register_engine(
        self,
        engine_id: int,
        dataset: str,
        split: str,
        n_cycles: int | None = None,
        true_rul: float | None = None,
    ) -> None:
        """
        Add an engine to the catalogue (INSERT OR REPLACE — idempotent).
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO engines
                (engine_id, dataset, split, n_cycles, true_rul)
            VALUES (?, ?, ?, ?, ?)
            """,
            (engine_id, dataset, split, n_cycles, true_rul),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Results storage
    # ------------------------------------------------------------------

    def store_run_result(
        self,
        exp_id: int,
        engine_id: int,
        dataset: str,
        rmse: float | None = None,
        mape: float | None = None,
        mean_ess: float | None = None,
        runtime_s: float | None = None,
    ) -> None:
        """
        Store per-engine summary metrics for one experiment run.
        Overwrites any existing row for the same (exp_id, engine_id, dataset).
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_results
                (exp_id, engine_id, dataset, rmse, mape, mean_ess, runtime_s)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (exp_id, engine_id, dataset, rmse, mape, mean_ess, runtime_s),
        )
        self._conn.commit()

    def store_rul_timeseries(
        self,
        exp_id: int,
        engine_id: int,
        dataset: str,
        cycle: int,
        rul_median: float,
        rul_p5: float,
        rul_p95: float,
        rul_mean: float,
        rul_std: float,
        health_mean: float | None = None,
        ess: float | None = None,
    ) -> None:
        """
        Store the RUL posterior summary at a single timestep.
        Call once per cycle inside the particle filter loop.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO rul_estimates
                (exp_id, engine_id, dataset, cycle,
                 rul_median, rul_p5, rul_p95, rul_mean, rul_std,
                 health_mean, ess)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (exp_id, engine_id, dataset, cycle,
             rul_median, rul_p5, rul_p95, rul_mean, rul_std,
             health_mean, ess),
        )
        # Commit in batches by calling commit() explicitly after the full engine run.

    def store_engine_parameters(
        self,
        exp_id: int,
        engine_id: int,
        dataset: str,
        growth_rate: float,
        sigma_v: float,
        source: str,
        accept_rate: float | None = None,
        n_iterations: int | None = None,
    ) -> None:
        """
        Store per-engine calibrated model parameters (from PMMH or fallback).
        Overwrites any existing row for the same (exp_id, engine_id, dataset).

        Parameters
        ----------
        source : {'pmmh', 'regression_fallback'}
        accept_rate : float or None — supply when source='pmmh', leave None
            when source='regression_fallback'.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO engine_parameters
                (exp_id, engine_id, dataset, growth_rate, sigma_v,
                 source, accept_rate, n_iterations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (exp_id, engine_id, dataset, growth_rate, sigma_v,
             source, accept_rate, n_iterations),
        )
        self._conn.commit()

    def commit(self) -> None:
        """Explicitly commit pending writes (use after a full engine run)."""
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_run_results(self, exp_id: int) -> list[dict]:
        """Return all per-engine results for an experiment."""
        rows = self._conn.execute(
            "SELECT * FROM run_results WHERE exp_id = ? ORDER BY engine_id",
            (exp_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_rul_timeseries(
        self, exp_id: int, engine_id: int, dataset: str
    ) -> list[dict]:
        """Return the full RUL timeseries for one engine in one experiment."""
        rows = self._conn.execute(
            """
            SELECT * FROM rul_estimates
            WHERE exp_id = ? AND engine_id = ? AND dataset = ?
            ORDER BY cycle
            """,
            (exp_id, engine_id, dataset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_engine_parameters(self, exp_id: int) -> list[dict]:
        """Return all per-engine parameter rows for an experiment."""
        rows = self._conn.execute(
            "SELECT * FROM engine_parameters WHERE exp_id = ? ORDER BY engine_id",
            (exp_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_experiments(self) -> list[dict]:
        """Return all experiments ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Commit any pending writes and close the connection."""
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
