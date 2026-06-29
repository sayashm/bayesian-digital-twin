"""
loader.py — C-MAPSS dataset reader
====================================
Loads all four C-MAPSS sub-datasets (FD001–FD004) from the NASA data files.

C-MAPSS column layout (no header in raw files):
  0        : engine_id
  1        : cycle
  2–4      : operational settings (op1, op2, op3)
  5–26     : sensor readings (s1 … s21)  [21 sensors]
  27 (test): RUL ground truth (only in RUL_FD00X.txt files)

Usage
-----
>>> from particle_twin.data.loader import CMAPSSLoader
>>> loader = CMAPSSLoader(data_dir="data/cmapss")
>>> train_df = loader.load_train("FD001")
>>> test_df  = loader.load_test("FD001")
>>> rul_df   = loader.load_rul("FD001")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Raw C-MAPSS files have no header; these are the canonical column names.
_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that carry degradation signal (selected by literature consensus).
# Sensors with near-zero variance are excluded.
INFORMATIVE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9",
                       "s11", "s12", "s13", "s14", "s15",
                       "s17", "s20", "s21"]

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


class CMAPSSLoader:
    """
    Reads and pre-processes NASA C-MAPSS turbofan run-to-failure data.

    Parameters
    ----------
    data_dir : str or Path
        Directory that contains the raw C-MAPSS text files
        (train_FD001.txt, test_FD001.txt, RUL_FD001.txt, …).
    """

    def __init__(self, data_dir: str | Path = "data/cmapss"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_train(self, dataset: str) -> pd.DataFrame:
        """
        Load the training split for *dataset* (e.g. 'FD001').

        Adds a 'rul' column computed from max cycle per engine
        (piece-wise linear capping is NOT applied here — the model
        will handle health index transformation).

        Returns
        -------
        pd.DataFrame with columns: engine_id, cycle, op1-3, s1-21, rul
        """
        self._check_dataset(dataset)
        df = self._read_txt(f"train_{dataset}.txt")
        df = self._add_rul_train(df)
        return df

    def load_test(self, dataset: str) -> pd.DataFrame:
        """
        Load the test split for *dataset*.

        Test sequences are *truncated* (we do not observe failure).
        No 'rul' column is added here; use load_rul() for ground truth.

        Returns
        -------
        pd.DataFrame with columns: engine_id, cycle, op1-3, s1-21
        """
        self._check_dataset(dataset)
        return self._read_txt(f"test_{dataset}.txt")

    def load_rul(self, dataset: str) -> pd.DataFrame:
        """
        Load the ground-truth RUL for every test engine in *dataset*.

        The file RUL_FD00X.txt contains one RUL value per line,
        where line i corresponds to test engine i+1.

        Returns
        -------
        pd.DataFrame with columns: engine_id (1-indexed), true_rul
        """
        self._check_dataset(dataset)
        path = self.data_dir / f"RUL_{dataset}.txt"
        rul_values = pd.read_csv(path, header=None, names=["true_rul"])
        rul_values.index = rul_values.index + 1   # engine IDs start at 1
        rul_values.index.name = "engine_id"
        return rul_values.reset_index()

    def get_engine_ids(self, dataset: str, split: str = "test") -> list[int]:
        """Return sorted list of engine IDs in the given split."""
        if split == "train":
            df = self.load_train(dataset)
        else:
            df = self.load_test(dataset)
        return sorted(df["engine_id"].unique().tolist())

    def get_engine(self, dataset: str, engine_id: int,
                   split: str = "test") -> pd.DataFrame:
        """
        Return time-series data for a single engine.

        Parameters
        ----------
        dataset   : e.g. 'FD001'
        engine_id : integer engine identifier (1-indexed)
        split     : 'train' or 'test'
        """
        if split == "train":
            df = self.load_train(dataset)
        else:
            df = self.load_test(dataset)
        engine_df = df[df["engine_id"] == engine_id].copy()
        if engine_df.empty:
            raise ValueError(
                f"Engine {engine_id} not found in {dataset} {split} split."
            )
        return engine_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dataset(dataset: str) -> None:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Choose from {DATASETS}."
            )

    def _read_txt(self, filename: str) -> pd.DataFrame:
        """Read a whitespace-delimited C-MAPSS file into a DataFrame."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"C-MAPSS file not found: {path}\n"
                "Download from: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data"
            )
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=_COLUMNS,
            engine="python",
        )
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        return df

    @staticmethod
    def _add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute piece-wise linear RUL for the training set.

        RUL at cycle t for engine e = (max_cycle_e - t).
        This gives the true RUL assuming failure at the last observed cycle.
        """
        max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycle, on="engine_id")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df.drop(columns=["max_cycle"], inplace=True)
        return df
