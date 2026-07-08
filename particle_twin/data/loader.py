"""
loader.py — C-MAPSS dataset reader
====================================
Loads all four C-MAPSS sub-datasets (FD001–FD004) from the NASA data files.

C-MAPSS column layout (no header in raw files):
  0    : unit_id
  1    : cycle
  2–4  : operational settings (setting_1, setting_2, setting_3)
  5–25 : sensor readings (T2 … W32)  [21 sensors, physical names]

Column names match the official C-MAPSS readme (Damage Propagation Modeling.pdf)
and are consistent with the IDA/EDA notebooks.

Usage
-----
>>> from particle_twin.data.loader import CMAPSSLoader
>>> loader = CMAPSSLoader(data_dir="data/cmapss")
>>> train_df = loader.load_train("FD001")
>>> test_df  = loader.load_test("FD001")
>>> rul_df   = loader.load_rul("FD001")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Raw C-MAPSS files have no header; column names follow the official readme.
_SENSOR_NAMES = [
    "T2",          # Total temperature at fan inlet (°R)
    "T24",         # Total temperature at LPC outlet (°R)
    "T30",         # Total temperature at HPC outlet (°R)
    "T50",         # Total temperature at LPT outlet (°R)
    "P2",          # Pressure at fan inlet (psia)
    "P15",         # Total pressure in bypass-duct (psia)
    "P30",         # Total pressure at HPC outlet (psia)
    "Nf",          # Physical fan speed (rpm)
    "Nc",          # Physical core speed (rpm)
    "epr",         # Engine pressure ratio (P50/P2)
    "Ps30",        # Static pressure at HPC outlet (psia)
    "phi",         # Ratio of fuel flow to Ps30
    "NRf",         # Corrected fan speed (rpm)
    "NRc",         # Corrected core speed (rpm)
    "BPR",         # Bypass ratio
    "farB",        # Burner fuel-air ratio
    "htBleed",     # Bleed enthalpy
    "Nf_dmd",      # Demanded fan speed (rpm)
    "PCNfR_dmd",   # Demanded corrected fan speed (rpm)
    "W31",         # HPT coolant bleed (lbm/s)
    "W32",         # LPT coolant bleed (lbm/s)
]

_COLUMNS = (
    ["unit_id", "cycle"]
    + ["setting_1", "setting_2", "setting_3"]
    + _SENSOR_NAMES
)

# Sensors confirmed informative by IDA (non-zero variance, carry degradation signal).
# Constant sensors excluded: T2, P2, P15, epr, farB, Nf_dmd, PCNfR_dmd.
INFORMATIVE_SENSORS = [
    "T24", "T30", "T50", "P30", "Nf", "Nc",
    "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32",
]

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

        Adds a 'rul' column: max_cycle_per_engine - current_cycle.
        Piece-wise linear RUL capping is NOT applied here — the state
        space model handles the health index transformation.

        Returns
        -------
        pd.DataFrame with columns: unit_id, cycle, setting_1-3,
                                   T2…W32 (21 sensors), rul
        """
        self._check_dataset(dataset)
        df = self._read_txt(f"train_{dataset}.txt")
        df = self._add_rul_train(df)
        return df

    def load_test(self, dataset: str) -> pd.DataFrame:
        """
        Load the test split for *dataset*.

        Test sequences are truncated (failure not observed).
        No 'rul' column is added; use load_rul() for ground-truth labels.

        Returns
        -------
        pd.DataFrame with columns: unit_id, cycle, setting_1-3, T2…W32
        """
        self._check_dataset(dataset)
        return self._read_txt(f"test_{dataset}.txt")

    def load_rul(self, dataset: str) -> pd.DataFrame:
        """
        Load the ground-truth RUL for every test engine in *dataset*.

        RUL_FD00X.txt contains one value per line; line i corresponds
        to test engine i+1 (unit_id is 1-indexed).

        Returns
        -------
        pd.DataFrame with columns: unit_id (1-indexed), true_rul
        """
        self._check_dataset(dataset)
        path = self.data_dir / f"RUL_{dataset}.txt"
        rul_values = pd.read_csv(path, header=None, names=["true_rul"])
        rul_values.index = rul_values.index + 1   # unit IDs start at 1
        rul_values.index.name = "unit_id"
        return rul_values.reset_index()

    def get_unit_ids(self, dataset: str, split: str = "test") -> list[int]:
        """Return sorted list of unit IDs in the given split."""
        if split == "train":
            df = self.load_train(dataset)
        else:
            df = self.load_test(dataset)
        return sorted(df["unit_id"].unique().tolist())

    def get_engine(self, dataset: str, unit_id: int,
                   split: str = "test") -> pd.DataFrame:
        """
        Return time-series data for a single engine.

        Parameters
        ----------
        dataset : e.g. 'FD001'
        unit_id : integer engine identifier (1-indexed)
        split   : 'train' or 'test'
        """
        if split == "train":
            df = self.load_train(dataset)
        else:
            df = self.load_test(dataset)
        engine_df = df[df["unit_id"] == unit_id].copy()
        if engine_df.empty:
            raise ValueError(
                f"unit_id {unit_id} not found in {dataset} {split} split."
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
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        return df

    @staticmethod
    def _add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RUL for the training set.
        RUL at cycle t for unit u = max_cycle_u - t.
        """
        max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycle, on="unit_id")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df.drop(columns=["max_cycle"], inplace=True)
        return df
