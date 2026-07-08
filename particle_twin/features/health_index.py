"""
health_index.py — Health Index (HI) construction from raw sensors
====================================================================
Turns a vector of raw C-MAPSS sensor readings into a single scalar Health
Index HI_t, on a 0-100 scale: HI_t = 100 -> fully healthy, HI_t = 0 ->
failure. This scalar HI_t plays the role of the noisy observation Y_t in
the state-space model (see particle_twin/models/state_space.py): the
degradation dynamics model fits the mean trajectory E[X_t] of HI_t, and
the measurement model characterises the residual noise Y_t - E[X_t].

Three construction methods are supported. They are NOT interchangeable
in one important respect — whether they can be evaluated at *test* time,
when an engine's true remaining life is unknown:

  'weighted'   Sensor-only. Each sensor is min-max normalised to [0,100]
               (range learned from the training fleet), oriented so it
               decreases as the engine degrades, then combined in a
               weighted average. Because it only ever looks at the
               sensor values of the current row, it can be applied to
               train AND test data with .transform().

  'pca'        Sensor-only. Sensors are z-scored (mean/std learned from
               the training fleet) and projected onto the first
               principal component, oriented and rescaled to [0,100].
               Also sensor-only -> usable with .transform() on test data.

  'industrial' Requires the true total life of the engine (RUL_t and
               L_u = total run length). This is only known for
               run-to-failure (training) sequences, or for test engines
               if the true life is supplied externally (oracle mode,
               e.g. to build a reference/ground-truth curve for
               comparison). It CANNOT be used as a live observation
               function during inference on censored test sequences,
               since RUL is precisely the quantity the particle filter
               is trying to estimate. Use this method to build a
               reference degradation-shape curve on the training fleet
               (e.g. to sanity-check which degradation_model shape --
               linear / polynomial / exponential -- best matches a
               formula-based prognostics HI), not as the pipeline's Y_t.
               Formula: HI_t = 100*(1 - exp(ln(beta)*RUL_t / ((1-beta)*L_u))),
               adapted from Eq. (3) in Zhevnenko & Makarov, "Optimizing
               State Monitoring With Domain Degradation Knowledge",
               IEEE Access, 2025 (doi:10.1109/ACCESS.2025.3573683) --
               see thesis chapt3.tex Sec. 3.1 / Thesis_bib.bib
               (zhevnenko2025ihie) for the full citation.

Usage
-----
>>> builder = HealthIndexBuilder(sensor_cols, method='pca').fit(train_df)
>>> train_df['HI'] = builder.transform(train_df)
>>> test_df['HI']  = builder.transform(test_df)     # 'weighted'/'pca' only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class HealthIndexBuilder:
    """
    Build a scalar Health Index (HI) in [0, 100] from raw sensor readings.

    Parameters
    ----------
    sensor_cols : list[str]
        Columns to combine into the HI.
    id_col : str
        Engine/unit identifier column.
    time_col : str
        Cycle index column.
    method : {'weighted', 'pca', 'industrial'}
    """

    METHODS = ("weighted", "pca", "industrial")

    def __init__(self, sensor_cols, id_col: str = "unit_id",
                 time_col: str = "cycle", method: str = "weighted"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.sensor_cols = list(sensor_cols)
        self.id_col = id_col
        self.time_col = time_col
        self.method = method

        self._fitted = False
        self.fit_params_: dict = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, weights=None, beta: float = 0.10) -> "HealthIndexBuilder":
        """Learn whatever normalisation/projection the method needs from train_df."""
        if self.method == "weighted":
            self._fit_weighted(train_df, weights)
        elif self.method == "pca":
            self._fit_pca(train_df)
        elif self.method == "industrial":
            self._fit_industrial(beta)
        self._fitted = True
        return self

    def _fit_weighted(self, df, weights):
        mins = df[self.sensor_cols].min()
        maxs = df[self.sensor_cols].max()
        span = (maxs - mins).replace(0, 1.0)  # guard against constant sensors

        # Normalise to [0, 100] using train-fleet range, then check
        # correlation with cycle to decide which sensors need flipping so
        # that ALL normalised sensors decrease as the engine degrades.
        normed = (df[self.sensor_cols] - mins) / span * 100.0
        flip = {}
        for col in self.sensor_cols:
            corr = np.corrcoef(normed[col], df[self.time_col])[0, 1]
            flip[col] = bool(corr > 0)  # increases with cycle -> flip

        if weights is None:
            weights = np.ones(len(self.sensor_cols)) / len(self.sensor_cols)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        self.fit_params_ = {
            "mins": mins, "span": span, "flip": flip, "weights": weights,
        }

    def _fit_pca(self, df):
        means = df[self.sensor_cols].mean()
        stds = df[self.sensor_cols].std().replace(0, 1.0)
        standardised = (df[self.sensor_cols] - means) / stds

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(standardised)[:, 0]

        flip = bool(np.corrcoef(pc1, df[self.time_col])[0, 1] > 0)
        if flip:
            pc1 = -pc1

        self.fit_params_ = {
            "means": means, "stds": stds, "pca": pca,
            "flip": flip, "pc1_min": pc1.min(), "pc1_max": pc1.max(),
        }

    def _fit_industrial(self, beta):
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie strictly in (0, 1).")
        self.fit_params_ = {"beta": beta}

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame, true_life: "pd.Series | dict | None" = None) -> pd.Series:
        """
        Compute HI_t for every row of df using the parameters learned in fit().

        Parameters
        ----------
        df : pd.DataFrame
            Must contain sensor_cols (weighted/pca) or id_col/time_col
            (industrial).
        true_life : optional, only used by 'industrial'
            Mapping unit_id -> true total run length L_u. Required if df
            is a censored (test) split; not needed for a run-to-failure
            (train) split, where L_u is simply the max observed cycle.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(train_df) before .transform().")

        if self.method == "weighted":
            return self._transform_weighted(df)
        elif self.method == "pca":
            return self._transform_pca(df)
        else:
            return self._transform_industrial(df, true_life)

    def _transform_weighted(self, df):
        p = self.fit_params_
        normed = (df[self.sensor_cols] - p["mins"]) / p["span"] * 100.0
        normed = normed.clip(lower=0.0, upper=100.0)
        for col, was_flipped in p["flip"].items():
            if was_flipped:
                normed[col] = 100.0 - normed[col]
        hi = np.average(normed.values, axis=1, weights=p["weights"])
        return pd.Series(hi, index=df.index, name="HI")

    def _transform_pca(self, df):
        p = self.fit_params_
        standardised = (df[self.sensor_cols] - p["means"]) / p["stds"]
        pc1 = p["pca"].transform(standardised)[:, 0]
        if p["flip"]:
            pc1 = -pc1
        hi = (pc1 - p["pc1_min"]) / (p["pc1_max"] - p["pc1_min"]) * 100.0
        hi = np.clip(hi, 0.0, 100.0)
        return pd.Series(hi, index=df.index, name="HI")

    def _transform_industrial(self, df, true_life):
        beta = self.fit_params_["beta"]
        d = df.copy()

        if true_life is not None:
            life = d[self.id_col].map(true_life)
            if life.isna().any():
                raise ValueError("true_life is missing entries for some unit_id values in df.")
            d["L_u"] = life.values
        else:
            d["L_u"] = d.groupby(self.id_col)[self.time_col].transform("max")

        d["RUL"] = d["L_u"] - d[self.time_col]
        exponent = np.log(beta) * d["RUL"] / ((1.0 - beta) * d["L_u"])
        hi = (1.0 - np.exp(exponent)) * 100.0
        return pd.Series(hi.values, index=df.index, name="HI")

    def fit_transform(self, train_df, **kwargs) -> pd.Series:
        return self.fit(train_df, **kwargs).transform(train_df)
