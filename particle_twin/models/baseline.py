"""
baseline.py — Simple linear regression baseline for RUL (Day 9)
====================================================================
A non-Bayesian point-estimate comparison for the particle filter: one
ordinary least squares fit, pooled across the whole training fleet,
regressing RUL directly on the same sensor readings the PF's health
index is built from (SENSOR_COLS) -- NOT the latent health index
itself, since the baseline has no state-space model or filtering step
at all.

    RUL_t ~= beta_0 + sum_i beta_i * sensor_i,t

Fit by ordinary least squares (minimising sum of squared residuals) on
every (cycle, sensor readings, rul) row across ALL training engines,
pooled together -- same "pool across the fleet" philosophy as
DegradationModelLearner (models/degradation_learner.py), so the
§4.6 comparison isolates the modelling architecture (state-space +
particle filter vs. flat regression), not a difference in how much
data each model got to see.

At test time, predict_engine() returns ONE point estimate from the
sensor reading at the engine's LAST observed cycle -- matching exactly
what the particle filter sees (data up to the last observed cycle,
nothing from the future). No distribution, no credible interval: that
missing piece is precisely the gap the thesis argues the particle
filter closes.

Known simplification (deliberate, for scope -- see Thesis_claude.md
§7 "do not let Sajjad go down a rabbit hole"): no piecewise-linear RUL
capping is applied (the common C-MAPSS-literature trick of capping
training RUL at ~125 cycles, since a raw linear regression struggles
with the near-flat early-life RUL curve). loader.py's load_train()
already returns an uncapped 'rul' column for the same reason the PF
pipeline doesn't cap it -- see the note there. This means the baseline
may predict physically-impossible negative RUL for some engines; that
is an honest, reportable limitation for §4.6, not a bug to silently
patch here.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRULBaseline:
    """
    Ordinary least squares baseline: RUL regressed directly on sensor
    readings, pooled across the training fleet.

    Parameters
    ----------
    sensor_cols : list[str] -- which sensor columns to use as features.
        Use the SAME list the particle filter's health index is built
        from (see e.g. run_full_fd001.py's SENSOR_COLS), so both models
        see identical information -- a fair comparison.
    """

    def __init__(self, sensor_cols: list[str]):
        self.sensor_cols = sensor_cols
        self.model = LinearRegression()

    def fit(self, train_df: pd.DataFrame) -> "LinearRULBaseline":
        """
        Fit on every (cycle, sensor readings, rul) row across all
        training engines, pooled together. train_df must already have
        an 'rul' column -- see CMAPSSLoader.load_train().
        """
        X = train_df[self.sensor_cols].values
        y = train_df['rul'].values
        self.model.fit(X, y)
        return self

    def predict_engine(self, engine_df: pd.DataFrame) -> float:
        """
        Point-estimate RUL prediction for one engine, from its LAST
        observed cycle's sensor readings only (matches what the
        particle filter sees -- data up to the last observed cycle,
        no future information).
        """
        last_row = engine_df.iloc[[-1]]
        X = last_row[self.sensor_cols].values
        return float(self.model.predict(X)[0])
