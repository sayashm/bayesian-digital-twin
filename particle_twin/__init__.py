"""
particle_twin
=============
A Bayesian digital twin for turbofan engine prognostics using Sequential Monte Carlo.

Package layout
--------------
data/         -- C-MAPSS data loading and SQLite result storage
models/       -- State space model (transition + observation equations)
filters/      -- Bootstrap particle filter (per-engine)
inference/    -- PMMH parameter estimation
analysis/     -- RUL extraction and evaluation metrics
visualization/-- Thesis-quality figures
"""

__version__ = "0.1.0"
__author__ = "Sajjad Ayashmand"
