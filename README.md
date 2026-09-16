# Bayesian Digital Twin for Predictive Maintenance

A particle-filtering (Sequential Monte Carlo) framework for building Bayesian
digital twins of turbofan engines, applied to anomaly detection and Remaining
Useful Life (RUL) estimation on the NASA C-MAPSS dataset.

This repository contains the implementation only. The written thesis
("A Bayesian Approach to Digital Twins with Applications to Predictive
Maintenance", Ghent University, Advanced Master's in Statistical Data
Analysis, 2026) is kept private — feel free to reach out if you'd like to
read it.

## Overview

Each engine's health is tracked as a scalar Health Index through a nonlinear
state-space model:

- **Health Index construction** (`particle_twin/features/`) — builds a scalar
  health signal from raw sensor channels.
- **Degradation & measurement models** (`particle_twin/models/`) — fit the
  state transition and observation models per engine / fleet.
- **Bootstrap particle filter** (`particle_twin/filters/`) — sequential Monte
  Carlo filtering of the Health Index.
- **RUL extraction** (`particle_twin/analysis/`) — forward-simulates
  particles against a random failure threshold to get a full posterior over
  Remaining Useful Life, not just a point estimate.
- **PMMH calibration** (`particle_twin/inference/`) — Particle Marginal
  Metropolis-Hastings for per-engine parameter estimation.
- **Similarity library** (`particle_twin/library/`) — a one-shot, MMD-matched
  library of fitted degradation rates as an alternative to pooled/PMMH
  parameters.

`experiments/` contains the scripts and notebooks used to run and evaluate
the three model configurations (fixed pooled parameters / per-engine PMMH /
similarity library) on the FD001 and FD003 C-MAPSS sub-datasets.

## Stack

Python, NumPy/SciPy/pandas, scikit-learn, particle filtering / SMC, PMMH.

## Author

Sajjad Ayashm — Advanced Master's in Statistical Data Analysis, Ghent
University (2026).
