"""
bootstrap.py — Bootstrap Particle Filter (SIR)
===============================================
Runs a per-engine Sequential Importance Resampling (SIR) particle filter
over the health-index state space defined in models/state_space.py.

Algorithm (Day 3):
  1. Initialise N particles from the prior p(h_0)
  2. For each cycle t:
     a. Predict:  propagate each particle through the transition model
     b. Weight:   update weights using the observation likelihood
     c. Resample: systematic resampling when ESS drops below threshold
  3. Return particle cloud at each timestep
"""

import pandas as pd
import numpy as np
from particle_twin.models.state_space import DegradationModel
from scipy.special import logsumexp


class BootstrapPF:
  def __init__(self, model: DegradationModel, n_particles=1000, ess_threshold=0.5, verbose=False):
    self.model = model
    self.n_particles = n_particles
    self.ess_threshold = ess_threshold
    self.verbose = verbose

    self.history = []

  def normalize(self, log_w):
      log_w_norm = log_w - logsumexp(log_w)
      return np.exp(log_w_norm)

  def ESS(self, log_w):
      w_norm = self.normalize(log_w)
      return 1.0 / np.sum(w_norm ** 2)

  def resample(self, h, log_w):
      w = self.normalize(log_w)
      cumw = np.cumsum(w)
      u = np.random.uniform(0, 1 / self.n_particles)
      positions = u + np.arange(self.n_particles) / self.n_particles
      idx = np.searchsorted(cumw, positions)
      h_new = h[idx]
      log_w_new = np.full(self.n_particles, -np.log(self.n_particles))
      return h_new, log_w_new

  def run(self, engine_df: pd.DataFrame):
    self.history = []
    engine_df = engine_df.sort_values('cycle')
    cycles = engine_df['cycle'].to_numpy()
    y = self.model.observe(df=engine_df).to_numpy()
    h = self.model.sample_initial(self.n_particles)
    log_w = np.full(self.n_particles, -np.log(self.n_particles))

    self.history.append({'cycle_number': 0, 'particles': h, 'weights': log_w, 'ESS': self.ESS(log_w)})

    for i, t in enumerate(cycles):
      h = self.model.transition(h)
      log_w = log_w + self.model.log_likelihood(h, y[i])
      if self.ESS(log_w=log_w) < self.ess_threshold * self.n_particles:
        h, log_w = self.resample(h, log_w)

      self.history.append({'cycle_number': t, 'particles': h, 'weights': log_w, 'ESS': self.ESS(log_w)})

      if self.verbose:
        print(f't: {t}, h_est = {np.average(h, weights=self.normalize(log_w)):.2f}, '
              f'ESS: {self.ESS(log_w):.2f}, N/2 = {self.n_particles * self.ess_threshold}')