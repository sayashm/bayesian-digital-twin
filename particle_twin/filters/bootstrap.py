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

TODO (Day 3): implement BootstrapPF class.
"""

raise NotImplementedError("bootstrap.py will be implemented on Day 3.")
