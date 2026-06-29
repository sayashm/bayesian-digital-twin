"""
state_space.py — Degradation state space model
================================================
Defines the transition (process) and observation equations for the
turbofan health-index model.  Implemented on Day 2.

State:  h_t  (scalar health index ∈ [0, 1], 1 = healthy, 0 = failed)
Obs:    y_t  (vector of informative sensor readings at cycle t)

Transition:  h_t = h_{t-1} - delta + v_t,   v_t ~ N(0, sigma_v^2)
Observation: y_t = g(h_t) + w_t,            w_t ~ N(0, Sigma_w)

where delta is the mean degradation rate and g(.) maps health to
expected sensor values (linear approximation from training data).

TODO (Day 2): implement DegradationModel class.
"""

raise NotImplementedError("state_space.py will be implemented on Day 2.")
