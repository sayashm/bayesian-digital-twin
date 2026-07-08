"""
state_space.py — Degradation state space model (v2 — Health Index pipeline)
=============================================================================
State:        x_t   scalar Health Index (HI) in [0, 100]
              h_t = x_t / 100 in [0, 1],  1 = healthy, 0 = failed
Transition:   x_t = f_theta(x_{t-1}) + v_t,    v_t ~ N(0, sigma_v^2)
              f_theta is ONE of three learned shapes -- linear, polynomial,
              or exponential -- fitted by DegradationModelLearner on the
              fleet-pooled Health Index series (see features/health_index.py).
Observation:  y_t = x_t + w_t,                 w_t ~ p_v(.)
              y_t is the scalar Health Index built from raw sensors by
              HealthIndexBuilder ('weighted' or 'pca' — sensor-only, so it
              can be computed on train AND test data). p_v is the residual
              noise distribution learned by MeasurementModelLearner
              (gaussian / empirical / custom).

Why this replaces the Day-2 prototype
--------------------------------------
The original model (i) only supported a linear transition
h_t = h_{t-1} - delta + v_t, and (ii) observed the 21 raw C-MAPSS sensors
directly through per-sensor OLS regressions against a hand-built linear
time proxy tilde_h_t = 1 - t/T_max. That gave the thesis a single,
un-argued modelling choice at both the transition AND the observation
step. This version lets three degradation *shapes* be fit and compared
against data (Section 3.1), and reduces the observation step to a single,
interpretable Health Index rather than 21 separate sensor regressions,
with the noise model itself chosen by AIC/KS diagnostics rather than
assumed Gaussian.

C, d, sigma_w (the old per-sensor regression parameters) no longer exist:
the observation equation is now y = x + w instead of y = Cx + d + w,
because x IS the Health Index (same units as y), not an abstract [0,1]
index that needs a sensor-specific linear map.
"""

from __future__ import annotations

import numpy as np

from particle_twin.features.health_index import HealthIndexBuilder
from particle_twin.models.degradation_learner import DegradationModelLearner
from particle_twin.models.measurement_learner import MeasurementModelLearner


class DegradationModel:
    """
    Probabilistic degradation model for a turbofan engine.

    Parameters
    ----------
    hi_method : {'weighted', 'pca'}
        How the scalar Health Index (the observation y_t) is built from
        raw sensors. NOTE: 'industrial' is intentionally not accepted
        here — it requires the true RUL and so cannot serve as a live
        observation function; use HealthIndexBuilder('industrial')
        directly as a training-time reference/diagnostic curve instead.
    degradation_model : {'linear', 'polynomial', 'exponential'}
        Shape of the mean degradation trajectory E[X_t] (the transition).
        Default 'exponential' — chosen for physical reasoning (wear-out
        failure processes accelerate) and supported by the training-only
        'industrial' (true-RUL) HI, which is fit best by exponential
        (R^2=0.82 vs 0.75 linear/polynomial). The default hi_method
        'weighted' is the better-fitting sensor-only pairing for
        exponential specifically (R^2=0.442 vs 0.409 for pca+exponential
        on FD001) — see thesis/chapt3.tex Sec. 3.1 model-selection notes
        and experiments/test_state_space.py for the full comparison.
    measurement_method : {'gaussian', 'empirical', 'custom'}
        Distribution family for the observation noise w_t = Y_t - E[X_t].
    sigma_v : float
        Process noise std, in HI units (0-100 scale), added at each
        transition step. Free parameter — see Section 3.4 (PMMH).
    sigma_0 : float
        Std dev of the initial health distribution (h-scale, 0-1).
    sensor_cols : list[str] or None
        Sensors combined into the Health Index. Defaults to a fixed
        informative subset identified during IDA if not provided.
    """

    DEFAULT_SENSORS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']
    _VALID_HI_METHODS = ('weighted', 'pca')

    def __init__(self, hi_method: str = 'weighted', degradation_model: str = 'exponential',
                 measurement_method: str = 'gaussian', sigma_v: float = 0.3,
                 sigma_0: float = 0.05, sensor_cols: list[str] | None = None):
        if hi_method not in self._VALID_HI_METHODS:
            raise ValueError(
                f"hi_method must be one of {self._VALID_HI_METHODS}. "
                "'industrial' requires the true RUL and cannot be used as a "
                "live observation function — build it separately via "
                "HealthIndexBuilder('industrial') for diagnostic comparison only."
            )

        self.hi_method = hi_method
        self.degradation_model = degradation_model
        self.measurement_method = measurement_method
        self.sigma_v = sigma_v
        self.sigma_0 = sigma_0
        self.sensor_cols = sensor_cols or list(self.DEFAULT_SENSORS)

        # set by fit() — None until training data is provided
        self.hi_builder: HealthIndexBuilder | None = None
        self.degradation_learner: DegradationModelLearner | None = None
        self.measurement_learner: MeasurementModelLearner | None = None
        self.fit_report: dict | None = None

    # ------------------------------------------------------------------
    # Fitting: build HI, fit dynamics, fit measurement noise
    # ------------------------------------------------------------------

    def fit(self, train_df) -> "DegradationModel":
        """
        Fit the full pipeline on run-to-failure training data.

          1. Build the scalar Health Index Y_t for every (engine, cycle)
             row via HealthIndexBuilder (self.hi_method), fit on train_df.
          2. Fit the fleet-pooled mean degradation trajectory E[X_t]
             (self.degradation_model shape) on the pooled (cycle, HI)
             observations across all engines via DegradationModelLearner.
          3. Fit the observation noise model on residuals Y_t - E[X_t]
             via MeasurementModelLearner (self.measurement_method).

        Stores: self.hi_builder, self.degradation_learner,
                self.measurement_learner, self.fit_report
        """
        self.hi_builder = HealthIndexBuilder(
            self.sensor_cols, id_col='unit_id', time_col='cycle', method=self.hi_method
        ).fit(train_df)
        hi = self.hi_builder.transform(train_df).to_numpy()
        cycles = train_df['cycle'].to_numpy(dtype=float)

        self.degradation_learner = DegradationModelLearner(
            health_series=hi, time=cycles,
            model_type=self.degradation_model, sigma_v=self.sigma_v,
        )

        residuals = hi - self.degradation_learner.predict(cycles)
        self.measurement_learner = MeasurementModelLearner(
            residuals=residuals, method=self.measurement_method
        )

        self.fit_report = {
            'hi_method': self.hi_method,
            'degradation_model': self.degradation_learner.get_fit_quality(),
            'measurement_method': self.measurement_learner.get_fit_quality(),
        }
        return self

    # ------------------------------------------------------------------
    # Three methods the bootstrap particle filter will call
    # ------------------------------------------------------------------

    def sample_initial(self, n_particles: int) -> np.ndarray:
        """
        Draw initial health samples: h_0^(i) ~ N(1, sigma_0^2)
        Returns array of shape (n_particles,), values in [0, 1].
        """
        return np.random.normal(1, self.sigma_0, n_particles)

    def transition(self, h_prev: np.ndarray) -> np.ndarray:
        """
        Propagate particles one cycle forward via the learned dynamics
        f_theta (linear / polynomial / exponential), applied on the HI
        (0-100) scale internally and returned on the h (0-1) scale.

        Parameters
        ----------
        h_prev : shape (n_particles,), values in [0, 1]

        Returns
        -------
        h_next : shape (n_particles,), values in [0, 1]
        """
        if self.degradation_learner is None:
            raise RuntimeError("Call .fit(train_df) before .transition().")
        hi_prev = np.clip(h_prev, 0.0, 1.0) * 100.0
        hi_next = self.degradation_learner.dynamics_func(hi_prev, dt=1)
        return np.clip(hi_next / 100.0, 0.0, 1.0)

    def log_likelihood(self, h: np.ndarray, y: float) -> np.ndarray:
        """
        Evaluate log p(y_t | h_t^(i)) for each particle.

        The observation model is y = x + w  (w ~ learned noise p_v), so
            log p(y | h) = log p_v(y*100 - h*100)

        Parameters
        ----------
        h : shape (n_particles,)  -- current health of each particle, in [0, 1]
        y : float                 -- observed Health Index at this cycle,
                                     on the SAME [0, 1] scale as h (i.e. the
                                     raw HealthIndexBuilder output / 100).

        Returns
        -------
        log_liks : shape (n_particles,)
        """
        if self.measurement_learner is None:
            raise RuntimeError("Call .fit(train_df) before .log_likelihood().")
        hi = np.clip(h, 0.0, 1.0) * 100.0
        residual = (y * 100.0) - hi
        log_lik_func = self.measurement_learner.get_log_likelihood_function()
        return log_lik_func(residual)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def observe(self, df):
        """Compute the scalar Health Index observation (h-scale, [0,1]) for
        arbitrary new rows (train or test) using the fitted HI builder."""
        if self.hi_builder is None:
            raise RuntimeError("Call .fit(train_df) before .observe().")
        return self.hi_builder.transform(df) / 100.0
