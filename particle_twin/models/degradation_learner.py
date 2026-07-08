"""
degradation_learner.py — Learn the mean degradation trajectory E[X_t]
=========================================================================
Fits the DETERMINISTIC mean trajectory of a Health Index (HI) time series
using one of three shapes, and returns:

  1. A `dynamics_func(health, dt=1)` — an autonomous (Markov) one-step
     transition used as the particle filter's transition model:
         X_t = f_theta(X_{t-1}) + v_t,   v_t ~ N(0, sigma_v^2)
     f_theta is the deterministic mean shape fitted here; sigma_v is the
     process-noise scale (a free parameter — see `sigma_v` below and
     Section 3.4, estimated later via PMMH).

  2. A `predict(t)` method — the smooth fitted curve evaluated at
     arbitrary (possibly pooled, unsorted) cycle values, used to compute
     residuals Y_t - E[X_t] for MeasurementModelLearner.

Three model shapes
-------------------
  'linear'      : health(t) = h0 - rate * t                  (constant rate)
  'polynomial'  : health(t) = h0 - (base + accel * t) * t    (accelerating)
  'exponential' : health(t) = 100 - D * exp(k * t)           (wear-out)

Author: Thesis Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


class DegradationModelLearner:
    """
    Learn a degradation dynamics model from a Health Index time series.

    Parameters
    ----------
    health_series : array-like
        HI values over time (0-100 scale), e.g. one engine's trajectory,
        or a *pooled* concatenation across many engines (in which case
        `time` must be supplied explicitly — see below).
    model_type : {'linear', 'polynomial', 'exponential'}
    time : array-like or None
        Cycle number for each entry of health_series. If None, assumes a
        single clean per-cycle sequence and uses time = 0, 1, ..., n-1
        (original single-engine use case). Pass explicit cycle numbers to
        fit one SHARED fleet-level dynamics model on pooled
        (engine, cycle, HI) observations from many engines at once.
    sigma_v : float
        Process noise std added at each transition step (models the fact
        that degradation does not follow the fitted mean shape exactly).
        This was a hardcoded magic number (0.3) in the original prototype;
        it is now an explicit, inspectable parameter that can later be
        refined via PMMH (Section 3.4).
    """

    def __init__(self, health_series, model_type: str = "exponential",
                 time=None, sigma_v: float = 0.3):
        self.health_series = np.asarray(health_series, dtype=float)
        self.n_cycles = len(self.health_series)
        self.time = np.arange(self.n_cycles, dtype=float) if time is None else np.asarray(time, dtype=float)
        if len(self.time) != len(self.health_series):
            raise ValueError("time and health_series must have the same length.")

        self.model_type = model_type
        self.sigma_v = sigma_v
        self.params = None
        self.fit_quality = None
        self.dynamics_func = None

        self._fit_model()

    def _fit_model(self):
        if self.model_type == "linear":
            self._fit_linear()
        elif self.model_type == "polynomial":
            self._fit_polynomial()
        elif self.model_type == "exponential":
            self._fit_exponential()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ============================================================
    # MODEL 1: LINEAR — health(t) = h0 - rate*t
    # ============================================================

    def _fit_linear(self):
        slope, intercept, r_value, p_value, std_err = linregress(self.time, self.health_series)

        degradation_rate = -slope
        initial_health = intercept

        self.params = {"initial_health": initial_health, "degradation_rate": degradation_rate}
        self.fit_quality = {
            "r_squared": r_value ** 2,
            "model_type": "linear",
            "equation": f"health(t) = {initial_health:.2f} - {degradation_rate:.4f}*t",
        }

        sigma_v = self.sigma_v

        def linear_dynamics(health, dt=1):
            noise = np.random.normal(0, sigma_v, size=np.shape(health))
            new_health = health - degradation_rate * dt + noise
            return np.clip(new_health, 0, 100)

        self.dynamics_func = linear_dynamics

    # ============================================================
    # MODEL 2: POLYNOMIAL — health(t) = h0 - (base + accel*t)*t
    # ============================================================

    def _fit_polynomial(self):
        coeffs = np.polyfit(self.time, self.health_series, 2)
        poly_model = np.poly1d(coeffs)

        const_term, linear_coeff, quadratic_coeff = coeffs[2], coeffs[1], coeffs[0]
        base_rate = -linear_coeff
        accel_coeff = -2 * quadratic_coeff

        self.params = {
            "initial_health": const_term,
            "base_rate": base_rate,
            "acceleration_coefficient": accel_coeff,
        }

        y_pred = poly_model(self.time)
        ss_res = np.sum((self.health_series - y_pred) ** 2)
        ss_tot = np.sum((self.health_series - np.mean(self.health_series)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        self.fit_quality = {
            "r_squared": r_squared,
            "model_type": "polynomial",
            "equation": f"health(t) = {const_term:.2f} - ({base_rate:.4f} + {accel_coeff:.6f}*t)*t",
        }

        sigma_v = self.sigma_v

        def polynomial_dynamics(health, dt=1):
            """
            Autonomous approximation: acceleration is driven by the current
            health level (100 - health, normalised) as a proxy for elapsed
            life-fraction, since the particle filter transition must be a
            function of the current state, not absolute cycle number.
            """
            base_degrad = base_rate * dt
            health_remaining_normalized = (100 - health) / 100
            accel_degrad = accel_coeff * health_remaining_normalized * dt
            noise = np.random.normal(0, sigma_v, size=np.shape(health))
            new_health = health - (base_degrad + accel_degrad) + noise
            return np.clip(new_health, 0, 100)

        self.dynamics_func = polynomial_dynamics

    # ============================================================
    # MODEL 3: EXPONENTIAL — health(t) = 100 - D*exp(k*t)
    # ============================================================

    def _fit_exponential(self):
        damage = 100 - self.health_series
        damage_safe = np.maximum(damage, 0.1)
        log_damage = np.log(damage_safe)

        slope, intercept, r_value, p_value, std_err = linregress(self.time, log_damage)
        growth_rate = slope
        D = np.exp(intercept)

        self.params = {"initial_health": 100.0, "D": D, "growth_rate": growth_rate}
        self.fit_quality = {
            "r_squared": r_value ** 2,
            "model_type": "exponential",
            "equation": f"health(t) = 100 - {D:.4f} * exp({growth_rate:.6f}*t)",
            "growth_rate": growth_rate,
        }

        sigma_v = self.sigma_v

        def exponential_dynamics(health, dt=1):
            """Damage grows proportional to current damage (accelerating wear).
            Fully autonomous: depends only on the current health, not on t."""
            damage = 100 - health
            new_damage = damage * np.exp(growth_rate * dt)
            noise = np.random.normal(0, sigma_v, size=np.shape(health))
            return np.clip(100 - new_damage + noise, 0, 100)

        self.dynamics_func = exponential_dynamics

    # ============================================================
    # PUBLIC METHODS
    # ============================================================

    def predict(self, t):
        """Evaluate the smooth fitted mean curve E[X_t] at arbitrary cycle values."""
        t = np.asarray(t, dtype=float)
        if self.model_type == "linear":
            return self.params["initial_health"] - self.params["degradation_rate"] * t
        elif self.model_type == "polynomial":
            h0, base, accel = self.params["initial_health"], self.params["base_rate"], self.params["acceleration_coefficient"]
            return h0 - (base + accel * t) * t
        elif self.model_type == "exponential":
            D, k = self.params["D"], self.params["growth_rate"]
            return 100 - D * np.exp(k * t)

    def get_dynamics_function(self):
        """Return dynamics_func(health, dt=1) -> new_health for the particle filter."""
        return self.dynamics_func

    def get_parameters(self):
        return self.params

    def get_fit_quality(self):
        return self.fit_quality

    def simulate_trajectory(self, initial_health=100, max_cycles=300, failure_threshold=10):
        trajectory = [initial_health]
        health = initial_health
        for _ in range(max_cycles):
            health = self.dynamics_func(health, dt=1)
            trajectory.append(health)
            if health < failure_threshold:
                break
        return np.array(trajectory)

    def plot_fit(self, figsize=(14, 5)):
        order = np.argsort(self.time)
        t_sorted = self.time[order]
        fitted_sorted = self.predict(t_sorted)

        degradation_rate = np.abs(np.diff(self.health_series[order]))

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax = axes[0]
        ax.plot(self.time, self.health_series, "ko", alpha=0.4, markersize=3, label="Actual (HI) data")
        ax.plot(t_sorted, fitted_sorted, "b-", linewidth=2, label=f"Fitted ({self.model_type})")
        ax.axhline(y=10, color="r", linestyle="--", alpha=0.5, label="Failure threshold")
        ax.set_xlabel("Cycles"); ax.set_ylabel("Health index (HI)")
        ax.set_title(f"{self.model_type.capitalize()} Model Fit")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        residuals = self.health_series - self.predict(self.time)
        ax.plot(self.time, residuals, "go", alpha=0.5, markersize=3)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Cycles"); ax.set_ylabel("Residual (Actual - Fitted)")
        ax.set_title(f"Residuals (R² = {self.fit_quality['r_squared']:.4f})")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(t_sorted[:-1], degradation_rate, "mo", alpha=0.5, markersize=3, label="Actual rate")
        if self.model_type == "linear":
            fitted_rate = np.full_like(t_sorted[:-1], self.params["degradation_rate"])
        elif self.model_type == "polynomial":
            fitted_rate = self.params["base_rate"] + self.params["acceleration_coefficient"] * t_sorted[:-1]
        else:
            D, k = self.params["D"], self.params["growth_rate"]
            fitted_rate = D * k * np.exp(k * t_sorted[:-1])
        ax.plot(t_sorted[:-1], fitted_rate, "b-", linewidth=2, label="Fitted rate")
        ax.set_xlabel("Cycles"); ax.set_ylabel("Degradation rate (HI/cycle)")
        ax.set_title("Degradation Rate Over Time")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def print_summary(self):
        print("=" * 70)
        print(f"DEGRADATION MODEL LEARNER - {self.model_type.upper()}")
        print("=" * 70)
        print(f"\nData Summary:\n  n observations: {self.n_cycles}")
        print(f"  Model Parameters:")
        for key, value in self.params.items():
            print(f"    {key}: {value:.6f}")
        print(f"  Fit Quality:")
        for key, value in self.fit_quality.items():
            if key != "equation":
                print(f"    {key}: {value}")
        print(f"  Model Equation:\n    {self.fit_quality['equation']}")
        print("=" * 70)
