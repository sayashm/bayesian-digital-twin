"""
measurement_learner.py — Learn the observation noise model from residuals
============================================================================
Characterises the measurement noise v_t between the noisy Health Index
observations Y_t (from HealthIndexBuilder) and the smooth degradation
mean-trajectory fit E[X_t] (from DegradationModelLearner):

    Y_t = X_t + v_t   =>   residuals = Y_t - E[X_t]  ~  samples of v_t

This class fits p(v_t) using one of three approaches and returns a
likelihood function p(y|x) = p_v(y - x) for use in the particle filter's
weight-update step.

Author: Thesis Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class MeasurementModelLearner:
    """
    Learn measurement/observation noise model from residuals.

    Supports three approaches:
    1. 'gaussian'  : v_t ~ N(mu, sigma^2)                    (parametric, 2 params)
    2. 'empirical' : v_t ~ KDE(residuals)                    (nonparametric)
    3. 'custom'    : v_t ~ best of {norm, t, skewnorm, laplace} by AIC
    """

    CUSTOM_CANDIDATES = ("norm", "t", "skewnorm", "laplace")

    def __init__(self, residuals, method: str = "gaussian"):
        self.residuals = np.asarray(residuals, dtype=float)
        self.n = len(self.residuals)
        self.method = method

        self.params = None
        self.fit_quality = None
        self.likelihood_func = None
        self.log_likelihood_func = None

        self.residual_stats = {
            "mean": np.mean(self.residuals),
            "std": np.std(self.residuals, ddof=1),
            "skewness": stats.skew(self.residuals),
            "excess_kurtosis": stats.kurtosis(self.residuals),
        }

        self._fit_model()

    def _fit_model(self):
        if self.method == "gaussian":
            self._fit_gaussian()
        elif self.method == "empirical":
            self._fit_empirical()
        elif self.method == "custom":
            self._fit_custom()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ============================================================
    # METHOD 1: GAUSSIAN
    # ============================================================

    def _fit_gaussian(self):
        mu, sigma = stats.norm.fit(self.residuals)
        sigma = max(sigma, 1e-8)
        self.params = {"mu": mu, "sigma": sigma}

        log_lik = np.sum(stats.norm.logpdf(self.residuals, mu, sigma))
        k = 2
        aic = 2 * k - 2 * log_lik
        # NOTE: pass the CDF as a callable rather than kstest(..., args=(mu, sigma)).
        # Some SciPy builds raise "ndtr() takes from 1 to 2 positional arguments
        # but 3 were given" when args= dispatches through the array-api backend;
        # the callable form sidesteps that internal dispatch entirely.
        ks_stat, ks_p = stats.kstest(self.residuals, lambda x: stats.norm.cdf(x, mu, sigma))

        self.fit_quality = {
            "method": "gaussian", "log_likelihood": log_lik, "aic": aic,
            "ks_stat": ks_stat, "ks_pvalue": ks_p,
            "equation": f"v ~ N({mu:.4f}, {sigma:.4f}^2)",
        }

        self.likelihood_func = lambda residual: stats.norm.pdf(residual, mu, sigma)
        self.log_likelihood_func = lambda residual: stats.norm.logpdf(residual, mu, sigma)

    # ============================================================
    # METHOD 2: EMPIRICAL (KDE)
    # ============================================================

    def _fit_empirical(self, bw_method=None):
        kde = stats.gaussian_kde(self.residuals, bw_method=bw_method)
        self._kde = kde

        bandwidth = kde.factor * self.residuals.std(ddof=1)
        self.params = {"bandwidth": bandwidth, "n_samples": self.n}

        log_lik = np.sum(kde.logpdf(self.residuals))
        self.fit_quality = {
            "method": "empirical", "log_likelihood": log_lik,
            "aic": None, "ks_stat": None, "ks_pvalue": None,
            "equation": f"v ~ KDE(n={self.n}, bandwidth={bandwidth:.4f})",
        }

        self.likelihood_func = lambda residual: kde.evaluate(residual)
        self.log_likelihood_func = lambda residual: kde.logpdf(residual)

    # ============================================================
    # METHOD 3: CUSTOM (best parametric non-Gaussian by AIC)
    # ============================================================

    def _fit_custom(self, candidates=None):
        if candidates is None:
            candidates = self.CUSTOM_CANDIDATES

        results = {}
        for name in candidates:
            dist = getattr(stats, name)
            try:
                params_d = dist.fit(self.residuals)
            except Exception:
                continue
            log_lik = np.sum(dist.logpdf(self.residuals, *params_d))
            k = len(params_d)
            aic = 2 * k - 2 * log_lik
            # See NOTE in _fit_gaussian -- callable CDF avoids the args= dispatch bug.
            ks_stat, ks_p = stats.kstest(self.residuals, lambda x, p=params_d: dist.cdf(x, *p))
            results[name] = {"params": params_d, "log_likelihood": log_lik,
                              "aic": aic, "ks_stat": ks_stat, "ks_pvalue": ks_p}

        if not results:
            raise RuntimeError("No candidate distribution could be fit.")

        best_name = min(results, key=lambda nm: results[nm]["aic"])
        best = results[best_name]
        dist = getattr(stats, best_name)

        self.params = {"distribution": best_name, "params": best["params"]}
        self.fit_quality = {
            "method": "custom", "best_distribution": best_name,
            "log_likelihood": best["log_likelihood"], "aic": best["aic"],
            "ks_stat": best["ks_stat"], "ks_pvalue": best["ks_pvalue"],
            "all_candidates_aic": {nm: r["aic"] for nm, r in results.items()},
            "equation": f"v ~ {best_name}{tuple(round(p, 4) for p in best['params'])}",
        }

        self.likelihood_func = lambda residual: dist.pdf(residual, *best["params"])
        self.log_likelihood_func = lambda residual: dist.logpdf(residual, *best["params"])

    # ============================================================
    # PUBLIC METHODS
    # ============================================================

    def get_likelihood_function(self):
        """p(y|x) = p_v(y - x). Use as: w_i = f(y_t - x_t_particle_i)."""
        return self.likelihood_func

    def get_log_likelihood_function(self):
        """Log version -- preferred for particle filter weight updates."""
        return self.log_likelihood_func

    def get_parameters(self):
        return self.params

    def get_fit_quality(self):
        return self.fit_quality

    def plot_fit(self, figsize=(14, 5)):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax = axes[0]
        ax.hist(self.residuals, bins=30, density=True, alpha=0.5, color="gray", label="Residuals")
        x_grid = np.linspace(self.residuals.min(), self.residuals.max(), 300)
        ax.plot(x_grid, self.likelihood_func(x_grid), "b-", linewidth=2, label=f"Fitted ({self.method})")
        ax.set_xlabel("Residual (Y - E[X])"); ax.set_ylabel("Density")
        ax.set_title(f"{self.method.capitalize()} Noise Model")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        stats.probplot(self.residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot vs Normal"); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(self.residuals, "go", alpha=0.4, markersize=3)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Index"); ax.set_ylabel("Residual")
        ax.set_title("Residuals over Time"); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def print_summary(self):
        print("=" * 70)
        print(f"MEASUREMENT MODEL LEARNER - {self.method.upper()}")
        print("=" * 70)
        print(f"\nResidual Summary:\n  n: {self.n}")
        for k, v in self.residual_stats.items():
            print(f"    {k}: {v:.6f}")
        print("\nModel Parameters:")
        for k, v in self.params.items():
            print(f"    {k}: {v}")
        print("\nFit Quality:")
        for k, v in self.fit_quality.items():
            if k not in ("equation", "all_candidates_aic"):
                print(f"    {k}: {v}")
        print(f"\nModel Equation:\n    {self.fit_quality['equation']}")
        print("=" * 70)
