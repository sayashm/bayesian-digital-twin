<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Python code for C-MAPSS RUL estimation using exponential model

Here is a clean Python baseline for C-MAPSS RUL estimation using an exponential health-index model. It builds a simple HI from sensors, fits $HI(t)=a e^{bt}$, then estimates RUL as the time until the curve reaches a failure threshold.[^1][^2]

## Python code

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# ----------------------------
# Load C-MAPSS-style data
# ----------------------------
# Expected columns:
# engine_id, cycle, setting1, setting2, setting3, s1 ... s21
train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None)
test  = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)
rul   = pd.read_csv("RUL_FD001.txt", sep=r"\s+", header=None)

cols = ["engine_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
train.columns = cols
test.columns = cols
rul.columns = ["RUL"]

# ----------------------------
# Helpers
# ----------------------------
def add_true_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df = df.copy()
    df["true_RUL"] = max_cycle - df["cycle"]
    return df

def linear_health_index(df, sensor_cols):
    # Simple HI = weighted first PCA component proxy via mean z-scored sensors
    scaler = StandardScaler()
    X = scaler.fit_transform(df[sensor_cols])
    hi = X.mean(axis=1)
    return hi

def exp_model(t, a, b):
    return a * np.exp(b * t)

def fit_exponential_hi(t, hi):
    # Ensure positive HI for log/exponential fit
    hi = np.asarray(hi, dtype=float)
    hi_min = hi.min()
    if hi_min <= 0:
        hi = hi - hi_min + 1e-6

    p0 = [hi[^0], -0.01]
    params, _ = curve_fit(exp_model, t, hi, p0=p0, maxfev=10000)
    return params, hi

def predict_rul_from_hi(t_hist, hi_hist, t_now, theta=0.1):
    a, b = fit_exponential_hi(t_hist, hi_hist)[^0]

    if b >= 0:
        return np.nan

    if theta <= 0:
        theta = 1e-6

    tf = np.log(theta / a) / b
    return max(tf - t_now, 0.0)

# ----------------------------
# Prepare training data
# ----------------------------
train = add_true_rul(train)

# Typical C-MAPSS practice: remove constant sensors
sensor_cols = [c for c in train.columns if c.startswith("s")]
constant_sensors = ["s1", "s5", "s6", "s10", "s16", "s18", "s19", "s20"]
sensor_cols = [c for c in sensor_cols if c not in constant_sensors]

# Build a simple HI for each engine using only sensor data
train["HI"] = np.nan
for eng_id, grp_idx in train.groupby("engine_id").groups.items():
    grp = train.loc[grp_idx].sort_values("cycle")
    hi = linear_health_index(grp, sensor_cols)
    train.loc[grp.index, "HI"] = hi

# Optional: make HI decrease with degradation
# If needed, flip sign so it trends downward
train["HI"] = -train["HI"]

# ----------------------------
# Estimate RUL per engine at last observed cycle
# ----------------------------
results = []
theta = 0.1

for eng_id, grp in train.groupby("engine_id"):
    grp = grp.sort_values("cycle")
    t = grp["cycle"].to_numpy(dtype=float)
    hi = grp["HI"].to_numpy(dtype=float)
    t_now = t[-1]
    true_rul = grp["true_RUL"].iloc[-1]

    pred_rul = predict_rul_from_hi(t, hi, t_now, theta=theta)

    results.append({
        "engine_id": eng_id,
        "true_RUL": true_rul,
        "pred_RUL": pred_rul
    })

res = pd.DataFrame(results)

# ----------------------------
# Evaluate
# ----------------------------
valid = res.dropna()
rmse = mean_squared_error(valid["true_RUL"], valid["pred_RUL"], squared=False)

print(f"RMSE: {rmse:.4f}")
print(valid.head())
```


## How it works

- The code first creates a simple health index by standardizing sensors and averaging them.
- It then fits an exponential decay curve $HI(t)=a e^{bt}$ with $b<0$.
- Failure time is the point where the fitted curve reaches a threshold $\theta$, and RUL is $t_f-t$.[^3][^2]
- This is a baseline, not state of the art, but it is easy to explain and extend.[^2][^1]


## Better version

A stronger version usually does this:

1. Remove constant sensors.
2. Normalize by operating regime.
3. Build HI with PCA, autoencoder, or regression instead of a raw average.
4. Smooth HI before fitting.
5. Fit the exponential only on the last degradation segment, not the entire life.
6. Use $RUL = t_f - t$ after solving the threshold-crossing time.[^4][^2]

## Small fix you may need

For test data, C-MAPSS gives the final-cycle RUL separately in `RUL_FD001.txt`, so evaluation usually appends the unknown post-test cycles before computing the last-cycle target. If you want, I can give you a full FD001 training/testing script that reads all three files and produces submission-style predictions.[^5][^2]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/ozogxyz/cmapss

[^2]: https://ntrs.nasa.gov/api/citations/20150007677/downloads/20150007677.pdf

[^3]: https://recipp.ipp.pt/bitstreams/159d4a23-4ef4-4a78-aa67-f62698afedec/download

[^4]: https://research.tudelft.nl/files/138848617/1_s2.0_S0952197622005723_main.pdf

[^5]: https://scholarworks.sjsu.edu/etd_projects/1360/

[^6]: https://arxiv.org/pdf/2604.13459.pdf

[^7]: https://c3.ndc.nasa.gov/dashlink/static/media/publication/2008_IEEEPHM_CMAPPSDamagePropagation.pdf

[^8]: https://www.academia.edu/123470313/ICAAIML

[^9]: https://c3.ndc.nasa.gov/dashlink/static/media/publication/2018_DegradationModelingRULEnsemble_Wu.pdf

[^10]: https://github.com/UtkarshPanara/Remaining-Useful-Life-Prediction-for-Turbofan-Engines

[^11]: https://papers.phmsociety.org/index.php/phme/article/download/3359/1923

[^12]: https://github.com/boemer00/jet-engine-degradation-prediction

[^13]: https://github.com/topics/rul-prediction?o=desc\&s=forks

[^14]: https://www.reddit.com/r/DSP/comments/1uzipsi/case_study_on_the_nasa_cmapss_turbofan_engine/

[^15]: https://arxiv.org/html/2604.27234v1

