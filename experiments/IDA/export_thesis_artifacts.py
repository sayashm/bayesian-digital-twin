"""
export_thesis_artifacts.py — Day 9: export IDA/EDA tables & figures for the thesis
=====================================================================================
Turns the analysis already done in Improved_IDA.ipynb / Improved_EDA.ipynb into
thesis-ready CSV tables and PNG figures, restricted to what §4.1 (Dataset
description) actually needs: the two datasets used in this thesis (FD001,
FD003 -- not all four) and the sensor-selection pipeline that produced
SENSOR_COLS, the 10-sensor list used everywhere in particle_twin/.

This is NOT a re-run of the full notebooks -- it re-derives only the pieces
needed here (dataset dimensions, cycle-length stats, and the 5-criterion
sensor ranking), using the exact same method as Improved_EDA.ipynb's
"Final Sensor Selection" section, so the numbers match what's already in
experiments/IDA/output/tables/ if those were regenerated.

Outputs
-------
  experiments/IDA/output/tables/dataset_dimensions.csv
      FD001 vs FD003: train/test rows, engines, cycle-length stats,
      fault-mode count -- the numbers §4.1 reports.
  experiments/IDA/output/tables/sensor_selection_fd001.csv
      The 5-criterion composite ranking (Mann-Kendall significance
      fraction, Monotonicity, Trendability, Mutual Information, VIF)
      for the 14 informative sensors, with the final 10 selected
      flagged -- the table that documents "why these 10 sensors,"
      the open question from earlier in this session.
  thesis/Fig/fig_sensor_composite_ranking.png
      Stacked bar chart of the 5 normalised criteria per sensor
      (mirrors Improved_EDA.ipynb's composite_ranking_FD001.png).
  thesis/Fig/fig_selected_sensor_trajectories.png
      Degradation trajectories (vs. RUL) for the 10 selected sensors,
      6 sample FD001 engines -- the "what does the signal look like"
      figure for §4.1.

Sanity check: this script asserts its derived 10-sensor list matches
particle_twin's SENSOR_COLS exactly (['Ps30', 'T50', 'BPR', 'phi', 'P30',
'htBleed', 'T30', 'T24', 'W32', 'W31']) -- if the EDA methodology or
thresholds ever change, this assertion will catch a mismatch before it
silently propagates into the thesis text.

Run from the repo root:
    python experiments/IDA/export_thesis_artifacts.py

Requires: pymannkendall, statsmodels (already used in Improved_EDA.ipynb).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pymannkendall as mk

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/cmapss")
TABLE_DIR = Path("experiments/IDA/output/tables")
FIG_DIR = Path("thesis/Fig")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = sns.color_palette("tab10")

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32",
]
ALL_COLS = ["unit_id", "cycle"] + SETTING_COLS + SENSOR_NAMES

CONSTANT_SENSORS = ["T2", "P2", "P15", "epr", "farB", "Nf_dmd", "PCNfR_dmd"]
INFORMATIVE = [s for s in SENSOR_NAMES if s not in CONSTANT_SENSORS]  # 14

FAULT_MODES = {"FD001": "HPC degradation", "FD003": "HPC + Fan degradation"}
OP_CONDITIONS = {"FD001": 1, "FD003": 1}

EXPECTED_SENSOR_COLS = ['Ps30', 'T50', 'BPR', 'phi', 'P30', 'htBleed', 'T30', 'T24', 'W32', 'W31']


def load_dataset(name: str):
    train = pd.read_csv(DATA_DIR / f"train_{name}.txt", sep=r"\s+", header=None, names=ALL_COLS)
    test = pd.read_csv(DATA_DIR / f"test_{name}.txt", sep=r"\s+", header=None, names=ALL_COLS)
    rul = pd.read_csv(DATA_DIR / f"RUL_{name}.txt", sep=r"\s+", header=None, names=["true_rul"])["true_rul"]
    max_cyc = train.groupby("unit_id")["cycle"].transform("max")
    train["RUL"] = max_cyc - train["cycle"]
    return train, test, rul


# ══════════════════════════════════════════════════════════════════════════
# 1. Dataset dimensions table (FD001 vs FD003 -- the two used in this thesis)
# ══════════════════════════════════════════════════════════════════════════
data = {}
rows = []
for ds in ["FD001", "FD003"]:
    train, test, rul = load_dataset(ds)
    data[ds] = {"train": train, "test": test, "rul": rul}
    train_lengths = train.groupby("unit_id")["cycle"].max()
    test_lengths = test.groupby("unit_id")["cycle"].max()
    rows.append({
        "Dataset": ds,
        "Operating conditions": OP_CONDITIONS[ds],
        "Fault mode(s)": FAULT_MODES[ds],
        "Train engines": train["unit_id"].nunique(),
        "Test engines": test["unit_id"].nunique(),
        "Train rows": len(train),
        "Test rows": len(test),
        "Train cycle length (min/median/max)": f"{int(train_lengths.min())}/{int(train_lengths.median())}/{int(train_lengths.max())}",
        "Test cycle length (min/median/max)": f"{int(test_lengths.min())}/{int(test_lengths.median())}/{int(test_lengths.max())}",
        "Test true_rul (min/median/max)": f"{int(rul.min())}/{int(rul.median())}/{int(rul.max())}",
    })

dim_df = pd.DataFrame(rows).set_index("Dataset")
dim_df.to_csv(TABLE_DIR / "dataset_dimensions.csv")
print(f"[OK] Wrote {TABLE_DIR / 'dataset_dimensions.csv'}")
print(dim_df.T)

# ══════════════════════════════════════════════════════════════════════════
# 2. Sensor selection pipeline -- same methodology as Improved_EDA.ipynb.
#    IMPORTANT: the composite-score THRESHOLD (step 1) is applied to the
#    MEAN composite score across ALL FOUR sub-datasets (FD001-FD004), not
#    FD001 alone -- that's how Improved_EDA.ipynb actually selected
#    SENSOR_COLS (its 'Nf'/'NRf' etc. only get excluded once FD002/FD004's
#    six-operating-condition noise is folded into the average). VIF and
#    the final |r|>0.90 pruning are then done on FD001 alone, exactly as
#    the notebook did (§8/§10 there use FD001 for those steps). Missing
#    this distinction is exactly what caused the assertion below to fail
#    on the first pass of this script.
# ══════════════════════════════════════════════════════════════════════════
train_fd001 = data["FD001"]["train"].copy()


def assign_op_cluster(df):
    key = df["setting_1"].round(0).astype(int).astype(str) + "_" + df["setting_2"].round(2).astype(str)
    cluster_map = {v: i for i, v in enumerate(sorted(key.unique()))}
    return key.map(cluster_map)


def normalise_by_cluster(df, sensors):
    df = df.copy()
    df["op_cluster"] = assign_op_cluster(df)
    for s in sensors:
        df[s + "_norm"] = df.groupby("op_cluster")[s].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return df


df_n = normalise_by_cluster(train_fd001, INFORMATIVE)
NORM_SENSORS = [s + "_norm" for s in INFORMATIVE]


def mann_kendall_sensor(df, sensor_col, alpha=0.05):
    sigs = []
    for _, grp in df.groupby("unit_id"):
        series = grp.sort_values("cycle")[sensor_col].values
        if len(series) < 4:
            continue
        result = mk.original_test(series, alpha=alpha)
        sigs.append(1 if result.p < alpha else 0)
    return np.mean(sigs)


def monotonicity_score(series):
    d = np.diff(series)
    n = len(d)
    return np.nan if n == 0 else abs((d > 0).sum() - (d < 0).sum()) / n


def sensor_monotonicity(df, sensor_col):
    scores = []
    for _, grp in df.groupby("unit_id"):
        series = grp.sort_values("cycle")[sensor_col].values
        if len(series) > 1:
            scores.append(monotonicity_score(series))
    return np.mean(scores)


def sensor_trendability(df, sensor_col):
    corrs = []
    for _, grp in df.groupby("unit_id"):
        grp = grp.sort_values("cycle")
        n = len(grp)
        if n < 4:
            continue
        t_norm = np.linspace(0, 1, n)
        r, _ = stats.pearsonr(grp[sensor_col].values, t_norm)
        if not np.isnan(r):
            corrs.append(abs(r))
    return np.mean(corrs) if corrs else np.nan


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def compute_sensor_composite(train_df, sensor_cols):
    """One dataset's 5-criterion table -- MK/mono/trend/MI/VIF, each
    normalised WITHIN this dataset, then averaged into a Composite column.
    Mirrors the per-dataset loop in Improved_EDA.ipynb's §9."""
    dfn = normalise_by_cluster(train_df, sensor_cols)
    norm_cols = [s + "_norm" for s in sensor_cols]

    mk_sig = {s: mann_kendall_sensor(dfn, s + "_norm") for s in sensor_cols}
    mono_ = {s: sensor_monotonicity(dfn, s + "_norm") for s in sensor_cols}
    trend_ = {s: sensor_trendability(dfn, s + "_norm") for s in sensor_cols}

    X_mi = dfn[norm_cols].dropna()
    y_mi = dfn.loc[X_mi.index, "RUL"]
    mi_vals = mutual_info_regression(X_mi.values, y_mi.values, n_neighbors=5, random_state=42)
    mi_ = dict(zip(sensor_cols, mi_vals))

    X_vif = dfn[norm_cols].dropna()
    if len(X_vif) > 10_000:
        X_vif = X_vif.sample(10_000, random_state=42)
    vif_vals = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    vif_ = dict(zip(sensor_cols, vif_vals))

    out = pd.DataFrame({
        "MK_sig_frac": pd.Series(mk_sig), "Monotonicity": pd.Series(mono_),
        "Trendability": pd.Series(trend_), "MI": pd.Series(mi_), "VIF": pd.Series(vif_),
    })
    out["MK_norm"] = minmax(out["MK_sig_frac"])
    out["Mono_norm"] = minmax(out["Monotonicity"])
    out["Trend_norm"] = minmax(out["Trendability"])
    out["MI_norm"] = minmax(out["MI"])
    out["VIF_norm"] = 1 - minmax(out["VIF"])
    out["Composite"] = out[["MK_norm", "Mono_norm", "Trend_norm", "MI_norm", "VIF_norm"]].mean(axis=1)
    return out


# FD001's own table -- used for the final displayed criteria values, the
# VIF<=10 filter, and the pairwise-correlation dedup step (matches
# Improved_EDA.ipynb, which does all three on FD001 specifically).
comp = compute_sensor_composite(train_fd001, INFORMATIVE)

# Step 1 threshold uses the MEAN composite across ALL FOUR sub-datasets
# (see the block comment above) -- so FD002/FD004 must be loaded too, even
# though this thesis's experiments only use FD001/FD003.
print("[OK] Computing sensor composite scores on FD002/FD004 too "
      "(needed only for the cross-dataset mean-composite threshold, per "
      "Improved_EDA.ipynb's actual selection rule) -- this takes a while, "
      "so each dataset's result is cached to disk and skipped on rerun "
      "(FD002/FD004 have 250+ engines each, and Mann-Kendall is run "
      "per-engine-per-sensor, so this doesn't fit in one short call).")
CACHE_DIR = TABLE_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

composite_by_dataset = {"FD001": comp["Composite"]}
for ds in ["FD002", "FD003", "FD004"]:
    cache_path = CACHE_DIR / f"composite_{ds}.csv"
    if cache_path.exists():
        composite_by_dataset[ds] = pd.read_csv(cache_path, index_col=0)["Composite"]
        print(f"  [OK] {ds} composite scores loaded from cache ({cache_path})")
        continue
    train_ds, _, _ = load_dataset(ds)
    ds_comp = compute_sensor_composite(train_ds, INFORMATIVE)
    ds_comp.to_csv(cache_path)
    composite_by_dataset[ds] = ds_comp["Composite"]
    print(f"  [OK] {ds} composite scores computed and cached ({cache_path})")

mean_composite = pd.DataFrame(composite_by_dataset).mean(axis=1)
comp["MeanCompositeAllDatasets"] = mean_composite

# Selection rules (identical to Improved_EDA.ipynb §10): mean composite
# (across FD001-FD004) >= 0.40, then FD001's own VIF <= 10, then drop the
# lower-composite member of any FD001 pair with |r| > 0.90.
step1 = mean_composite[mean_composite >= 0.40].index.tolist()
step2 = [s for s in step1 if comp.loc[s, "VIF"] <= 10]

corr_fd001 = df_n[[s + "_norm" for s in step2]].rename(columns=lambda c: c.replace("_norm", "")).corr().abs()
to_drop = set()
for i, s1 in enumerate(step2):
    for s2 in step2[i + 1:]:
        if s1 in to_drop or s2 in to_drop:
            continue
        r = corr_fd001.loc[s1, s2]
        if r > 0.90:
            drop = s1 if mean_composite[s1] < mean_composite[s2] else s2
            to_drop.add(drop)

selected = [s for s in step2 if s not in to_drop]
comp = comp.sort_values("Composite", ascending=False)
comp["Selected"] = comp.index.isin(selected)

assert sorted(selected) == sorted(EXPECTED_SENSOR_COLS), (
    f"Derived sensor selection {sorted(selected)} does not match particle_twin's "
    f"SENSOR_COLS {sorted(EXPECTED_SENSOR_COLS)} -- methodology or data has changed; "
    f"check before using this table in the thesis."
)
print(f"\n[OK] Sensor selection matches particle_twin's SENSOR_COLS exactly: {selected}")

comp.round(4).to_csv(TABLE_DIR / "sensor_selection_fd001.csv")
print(f"[OK] Wrote {TABLE_DIR / 'sensor_selection_fd001.csv'}")
print(comp[["MK_sig_frac", "Monotonicity", "Trendability", "MI", "VIF",
            "Composite", "MeanCompositeAllDatasets", "Selected"]].round(4))

# ══════════════════════════════════════════════════════════════════════════
# 3. Figure: composite ranking stacked bar chart
# ══════════════════════════════════════════════════════════════════════════
criteria_cols = ["MK_norm", "Mono_norm", "Trend_norm", "MI_norm", "VIF_norm"]
criteria_labels = ["Mann-Kendall", "Monotonicity", "Trendability", "Mut. Info.", "VIF (inv.)"]
plot_df = comp[criteria_cols].sort_values("MK_norm", ascending=True)
plot_df.columns = criteria_labels

fig, ax = plt.subplots(figsize=(9, 6))
plot_df.plot(kind="barh", stacked=True, ax=ax, colormap="tab10", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Normalised score (sum of 5 criteria; max = 5.0)")
ax.set_title("Composite Sensor Score — FD001 (14 informative sensors)", fontsize=11)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_sensor_composite_ranking.png", bbox_inches="tight")
plt.close(fig)
print(f"[OK] Wrote {FIG_DIR / 'fig_sensor_composite_ranking.png'}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Figure: degradation trajectories for the 10 selected sensors
# ══════════════════════════════════════════════════════════════════════════
sample_units = list(train_fd001["unit_id"].unique()[:6])
n_sel = len(selected)
ncols = 3
nrows = int(np.ceil(n_sel / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.0))
axes = np.array(axes).flatten()
fig.suptitle("Selected Sensors — Normalised Degradation Trajectories (FD001, 6 engines)", fontsize=11)

for i, s in enumerate(selected):
    ax = axes[i]
    for j, uid in enumerate(sample_units):
        grp = df_n[df_n["unit_id"] == uid].sort_values("cycle")
        ax.plot(grp["RUL"], grp[s + "_norm"], linewidth=0.9, alpha=0.8, color=PALETTE[j], label=f"Unit {uid}")
    ax.invert_xaxis()
    ax.set_title(s, fontsize=10, fontweight="bold")
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Normalised value")
    if i == 0:
        ax.legend(fontsize=7, ncol=2)

for j in range(n_sel, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_selected_sensor_trajectories.png", bbox_inches="tight")
plt.close(fig)
print(f"[OK] Wrote {FIG_DIR / 'fig_selected_sensor_trajectories.png'}")

print("\n[DONE] All Day 9 thesis artifacts exported.")
