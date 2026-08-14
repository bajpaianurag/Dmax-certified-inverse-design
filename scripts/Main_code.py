# Imports
import os, sys, json, time, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import StratifiedShuffleSplit
from math import sqrt
from sklearn.metrics import roc_auc_score
from math import lgamma
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from scipy.special import gammaln, logsumexp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap
import hashlib
from itertools import chain
import gc
import matplotlib as mpl
import joblib, subprocess, platform, datetime as _dt
from sklearn.metrics import mean_pinball_loss, mean_absolute_error, mean_squared_error, r2_score
from skopt import forest_minimize
from skopt import gp_minimize
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from matplotlib import ticker as mticker
from scipy.special import gammaln
from math import lgamma as _lgamma
from pandas.api.types import CategoricalDtype
from sklearn.linear_model import LogisticRegression
from typing import Optional, List, Dict, Tuple, Iterable
warnings.filterwarnings("once", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.copy_on_write = True
plt.rcParams.update({
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)

RESULTS_ROOT = Path(os.getenv("RESULTS_DIR", "/results"))
OUTDIR = RESULTS_ROOT / "project_output"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Input data and parameters
RAW_CSV = "Final_MMG_Dmax_dataset.csv"

SEED = 1

def set_all_seeds(seed: int = 1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_all_seeds(SEED)

rng = np.random.default_rng(SEED)

ALPHA = 0.15
DRIFT_EPS_ATPCT     = 1.00
ROBUST_EPS          = DRIFT_EPS_ATPCT / 100.0
DRIFT_SUPPORT_ONLY  = True 
DRIFT_BOUNDARY_FRAC = 0.75    
DRIFT_SEED          = 1000    

ROBUST_SAMPLES = 256         
QT_TAU = 0.95
ROBUST_METHOD = "mc"
ADV_MAX_ITERS = 300
ADV_STEP = 0.001             
ADV_TOPK = 5
THRESHOLDS_MM = tuple(sorted({2.0, 5.0, 7.0, 10.0, 15.0, 20.0}))
MIN_GROUP_N     = 30
FAMILY_MIN_TEST = 10

(OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
RUN_INFO = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "host": platform.node(),
    "python": sys.version.split()[0],
    "versions": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "catboost": __import__("catboost").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "seaborn": __import__("seaborn").__version__,
    },
    "params": {
        "SEED": SEED,
        "ALPHA": ALPHA,
        "ROBUST_EPS": ROBUST_EPS,
        "DRIFT_EPS_ATPCT": DRIFT_EPS_ATPCT,
        "DRIFT_SUPPORT_ONLY": DRIFT_SUPPORT_ONLY,
        "DRIFT_BOUNDARY_FRAC": DRIFT_BOUNDARY_FRAC,
        "DRIFT_SEED": DRIFT_SEED,
        "GRID_STEP_ATPCT": GRID_STEP_ATPCT,
        "ROBUST_SAMPLES": ROBUST_SAMPLES,
        "QT_TAU": QT_TAU,
        "ADV_STEP": ADV_STEP,
        "THRESHOLDS_MM": THRESHOLDS_MM,
        "MIN_GROUP_N": MIN_GROUP_N,
        "FAMILY_MIN_TEST": FAMILY_MIN_TEST,
    }
}
with open(OUTDIR / "reports" / "run_config.json", "w") as f:
    json.dump(RUN_INFO, f, indent=2)
print("Run config written to:", OUTDIR / "reports" / "run_config.json")


def weighted_quantile(values, q, sample_weight=None):
    """
    Return the q-quantile of `values` with optional weights (0<=q<=1).
    Uses a right-continuous definition (upper order statistic under ties).

    Notes:
    - NaN-safe: drops NaNs in values/weights.
    - q=0 → min, q=1 → max.
    - Right quantile matches one-sided conformal score calibration well.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1].")

    v = np.asarray(values, dtype=float)

    if sample_weight is None:
        v = v[~np.isnan(v)]
        if v.size == 0:
            return np.nan
        if q <= 0.0:
            return float(np.nanmin(v))
        if q >= 1.0:
            return float(np.nanmax(v))
        return float(np.quantile(v, q, method="higher"))

    w = np.asarray(sample_weight, dtype=float)
    if v.shape[0] != w.shape[0]:
        raise ValueError("values and sample_weight must have the same length.")

    mask = (~np.isnan(v)) & (~np.isnan(w)) & (w > 0)
    v, w = v[mask], w[mask]
    if v.size == 0:
        return np.nan

    sorter = np.argsort(v, kind="mergesort")
    v, w = v[sorter], w[sorter]

    cw = np.cumsum(w)
    if q <= 0.0:
        return float(v[0])
    if q >= 1.0:
        return float(v[-1])

    cutoff = q * cw[-1]
    idx = np.searchsorted(cw, cutoff, side="right")
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def project_to_simplex_with_caps(x, caps=None, tol=1e-12, max_iter=1000):
    """
    Project x (fractions) onto { z : z>=0, sum z = 1 } with optional per-dim caps (0<=z_i<=cap_i).
    Solves sum_i clip(x_i - tau, 0, u_i) = 1 via bisection (water-filling).
    - If caps infeasible (sum finite caps < 1), raises ValueError.
    - Returns a feasible point even if x is pathological (uniform under caps).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    d = x.size

    if caps is None:
        u = np.full(d, np.inf)
    else:
        vals = np.asarray(caps["values"], dtype=object)
        if vals.size != d:
            raise ValueError("caps['values'] must have the same length as x.")
        u = np.array([np.inf if v is None else float(v) for v in vals], dtype=float)
        if np.any(u < 0):
            raise ValueError("Caps must be nonnegative.")
        finite_u = u[np.isfinite(u)]
        if finite_u.size > 0 and finite_u.sum() + tol < 1.0:
            raise ValueError("Infeasible caps: sum(caps) < 1, cannot satisfy sum(z)=1.")

    def S(tau):
        return np.clip(x - tau, 0.0, u).sum()

    lo = np.min(x - u[np.isfinite(u)]) if np.any(np.isfinite(u)) else (np.min(x) - 1.0)
    hi = np.max(x)

    while S(lo) < 1.0:
        lo -= max(1.0, np.nanmax(u[np.isfinite(u)]) if np.any(np.isfinite(u)) else 1.0)
        if not np.isfinite(lo):
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if S(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < tol:
            break

    tau = 0.5 * (lo + hi)
    z = np.clip(x - tau, 0.0, u)

    s = z.sum()
    if s <= 0 or not np.isfinite(s):
        z = np.minimum(u, 1.0 / d)
        z /= z.sum()
    else:
        z /= s
    return z


def _drift_support(x, tol=1e-12):
    return np.where(np.asarray(x, float) > tol)[0]


def sample_drift_neighborhood(x, eps, K, rng, include_x=True,
                              support_only=None, boundary_frac=None,
                              enumerate_vertices=True):
    """
    Sample K compositions from the admissible drift neighbourhood

        B_eps(x) = { x' in simplex : d_drift(x, x') <= eps },
        d_drift(x, x') = 0.5 * ||x' - x||_1            (manuscript Eq. 1, Eq. 6)

    `eps` is in the d_drift metric, FRACTION units:
        eps = 0.01  <=>  1.00 at.% transferred mass  <=>  ||x'-x||_1 <= 0.02

    Every returned sample is a mass-conserving donor -> acceptor transfer among
    the elements of supp(x): total mass r <= eps is removed from a donor subset
    D and added to a disjoint acceptor subset A.  No element outside supp(x) is
    ever populated, and no coordinate becomes negative.

    The K rows are composed as:
      1  x itself                                  (x in B_eps(x); Eq. 8)
      +  ALL single-pair vertices x + d*(e_j - e_i), i,j in supp(x)
         -- the extreme points of the mass-conserving budget, and exactly the
            transfers scored by the tornado analysis
      +  random donor/acceptor mixture transfers; `boundary_frac` of them at
         d_drift == eps exactly, the remainder at a uniformly random radius
         in (0, eps].
    """
    if support_only is None:
        support_only = bool(globals().get("DRIFT_SUPPORT_ONLY", True))
    if boundary_frac is None:
        boundary_frac = float(globals().get("DRIFT_BOUNDARY_FRAC", 0.75))

    x = np.asarray(x, float)
    d = x.size
    s = x.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Base composition must have positive finite sum.")
    x = np.clip(x, 0.0, None) / s

    K = int(K)
    eps = float(max(0.0, eps))
    if eps <= 0.0 or K <= 0:
        return np.broadcast_to(x, (max(K, 1), d)).copy()

    S = _drift_support(x) if support_only else np.arange(d)
    m = int(S.size)

    rows = []
    if include_x:
        rows.append(x.copy())

    if m >= 2:
        if enumerate_vertices:
            for i in S:
                delta = min(eps, float(x[i]))
                if delta <= 0.0:
                    continue
                for j in S:
                    if i == j or len(rows) >= K:
                        continue
                    xp = x.copy()
                    xp[i] -= delta
                    xp[j] += delta
                    rows.append(xp)
                if len(rows) >= K:
                    break

        while len(rows) < K:
            n_D = int(rng.integers(1, m)) 
            perm = rng.permutation(m)
            D = S[perm[:n_D]]
            A = S[perm[n_D:]]

            a = rng.dirichlet(np.ones(D.size))         # donor split
            b = rng.dirichlet(np.ones(A.size))         # acceptor split
            r = eps if rng.random() < boundary_frac else eps * float(rng.random())

            take = r * a
            over = take > x[D]
            if np.any(over): 
                scale = float(np.min(x[D][over] / np.maximum(take[over], 1e-300)))
                take = take * scale

            xp = x.copy()
            xp[D] -= take
            xp[A] += take.sum() * b
            xp = np.clip(xp, 0.0, None)
            t = xp.sum()
            rows.append(xp / (t if t > 0 else 1.0))
    else:
        while len(rows) < K:
            rows.append(x.copy())

    out = np.asarray(rows[:K], float)
    if out.shape[0] < K:
        out = np.vstack([out, np.broadcast_to(x, (K - out.shape[0], d))])

    dd = 0.5 * np.abs(out - x[None, :]).sum(axis=1)
    if not np.all(dd <= eps + 1e-9):
        raise AssertionError(f"drift budget violated: max d_drift={dd.max():.3e} > eps={eps:.3e}")
    return out


def jitter_in_L1_ball_simplex(x, eps, K, rng, include_x=True):
    return sample_drift_neighborhood(x, eps, K, rng, include_x=include_x)


def jitter_allowed_simplex(x_allowed, eps, K, rng):
    return sample_drift_neighborhood(x_allowed, eps, K, rng, include_x=True)


def _run_drift_sampler_selftest(n_dims=33):
    rng_t = np.random.default_rng(0)
    x = np.zeros(n_dims)
    supp = [0, 1, 2, 3, 4]
    for i, v in zip(supp, [0.26, 0.06, 0.08, 0.21, 0.39]):
        x[i] = v

    S = sample_drift_neighborhood(x, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_t)
    dd = 0.5 * np.abs(S - x[None, :]).sum(axis=1)
    off = [j for j in range(n_dims) if j not in supp]

    assert np.allclose(S.sum(axis=1), 1.0),        "simplex closure violated"
    assert (S >= -1e-12).all(),                    "negative atomic fraction produced"
    assert dd.max() <= ROBUST_EPS + 1e-9,          "drift budget exceeded"
    assert np.allclose(S[:, off], 0.0),            "support not preserved"
    assert np.isclose(dd.min(), 0.0),              "nominal composition not included"
    assert (dd > 0.999 * ROBUST_EPS).mean() > 0.5, "boundary not emphasised"

    want = {(i, j) for i in supp for j in supp if i != j}
    got = set()
    for r in S:
        dvec = r - x
        nz = np.where(np.abs(dvec) > 1e-10)[0]
        if nz.size == 2:
            got.add((int(nz[np.argmin(dvec[nz])]), int(nz[np.argmax(dvec[nz])])))
    assert want <= got, f"missing single-pair vertices: {sorted(want - got)}"

    n_dir = len({tuple(np.sign(np.round(r - x, 12))) for r in S})
    print(f"[drift sampler] max d_drift   = {dd.max()*100:.4f} at.% (target {DRIFT_EPS_ATPCT})")
    print(f"[drift sampler] boundary frac = {(dd > 0.999*ROBUST_EPS).mean():.3f}")
    print(f"[drift sampler] vertices {len(want & got)}/{len(want)} | directions = {n_dir}")

_run_drift_sampler_selftest()

PT_SYMBOLS = [
    "Ag","Al","Au","B","Be","C","Ca","Ce","Co","Cr","Cu","Dy","Er","Fe","Ga",
    "Gd","Hf","La","Mg","Mn","Mo","Nb","Nd","Ni","P","Pd","Pr","Pt","Sc","Si",
    "Sm","Sn","Ta","Tb","Ti","Tm","V","W","Y","Zn","Zr"
]

df_raw = pd.read_csv(RAW_CSV, low_memory=False)
if df_raw.shape[0] == 0:
    raise ValueError(f"File '{RAW_CSV}' loaded but has zero rows.")

df_raw.columns = [str(c).strip() for c in df_raw.columns]

def _normalize(c: str) -> str:
    return re.sub(r"[\s_\-\(\)\[\]]+", "", c.lower())

norm_map = {c: _normalize(c) for c in df_raw.columns}

dmax_like = []
for c, nc in norm_map.items():
    if re.fullmatch(r"dmax(mm)?", nc) or re.fullmatch(r"d_?max(mm)?", nc):
        dmax_like.append(c)

if "Dmax_mm" in df_raw.columns and "Dmax_mm" not in dmax_like:
    dmax_like.insert(0, "Dmax_mm")

if len(dmax_like) == 0:
    raise KeyError(
        "Could not find a Dmax column. Expected something like "
        "'Dmax_mm', 'Dmax', 'D_max', or 'Dmax (mm)'. "
        f"Available columns: {list(df_raw.columns)[:20]}{' ...' if len(df_raw.columns)>20 else ''}"
    )
elif len(dmax_like) > 1:
    prefer = [c for c in dmax_like if _normalize(c) == "dmaxmm" or c == "Dmax_mm"]
    dmax_col = prefer[0] if len(prefer) else dmax_like[0]
    warnings.warn(f"Multiple Dmax-like columns found {dmax_like}; using '{dmax_col}'.")
else:
    dmax_col = dmax_like[0]

def _to_symbol(c: str) -> str:
    s = str(c).strip()
    return (s[:1].upper() + s[1:].lower()) if s else s

sym_to_cols = defaultdict(list)
for c in df_raw.columns:
    sym = _to_symbol(c)
    if sym in PT_SYMBOLS:
        sym_to_cols[sym].append(c)

if len(sym_to_cols) < 5:
    raise AssertionError(
        f"Too few element columns detected ({len(sym_to_cols)}). "
        f"First 20 columns: {list(df_raw.columns)[:20]}"
    )

duplicates = {sym: cols for sym, cols in sym_to_cols.items() if len(cols) > 1}
if duplicates:
    for sym, cols in duplicates.items():
        df_raw[sym] = df_raw[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        to_drop = [c for c in cols if c != sym]
        if to_drop:
            df_raw.drop(columns=to_drop, inplace=True)
    sym_to_cols = {sym: [sym] for sym in sym_to_cols.keys()}

elem_cols = [sym for sym in PT_SYMBOLS if sym in sym_to_cols]

print(f"Dmax column: {dmax_col}")
print(f"{len(elem_cols)} element columns detected (PT order): {elem_cols[:15]}{' ...' if len(elem_cols) > 15 else ''}")

missing_syms = [sym for sym in PT_SYMBOLS if sym not in elem_cols]
if missing_syms:
    print(f"PT symbols not present in this dataset: {missing_syms[:20]}{' ...' if len(missing_syms) > 20 else ''}")


SUM_TOL = 1e-4
NEG_TOL = 1e-12 
FORCE_EXACT_SUM = True

df = df_raw.copy()

for c in elem_cols + [dmax_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n_neg_before = int((df[elem_cols] < 0).sum().sum())
df[elem_cols] = df[elem_cols].fillna(0.0)
n_hard_neg = int((df[elem_cols] < -NEG_TOL).sum().sum())
df[elem_cols] = df[elem_cols].clip(lower=0.0)

sum_before = df[elem_cols].sum(axis=1)
n_all_zero = int((sum_before <= 0).sum())
df = df.loc[sum_before > 0].reset_index(drop=True)

sums = df[elem_cols].sum(axis=1)
need = (sums > 0) & ~sums.between(1.0 - SUM_TOL, 1.0 + SUM_TOL)
n_rescaled = int(need.sum())
if n_rescaled > 0:
    df.loc[need, elem_cols] = df.loc[need, elem_cols].div(sums[need], axis=0)

if FORCE_EXACT_SUM:
    sums2 = df[elem_cols].sum(axis=1)
    df.loc[:, elem_cols] = df.loc[:, elem_cols].div(np.where(sums2 > 0, sums2, 1.0), axis=0)

sums_after = df[elem_cols].sum(axis=1)
off_after = ~sums_after.between(1.0 - SUM_TOL, 1.0 + SUM_TOL)
n_off_after = int(off_after.sum())

dmax_series = pd.to_numeric(df[dmax_col], errors="coerce")
dmax_valid = dmax_series.dropna().values
dmax_stats = {
    "count": int(np.isfinite(dmax_series).sum()),
    "min":   float(np.nanmin(dmax_series)) if len(dmax_valid) else np.nan,
    "p50":   float(np.nanmedian(dmax_series)) if len(dmax_valid) else np.nan,
    "p90":   float(np.nanpercentile(dmax_series, 90)) if len(dmax_valid) else np.nan,
    "max":   float(np.nanmax(dmax_series)) if len(dmax_valid) else np.nan,
    "n_nonpositive": int((dmax_series.fillna(np.inf) <= 0).sum())
}

schema = {
    "n_rows": int(len(df)),
    "n_cols": int(df.shape[1]),
    "dmax_col": dmax_col,
    "n_element_cols": len(elem_cols),
    "composition_qc": {
        "neg_entries_before_total": n_neg_before,
        "hard_neg_entries_clipped_<-NEG_TOL": n_hard_neg,
        "rows_all_zero_dropped": n_all_zero,
        "tolerance_SUM_TOL": SUM_TOL,
        "n_rescaled_outside_tolerance": n_rescaled,
        "n_rows_outside_tolerance_after_rescale": n_off_after,
        "sum_after": {
            "mean": float(sums_after.mean()),
            "min":  float(sums_after.min()),
            "max":  float(sums_after.max())
        },
        "force_exact_sum_applied": bool(FORCE_EXACT_SUM)
    },
    "dmax_stats": dmax_stats
}

(OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
with open(OUTDIR / "reports" / "schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print(json.dumps(schema, indent=2))


GRID_STEP_ATPCT = 0.5
GRID_STEP_FRAC  = GRID_STEP_ATPCT / 100.0

def mean_abs_dev_mean(s):
    s = np.asarray(s, float)
    m = np.nanmean(s)
    return float(np.nanmean(np.abs(s - m)))

def mad_about_median(s):
    s = np.asarray(s, float)
    med = np.nanmedian(s)
    return float(np.nanmedian(np.abs(s - med)))

def _snap_to_grid(comp, step=None):
    """Round each atomic fraction to the nearest multiple of `step`."""
    if step is None:
        step = GRID_STEP_FRAC
    comp = np.clip(np.asarray(comp, float), 0.0, None)
    return np.rint(comp / float(step)).astype(np.int64)

def composition_signature(df, elem_cols, step=None):
    g = _snap_to_grid(df[elem_cols].to_numpy(float), step)
    parts = [";".join(map(str, row)) for row in g]
    return pd.Series(parts, index=df.index, name="signature")

def composition_signature_int(df, elem_cols, step=None):
    return _snap_to_grid(df[elem_cols].to_numpy(float), step)

def composition_signature_hash(sig_int, algo="md5"):
    hashes = []
    for row in sig_int:
        b = row.tobytes()
        h = hashlib.md5(b).hexdigest() if algo == "md5" else hashlib.sha1(b).hexdigest()
        hashes.append(h)
    return pd.Series(hashes, name="signature_hash")

df["signature"] = composition_signature(df, elem_cols)
sig_int = composition_signature_int(df, elem_cols)
df["signature_hash"] = composition_signature_hash(sig_int, algo="md5")

def iqr(a):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, 75) - np.nanpercentile(a, 25))

dup_stats = (
    df.groupby("signature", sort=False)[dmax_col]
      .agg(
           count="size",
           mean="mean",
           median="median",
           std="std",
           mad_mean=mean_abs_dev_mean,
           mad_med=mad_about_median,
           iqr=lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25)),
           min="min",
           max="max"
      )
      .sort_values("count", ascending=False)
      .reset_index()
)


replicates_resolved = (
    df.groupby(["signature"], as_index=False)
      .agg({dmax_col: "median"})
      .rename(columns={dmax_col: f"{dmax_col}_median"})
)

OUTDIR.joinpath("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTDIR / "data" / "processed" / "with_signatures.csv", index=False)
dup_stats.to_csv(OUTDIR / "reports" / "dups.csv", index=False)
replicates_resolved.to_csv(OUTDIR / "data" / "processed" / "replicates_resolved.csv", index=False)

n_dups = int((dup_stats["count"] > 1).sum())
print(f"[Signature grid = {GRID_STEP_ATPCT} at.%] compositions: {df.shape[0]} rows, "
      f"{dup_stats.shape[0]} unique signatures; duplicates (count>1): {n_dups}")
dup_stats.head(10)


# In[9]:


def heaping_report_plus(y, tol=5e-3):
    s = pd.to_numeric(y, errors="coerce").to_numpy()
    s = s[np.isfinite(s)]
    rep = {"n": int(s.size)}
    if rep["n"] == 0:
        return rep

    rep.update({
        "min": float(np.nanmin(s)),
        "p50": float(np.nanpercentile(s, 50)),
        "p90": float(np.nanpercentile(s, 90)),
        "max": float(np.nanmax(s)),
    })

    vals  = np.unique(np.round(s, 3))
    diffs = np.diff(vals) if vals.size > 1 else np.array([])
    if diffs.size > 2:
        step_mode_candidates = pd.Series(np.round(diffs, 3)).value_counts()
        step_mode = float(step_mode_candidates.index[0]) if len(step_mode_candidates) else None
    else:
        step_mode = None
    rep["mode_step_estimate"] = step_mode

    fr = np.mod(s, 1.0)
    frac_zero = float(np.mean(np.abs(fr - 0.0)  <= tol))
    frac_q25  = float(np.mean(np.abs(fr - 0.25) <= tol))
    frac_q50  = float(np.mean(np.abs(fr - 0.50) <= tol))
    frac_q75  = float(np.mean(np.abs(fr - 0.75) <= tol))
    frac_int  = float(np.mean(np.abs(fr - np.round(fr)) <= tol))

    rep.update({
        "frac_zero_bin": frac_zero,
        "frac_half_bin": frac_q50,
        "frac_quarter_bins": {
            "0.00": frac_zero,
            "0.25": frac_q25,
            "0.50": frac_q50,
            "0.75": frac_q75
        },
        "integer_heaping_rate": frac_int
    })

    ints = np.round(s).astype(int)
    vc = pd.Series(ints).value_counts().sort_values(ascending=False).head(20)
    top_bins = {int(k): int(v) for k, v in vc.items()}
    rep["top_integer_bins"] = top_bins

    first_dec_digit = np.mod(np.round(s * 10).astype(int), 10)
    counts = np.bincount(first_dec_digit, minlength=10)
    total = int(counts.sum())
    expected = total / 10.0 if total > 0 else np.nan
    chi2 = float(np.sum((counts - expected)**2 / expected)) if (total > 0 and expected > 0) else np.nan
    rep["first_decimal_digit"] = {
        "counts": counts.tolist(),
        "total": total,
        "chi2_stat": chi2,
        "df": 9
    }

    hist_counts, _ = np.histogram(fr, bins=np.linspace(0.0, 1.0, 11))
    rep["fraction_histogram_10"] = hist_counts.astype(int).tolist()

    return rep

heap = heaping_report_plus(df[dmax_col], tol=5e-3)
(OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
with open(OUTDIR / "reports" / "heaping.json", "w") as f:
    json.dump(heap, f, indent=2)
heap


# In[10]:


def plot_heaping_panels(y, out_png):
    s = pd.to_numeric(y, errors="coerce").to_numpy()
    s = s[np.isfinite(s)]
    if s.size == 0:
        print("No finite values to plot."); return

    fr = np.mod(s, 1.0)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(s, bins=40, kde=True)
    p50, p90 = np.percentile(s, [50, 90])
    plt.axvline(p50, ls="--", lw=1.2, label=f"Median = {p50:.1f} mm")
    plt.axvline(p90, ls=":",  lw=1.2, label=f"P90 = {p90:.1f} mm")
    plt.xlabel("Dmax (mm)"); plt.ylabel("Count"); plt.title("Distribution of Dmax")
    plt.legend(frameon=False, fontsize=9, loc="upper right")
    plt.subplot(1, 2, 2)
    bins = np.linspace(0, 1, 11)
    sns.histplot(fr, bins=bins, discrete=False)
    for xline in (0.0, 0.25, 0.50, 0.75):
        plt.axvline(xline, ls="--", lw=1.0)
    plt.xlabel("Fractional part of Dmax"); plt.ylabel("Count")
    plt.title("Heaping in fractional part (0–1)")
    plt.tight_layout()

    out_png = Path(out_png)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

plot_heaping_panels(df[dmax_col], OUTDIR / "reports" / "heaping_panels.png")


# In[11]:


# EDA plots
FIGDIR = OUTDIR / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
sns.set_palette("colorblind")

def savefig_both(path_png):
    path_png = Path(path_png)
    path_pdf = path_png.with_suffix(".pdf")
    path_svg = path_png.with_suffix(".svg")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.savefig(path_svg, bbox_inches="tight")
    plt.close()

def annotate_bars(ax, fmt="{:d}"):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt.format(int(round(h))),
                        (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

def add_percent_axis_top(ax, total):
    if total <= 0:
        return
    def count_to_pct(x):
        return 100.0 * x / total
    def pct_to_count(p):
        return p * total / 100.0
    secax = ax.secondary_xaxis('top', functions=(count_to_pct, pct_to_count))
    secax.set_xlabel("Percent of dataset (%)")
    secax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

def freedman_diaconis_bins(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return 10
    q75, q25 = np.percentile(a, [75, 25])
    iqr = max(q75 - q25, 1e-12)
    bw = 2 * iqr * (n ** (-1/3))
    if bw <= 0:
        return 40
    k = int(np.clip(np.ceil((a.max() - a.min()) / bw), 10, 120))
    return k

total_rows = int(len(df))

# Dmax distribution (hist + KDE)
d = pd.to_numeric(df[dmax_col], errors="coerce").dropna()
if len(d) > 0:
    plt.figure(figsize=(6.4, 4.2))
    bins = freedman_diaconis_bins(d.to_numpy())
    ax = sns.histplot(d, bins=bins, kde=True)
    ax.set_xlabel("Dmax (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Dmax")
    p50, p90 = np.percentile(d, [50, 90])
    ax.axvline(p50, ls="--", lw=1.6, label=f"Median = {p50:.1f} mm")
    ax.axvline(p90, ls=":",  lw=1.6, label=f"P90 = {p90:.1f} mm")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    savefig_both(FIGDIR / "dmax_hist_kde.png")

# Element prevalence bar plot
preval = (df[elem_cols] > 0).sum().sort_values(ascending=False)
N = 41 if len(preval) > 25 else int(len(preval))
top = preval.head(N).sort_values(ascending=False)
others = int(preval.iloc[N:].sum()) if N < len(preval) else 0

plt.figure(figsize=(7.2, 6.5 if N > 20 else 5.0))
ax = sns.barplot(x=top.values, y=top.index, orient="h")
ax.set_xlabel("Non-zero count")
ax.set_ylabel("Element")
title = f"Element prevalence (top {N}" + (f" + others={others}" if others > 0 else "") + ")"
ax.set_title(title)
annotate_bars(ax, fmt="{:d}")
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
add_percent_axis_top(ax, total_rows)
sns.despine(left=False, bottom=False)
savefig_both(FIGDIR / "element_prevalence_topN.png")

# Solvent-family counts bar plot
comp = df[elem_cols].fillna(0).to_numpy()
fam = pd.Series(np.array(elem_cols)[comp.argmax(axis=1)], name="family")
fc = fam.value_counts().sort_values(ascending=False)

plt.figure(figsize=(7.2, 6.0 if len(fc) > 14 else 4.8))
ax = sns.barplot(x=fc.values, y=fc.index, orient="h")
ax.set_xlabel("Count")
ax.set_ylabel("Solvent family (max-fraction element)")
ax.set_title("Solvent-family counts")
annotate_bars(ax, fmt="{:d}")
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
add_percent_axis_top(ax, total_rows)
sns.despine(left=False, bottom=False)
savefig_both(FIGDIR / "solvent_family_counts_h.png")


# In[12]:


# Splits (random, family-out)
if "family" not in df.columns:
    comp = df[elem_cols].fillna(0).to_numpy()
    df["family"] = pd.Series(np.array(elem_cols)[comp.argmax(axis=1)], name="family")

if "signature" not in df.columns:
    df["signature"] = composition_signature(df, elem_cols)

def balanced_assign_groups(sig_to_indices, targets, seed=SEED, universe_index=None):
    rng_local = np.random.default_rng(seed)

    if universe_index is None:
        if len(sig_to_indices) == 0:
            universe = []
        else:
            universe = np.concatenate(list(sig_to_indices.values())).tolist()
    else:
        universe = np.array(universe_index, dtype=int).tolist()

    sig_sizes = [(s, len(ix)) for s, ix in sig_to_indices.items()]
    rng_local.shuffle(sig_sizes)
    sig_sizes.sort(key=lambda t: t[1], reverse=True)

    loads = {k: 0 for k in targets}
    assigned_sigs = {k: [] for k in targets}

    def remaining(k):
        return targets[k] - loads[k]

    for sig, sz in sig_sizes:
        k_best = max(targets.keys(), key=lambda k: remaining(k))
        assigned_sigs[k_best].append(sig)
        loads[k_best] += sz

    split_idx = {}
    for k, sigs in assigned_sigs.items():
        split_idx[k] = (np.concatenate([sig_to_indices[s] for s in sigs]).tolist()
                        if sigs else [])

    def move_one_smallest(from_k, to_k):
        if not assigned_sigs[from_k]:
            return False
        small_sig = min(assigned_sigs[from_k], key=lambda s: len(sig_to_indices[s]))
        assigned_sigs[from_k].remove(small_sig)
        assigned_sigs[to_k].append(small_sig)
        for kk in targets:
            split_idx[kk] = (np.concatenate([sig_to_indices[s] for s in assigned_sigs[kk]]).tolist()
                             if assigned_sigs[kk] else [])
            loads[kk] = sum(len(sig_to_indices[s]) for s in assigned_sigs[kk])
        return True

    for must_have in ["cal", "test"]:
        if must_have in targets and len(split_idx[must_have]) == 0 and len(split_idx["train"]) > 0:
            move_one_smallest("train", must_have)

    all_indices = sum((split_idx[k] for k in targets), [])
    assert len(all_indices) == len(set(all_indices)), "Split leakage: indices overlap between splits."
    assert set(all_indices) == set(universe), "Split coverage error: some rows not assigned."

    return split_idx

# Random split (group by signature)
sig_groups = df.groupby("signature").indices
all_signatures = list(sig_groups.keys())

N = len(df)
n_train = int(0.75 * N)
n_cal   = int(0.15 * N)
n_test  = N - n_train - n_cal
targets = {"train": n_train, "cal": n_cal, "test": n_test}

split_random = balanced_assign_groups(
    sig_groups,
    targets={"train": n_train, "cal": n_cal, "test": n_test},
    seed=SEED,
    universe_index=df.index.values,
)

(OUTDIR / "splits").mkdir(parents=True, exist_ok=True)

with open(OUTDIR / "splits_random.json", "w") as f:
    json.dump(split_random, f, indent=2)

# Family-out splits (one held-out family at a time)
split_family_out = {}
fam_counts = df["family"].value_counts()

for fam_name, fam_count in fam_counts.items():
    if fam_count < FAMILY_MIN_TEST:
        continue

    test_idx = df.index[df["family"] == fam_name].tolist()
    remain_mask = (df["family"] != fam_name) & df["signature"].notna()
    remain_idx = df.index[remain_mask].tolist()

    sig_groups_remain = df.loc[remain_idx].groupby("signature").indices
    N_rem = len(remain_idx)
    targets_rem = {"train": int(0.80 * N_rem), "cal": N_rem - int(0.80 * N_rem)}
    splits_rem = balanced_assign_groups(sig_groups_remain, targets_rem, seed=SEED)

    split_family_out[fam_name] = {
        "train": splits_rem["train"],
        "cal":   splits_rem["cal"],
        "test":  test_idx,
    }

with open(OUTDIR / "splits_family_out.json", "w") as f:
    json.dump(split_family_out, f, indent=2)

summary = {
    "random_counts": {k: len(v) for k, v in split_random.items()},
    "random_signature_counts": {k: int(len(df.loc[v, "signature"].unique())) for k, v in split_random.items()},
    "families_available": fam_counts.to_dict(),
    "family_out_keys": list(split_family_out.keys()),
    "notes": {
        "grouping": "by composition signature (no leakage across splits)",
        "random_ratio": "train 75%, cal 15%, test 10% (balanced by group size)",
        "family_out_ratio_on_remainder": "train 80%, cal 20% (balanced by group size)"
    },
    "seed": int(SEED)
}
with open(OUTDIR / "splits" / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

split_labels = np.full(N, "unassigned", dtype=object)
for k, idxs in split_random.items():
    split_labels[np.array(idxs, int)] = k
df_splits = pd.DataFrame({"split": split_labels, "family": df["family"].values})
fam_by_split = df_splits.groupby(["split", "family"]).size().unstack(fill_value=0)
fam_by_split.to_csv(OUTDIR / "splits" / "family_distribution_random.csv")

print("Saved splits: random and family-out.")
print("Random counts (rows):", summary["random_counts"])
print("Family-out splits for families:", ", ".join(summary["family_out_keys"]) or "(none met threshold)")


# In[13]:


# UMAP of all compositions + dominant elements (>=25 at.%)
SRC_DIR = OUTDIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)
FIGDIR = OUTDIR / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# 1) Build UMAP on elemental fractions only
X_all = df[elem_cols].to_numpy()
reducer = umap.UMAP(
    random_state=SEED,
    n_neighbors=30,
    min_dist=0.1,
    metric="manhattan"
)
Z = reducer.fit_transform(X_all)  # shape (n_samples, 2)

# 2) Dominant element(s): all elements with >=25 at.% (else top element)
def dominant_elements(row, threshold=0.25):
    els = [el for el in elem_cols if row[el] >= threshold]
    if els:
        return "|".join(els)
    return row[elem_cols].idxmax()

dominants = df.apply(dominant_elements, axis=1)

# 3) Assemble output: composition + UMAP + dominant elements
umap_out = df[elem_cols].copy()
umap_out["UMAP1"] = Z[:, 0]
umap_out["UMAP2"] = Z[:, 1]
umap_out["dominant_elements_ge25atpct"] = dominants

# 4) Save CSV
out_csv = SRC_DIR / "umap_alloys_with_coords_and_dominant_elements.csv"
umap_out.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# 5) Optional quick scatter (all points, single color to avoid legend clutter)
plt.figure(figsize=(6.4, 5.4))
plt.scatter(umap_out["UMAP1"], umap_out["UMAP2"], s=14, alpha=0.85, edgecolors="none")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.title("UMAP of compositions (all data)")
plt.tight_layout()

# If you have your own helper; else use standard savefig:
# savefig_both(FIGDIR / "umap_all_compositions.png")
plt.savefig(FIGDIR / "umap_all_compositions.png", dpi=300, bbox_inches="tight")
plt.close()


# In[14]:


# Feature engineering
RAD = {# metallic radius in Å
 'Ag':1.44,'Al':1.43,'Au':1.44,'B':0.85,'Be':1.12,'C':0.77,'Ca':1.97,'Ce':1.82,'Co':1.25,'Cr':1.28,'Cu':1.28,
 'Dy':1.78,'Er':1.76,'Fe':1.26,'Ga':1.36,'Gd':1.80,'Hf':1.58,'La':1.88,'Mg':1.60,'Mn':1.27,'Mo':1.39,
 'Nb':1.43,'Nd':1.82,'Ni':1.24,'P':1.06,'Pd':1.37,'Pr':1.82,'Pt':1.39,'Sc':1.64,'Si':1.11,'Sm':1.85,'Sn':1.62,
 'Ta':1.43,'Tb':1.78,'Ti':1.47,'Tm':1.75,'V':1.34,'W':1.37,'Y':1.80,'Zn':1.33,'Zr':1.60
}
EN = { # Pauling electronegativity
 'Ag':1.93,'Al':1.61,'Au':2.54,'B':2.04,'Be':1.57,'C':2.55,'Ca':1.00,'Ce':1.12,'Co':1.88,'Cr':1.66,'Cu':1.90,
 'Dy':1.22,'Er':1.24,'Fe':1.83,'Ga':1.81,'Gd':1.20,'Hf':1.30,'La':1.10,'Mg':1.31,'Mn':1.55,'Mo':2.16,  
 'Nb':1.60,'Nd':1.14,'Ni':1.91,'P':2.19,'Pd':2.20,'Pr':1.13,'Pt':2.28,'Sc':1.36,'Si':1.90,'Sm':1.17,'Sn':1.96,
 'Ta':1.50,'Tb':1.10,'Ti':1.54,'Tm':1.25,'V':1.63,'W':2.36,'Y':1.22,'Zn':1.65,'Zr':1.33
}
VEC = { # valence electron counts
 'Ag':1,'Al':3,'Au':1,'B':3,'Be':2,'C':4,'Ca':2,'Ce':3,'Co':9,'Cr':6,'Cu':1,
 'Dy':3,'Er':3,'Fe':8,'Ga':3,'Gd':3,'Hf':4,'La':3,'Mg':2,'Mn':7,'Mo':6,'Nb':5,
 'Nd':3,'Ni':10,'P':5,'Pd':10,'Pr':3,'Pt':10,'Sc':3,'Si':4,'Sm':3,'Sn':4,
 'Ta':5,'Tb':3,'Ti':4,'Tm':3,'V':5,'W':6,'Y':3,'Zn':2,'Zr':4
}

_missing = {
    "RAD": [e for e in elem_cols if e not in RAD],
    "EN":  [e for e in elem_cols if e not in EN],
    "VEC": [e for e in elem_cols if e not in VEC],
}
for k, vals in _missing.items():
    if vals:
        raise KeyError(f"{k} table missing values for: {', '.join(vals)}")

def prop_vector(prop_dict, elems=elem_cols):
    """Return property array aligned to elem_cols; safe-casts to float."""
    return np.array([float(prop_dict.get(e)) for e in elems], dtype=float)

RAD_VEC = prop_vector(RAD)
EN_VEC  = prop_vector(EN)
VEC_VEC = prop_vector(VEC)


# In[15]:


# Features builder
def comp_stats_from_props_ROWWISE(c_row, p_vec):
    c = np.asarray(c_row, float)
    p = np.asarray(p_vec, float)
    mask = (c > 0) & (~np.isnan(p))
    if not np.any(mask):
        return [np.nan]*4
    c = c[mask]; p = p[mask]
    w = c / c.sum()
    mu  = float(np.sum(w * p))
    var = float(np.sum(w * (p - mu)**2))
    return [mu, var, float(p.min()), float(p.max())]

def delta_size_mismatch_ROWWISE(c_row, r_vec):
    c = np.asarray(c_row, float)
    r = np.asarray(r_vec, float)
    mask = (c > 0) & np.isfinite(r)
    if not np.any(mask) or c[mask].sum() <= 0:
        return np.nan
    w = c[mask] / c[mask].sum()
    rbar = float(np.dot(w, r[mask]))
    return 100.0 * float(np.sqrt(np.dot(w, (1.0 - r[mask] / rbar)**2)))

def build_features(df_or_arr, elem_cols):
    if isinstance(df_or_arr, pd.DataFrame):
        C = df_or_arr[elem_cols].to_numpy(dtype=float, copy=False)
    else:
        C = np.asarray(df_or_arr, dtype=float)
        if C.ndim != 2 or C.shape[1] != len(elem_cols):
            raise ValueError("Input must have shape (n_samples, len(elem_cols)).")

    C = np.clip(C, 0.0, None)
    row_sums = C.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    W = C / safe_sums

    n, d = C.shape
    rad = RAD_VEC
    en  = EN_VEC
    vec = VEC_VEC

    def stats_from_prop_vectorized(prop_vec):
        mask = ~np.isnan(prop_vec)
        if not np.any(mask):
            mu=var=pmin=pmax=np.full(n, np.nan); return mu, var, pmin, pmax
        P   = prop_vec[mask]
        C_m = C[:, mask]
        W_m = W[:, mask]
        present = (C_m > 0)
        wsum = np.where(present, W_m, 0.0).sum(axis=1, keepdims=True)
        wsum = np.where(wsum > 0, wsum, 1.0)
        Wn = np.where(present, W_m / wsum, 0.0)
        mu  = (Wn * P).sum(axis=1)
        var = (Wn * (P - mu[:, None])**2).sum(axis=1)
        has_any = present.any(axis=1)
        mu  = np.where(has_any, mu,  np.nan)
        var = np.where(has_any, var, np.nan)
        P_tiled   = np.broadcast_to(P, (n, P.shape[0]))
        P_mask_min = np.where(present, P_tiled,  np.inf)
        P_mask_max = np.where(present, P_tiled, -np.inf)
        pmin = np.min(P_mask_min, axis=1)
        pmax = np.max(P_mask_max, axis=1)
        pmin = np.where(has_any, pmin, np.nan)
        pmax = np.where(has_any, pmax, np.nan)
        return mu, var, pmin, pmax

    rad_mu, rad_var, rad_min, rad_max = stats_from_prop_vectorized(rad)
    en_mu,  en_var,  en_min,  en_max  = stats_from_prop_vectorized(en)
    vec_mu, vec_var, vec_min, vec_max = stats_from_prop_vectorized(vec)

    # δ-size mismatch
    rmask = ~np.isnan(rad)
    if np.any(rmask):
        r = rad[rmask]
        W_r = W[:, rmask]
        present_r = (C[:, rmask] > 0)
        wsum_r = np.where(present_r, W_r, 0.0).sum(axis=1, keepdims=True)
        wsum_r = np.where(wsum_r > 0, wsum_r, 1.0)
        Wn_r = np.where(present_r, W_r / wsum_r, 0.0)
        rbar = (Wn_r * r).sum(axis=1)
        delta_size = 100.0 * np.sqrt((Wn_r * (1.0 - r / rbar[:, None])**2).sum(axis=1))
        has_any_r = present_r.any(axis=1)
        delta_size = np.where(has_any_r, delta_size, np.nan)
    else:
        delta_size = np.full(n, np.nan)
    
    elem_block = {f"at_{el}": C[:, j] for j, el in enumerate(elem_cols)}
    out = {**elem_block,
           "rad_mu": rad_mu, "rad_var": rad_var, "rad_min": rad_min, "rad_max": rad_max,
           "en_mu":  en_mu,  "en_var":  en_var,  "en_min":  en_min,  "en_max":  en_max,
           "vec_mu": vec_mu, "vec_var": vec_var, "vec_min": vec_min, "vec_max": vec_max,
           "delta_size": delta_size}
    return pd.DataFrame(out)

# Add family if missing
if "family" not in df.columns:
    comp = df[elem_cols].to_numpy()
    df["family"] = pd.Series(np.array(elem_cols)[comp.argmax(axis=1)], name="family")

# Build training features (engineered + family one-hots)
X_engineered = build_features(df, elem_cols)
FAM_LEVELS = sorted(pd.Series(df["family"], dtype="string").unique().tolist())
fam_dtype = pd.CategoricalDtype(categories=FAM_LEVELS)
X_fam = pd.get_dummies(df["family"].astype(fam_dtype), prefix="fam", dtype=float)
X_full = pd.concat([X_engineered.reset_index(drop=True),
                    X_fam.reset_index(drop=True)], axis=1)

if not np.isfinite(X_full.to_numpy()).all():
    raise ValueError("Non-finite values detected in engineered features (check descriptor tables and input).")

X_COLUMNS = list(X_full.columns)

# Parallel no-family feature set for ablations
X_NOFAM_COLUMNS = list(X_engineered.columns)

def make_features_no_family(X_comp):
    X_arr = np.asarray(X_comp, float)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    X_arr = np.clip(X_arr, 0.0, None)
    rs = X_arr.sum(axis=1, keepdims=True)
    X_arr = X_arr / np.where(rs > 0, rs, 1.0)
    df_tmp = pd.DataFrame(X_arr, columns=elem_cols)
    feats  = build_features(df_tmp, elem_cols)
    feats  = feats.reindex(columns=X_NOFAM_COLUMNS, fill_value=0.0)
    if not np.isfinite(feats.to_numpy()).all():
        raise ValueError("Non-finite values in no-family inference features.")
    return feats


# In[16]:


def make_features_from_compositions(X_comp):
    X_arr = np.asarray(X_comp, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    if X_arr.ndim != 2 or X_arr.shape[1] != len(elem_cols):
        raise ValueError(f"X_comp must have shape (n, {len(elem_cols)}); got {X_arr.shape}.")

    X_arr = np.clip(X_arr, 0.0, None)
    rs = X_arr.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    X_arr = X_arr / rs

    df_tmp = pd.DataFrame(X_arr, columns=elem_cols)
    feats  = build_features(df_tmp, elem_cols)

    # family one-hots over frozen vocabulary
    top_idx  = np.argmax(X_arr, axis=1)
    fam_pred = np.array(elem_cols, dtype=object)[top_idx]
    fam_cat  = pd.Categorical(fam_pred, categories=FAM_LEVELS)
    fam_oh   = pd.get_dummies(fam_cat, prefix="fam", dtype=float)

    all_fam_cols = [f"fam_{f}" for f in FAM_LEVELS]
    fam_oh = fam_oh.reindex(columns=all_fam_cols, fill_value=0.0)

    # concat + enforce EXACT training schema & order
    feats_full = pd.concat([feats.reset_index(drop=True),
                            fam_oh.reset_index(drop=True)], axis=1)
    feats_full = feats_full.reindex(columns=X_COLUMNS, fill_value=0.0)

    if not np.isfinite(feats_full.to_numpy()).all():
        raise ValueError("Non-finite values detected in inference features.")

    return feats_full


# In[17]:


def _family_dummies_from_composition_matrix(X_comp):
    """
    Build fam_* one-hots from compositions, aligned to the frozen vocabulary FAM_LEVELS.

    Parameters
    ----------
    X_comp : array-like, shape (n_samples, len(elem_cols)) or (len(elem_cols),)
        Compositions (fractions). Values are clipped to [0, ∞) and NaNs -> 0
        before selecting the top-element family.

    Returns
    -------
    fam_dm : pandas.DataFrame, shape (n_samples, len(FAM_LEVELS))
        One-hot matrix with columns exactly ['fam_<family>' for family in FAM_LEVELS].
    """
    X = np.asarray(X_comp, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != len(elem_cols):
        raise ValueError(f"X_comp must have shape (n, {len(elem_cols)}); got {X.shape}.")

    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, 0.0, None)

    top_idx = np.argmax(X, axis=1)
    fams    = np.array(elem_cols, dtype=object)[top_idx]

    fam_cat = pd.Categorical(fams, categories=FAM_LEVELS)
    fam_dm  = pd.get_dummies(fam_cat, prefix="fam", dtype=float)

    all_fam_cols = [f"fam_{f}" for f in FAM_LEVELS]
    fam_dm = fam_dm.reindex(columns=all_fam_cols, fill_value=0.0)

    if not np.isfinite(fam_dm.to_numpy()).all():
        raise ValueError("Non-finite values in family one-hots.")
    return fam_dm


# In[18]:


def robust_L_for_comps_hi(
    X_elem,
    *,
    q_cal_robust,
    model=None,
    eps=ROBUST_EPS,
    K=ROBUST_SAMPLES,
    rng=None
):

    if not np.isfinite(q_cal_robust):
        raise ValueError("q_cal_robust must be a finite float.")

    if model is None:
        if "cat_qt_hi" not in globals():
            raise ValueError("Provide `model` or ensure `cat_qt_hi` exists.")
        model = cat_qt_hi

    if rng is None:
        rng = np.random.default_rng(SEED + DRIFT_SEED)

    X = np.asarray(X_elem, float)
    if X.ndim == 1:
        X = X[None, :]

    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, 0.0, None)
    rs = X.sum(axis=1, keepdims=True)
    X = X / np.where(rs > 0, rs, 1.0)

    if eps <= 0.0 or K <= 1:
        feats = make_features_from_compositions(X)
        qhat  = np.asarray(model.predict(feats), float).ravel()
        return np.exp(qhat - float(q_cal_robust))

    L_mm = np.empty(X.shape[0], float)
    for i, x in enumerate(X):
        rng_i = np.random.default_rng(SEED + DRIFT_SEED)
        Xj = jitter_in_L1_ball_simplex(x, eps=float(eps), K=int(K), rng=rng_i)
        rsj = Xj.sum(axis=1, keepdims=True)
        Xj  = Xj / np.where(rsj > 0, rsj, 1.0)

        feats_j = make_features_from_compositions(Xj)
        qj      = np.asarray(model.predict(feats_j), float)
        q_min   = float(np.nanmin(qj))
        if not np.isfinite(q_min):
            q_min = float(np.asarray(model.predict(
                make_features_from_compositions(x[None, :])
            ), float).ravel()[0])

        L_mm[i] = float(np.exp(q_min - float(q_cal_robust)))
    return L_mm


# Training (ET point, CatBoost τ-quantile)
CV_FOLDS          = 5
HPO_BUDGET_ET     = 300
HPO_BUDGET_CAT    = 300
N_BOOT            = 2000
METRICS_DIR       = OUTDIR / "reports" / "metrics"
MODELS_DIR        = OUTDIR / "models"
HPO_DIR           = OUTDIR / "reports" / "hpo"
for p in [METRICS_DIR, MODELS_DIR, HPO_DIR]:
    p.mkdir(parents=True, exist_ok=True)

X = X_full
y_log = np.log(df[dmax_col].astype(float).values)

idx_train = split_random["train"]; idx_cal = split_random["cal"]; idx_test = split_random["test"]
X_train, y_train = X.iloc[idx_train], y_log[idx_train]
X_cal,   y_cal   = X.iloc[idx_cal],   y_log[idx_cal]
X_test,  y_test  = X.iloc[idx_test],  y_log[idx_test]

sig_tr = set(df.loc[idx_train, "signature"])
sig_ca = set(df.loc[idx_cal,   "signature"])
sig_te = set(df.loc[idx_test,  "signature"])
assert sig_tr.isdisjoint(sig_ca) and sig_tr.isdisjoint(sig_te) and sig_ca.isdisjoint(sig_te), \
    "Signature leakage across splits."
assert len({*idx_train, *idx_cal, *idx_test}) == len(idx_train) + len(idx_cal) + len(idx_test), \
    "Index overlap across splits."

# Family reweighting (class imbalance)
fam_counts_tr = df.loc[idx_train, "family"].value_counts()
w_train = df.loc[idx_train, "family"].map(lambda f: 1.0 / np.sqrt(fam_counts_tr.get(f, 1))).to_numpy(float)
w_train = w_train * (len(w_train) / w_train.sum())

# Grouped CV by composition signature
groups_train = df.loc[idx_train, "signature"].values
cv = GroupKFold(n_splits=CV_FOLDS)

# Extra Trees (point model)
et_scoring = "neg_mean_absolute_error"
et_space = {
    'n_estimators'     : (10, 400),
    'max_depth'        : (2, 30),
    'min_samples_split': (2, 30),
    'min_samples_leaf' : (1, 30),
    'max_features'     : (0.1, 1.0),
    'bootstrap'        : [True, False],
    'criterion'        : ['squared_error', 'absolute_error'],
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    et_bayes = BayesSearchCV(
        estimator=ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
        search_spaces=et_space,
        n_iter=HPO_BUDGET_ET,
        scoring=et_scoring,
        cv=cv,
        n_jobs=1,
        random_state=SEED,
        verbose=1,
        refit=True,
        return_train_score=False,
        error_score="raise"
    )
    et_bayes.fit(X_train, y_train, groups=groups_train, sample_weight=w_train)

et = et_bayes.best_estimator_
pd.DataFrame(et_bayes.cv_results_).to_csv(HPO_DIR / "et_bayes_cv_results.csv", index=False)
with open(HPO_DIR / "et_bayes_best_params.json", "w") as f:
    json.dump(et_bayes.best_params_, f, indent=2)

yhat_train = et.predict(X_train)
yhat_cal   = et.predict(X_cal)
yhat_test  = et.predict(X_test)

et_r2   = r2_score(y_test, yhat_test)
et_mae  = mean_absolute_error(y_test, yhat_test)
et_rmse = np.sqrt(mean_squared_error(y_test, yhat_test))
print(f"[ET Bayes→TEST] R2={et_r2:.4f} | MAE={et_mae:.4f} | RMSE={et_rmse:.4f} (log)")

y_test_mm    = np.exp(y_test)
yhat_test_mm = np.exp(yhat_test)
et_mae_mm    = float(mean_absolute_error(y_test_mm, yhat_test_mm))
et_rmse_mm   = float(np.sqrt(mean_squared_error(y_test_mm, yhat_test_mm)))

joblib.dump(et, MODELS_DIR / "et_point.pkl")

# CatBoost (τ-quantile model)
def neg_pinball_scorer_tau(estimator, X, y, alpha=QT_TAU):
    yp = estimator.predict(X)
    return -mean_pinball_loss(y, yp, alpha=alpha)

catb_base = CatBoostRegressor(
    loss_function=f"Quantile:alpha={QT_TAU}",
    eval_metric=f"Quantile:alpha={QT_TAU}",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
    od_type="Iter",
    od_wait=100
)

space_bayesian = {
    'iterations'         : Integer(500, 1000),
    'depth'              : Integer(2, 10),
    'learning_rate'      : Real(1e-3, 3e-1, prior='log-uniform'),
    'l2_leaf_reg'        : Real(1e-3, 10.0, prior='log-uniform'),
    'random_strength'    : Real(0.0, 1.0),
    'rsm'                : Real(0.5, 1.0),
    'border_count'       : Integer(32, 255),
    'bootstrap_type'     : Categorical(['Bayesian']),
    'bagging_temperature': Real(0.0, 1.0),
}
space_bernoulli = {
    'iterations'     : Integer(500, 1000),
    'depth'          : Integer(2, 10),
    'learning_rate'  : Real(1e-3, 3e-1, prior='log-uniform'),
    'l2_leaf_reg'    : Real(1e-3, 10.0, prior='log-uniform'),
    'random_strength': Real(0.0, 1.0),
    'rsm'            : Real(0.5, 1.0),
    'border_count'   : Integer(32, 255),
    'bootstrap_type' : Categorical(['Bernoulli']),
    'subsample'      : Real(0.5, 1.0),
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    catb_bayes = BayesSearchCV(
        estimator=catb_base,
        search_spaces=[(space_bayesian, HPO_BUDGET_CAT//2),
                       (space_bernoulli, HPO_BUDGET_CAT//2)],
        scoring=neg_pinball_scorer_tau,
        cv=cv,
        n_jobs=1,
        random_state=SEED,
        verbose=1,
        refit=True,
        return_train_score=False,
        error_score=np.nan
    )
    catb_bayes.fit(X_train, y_train, groups=groups_train, sample_weight=w_train)

cat_qt = catb_bayes.best_estimator_
pd.DataFrame(catb_bayes.cv_results_).to_csv(HPO_DIR / "catboost_tau_bayes_cv_results.csv", index=False)
with open(HPO_DIR / "catboost_tau_bayes_best_params.json", "w") as f:
    json.dump(catb_bayes.best_params_, f, indent=2)

# Predictions
qhat_train = cat_qt.predict(X_train)
qhat_cal   = cat_qt.predict(X_cal)
qhat_test  = cat_qt.predict(X_test)

# Calibration & loss for τ
obs_tau_cal  = float(np.mean(y_cal  <= qhat_cal))
obs_tau_test = float(np.mean(y_test <= qhat_test))
pin_cal      = float(mean_pinball_loss(y_cal,  qhat_cal,  alpha=QT_TAU))
pin_test     = float(mean_pinball_loss(y_test, qhat_test, alpha=QT_TAU))

print(f"[CatBoost τ={QT_TAU:.2f}] P(Y≤q̂): CAL={obs_tau_cal:.3f}, TEST={obs_tau_test:.3f} | "
      f"pinball: CAL={pin_cal:.5f}, TEST={pin_test:.5f}")

# Save model
cat_qt.save_model(str(MODELS_DIR / f"cat_qt_tau{QT_TAU:.2f}.cbm"))

# Out-of-fold (OOF) estimates (same hyperparams across folds)
def oof_predictions(estimator_ctor, params, X_df, y_vec, groups, sample_weight=None):
    """
    Produce OOF predictions with GroupKFold (no leakage), refitting a fresh model each fold
    with the *fixed* best hyperparameters.
    """
    oof = np.full_like(y_vec, fill_value=np.nan, dtype=float)
    gkf = GroupKFold(n_splits=CV_FOLDS)
    for tr_idx, va_idx in gkf.split(X_df, y_vec, groups):
        model = estimator_ctor(**params)
        if sample_weight is None:
            model.fit(X_df.iloc[tr_idx], y_vec[tr_idx])
        else:
            model.fit(X_df.iloc[tr_idx], y_vec[tr_idx], sample_weight=sample_weight[tr_idx])
        oof[va_idx] = model.predict(X_df.iloc[va_idx])
    assert np.isfinite(oof).all()
    return oof

# ET OOF (point)
et_params_fixed = et.get_params(deep=False)
y_oof_et = oof_predictions(ExtraTreesRegressor, et_params_fixed, X_train, y_train, groups_train, sample_weight=w_train)

# CatBoost OOF (τ-quantile)
cat_params_fixed = cat_qt.get_params()

def _cat_ctor(**p):
    p = dict(p)
    verbose = p.pop("verbose", False)
    allow_writing_files = p.pop("allow_writing_files", False)
    return CatBoostRegressor(**p,
                             verbose=verbose,
                             allow_writing_files=allow_writing_files)

y_oof_cat = oof_predictions(_cat_ctor, cat_params_fixed,
                            X_train, y_train, groups_train,
                            sample_weight=w_train)

np.save(OUTDIR / "oof_et_log.npy", y_oof_et)
np.save(OUTDIR / "oof_cat_tau_log.npy", y_oof_cat)

# Bootstrap CIs for test metrics
rng_ci = np.random.default_rng(SEED + 123)

def bootstrap_ci(y_true, y_pred, fn, n_boot=N_BOOT, rng=rng_ci):
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(fn(y_true[idx], y_pred[idx]))
    vals = np.sort(vals)
    lo = float(vals[int(0.025 * n_boot)])
    hi = float(vals[int(0.975 * n_boot)])
    return lo, hi

# Log-space
et_mae_ci  = bootstrap_ci(y_test, yhat_test, mean_absolute_error)
et_rmse_ci = bootstrap_ci(y_test, yhat_test, lambda a,b: np.sqrt(mean_squared_error(a,b)))

# mm-space
et_mae_mm_ci  = bootstrap_ci(y_test_mm, yhat_test_mm, mean_absolute_error)
et_rmse_mm_ci = bootstrap_ci(y_test_mm, yhat_test_mm, lambda a,b: np.sqrt(mean_squared_error(a,b)))

# τ-quantile: pinball on test
pin_test_ci = bootstrap_ci(y_test, qhat_test, lambda a,b: mean_pinball_loss(a,b,alpha=QT_TAU))

# Record metrics
metrics = {
    "et": {
        "test_log": {"R2": float(et_r2), "MAE": float(et_mae), "RMSE": float(et_rmse),
                     "MAE_CI95": et_mae_ci, "RMSE_CI95": et_rmse_ci},
        "test_mm":  {"MAE": et_mae_mm, "RMSE": et_rmse_mm,
                     "MAE_CI95": et_mae_mm_ci, "RMSE_CI95": et_rmse_mm_ci},
        "best_params": et_bayes.best_params_,
    },
    "cat_tau": {
        "tau": float(QT_TAU),
        "calibration": {"CAL": obs_tau_cal, "TEST": obs_tau_test},
        "pinball": {"CAL": pin_cal, "TEST": pin_test, "TEST_CI95": pin_test_ci},
        "best_params": catb_bayes.best_params_,
    },
    "splits_random_counts": {k: len(v) for k, v in split_random.items()}
}
with open(METRICS_DIR / "summary_training.json", "w") as f:
    json.dump(metrics, f, indent=2)


# In[20]:


# Figures
FIGDIR = OUTDIR / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUTDIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

def savefig_both(path_png):
    path_png = Path(path_png)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

# Basic parity (mm)
y_test_mm    = np.asarray(np.exp(y_test)).ravel()
yhat_test_mm = np.asarray(np.exp(yhat_test)).ravel()

plt.figure(figsize=(5.6, 5.2))
ax = sns.scatterplot(x=y_test_mm, y=yhat_test_mm, s=18, alpha=0.6, edgecolor=None)
xmin = float(np.nanmin([y_test_mm.min(), yhat_test_mm.min()]))
xmax = float(np.nanmax([y_test_mm.max(), yhat_test_mm.max()]))
lims = [xmin, xmax]
ax.plot(lims, lims, linestyle="--")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect('equal', 'box')
ax.set_xlabel("True Dmax (mm)"); ax.set_ylabel("Predicted Dmax (mm)")
rmse_mm = float(np.sqrt(mean_squared_error(y_test_mm, yhat_test_mm)))
mae_mm  = float(mean_absolute_error(y_test_mm, yhat_test_mm))
ax.text(0.01, 0.99, f"RMSE={rmse_mm:.2f} mm\nMAE={mae_mm:.2f} mm",
        transform=ax.transAxes, ha="left", va="top", fontsize=10)
ax.set_title("Parity (mm)")
savefig_both(FIGDIR / "parity_mm_et.png")

# Hexbin (density parity) for cluttered regimes
plt.figure(figsize=(5.6, 5.2))
hb = plt.hexbin(y_test_mm, yhat_test_mm, gridsize=35, mincnt=1, linewidths=0.0)
plt.plot(lims, lims, linestyle="--")
plt.xlabel("True Dmax (mm)"); plt.ylabel("Predicted Dmax (mm)")
plt.title("Parity (mm) — density")
cb = plt.colorbar(hb); cb.set_label("Counts")
plt.axis('equal'); plt.xlim(lims); plt.ylim(lims)
savefig_both(FIGDIR / "parity_mm_hex_et.png")

pd.DataFrame({"true_mm": y_test_mm, "pred_mm": yhat_test_mm}) \
  .to_csv(SRC_DIR / "parity_et_test.csv", index=False)

#  Residual diagnostics (log)
res_log = np.asarray(y_test - yhat_test).ravel()
yhat_log = np.asarray(yhat_test).ravel()
m_res = np.isfinite(res_log)
m_sc  = m_res & np.isfinite(yhat_log)

# residual histogram + KDE
plt.figure(figsize=(6.2, 4.0))
ax = sns.histplot(res_log[m_res], bins=40, kde=True)
ax.axvline(0.0, ls="--")
ax.set_xlabel("Residual (true − pred) on log scale")
ax.set_ylabel("Count")
ax.set_title("Residuals (log scale)")
savefig_both(FIGDIR / "residual_hist_log_et.png")

# residual vs predicted with heteroscedasticity bands (median/IQR/±1σ)
df_sc = pd.DataFrame({"yhat_log": yhat_log[m_sc], "res_log": res_log[m_sc]})

df_sc["bin"] = pd.qcut(df_sc["yhat_log"], q=15, duplicates="drop")
bins_agg = df_sc.groupby("bin").agg(
    yhat_c=("yhat_log", "mean"),
    res_med=("res_log", "median"),
    res_p25=("res_log", lambda a: float(np.nanpercentile(a, 25))),
    res_p75=("res_log", lambda a: float(np.nanpercentile(a, 75))),
    res_std=("res_log", "std"),
    n=("res_log", "size"),
).reset_index(drop=True)

plt.figure(figsize=(6.6, 4.2))
ax = sns.scatterplot(x=df_sc["yhat_log"], y=df_sc["res_log"], s=14, alpha=0.35, edgecolor=None)
ax.axhline(0.0, ls="--", lw=1.0, c='k')

ax.plot(bins_agg["yhat_c"], bins_agg["res_med"], lw=2, label="Median residual")
ax.fill_between(
    bins_agg["yhat_c"], bins_agg["res_p25"], bins_agg["res_p75"],
    alpha=0.2, label="IQR"
)

ax.plot(bins_agg["yhat_c"], +bins_agg["res_std"], lw=1, ls=":", label="+1σ")
ax.plot(bins_agg["yhat_c"], -bins_agg["res_std"], lw=1, ls=":", label="-1σ")
ax.set_xlabel("Predicted log Dmax")
ax.set_ylabel("Residual (true − pred)")
ax.set_title("Residuals vs prediction (log) with dispersion bands")
ax.legend(frameon=False, loc="upper right")
savefig_both(FIGDIR / "residual_vs_pred_log_bands_et.png")

pd.DataFrame({"pred_log": yhat_log[m_sc], "residual_log": res_log[m_sc]}) \
  .to_csv(SRC_DIR / "residuals_et_test.csv", index=False)
bins_agg.to_csv(SRC_DIR / "residuals_binned_vs_pred.csv", index=False)

# Family-wise error (ET)
families_test = df.iloc[idx_test]["family"].values
df_res = pd.DataFrame({"family": families_test, "res_log": res_log})
top_fams = df_res["family"].value_counts().head(12).index.tolist()
df_top = df_res[df_res["family"].isin(top_fams)]

plt.figure(figsize=(7.6, 4.8))
ax = sns.boxplot(data=df_top, x="family", y="res_log", order=top_fams, showfliers=False)
ax.axhline(0.0, ls="--")
ax.set_xlabel("Family (top 12 by count)")
ax.set_ylabel("Residual (log)")
ax.set_title("Residuals by solvent family (test)")
plt.xticks(rotation=30, ha="right")
savefig_both(FIGDIR / "residuals_by_family_et.png")

pd.DataFrame({"family": families_test, "residual_log": res_log}) \
  .to_csv(SRC_DIR / "residuals_by_family_samples.csv", index=False)

# Quantile model: calibration with 95% Wilson CIs
qhat_cal  = np.asarray(qhat_cal).ravel()
qhat_test = np.asarray(qhat_test).ravel()

obs_tau_cal  = float(np.mean(y_cal  <= qhat_cal))
obs_tau_test = float(np.mean(y_test <= qhat_test))
pinball_cal  = float(mean_pinball_loss(y_cal,  qhat_cal, alpha=QT_TAU))
pinball_test = float(mean_pinball_loss(y_test, qhat_test, alpha=QT_TAU))

print(f"Quantile check (τ={QT_TAU:.2f}):")
print(f" CAL: observed P(Y≤q̂) = {obs_tau_cal:.3f} (target {QT_TAU:.2f}), pinball = {pinball_cal:.4f}")
print(f" TEST: observed P(Y≤q̂) = {obs_tau_test:.3f} (target {QT_TAU:.2f}), pinball = {pinball_test:.4f}")

def wilson_ci(k, n, alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # ~ N^{-1}(1 - alpha/2)
    phat = k / n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * np.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / denom
    return float(center - half), float(center + half)

k_test = int(np.sum(y_test <= qhat_test)); n_test = int(len(y_test))
lo_ci, hi_ci = wilson_ci(k_test, n_test, alpha=0.05)

plt.figure(figsize=(5.2, 3.8))
vals = [obs_tau_cal, obs_tau_test]
ax = sns.barplot(x=["Cal", "Test"], y=vals)
ax.axhline(QT_TAU, ls="--", label=f"Target τ={QT_TAU:.2f}")

ax.errorbar(1, obs_tau_test, yerr=[[obs_tau_test - lo_ci], [hi_ci - obs_tau_test]],
            fmt='none', capsize=3, lw=1.5, color='black', label="95% CI (Wilson)")
for i, v in enumerate(vals):
    ax.text(i, min(0.99, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 1)
ax.set_ylabel("Observed P(Y ≤ q̂τ)")
ax.set_title("Quantile calibration (single τ) with 95% CI")
ax.legend(frameon=False, loc="lower right")
savefig_both(FIGDIR / "quantile_calibration_bar_wilson.png")

pd.DataFrame({
    "set": ["Cal","Test"],
    "obs_prob": [obs_tau_cal, obs_tau_test],
    "target_tau": [QT_TAU, QT_TAU],
    "test_wilson_lo": [np.nan, lo_ci],
    "test_wilson_hi": [np.nan, hi_ci],
    "n_test": [len(y_cal), len(y_test)],
    "k_leq": [int(np.sum(y_cal <= qhat_cal)), k_test],
    "pinball": [pinball_cal, pinball_test]
}).to_csv(SRC_DIR / "quantile_calibration_single_tau.csv", index=False)

# Error vs distance-to-training in composition space 
X_train_elem = df.loc[idx_train, elem_cols].to_numpy(dtype=float)
X_test_elem  = df.loc[idx_test,  elem_cols].to_numpy(dtype=float)

def _row_normalize(A):
    A = np.clip(A, 0.0, None)
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)

X_train_elem = _row_normalize(X_train_elem)
X_test_elem  = _row_normalize(X_test_elem)

nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
nn.fit(X_train_elem)
dist_l1, _ = nn.kneighbors(X_test_elem, return_distance=True)
dist_l1 = dist_l1.ravel()

abs_err_mm = np.abs(y_test_mm - yhat_test_mm)
df_dist = pd.DataFrame({
    "dist_L1_to_train": dist_l1,
    "abs_error_mm": abs_err_mm,
    "residual_log": res_log,
    "pred_mm": yhat_test_mm,
    "true_mm": y_test_mm
})

# scatter + binned median trend
df_dist["bin"] = pd.qcut(df_dist["dist_L1_to_train"], q=12, duplicates="drop")
trend = df_dist.groupby("bin").agg(
    dist_c=("dist_L1_to_train","mean"),
    ae_med=("abs_error_mm","median"),
    ae_p75=("abs_error_mm", lambda a: float(np.nanpercentile(a, 75))),
    ae_p25=("abs_error_mm", lambda a: float(np.nanpercentile(a, 25))),
    n=("abs_error_mm","size")
).reset_index(drop=True)

plt.figure(figsize=(6.6, 4.2))
ax = sns.scatterplot(data=df_dist, x="dist_L1_to_train", y="abs_error_mm",
                     s=16, alpha=0.45, edgecolor=None)
ax.plot(trend["dist_c"], trend["ae_med"], lw=2, label="Median |error| (mm)")
ax.fill_between(trend["dist_c"], trend["ae_p25"], trend["ae_p75"], alpha=0.2, label="IQR")
ax.set_xlabel("L1 distance to nearest training composition")
ax.set_ylabel("|Error| in Dmax (mm)")
ax.set_title("Prediction error vs. distance to training")
ax.legend(frameon=False)
savefig_both(FIGDIR / "error_vs_dist_train.png")

df_dist.drop(columns=["bin"]).to_csv(SRC_DIR / "error_vs_dist_train.csv", index=False)
trend.to_csv(SRC_DIR / "error_vs_dist_train_trend.csv", index=False)

# Error vs composition entropy
def comp_entropy_norm(row):
    p = np.asarray(row, float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    H = -np.sum(p * np.log(p))
    return float(H / np.log(p.size))  # normalized to [0,1]

H_test = np.apply_along_axis(comp_entropy_norm, 1, X_test_elem)
df_H = pd.DataFrame({
    "entropy_norm": H_test,
    "abs_error_mm": abs_err_mm,
    "residual_log": res_log
}).dropna()

df_H["bin"] = pd.qcut(df_H["entropy_norm"], q=12, duplicates="drop")
trendH = df_H.groupby("bin").agg(
    H_c=("entropy_norm","mean"),
    ae_med=("abs_error_mm","median"),
    ae_p75=("abs_error_mm", lambda a: float(np.nanpercentile(a, 75))),
    ae_p25=("abs_error_mm", lambda a: float(np.nanpercentile(a, 25))),
    n=("abs_error_mm","size")
).reset_index(drop=True)

plt.figure(figsize=(6.4, 4.0))
ax = sns.scatterplot(data=df_H, x="entropy_norm", y="abs_error_mm",
                     s=16, alpha=0.45, edgecolor=None)
ax.plot(trendH["H_c"], trendH["ae_med"], lw=2, label="Median |error| (mm)")
ax.fill_between(trendH["H_c"], trendH["ae_p25"], trendH["ae_p75"], alpha=0.2, label="IQR")
ax.set_xlabel("Normalized composition entropy")
ax.set_ylabel("|Error| in Dmax (mm)")
ax.set_title("Prediction error vs. composition entropy")
ax.legend(frameon=False)
savefig_both(FIGDIR / "error_vs_entropy.png")

df_H.drop(columns=["bin"]).to_csv(SRC_DIR / "error_vs_entropy.csv", index=False)
trendH.to_csv(SRC_DIR / "error_vs_entropy_trend.csv", index=False)

# HPO traces
et_cv = pd.read_csv(HPO_DIR / "et_bayes_cv_results.csv")
plt.figure(figsize=(6.2, 3.8))
ax = plt.gca()
ax.plot(np.arange(len(et_cv)), -et_cv["mean_test_score"], lw=1.5)
ax.set_xlabel("Evaluation index")
ax.set_ylabel("MAE (log) – CV mean")
ax.set_title("ET HPO trace (lower is better)")
savefig_both(FIGDIR / "hpo_trace_et.png")
et_cv.to_csv(SRC_DIR / "hpo_trace_et.csv", index=False)

cb_cv = pd.read_csv(HPO_DIR / "catboost_tau_bayes_cv_results.csv")
plt.figure(figsize=(6.2, 3.8))
ax = plt.gca()
ax.plot(np.arange(len(cb_cv)), -cb_cv["mean_test_score"], lw=1.5)
ax.set_xlabel("Evaluation index")
ax.set_ylabel("Pinball loss (τ) – CV mean")
ax.set_title("CatBoost-τ HPO trace (lower is better)")
savefig_both(FIGDIR / "hpo_trace_cat_tau.png")
cb_cv.to_csv(SRC_DIR / "hpo_trace_cat_tau.csv", index=False)


if not pd.api.types.is_categorical_dtype(df_sc["bin"]):
    df_sc["bin"] = pd.Categorical(df_sc["bin"], ordered=True)

bins_full = (
    df_sc.groupby("bin", observed=True)
         .agg(
             yhat_c=("yhat_log", "mean"),
             res_med=("res_log", "median"),
             res_p25=("res_log", lambda a: float(np.nanpercentile(a, 25))),
             res_p75=("res_log", lambda a: float(np.nanpercentile(a, 75))),
             res_std=("res_log", "std"),
             n=("res_log", "size"),
         )
         .reset_index()
)

bins_full["bin_left"]  = bins_full["bin"].apply(lambda iv: float(iv.left))
bins_full["bin_right"] = bins_full["bin"].apply(lambda iv: float(iv.right))

# ±1σ curves used in the figure
bins_full["res_m1s"] = -bins_full["res_std"]
bins_full["res_p1s"] = +bins_full["res_std"]

# Raw scatter layer (Origin Layer 1)
df_sc.loc[:, ["yhat_log", "res_log"]].rename(
    columns={"yhat_log": "pred_log", "res_log": "residual_log"}
).to_csv(SRC_DIR / "residual_vs_pred_log_bands_et_scatter.csv", index=False)

# Binned summary (Origin Layers 2–4: median, IQR, ±1σ)
bins_full.drop(columns=["bin"]).to_csv(
    SRC_DIR / "residual_vs_pred_log_bands_et_binned.csv", index=False
)

_series_rows = []
for _, r in bins_full.iterrows():
    x = r["yhat_c"]
    _series_rows += [
        {"x": x, "y": r["res_med"], "series": "median"},
        {"x": x, "y": r["res_p25"], "series": "IQR_lo"},
        {"x": x, "y": r["res_p75"], "series": "IQR_hi"},
        {"x": x, "y": r["res_m1s"], "series": "minus1sigma"},
        {"x": x, "y": r["res_p1s"], "series": "plus1sigma"},
    ]
pd.DataFrame(_series_rows).to_csv(
    SRC_DIR / "residual_vs_pred_log_bands_et_series_long.csv", index=False
)

print("Saved enhanced figures to:", FIGDIR)
print("Saved (all) source data to:", SRC_DIR)


# In[21]:


# Learning Curves
LC_DIR = OUTDIR / "source_data"
LC_DIR.mkdir(parents=True, exist_ok=True)

def _row_normalize(A):
    A = np.clip(np.asarray(A, float), 0.0, None)
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)

def _subset_train_by_groups(groups, target_n, seed=SEED):
    """
    Select a subset of the training rows by whole 'signature' groups,
    accumulating until we reach >= target_n rows. Returns np.array of indices
    (relative to X_train/y_train arrays).
    """
    rng_loc = np.random.default_rng(seed)
    # map group -> list of row positions in training set
    g2idx = {}
    for i, g in enumerate(groups):
        g2idx.setdefault(g, []).append(i)
    shuffled_groups = list(g2idx.keys())
    rng_loc.shuffle(shuffled_groups)

    sel = []
    for g in shuffled_groups:
        sel.extend(g2idx[g])
        if len(sel) >= target_n:
            break
    return np.array(sorted(sel), dtype=int)

n_total = len(X_train)
grid_fracs = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
grid_sizes = np.unique(np.clip((grid_fracs * n_total).astype(int),  max(50, int(0.02*n_total)), n_total))

cb_best_params = None
try:
    cb_best_params = dict(catb_bayes.best_params_)
except Exception:
    pass
if cb_best_params is None:
    try:
        with open(HPO_DIR / "catboost_tau_bayes_best_params.json", "r") as f:
            cb_best_params = json.load(f)
    except Exception as e:
        raise RuntimeError("Cannot find CatBoost best params for learning curve refits.") from e

# Build a fresh CatBoost regressor with the same tuned hyperparams (but fixed τ = QT_TAU)
def _make_cat_model_for_tau(tau):
    return CatBoostRegressor(
        loss_function=f"Quantile:alpha={tau}",
        eval_metric=f"Quantile:alpha={tau}",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **cb_best_params
    )

et_base = clone(et)

lc_rows = []
for n_use in grid_sizes:
    sub_idx = _subset_train_by_groups(groups_train, target_n=int(n_use), seed=SEED + int(n_use))
    X_sub, y_sub = X_train.iloc[sub_idx], y_train[sub_idx]

    fam_counts_sub = df.iloc[idx_train].iloc[sub_idx]["family"].value_counts()
    w_sub = df.iloc[idx_train].iloc[sub_idx]["family"].map(lambda f: 1.0 / np.sqrt(fam_counts_sub.get(f, 1))).to_numpy(float)
    w_sub = w_sub * (len(w_sub) / w_sub.sum())

    # ET point model on subset
    et_lc = clone(et_base)
    et_lc.fit(X_sub, y_sub, sample_weight=w_sub)
    yhat_test_lc = et_lc.predict(X_test)

    # log-scale metrics
    rmse_log = float(np.sqrt(mean_squared_error(y_test, yhat_test_lc)))
    mae_log  = float(mean_absolute_error(y_test, yhat_test_lc))
    
    # mm-scale
    yhat_test_mm_lc = np.exp(yhat_test_lc)
    y_test_mm_lc    = np.exp(y_test)
    rmse_mm = float(np.sqrt(mean_squared_error(y_test_mm_lc, yhat_test_mm_lc)))
    mae_mm  = float(mean_absolute_error(y_test_mm_lc, yhat_test_mm_lc))

    lc_rows.append({
        "model": "ET",
        "train_size": int(len(sub_idx)),
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "rmse_mm": rmse_mm,
        "mae_mm": mae_mm
    })

    # CatBoost τ model on subset (pinball + calibration)
    cb_lc = _make_cat_model_for_tau(QT_TAU)
    cb_lc.fit(X_sub, y_sub, sample_weight=w_sub)
    qhat_test_lc = cb_lc.predict(X_test)
    qhat_cal_lc  = cb_lc.predict(X_cal)

    pin_test = float(mean_pinball_loss(y_test, qhat_test_lc, alpha=QT_TAU))
    pin_cal  = float(mean_pinball_loss(y_cal,  qhat_cal_lc,  alpha=QT_TAU))
    obs_cal  = float(np.mean(y_cal  <= qhat_cal_lc))
    obs_test = float(np.mean(y_test <= qhat_test_lc))

    lc_rows.append({
        "model": f"CatBoost_tau={QT_TAU:.2f}",
        "train_size": int(len(sub_idx)),
        "pinball_cal": pin_cal,
        "pinball_test": pin_test,
        "obs_prob_cal": obs_cal,
        "obs_prob_test": obs_test,
        "target_tau": QT_TAU
    })

df_lc = pd.DataFrame(lc_rows)
df_lc.to_csv(LC_DIR / "learning_curves.csv", index=False)

# Plots: Learning Curves
plt.figure(figsize=(6.4, 4.2))
sub = df_lc[df_lc["model"] == "ET"].sort_values("train_size")
plt.plot(sub["train_size"], sub["rmse_mm"], lw=2, label="ET RMSE (mm)")
plt.plot(sub["train_size"], sub["mae_mm"],  lw=2, linestyle="--", label="ET MAE (mm)")
plt.xlabel("Training size (by groups)")
plt.ylabel("Error (mm)")
plt.title("Learning curves — point model")
plt.legend(frameon=False)
savefig_both(FIGDIR / "learning_curves_et_mm.png")

plt.figure(figsize=(6.4, 4.2))
sub = df_lc[df_lc["model"] == f"CatBoost_tau={QT_TAU:.2f}"].sort_values("train_size")
plt.plot(sub["train_size"], sub["pinball_test"], lw=2, label=f"Pinball (test), τ={QT_TAU:.2f}")
plt.xlabel("Training size (by groups)")
plt.ylabel("Pinball loss (test)")
plt.title("Learning curves — CatBoost quantile")
plt.legend(frameon=False)
savefig_both(FIGDIR / "learning_curves_catboost_pinball.png")

print("Learning curves saved:",
      FIGDIR / "learning_curves_et_mm.png",
      FIGDIR / "learning_curves_catboost_pinball.png")
print("Learning curve CSV:", LC_DIR / "learning_curves.csv")


# In[22]:


# Multi-τ Sweep (0.80–0.99)
SWEEP_DIR = OUTDIR / "source_data"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

taus = np.round(np.arange(0.80, 0.991, 0.01), 2)

sweep_rows = []
for tau in taus:
    cb_t = _make_cat_model_for_tau(tau)
    cb_t.fit(X_train, y_train, sample_weight=w_train)

    q_cal  = cb_t.predict(X_cal)
    q_test = cb_t.predict(X_test)

    obs_cal  = float(np.mean(y_cal  <= q_cal))
    obs_test = float(np.mean(y_test <= q_test))
    pin_cal  = float(mean_pinball_loss(y_cal,  q_cal,  alpha=tau))
    pin_test = float(mean_pinball_loss(y_test, q_test, alpha=tau))

    sweep_rows.append({
        "tau": tau,
        "obs_prob_cal": obs_cal,
        "obs_prob_test": obs_test,
        "pinball_cal": pin_cal,
        "pinball_test": pin_test,
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_test))
    })

df_tau = pd.DataFrame(sweep_rows).sort_values("tau")
df_tau.to_csv(SWEEP_DIR / "multi_tau_sweep.csv", index=False)

# Plot τ vs observed coverage and pinball
plt.figure(figsize=(6.0, 4.0))
plt.plot(df_tau["tau"], df_tau["obs_prob_test"], lw=2, label="Observed P(Y≤q̂) (test)")
plt.plot([taus.min(), taus.max()], [taus.min(), taus.max()], ls="--", label="Ideal: y=x")
plt.xlabel("Target τ")
plt.ylabel("Observed coverage (test)")
plt.title("Quantile calibration sweep")
plt.legend(frameon=False)
savefig_both(FIGDIR / "tau_sweep_calibration.png")

plt.figure(figsize=(6.0, 4.0))
plt.plot(df_tau["tau"], df_tau["pinball_test"], lw=2, label="Pinball (test)")
plt.xlabel("τ")
plt.ylabel("Pinball loss (test)")
plt.title("Pinball vs τ")
plt.legend(frameon=False)
savefig_both(FIGDIR / "tau_sweep_pinball.png")

print("τ-sweep figures saved:",
      FIGDIR / "tau_sweep_calibration.png",
      FIGDIR / "tau_sweep_pinball.png")
print("τ-sweep CSV:", SWEEP_DIR / "multi_tau_sweep.csv")


# In[23]:


def robust_scores_lower(
    y_true_log,
    model,
    X_elem,
    eps=ROBUST_EPS,
    K=ROBUST_SAMPLES,
    rng=None,
    *,
    vectorized=True,
    chunk_size=16384,
    return_min_jitter=False,
):
    """
    Robust one-sided conformal scores for LOWER bounds on the log scale.

    For each calibration pair (x_i, y_i), sample K jittered compositions x'_{i,j}
    within an L1-ball of radius `eps` on the simplex, evaluate the τ-quantile
    model on them, take the worst-case (minimum) predicted quantile, and compute:

        S_i = max(0, min_j q̂_τ(x'_{i,j}) - y_i).

    Notes
    -----
    - If eps <= 0 or K <= 1, this reduces to the marginal one-sided score:
          S_i = max(0, q̂_τ(x_i) - y_i).
    - X_elem rows are clipped to [0, 1] and renormalized to sum to 1 (safety).
    - `vectorized=True` predicts all jitters in large batches with `chunk_size`
      to balance speed and memory; set False to use a per-sample loop.

    Parameters
    ----------
    y_true_log : array-like, shape (n,)
        Log targets for the calibration set.
    model : fitted regressor
        Must implement .predict(DataFrame) and output log-quantile predictions.
        Assumes `make_features_from_compositions` produces the exact training schema.
    X_elem : array-like, shape (n, d)
        Calibration compositions (fractions). Each row ~ sum to 1.
    eps : float
        L1 radius on the simplex (fraction units); e.g. 0.01 ≈ total 1 at.% drift.
    K : int
        Number of jitters per composition.
    rng : np.random.Generator or None
        RNG for reproducibility. If None, uses np.random.default_rng(SEED).
    vectorized : bool
        If True, predict jitters in vectorized batches (faster).
    chunk_size : int
        Max rows per prediction chunk when vectorized.
    return_min_jitter : bool
        If True, also return the worst-case jitter that attained min q̂_τ for each i.

    Returns
    -------
    S : np.ndarray, shape (n,)
        Robust one-sided scores for lower conformal calibration.
    (optional) X_star : np.ndarray, shape (n, d)
        The jitter composition per i that achieved the min predicted quantile.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    y_true_log = np.asarray(y_true_log, dtype=float).ravel()
    X_elem = np.asarray(X_elem, dtype=float)
    n = y_true_log.shape[0]
    if X_elem.ndim != 2 or X_elem.shape[0] != n:
        raise ValueError("X_elem must have shape (n_samples, n_elements) matching y_true_log length.")

    X_elem = np.clip(X_elem, 0.0, None)
    rs = X_elem.sum(axis=1, keepdims=True)
    X_elem = X_elem / np.where(rs > 0, rs, 1.0)

    K = int(K)
    eps = float(max(0.0, eps))

    if eps <= 0.0 or K <= 1:
        feats = make_features_from_compositions(X_elem)
        qhat = np.asarray(model.predict(feats), dtype=float).ravel()
        S = np.maximum(0.0, qhat - y_true_log)
        return (S, X_elem.copy()) if return_min_jitter else S

    if vectorized:
        J_list = []
        for i in range(n):
        rng_i = np.random.default_rng(SEED + DRIFT_SEED)
        Xj = jitter_in_L1_ball_simplex(x, eps=float(eps), K=int(K), rng=rng_i)
            # Normalize each jitter (safety)
            rj = Xj.sum(axis=1, keepdims=True)
            Xj = Xj / np.where(rj > 0, rj, 1.0)
            J_list.append(Xj.astype(float, copy=False))

        J_all = np.vstack(J_list)  # shape (n*K, d)

        q_all = np.empty(J_all.shape[0], dtype=float)
        start = 0
        while start < J_all.shape[0]:
            end = min(start + int(chunk_size), J_all.shape[0])
            feats = make_features_from_compositions(J_all[start:end])
            q_all[start:end] = np.asarray(model.predict(feats), dtype=float).ravel()
            start = end

        q_mat = q_all.reshape(n, K)
        q_min = np.nanmin(q_mat, axis=1)

        if return_min_jitter:
            argmin = np.nanargmin(q_mat, axis=1)
            X_star = np.empty_like(X_elem)
            off = 0
            for i in range(n):
                X_star[i] = J_list[i][argmin[i]]
                off += K
        S = np.maximum(0.0, q_min - y_true_log)
        return (S, X_star) if return_min_jitter else S

    # Non-vectorized fallback (per-sample loop)
    S = np.empty(n, dtype=float)
    X_star = np.empty_like(X_elem) if return_min_jitter else None
    for i, (y_i, x_i) in enumerate(zip(y_true_log, X_elem)):
        Xj = jitter_in_L1_ball_simplex(x_i, eps=eps, K=K, rng=rng)
        rj = Xj.sum(axis=1, keepdims=True)
        Xj = Xj / np.where(rj > 0, rj, 1.0)
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), dtype=float)
        if qj.size == 0 or not np.isfinite(qj).any():
            # rare fallback: use base comp prediction
            qj = np.asarray(model.predict(make_features_from_compositions(x_i[None, :])), dtype=float)
        j_star = int(np.nanargmin(qj))
        q_min = float(qj[j_star])
        S[i] = max(0.0, q_min - float(y_i))
        if return_min_jitter:
            X_star[i] = Xj[j_star]
    return (S, X_star) if return_min_jitter else S


# In[24]:


def _adversarial_min_quantile_log(model, x0, eps, *, step=0.02, max_iters=300, topk=5, rng=None):
    """
    Deterministic inner adversary on the simplex with an L1 budget.
    Greedy pairwise reassignments (donor->receiver) that most reduce q̂τ(x).
    Works with tree models (no gradients). Evaluates a small neighborhood per iter.
    Returns: (q_min, x_star)
    """
    if rng is None:
        rng = np.random.default_rng(SEED + 7)

    x = np.asarray(x0, float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    x = x / (s if s > 0 else 1.0)

    def _q_log(Xb):
        feats = make_features_from_compositions(Xb)
        return np.asarray(model.predict(feats), float).ravel()

    q_curr = float(_q_log(x[None, :])[0])
    budget = 2.0 * float(max(0.0, eps))
    if budget <= 0.0:
        return q_curr, x.copy()

    d = x.size
    supp = np.where(x > 1e-12)[0]
    if supp.size < 2:
        return q_curr, x.copy()
    
    donors    = supp[np.argsort(-x[supp])][:min(topk, supp.size)]
    receivers = supp[np.argsort( x[supp])][:min(max(topk, 2), supp.size)]
    
    x_star = x.copy()
    q_min = q_curr
    used = 0.0

    for _ in range(int(max_iters)):
        if used + 2*step > budget + 1e-12:
            step = max(1e-4, 0.5*(budget - used))
            if used + 2*step > budget + 1e-12:
                break

        best_pair, best_q = None, q_min
        cands = []
        for i in donors:
            if x_star[i] <= 1e-12: 
                continue
            delta_max = min(step, x_star[i])
            if delta_max <= 0: 
                continue
            for j in receivers:
                if i == j:
                    continue
                x_try = x_star.copy()
                x_try[i] -= delta_max
                x_try[j] += delta_max
                cands.append(x_try)

        if not cands:
            break

        Q = _q_log(np.stack(cands, axis=0))
        k = int(np.argmin(Q))
        q_try = float(Q[k])
        if q_try + 1e-12 < q_min:
            x_star = cands[k]
            q_min = q_try
            used += 2.0 * float(delta_max)
            donors    = supp[np.argsort(-x_star[supp])][:min(topk, supp.size)]
            receivers = supp[np.argsort( x_star[supp])][:min(max(topk, 2), supp.size)]
        else:
            break

    return q_min, x_star


def robust_scores_lower_adversarial(
    y_true_log, model, X_elem, eps=ROBUST_EPS, rng=None, return_min_jitter=False, return_meta=False
):
    if rng is None:
        rng = np.random.default_rng(SEED + 11)
    y = np.asarray(y_true_log, float).ravel()
    X = np.asarray(X_elem, float)
    X = np.clip(X, 0.0, None)
    rs = X.sum(axis=1, keepdims=True)
    X = X / np.where(rs > 0, rs, 1.0)

    S = np.empty_like(y)
    X_star = np.empty_like(X) if return_min_jitter else None
    meta = [] if return_meta else None
    for i, (yi, xi) in enumerate(zip(y, X)):
        q_min, x_star, tr = adversarial_min_with_report(model, xi, float(eps))
        S[i] = max(0.0, q_min - float(yi))
        if return_min_jitter: X_star[i] = x_star
        if return_meta: meta.append(tr)
    if return_meta:
        return S, X_star, meta if return_min_jitter else (S, None, meta)
    return (S, X_star) if return_min_jitter else S


# In[25]:


# Reporting wrapper: adversarial min with trace & stationarity flag
def adversarial_min_with_report(model, x0, eps, *, step=ADV_STEP, max_iters=ADV_MAX_ITERS, topk=ADV_TOPK, rng=None):
    """
    Run the deterministic inner adversary and report:
      - q_min (log), x_star
      - stationary (True if no improving donor->receiver move exists at final iterate)
      - n_iters, n_accept, best_improve, last_improve
      - a tiny 'gap' proxy = max(0, q_best_neighbor - q_min) with one extra neighborhood scan
    """
    if rng is None:
        rng = np.random.default_rng(SEED + 17)

    def _q_log(Xb):
        feats = make_features_from_compositions(Xb)
        return np.asarray(model.predict(feats), float).ravel()

    # Run the actual minimizer
    q_min, x_star = _adversarial_min_quantile_log(model, x0, eps, step=step, max_iters=max_iters, topk=topk, rng=rng)

    # One extra local neighborhood scan to test stationarity
    x = x_star.copy()
    d = x.size
    donors    = np.argsort(-x)[:min(topk, d)]
    receivers = np.argsort(x) [:min(max(topk,2), d)]
    cands = []
    for i in donors:
        if x[i] <= 1e-12: 
            continue
        delta = min(step, x[i])
        for j in receivers:
            if i == j: 
                continue
            xt = x.copy(); xt[i] -= delta; xt[j] += delta
            cands.append(xt)

    stationary = True
    gap_proxy  = 0.0
    best_improve = 0.0
    if cands:
        Q = _q_log(np.stack(cands, axis=0))
        q_best_neighbor = float(np.min(Q))
        if q_best_neighbor + 1e-12 < q_min:
            stationary = False
            best_improve = q_min - q_best_neighbor
        else:
            gap_proxy = max(0.0, q_best_neighbor - q_min)

    trace = {
        "eps": float(eps),
        "step": float(step),
        "topk": int(topk),
        "q_min": float(q_min),
        "stationary": bool(stationary),
        "best_improve": float(best_improve),
        "gap_proxy": float(gap_proxy),
    }
    return q_min, x_star, trace


def robust_scores_lower_dispatch(
    y_true_log, model, X_elem, *, eps, K, rng, method=None, return_meta=False
):
    """
    method: "mc" (sampling) or "adversarial"
    return_meta only applies to the adversarial backend (if it supports it).
    """
    m = (method or str(ROBUST_METHOD)).lower()
    if m == "adversarial":
        if return_meta:
            S, _, meta = robust_scores_lower_adversarial(
                y_true_log, model, X_elem, eps=eps, rng=rng, return_meta=True
            )
            return S, meta
        else:
            return robust_scores_lower_adversarial(
                y_true_log, model, X_elem, eps=eps, rng=rng
            )
    else:
        return robust_scores_lower(
            y_true_log, model, X_elem, eps=eps, K=K, rng=rng
        )


# In[27]:


def compute_density_ratio_weights(X_cal, X_test, seed=SEED, clip=1e6):
    """
    Return (w_cal, auc, ess), where w_cal ≈ p_test(x)/p_cal(x) estimated via logistic density ratio.
    """
    Xc = np.asarray(X_cal, float); Xt = np.asarray(X_test, float)
    X_cls = np.vstack([Xc, Xt])
    y_cls = np.hstack([np.zeros(len(Xc)), np.ones(len(Xt))])
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    clf.fit(X_cls, y_cls)
    proba_cal = clf.predict_proba(Xc)[:, 1]
    w = proba_cal / np.maximum(1.0 - proba_cal, 1e-12)
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=clip, neginf=0.0), 0.0, clip)

    # Effective sample size + AUC report
    ess = float((w.sum()**2) / (np.square(w).sum() + 1e-12)) if w.sum() > 0 else 0.0
    auc = float(roc_auc_score(y_cls, clf.predict_proba(X_cls)[:, 1]))
    return w, auc, ess


# In[28]:


# High-tail quantile model (τ = QT_TAU_HIGH) + conformal lower bounds
HPO_DIR = OUTDIR / "reports" / "hpo"
SRC_DIR = OUTDIR / "source_data"
for p in (HPO_DIR, SRC_DIR):
    p.mkdir(parents=True, exist_ok=True)

QT_TAU_HIGH = 0.99
groups_train = df.loc[idx_train, "signature"].values
cv = GroupKFold(n_splits=5)

def neg_pinball_tau_hi(estimator, X, y, alpha=QT_TAU_HIGH):
    y_pred = estimator.predict(X)
    return -mean_pinball_loss(y, y_pred, alpha=alpha)

# CatBoost high-τ
cb_hi_base = CatBoostRegressor(
    loss_function=f"Quantile:alpha={QT_TAU_HIGH}",
    eval_metric=f"Quantile:alpha={QT_TAU_HIGH}",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
    od_type="Iter",
    od_wait=100
)

catb_hi_space = {
    'iterations'        : Integer(500, 1000),
    'depth'             : Integer(2, 10),
    'learning_rate'     : Real(1e-3, 3e-1, prior='log-uniform'),
    'l2_leaf_reg'       : Real(1e-3, 10.0, prior='log-uniform'),
    'bootstrap_type'    : Categorical(['Bayesian']),
    'bagging_temperature': Real(0.0, 1.0),
    'random_strength'   : Real(0.0, 1.0),
    'border_count'      : Integer(32, 255),
    'rsm'               : Real(0.5, 1.0),
}
t0 = time.time()
cb_hi_bayes = BayesSearchCV(
    estimator=cb_hi_base,
    search_spaces=catb_hi_space,
    n_iter=300, 
    scoring=neg_pinball_tau_hi,
    cv=cv,
    n_jobs=-1, 
    random_state=SEED,
    verbose=1,
    refit=True,
    return_train_score=False
)
cb_hi_bayes.fit(X_train, y_train, groups=groups_train)
t_el = time.time() - t0

cat_qt_hi = cb_hi_bayes.best_estimator_
pd.DataFrame(cb_hi_bayes.cv_results_).to_csv(HPO_DIR / "catboost_hi_tau_bayes_cv_results.csv", index=False)
with open(HPO_DIR / "catboost_hi_tau_bayes_best_params.json", "w") as f:
    json.dump(cb_hi_bayes.best_params_, f, indent=2)
print(f"[CatBoost τ={QT_TAU_HIGH:.2f}] Best params: {cb_hi_bayes.best_params_} | time={t_el:.1f}s")

qhat_train_hi = cat_qt_hi.predict(X_train)
qhat_cal_hi   = cat_qt_hi.predict(X_cal)
qhat_test_hi  = cat_qt_hi.predict(X_test)

s_raw = qhat_cal_hi - y_cal
c_shift_rawtau = weighted_quantile(s_raw, 1.0 - QT_TAU_HIGH)
qhat_cal_hi_adj  = qhat_cal_hi  - c_shift_rawtau
qhat_test_hi_adj = qhat_test_hi - c_shift_rawtau

obs_tau_cal_hi  = float(np.mean(y_cal  <= qhat_cal_hi))
obs_tau_test_hi = float(np.mean(y_test <= qhat_test_hi))
pin_cal_hi      = float(mean_pinball_loss(y_cal,  qhat_cal_hi,  alpha=QT_TAU_HIGH))
pin_test_hi     = float(mean_pinball_loss(y_test, qhat_test_hi, alpha=QT_TAU_HIGH))
print(f"[Pre-CP τ={QT_TAU_HIGH:.2f}] P(Y≤q̂): CAL={obs_tau_cal_hi:.3f}, TEST={obs_tau_test_hi:.3f} "
      f"| pinball: CAL={pin_cal_hi:.5f}, TEST={pin_test_hi:.5f}")


# Conformal lower bounds (marginal & robust)
def conformal_qhat(S, alpha):
    """Split-conformal one-sided quantile with the finite-sample correction
    ceil((n+1)(1-alpha))/n required by manuscript Eq. (5)."""
    S = np.asarray(S, float)
    S = S[np.isfinite(S)]
    n = S.size
    if n == 0:
        return np.nan
    lvl = min(1.0, np.ceil((n + 1) * (1.0 - float(alpha))) / n)
    return float(np.quantile(S, lvl, method="higher"))

S_cal_hi = np.maximum(0.0, qhat_cal_hi - y_cal)
q_marginal_hi = conformal_qhat(S_cal_hi, ALPHA)
L_marginal_hi_mm = np.exp(qhat_test_hi - q_marginal_hi)

X_cal_elem  = df.loc[idx_cal,  elem_cols].to_numpy()
X_test_elem = df.loc[idx_test, elem_cols].to_numpy()

_method = str(ROBUST_METHOD).lower()
if _method in {"sampling", "mc", "none"}:
    _method = "mc"
elif _method != "adversarial":
    raise ValueError(f"ROBUST_METHOD must be 'adversarial' or 'mc', got '{ROBUST_METHOD}'")

S_cal_hi_rob = robust_scores_lower_dispatch(
    y_cal, cat_qt_hi, X_cal_elem,
    eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng,
    method=_method
)
    
S_cal_hi_rob = np.asarray(S_cal_hi_rob, float)
S_cal_hi_rob[~np.isfinite(S_cal_hi_rob)] = 0.0
S_cal_hi_rob = np.maximum(0.0, S_cal_hi_rob)

q_robust_hi = conformal_qhat(S_cal_hi_rob, ALPHA)
if not np.isfinite(q_robust_hi):
    q_robust_hi = conformal_qhat(S_cal_hi, ALPHA)
Q_ROBUST_HI_FROZEN = float(q_robust_hi)

def _assert_cp_frozen():
    assert np.isclose(float(q_robust_hi), Q_ROBUST_HI_FROZEN), (
        f"q_robust_hi = {q_robust_hi} but frozen value is {Q_ROBUST_HI_FROZEN}. "
        "A later cell reassigned the conformal constant."
    )

w_cal, auc_shift, ess_cal = compute_density_ratio_weights(X_cal, X_test, seed=SEED)
q_shiftweighted_hi = weighted_quantile(S_cal_hi, 1 - ALPHA, sample_weight=w_cal)
L_shiftweighted_hi_mm = np.exp(qhat_test_hi - q_shiftweighted_hi)
print(f"[Shift-weighted CP] q={q_shiftweighted_hi:.4f} | classifier AUC={auc_shift:.3f} | ESS={ess_cal:.1f}")

print(f"[Robust CP @ eps={ROBUST_EPS:.3f}] "
      f"score min/med/max = {np.nanmin(S_cal_hi_rob):.4f}/{np.nanmedian(S_cal_hi_rob):.4f}/{np.nanmax(S_cal_hi_rob):.4f}, "
      f"q_(1-α)={q_robust_hi:.4f}")

L_robust_hi_mm_test = robust_L_for_comps_hi(
    X_test_elem,
    q_cal_robust=q_robust_hi,
    model=cat_qt_hi,
    eps=ROBUST_EPS,
    K=ROBUST_SAMPLES,
    rng=rng
)

y_test_mm = np.exp(y_test)
cov_marg = float(np.mean(y_test_mm >= L_marginal_hi_mm))
cov_rob  = float(np.mean(y_test_mm >= L_robust_hi_mm_test))
print(f"[Post-CP τ={QT_TAU_HIGH:.2f}] Coverage@{(1-ALPHA):.2f} | "
      f"marginal={cov_marg:.3f}, robust={cov_rob:.3f} (expect ~{1-ALPHA:.2f})")

# Compute robust scores for both methods
S_cal_hi_rob_mc  = robust_scores_lower_dispatch(
    y_cal, cat_qt_hi, X_cal_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng, method="mc"
)
S_cal_hi_rob_adv = robust_scores_lower_dispatch(
    y_cal, cat_qt_hi, X_cal_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng, method="adversarial"
)

def post_cp(S_cal_hi_rob):
    S = np.maximum(0.0, np.nan_to_num(np.asarray(S_cal_hi_rob, float), nan=0.0))
    q = conformal_qhat(S, ALPHA)
    if not np.isfinite(q):  # fallback
        q = conformal_qhat(S_cal_hi, ALPHA)
    L = robust_L_for_comps_hi(
        X_test_elem, q_cal_robust=q, model=cat_qt_hi, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng
    )
    cov = float(np.mean(np.exp(y_test) >= L))
    return q, L, cov

q_mc,  L_mc,  cov_mc  = post_cp(S_cal_hi_rob_mc)
q_adv, L_adv, cov_adv = post_cp(S_cal_hi_rob_adv)

print(f"[Robust CP @ eps={ROBUST_EPS:.3f}] MC:   q_(1-α)={q_mc:.4f},  coverage={cov_mc:.3f}")
print(f"[Robust CP @ eps={ROBUST_EPS:.3f}] Adv:  q_(1-α)={q_adv:.4f}, coverage={cov_adv:.3f}")

# Alpha sweep (fixed eps = ROBUST_EPS)
alpha_grid = np.linspace(0.02, 0.30, 15)
rows_alpha = []
for a in alpha_grid:
    q_marg_a = weighted_quantile(np.maximum(0.0, qhat_cal_hi - y_cal), 1 - a)
    L_marg_a = np.exp(qhat_test_hi - q_marg_a)
    cov_marg_a = float(np.mean(y_test_mm >= L_marg_a))

    q_rob_a = conformal_qhat(S_cal_hi_rob, a)
    L_rob_a = robust_L_for_comps_hi(
        X_test_elem, q_cal_robust=q_rob_a, model=cat_qt_hi,
        eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng
    )
    cov_rob_a = float(np.mean(y_test_mm >= L_rob_a))

    rows_alpha.append({
        "alpha": float(a),
        "target_coverage": float(1 - a),
        "coverage_marginal": cov_marg_a,
        "coverage_robust":   cov_rob_a
    })

df_alpha = pd.DataFrame(rows_alpha)
df_alpha.to_csv(SRC_DIR / "coverage_vs_alpha.csv", index=False)

eps_grid = [0.000, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200]
rows_eps = []
for eps in eps_grid:
    S_eps = robust_scores_lower_dispatch(y_cal, cat_qt_hi, X_cal_elem, eps=eps, K=ROBUST_SAMPLES, rng=rng)
    S_eps = np.maximum(0.0, np.nan_to_num(S_eps, nan=0.0))
    q_eps = conformal_qhat(S_eps, ALPHA)
    L_eps = robust_L_for_comps_hi(X_test_elem, q_cal_robust=q_eps, model=cat_qt_hi,
                                  eps=eps, K=ROBUST_SAMPLES, rng=rng)
    cov_eps = float(np.mean(y_test_mm >= L_eps))
    rows_eps.append({"eps": float(eps), "q_robust": float(q_eps), "coverage_robust": cov_eps})
df_eps = pd.DataFrame(rows_eps)
df_eps.to_csv(SRC_DIR / "coverage_vs_epsilon.csv", index=False)

per_test = pd.DataFrame({
    "index": idx_test,
    "family": df.loc[idx_test, "family"].to_numpy(),
    "signature": df.loc[idx_test, "signature"].to_numpy(),
    "y_true_mm": y_test_mm,
    "qhat_hi_log": qhat_test_hi,
    "L_marginal_mm": L_marginal_hi_mm,
    "L_robust_mc_mm": L_mc,
    "L_robust_adv_mm": L_adv,
    "covered_marginal": (y_test_mm >= L_marginal_hi_mm).astype(int),
    "covered_robust_mc": (y_test_mm >= L_mc).astype(int),
    "covered_robust_adv": (y_test_mm >= L_adv).astype(int),
})
per_test.to_csv(SRC_DIR / "conformal_bounds_test.csv", index=False)

# Calibration scores (both S_robust variants)
cal_scores = pd.DataFrame({
    "index": idx_cal,
    "S_marginal": np.maximum(0.0, qhat_cal_hi - y_cal),
    "S_robust_mc":  np.maximum(0.0, np.nan_to_num(S_cal_hi_rob_mc,  nan=0.0)),
    "S_robust_adv": np.maximum(0.0, np.nan_to_num(S_cal_hi_rob_adv, nan=0.0)),
})
cal_scores.to_csv(SRC_DIR / "calibration_scores.csv", index=False)

with open(OUTDIR / "reports" / "robust_summary.json", "w") as f:
    json.dump({
        "tau_high": float(QT_TAU_HIGH),
        "alpha": float(ALPHA),
        "eps_used": float(ROBUST_EPS),
        "q_marginal_hi": float(q_marginal_hi),
        "q_robust_hi": float(q_robust_hi),
        "coverage_test": {"marginal": cov_marg, "robust": cov_rob},
        "pre_cp": {"obs_tau_cal": obs_tau_cal_hi, "obs_tau_test": obs_tau_test_hi,
                   "pin_cal": pin_cal_hi, "pin_test": pin_test_hi}
    }, f, indent=2)

print("Saved source data for figs:",
      (SRC_DIR / "conformal_bounds_test.csv").name, ",",
      (SRC_DIR / "calibration_scores.csv").name, ",",
      (SRC_DIR / "coverage_vs_alpha.csv").name, ",",
      (SRC_DIR / "coverage_vs_epsilon.csv").name)


# In[30]:


alpha_grid = np.linspace(0.02, 0.30, 15)
rows_alpha = []
for a in alpha_grid:
    # Marginal
    q_marg_a = weighted_quantile(np.maximum(0.0, qhat_cal_hi - y_cal), 1 - a)
    L_marg_a = np.exp(qhat_test_hi - q_marg_a)
    cov_marg_a = float(np.mean(y_test_mm >= L_marg_a))
    rows_alpha.append({"method":"marginal","alpha":float(a),"target_coverage":float(1-a),
                       "coverage":cov_marg_a})

    # MC (unweighted conformal quantile, manuscript Eq. 10)
    q_mc_a = conformal_qhat(np.maximum(0.0, np.nan_to_num(S_cal_hi_rob_mc, nan=0.0)), a)
    L_mc_a = robust_L_for_comps_hi(X_test_elem, q_cal_robust=q_mc_a, model=cat_qt_hi,
                                   eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng)
    cov_mc_a = float(np.mean(y_test_mm >= L_mc_a))
    rows_alpha.append({"method":"mc","alpha":float(a),"target_coverage":float(1-a),
                       "coverage":cov_mc_a})

    # Adversarial (unweighted conformal quantile, manuscript Eq. 10)
    q_adv_a = conformal_qhat(np.maximum(0.0, np.nan_to_num(S_cal_hi_rob_adv, nan=0.0)), a)
    L_adv_a = robust_L_for_comps_hi(X_test_elem, q_cal_robust=q_adv_a, model=cat_qt_hi,
                                    eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng)
    cov_adv_a = float(np.mean(y_test_mm >= L_adv_a))
    rows_alpha.append({"method":"adversarial","alpha":float(a),"target_coverage":float(1-a),
                       "coverage":cov_adv_a})

df_alpha = pd.DataFrame(rows_alpha)
df_alpha.to_csv(SRC_DIR / "coverage_vs_alpha.csv", index=False)

eps_grid = [0.000, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200]
rows_eps = []
for eps in eps_grid:
    # MC
    S_mc = robust_scores_lower_dispatch(y_cal, cat_qt_hi, X_cal_elem,
                                        eps=eps, K=ROBUST_SAMPLES, rng=rng, method="mc")
    S_mc = np.maximum(0.0, np.nan_to_num(S_mc, 0.0))
    q_mc_eps = conformal_qhat(S_mc, ALPHA)
    L_mc_eps = robust_L_for_comps_hi(X_test_elem, q_cal_robust=q_mc_eps, model=cat_qt_hi,
                                     eps=eps, K=ROBUST_SAMPLES, rng=rng)
    cov_mc_eps = float(np.mean(y_test_mm >= L_mc_eps))
    rows_eps.append({"method":"mc","eps":float(eps),"q_robust":float(q_mc_eps),
                     "coverage":cov_mc_eps})

    # Adversarial
    S_adv = robust_scores_lower_dispatch(y_cal, cat_qt_hi, X_cal_elem,
                                         eps=eps, K=ROBUST_SAMPLES, rng=rng, method="adversarial")
    S_adv = np.maximum(0.0, np.nan_to_num(S_adv, 0.0))
    q_adv_eps = conformal_qhat(S_adv, ALPHA)
    L_adv_eps = robust_L_for_comps_hi(X_test_elem, q_cal_robust=q_adv_eps, model=cat_qt_hi,
                                      eps=eps, K=ROBUST_SAMPLES, rng=rng)
    cov_adv_eps = float(np.mean(y_test_mm >= L_adv_eps))
    rows_eps.append({"method":"adversarial","eps":float(eps),"q_robust":float(q_adv_eps),
                     "coverage":cov_adv_eps})

df_eps = pd.DataFrame(rows_eps)
df_eps.to_csv(SRC_DIR / "coverage_vs_epsilon.csv", index=False)


# In[32]:


with open(OUTDIR / "reports" / "robust_summary.json", "w") as f:
    json.dump({
        "tau_high": float(QT_TAU_HIGH),
        "alpha": float(ALPHA),
        "eps_used": float(ROBUST_EPS),
        "q_marginal_hi": float(q_marginal_hi),
        "density_ratio": {"auc": float(auc_shift), "ess": float(ess_cal)},
        "mc":  {"q_robust": float(q_mc),  "coverage_test": float(cov_mc)},
        "adv": {"q_robust": float(q_adv), "coverage_test": float(cov_adv)},
        "pre_cp": {"obs_tau_cal": obs_tau_cal_hi, "obs_tau_test": obs_tau_test_hi,
                   "pin_cal": pin_cal_hi, "pin_test": pin_test_hi}
    }, f, indent=2)


# In[33]:


FIGDIR = OUTDIR / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Coverage vs alpha
dfa = pd.read_csv(SRC_DIR / "coverage_vs_alpha.csv")
plt.figure()
for m in ["marginal","mc","adversarial"]:
    d = dfa[dfa["method"]==m].sort_values("alpha")
    plt.plot(d["alpha"], d["coverage"], label=m)
plt.plot(dfa["alpha"].unique(), 1 - dfa["alpha"].unique(), linestyle="--", label="target")
plt.xlabel("alpha"); plt.ylabel("coverage"); plt.legend(); plt.grid(True, alpha=0.3)
plt.title(f"Coverage vs alpha (eps={ROBUST_EPS})")
plt.savefig(FIGDIR / "coverage_vs_alpha_mc_adv.png", dpi=300, bbox_inches="tight"); plt.close()

# Coverage vs epsilon
dfe = pd.read_csv(SRC_DIR / "coverage_vs_epsilon.csv")
plt.figure()
for m in ["mc","adversarial"]:
    d = dfe[dfe["method"]==m].sort_values("eps")
    plt.plot(d["eps"], d["coverage"], marker="o", label=m)
plt.axhline(1 - ALPHA, linestyle="--", label="target")
plt.xlabel("epsilon"); plt.ylabel("coverage"); plt.legend(); plt.grid(True, alpha=0.3)
plt.title(f"Coverage vs epsilon (alpha={ALPHA})")
plt.savefig(FIGDIR / "coverage_vs_epsilon_mc_adv.png", dpi=300, bbox_inches="tight"); plt.close()


# In[34]:


# No-family ablation: identical splits, identical targets
if True:
    X_train_nf = X_engineered.iloc[idx_train].reindex(columns=X_NOFAM_COLUMNS)
    X_cal_nf   = X_engineered.iloc[idx_cal  ].reindex(columns=X_NOFAM_COLUMNS)
    X_test_nf  = X_engineered.iloc[idx_test ].reindex(columns=X_NOFAM_COLUMNS)

    # Quantile model
    _best = dict(
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        border_count=32,
        depth=10,
        iterations=1000,
        l2_leaf_reg=0.001,
        learning_rate=0.0031,
        random_strength=1.0,
        rsm=0.5,
    )
    
    # Estimator
    cat_nf = CatBoostRegressor(
        loss_function=f"Quantile:alpha={QT_TAU}",
        eval_metric=f"Quantile:alpha={QT_TAU}",
        random_seed=SEED + 101,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        od_type="Iter",
        od_wait=100,
        **_best,
    )
    
    cat_nf.fit(X_train_nf, y_log[idx_train])
    
    class _ReindexPredictor:
        def __init__(self, base, cols):
            self.base = base
            self.cols = list(cols)
        def predict(self, feats):
            feats = feats.reindex(columns=self.cols, fill_value=0.0)
            return self.base.predict(feats)
    
    cat_nf_wrap = _ReindexPredictor(cat_nf, X_NOFAM_COLUMNS)
    
    S_cal_nf = robust_scores_lower(
        y_true_log=y_log[idx_cal],
        model=cat_nf_wrap,
        X_elem=df.loc[idx_cal, elem_cols].to_numpy(),
        eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=np.random.default_rng(SEED+202)
    )
    
    w_nf, auc_nf, ess_nf = compute_density_ratio_weights(X_cal_nf, X_test_nf, seed=SEED+303)
    q_nf = weighted_quantile(S_cal_nf, 1 - ALPHA, sample_weight=w_nf)
    
    L_nf = robust_L_for_comps_hi(
        df.loc[idx_test, elem_cols].to_numpy(),
        q_cal_robust=float(q_nf),
        model=cat_nf_wrap,
        eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=np.random.default_rng(SEED+404)
    )

    # Shift weights for no-family feature space
    w_nf, auc_nf, ess_nf = compute_density_ratio_weights(X_cal_nf, X_test_nf, seed=SEED+303)
    q_nf = weighted_quantile(S_cal_nf, 1 - ALPHA, sample_weight=w_nf)

    # Coverage on TEST
    L_nf = robust_L_for_comps_hi(
        df.loc[idx_test, elem_cols].to_numpy(),
        q_cal_robust=float(q_nf),
        model=cat_nf,
        eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=np.random.default_rng(SEED+404)
    )
    y_test_mm = np.exp(y_log[idx_test])
    cov_nf = float(np.mean(y_test_mm >= L_nf))

    Path(OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
    Path(OUTDIR / "source_data").mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "reports" / "no_family_ablation.json", "w") as f:
        json.dump({
            "coverage_no_family": cov_nf,
            "auc_shift_no_family": auc_nf,
            "ess_no_family": ess_nf,
            "alpha": ALPHA, "eps": ROBUST_EPS, "tau": QT_TAU,
        }, f, indent=2)


# In[35]:


# Visualization for τ=0.99 quantile + (marginal/robust) conformal
FIGDIR = OUTDIR / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUTDIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)

def savefig_both(path_png):
    path_png = Path(path_png)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

_rng_plot = np.random.default_rng(SEED + 42)

X_elem_test = df.loc[idx_test, elem_cols].to_numpy()
USE_FAMILY_AWARE = False

if USE_FAMILY_AWARE and ('robust_L_for_comps_hi_famaware' in globals()):
    fam_test = df.loc[idx_test, "family"].to_numpy()
    L_robust_hi_mm_test = robust_L_for_comps_hi_famaware(
        X_elem_test, fam_test,
        q_cal_robust=q_robust_hi,
        model=cat_qt_hi,
        eps=ROBUST_EPS,
        K=ROBUST_SAMPLES,
        rng=_rng_plot
    )
else:
    L_robust_hi_mm_test = robust_L_for_comps_hi(
        X_elem_test,
        q_cal_robust=q_robust_hi,
        model=cat_qt_hi,
        eps=ROBUST_EPS,
        K=ROBUST_SAMPLES,
        rng=_rng_plot
    )

# Quantile calibration @ τ (CAL vs TEST)
obs_tau_cal  = float(np.mean(y_cal  <= qhat_cal_hi_adj))
obs_tau_test = float(np.mean(y_test <= qhat_test_hi_adj))
pin_cal      = mean_pinball_loss(y_cal,  qhat_cal_hi_adj,  alpha=QT_TAU_HIGH)
pin_test     = mean_pinball_loss(y_test, qhat_test_hi_adj, alpha=QT_TAU_HIGH)
print(f"[τ={QT_TAU_HIGH:.2f}] P(Y≤q̂): CAL={obs_tau_cal:.3f}, TEST={obs_tau_test:.3f} (target {QT_TAU_HIGH:.2f})")
print(f"Pinball loss: CAL={pin_cal:.4f}, TEST={pin_test:.4f}")

plt.figure(figsize=(5.2,3.6))
ax = sns.barplot(x=["Cal","Test"], y=[obs_tau_cal, obs_tau_test])
ax.axhline(QT_TAU_HIGH, ls="--", label=f"Target τ={QT_TAU_HIGH:.2f}")
for i, v in enumerate([obs_tau_cal, obs_tau_test]):
    ax.text(i, min(0.99, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0,1)
ax.set_ylabel("Observed P(Y ≤ q̂τ)")
ax.set_title(f"Quantile calibration (τ = {QT_TAU_HIGH:.2f})")
ax.legend(frameon=False, loc="lower right")
savefig_both(FIGDIR / "hi_tau_quantile_calibration.png")

pd.DataFrame({
    "split": ["cal","test"],
    "observed_prob": [obs_tau_cal, obs_tau_test],
    "target_tau": [QT_TAU_HIGH, QT_TAU_HIGH],
    "pinball_loss": [pin_cal, pin_test]
}).to_csv(SRC_DIR / "hi_tau_quantile_calibration_bar.csv", index=False)

# Coverage bars: marginal vs robust (TEST)
def wilson_ci_counts(k, n, z=1.96):
    """Wilson CI from integer successes k out of n; returns (p_hat, lo, hi)."""
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom  = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(p), float(lo), float(hi)

k_m = int(np.sum(y_test_mm >= L_marginal_hi_mm))
k_r = int(np.sum(y_test_mm >= L_robust_hi_mm_test))
n   = int(len(y_test_mm))

p_m, lo_m, hi_m = wilson_ci_counts(k_m, n)
p_r, lo_r, hi_r = wilson_ci_counts(k_r, n)

vals     = np.array([p_m, p_r])
low_err  = np.maximum(0.0, vals - np.array([lo_m, lo_r]))
high_err = np.maximum(0.0, np.array([hi_m, hi_r]) - vals)

plt.figure(figsize=(5.8,3.8))
ax = sns.barplot(x=["Marginal","Robust"], y=vals)
ax.errorbar([0,1], vals, yerr=[low_err, high_err], fmt="none", capsize=4, color="black")
ax.axhline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
for i, v in enumerate(vals):
    ax.text(i, min(0.99, v + 0.01), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0,1)
ax.set_ylabel("Coverage (fraction ≥ L)")
ax.set_title(f"Coverage@{int((1-ALPHA)*100)}% (test)")
ax.legend(frameon=False, loc="lower right")
savefig_both(FIGDIR / "hi_tau_coverage_marg_vs_rob.png")

pd.DataFrame({
    "type":     ["marginal","robust"],
    "coverage": [p_m, p_r],
    "ci_lo":    [lo_m, lo_r],
    "ci_hi":    [hi_m, hi_r],
    "target":   [1-ALPHA, 1-ALPHA],
    "n_test":   [n, n]
}).to_csv(SRC_DIR / "hi_tau_coverage_bars.csv", index=False)

# Sharpness: gap = exp(q̂τ) − L
qhat_test_hi_mm = np.exp(qhat_test_hi)
gap_marg = qhat_test_hi_mm - L_marginal_hi_mm
gap_rob  = qhat_test_hi_mm - L_robust_hi_mm_test

plt.figure(figsize=(6.2,3.9))
ax = sns.boxplot(data=pd.DataFrame({"Marginal": gap_marg, "Robust": gap_rob}), showfliers=False)
ax.set_ylabel("Gap (mm)")
ax.set_title(f"Sharpness of lower bounds (τ = {QT_TAU_HIGH:.2f})")
savefig_both(FIGDIR / "hi_tau_sharpness_box.png")

pd.DataFrame({"gap_marg_mm": gap_marg, "gap_rob_mm": gap_rob}).to_csv(
    SRC_DIR / "hi_tau_sharpness_gaps.csv", index=False
)

# Reliability by predicted robust L
try:
    binned = pd.qcut(pd.Series(L_robust_hi_mm_test), q=7, duplicates="drop")
    df_rel = (
        pd.DataFrame({"bin": binned, "ok": (y_test_mm >= L_robust_hi_mm_test)})
        .groupby("bin").agg(n=("ok","size"), coverage=("ok","mean"))
        .reset_index()
    )
    centers = df_rel["bin"].apply(lambda iv: 0.5*(iv.left + iv.right)).astype(float)
    lefts   = df_rel["bin"].apply(lambda iv: float(iv.left))
    rights  = df_rel["bin"].apply(lambda iv: float(iv.right))
    df_rel_plot = df_rel.assign(bin_center=centers, bin_left=lefts, bin_right=rights).drop(columns=["bin"])
except Exception:
    bins = np.quantile(L_robust_hi_mm_test, np.linspace(0,1,8))
    bins = np.unique(bins)
    bins[0] -= 1e-9; bins[-1] += 1e-9
    bin_ids = np.digitize(L_robust_hi_mm_test, bins) - 1
    rows = []
    for b in range(len(bins)-1):
        m = (bin_ids == b)
        if m.sum() == 0: 
            continue
        rows.append({
            "n": int(m.sum()),
            "coverage": float(np.mean(y_test_mm[m] >= L_robust_hi_mm_test[m])),
            "bin_left": float(bins[b]),
            "bin_right": float(bins[b+1]),
            "bin_center": float(0.5*(bins[b]+bins[b+1]))
        })
    df_rel_plot = pd.DataFrame(rows)

plt.figure(figsize=(6.8,4.2))
ax = sns.lineplot(x=df_rel_plot["bin_center"], y=df_rel_plot["coverage"], marker="o")
for xv, yv, nn in zip(df_rel_plot["bin_center"], df_rel_plot["coverage"], df_rel_plot["n"]):
    ax.text(xv, min(0.99, yv + 0.03), f"n={nn}", ha="center", fontsize=9)
ax.axhline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
ax.set_xlabel("Predicted robust L (mm), bin centers")
ax.set_ylabel("Empirical coverage in bin")
ax.set_title("Reliability by predicted L (robust)")
ax.set_ylim(0,1)
ax.legend(frameon=False)
savefig_both(FIGDIR / "hi_tau_reliability_by_L.png")

df_rel_plot.to_csv(SRC_DIR / "hi_tau_reliability_by_bin.csv", index=False)

# Precision@k curves (rank by robust L)
def precision_k_with_baseline(L_mm, y_mm, T, ks=(5,10,20,50,100)):
    pos = (y_mm >= T).astype(int)
    P, N = int(pos.sum()), len(pos)
    order = np.argsort(-L_mm)
    prec = []
    for k in ks:
        sel = order[:min(k, N)]
        prec.append(float(pos[sel].mean()))
    baseline = P / N
    ceiling  = [min(1.0, P/k) for k in ks]
    return np.array(ks), np.array(prec), baseline, np.array(ceiling), P, N

thresholds = sorted(set(map(float, list(THRESHOLDS_MM))))

prec_rows = []
for Dstar in thresholds:
    ks, vals, base, ceil, P, N = precision_k_with_baseline(L_robust_hi_mm_test, y_test_mm, Dstar)
    plt.figure(figsize=(6.4,4.0))
    sns.lineplot(x=ks, y=vals, marker="o", label="Model")
    plt.hlines(base, ks.min(), ks.max(), linestyles="--", label=f"Prevalence baseline ({P}/{N})")
    plt.plot(ks, ceil, linestyle=":", marker="x", label="Best possible at k")
    plt.ylim(0,1); plt.xlabel("k"); plt.ylabel(f"Precision@k (≥ {int(Dstar)} mm)")
    plt.title(f"Tail discovery (robust L, τ={QT_TAU_HIGH:.2f})")
    plt.legend(frameon=False, loc="upper right")
    savefig_both(FIGDIR / f"hi_tau_precision_k_{int(Dstar)}mm.png")
    for k, v, c in zip(ks, vals, ceil):
        prec_rows.append({
            "threshold_mm": Dstar, "k": int(k), "precision": float(v),
            "baseline": base, "ceiling": float(c), "P": P, "N": N
        })

pd.DataFrame(prec_rows).to_csv(SRC_DIR / "hi_tau_precision_at_k.csv", index=False)

rank_df = df.loc[idx_test, ["signature","family"]].copy()
rank_df["true_mm"]  = y_test_mm
rank_df["L_rob_mm"] = L_robust_hi_mm_test
rank_df = rank_df.sort_values("L_rob_mm", ascending=False).reset_index(drop=True)
rank_df.to_csv(SRC_DIR / "hi_tau_test_ranking_by_Lrob.csv", index=False)

# Per-family robust coverage (TEST)
families_test = df.iloc[idx_test]["family"].values
rows = []
for fam in pd.unique(families_test):
    m = (families_test == fam)
    if m.sum() < 5:
        continue
    rows.append({"family": fam, "n": int(m.sum()),
                 "coverage_robust": float(np.mean(y_test_mm[m] >= L_robust_hi_mm_test[m]))})
df_fam_cov = pd.DataFrame(rows).sort_values("n", ascending=False)

plt.figure(figsize=(7.2, max(3.6, 0.28*len(df_fam_cov))))
ax = sns.barplot(data=df_fam_cov, x="coverage_robust", y="family", orient="h")
ax.axvline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
ax.set_xlabel("Coverage (robust)"); ax.set_ylabel("Solvent family")
ax.set_title("Per-family robust coverage (test)")
for i, v in enumerate(df_fam_cov["coverage_robust"]):
    ax.text(min(0.99, v), i, f" {v:.2f}", va="center")
ax.legend(frameon=False, loc="lower right")
savefig_both(FIGDIR / "hi_tau_per_family_coverage.png")

df_fam_cov.to_csv(SRC_DIR / "hi_tau_coverage_by_family.csv", index=False)

# Row-wise test data with bounds
src_rows = df.loc[idx_test, ["signature","family"]].copy()
src_rows["true_mm"]    = y_test_mm
src_rows["qhat_mm"]    = qhat_test_hi_mm
src_rows["L_marg_mm"]  = L_marginal_hi_mm
src_rows["L_rob_mm"]   = L_robust_hi_mm_test
src_rows.to_csv(SRC_DIR / "hi_tau_test_bounds.csv", index=False)

print("Saved figures to:", FIGDIR)
print("Saved source data to:", SRC_DIR)


# In[36]:


# Conformal score histograms with cutoff lines
plt.figure(figsize=(6.4, 4.0))
sns.histplot(np.maximum(0.0, qhat_cal_hi - y_cal), bins=40, kde=True, label="S_marginal", alpha=0.6)
sns.histplot(S_cal_hi_rob, bins=40, kde=True, label="S_robust", alpha=0.6)
plt.axvline(weighted_quantile(np.maximum(0.0, qhat_cal_hi - y_cal), 1 - ALPHA), ls="--", label="q_(1-α) marginal")
plt.axvline(q_robust_hi, ls=":", label="q_(1-α) robust")
plt.xlabel("Conformal score"); plt.ylabel("Count")
plt.title(f"Conformal score distributions (τ = {QT_TAU_HIGH:.2f})")
plt.legend(frameon=False)
savefig_both(FIGDIR / "hi_tau_conformal_scores_hist.png")

pd.DataFrame({
    "S_marginal": np.maximum(0.0, qhat_cal_hi - y_cal),
    "S_robust": S_cal_hi_rob
}).to_csv(SRC_DIR / "hi_tau_conformal_scores.csv", index=False)


# In[37]:


# Enrichment (cumulative hits) vs k at key thresholds
def cumulative_hits_curve(scores_desc, y_mm, T):
    order = np.argsort(-scores_desc)
    y_bin = (y_mm >= T).astype(int)[order]
    cum_hits = np.cumsum(y_bin)
    k = np.arange(1, len(y_bin)+1)
    baseline_rate = y_bin.mean()
    return k, cum_hits, baseline_rate

for Dstar in thresholds:
    k, cum_hits, base = cumulative_hits_curve(L_robust_hi_mm_test, y_test_mm, Dstar)
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(k, cum_hits, label="Model (ranked by L_robust)")
    plt.plot(k, base*k, ls="--", label="Random baseline")
    plt.xlabel("k"); plt.ylabel(f"Cumulative hits (≥ {int(Dstar)} mm)")
    plt.title(f"Enrichment curve (τ={QT_TAU_HIGH:.2f})")
    plt.legend(frameon=False)
    savefig_both(FIGDIR / f"hi_tau_enrichment_curve_{int(Dstar)}mm.png")

    pd.DataFrame({"k": k, "cum_hits": cum_hits, "baseline": base*k}) \
      .to_csv(SRC_DIR / f"hi_tau_enrichment_{int(Dstar)}mm.csv", index=False)


# In[38]:


# OOD distance (composition L1) and coverage vs distance
Xtr = df.loc[idx_train, elem_cols].to_numpy()
Xte = df.loc[idx_test,  elem_cols].to_numpy()

def min_l1_to_train(X_train, X_query, max_train=5000):
    Xt = X_train if len(X_train) <= max_train else X_train[np.random.RandomState(SEED).choice(len(X_train), max_train, replace=False)]
    dists = []
    for x in X_query:
        d = np.abs(Xt - x).sum(axis=1).min()
        dists.append(float(d))
    return np.array(dists)

l1_ood = min_l1_to_train(Xtr, Xte)
bins = np.quantile(l1_ood, np.linspace(0,1,8)); bins = np.unique(bins)
bins[0] -= 1e-9; bins[-1] += 1e-9
idx_bin = np.digitize(l1_ood, bins) - 1

rows = []
for b in range(len(bins)-1):
    m = (idx_bin == b)
    if m.sum() == 0: continue
    rows.append({
        "bin_left": float(bins[b]),
        "bin_right": float(bins[b+1]),
        "bin_center": float(0.5*(bins[b]+bins[b+1])),
        "n": int(m.sum()),
        "coverage_marginal": float(np.mean(y_test_mm[m] >= L_marginal_hi_mm[m])),
        "coverage_robust": float(np.mean(y_test_mm[m] >= L_robust_hi_mm_test[m]))
    })
df_ood = pd.DataFrame(rows)

plt.figure(figsize=(6.8, 4.2))
sns.lineplot(x="bin_center", y="coverage_marginal", data=df_ood, marker="o", label="Marginal")
sns.lineplot(x="bin_center", y="coverage_robust", data=df_ood, marker="o", label="Robust")
plt.axhline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
plt.xlabel("Min L1 distance to training compositions"); plt.ylabel("Coverage (test)")
plt.title("Coverage vs OOD distance (composition space)")
plt.ylim(0,1); plt.legend(frameon=False)
savefig_both(FIGDIR / "hi_tau_coverage_vs_ood_distance.png")

df_ood.to_csv(SRC_DIR / "hi_tau_coverage_vs_ood_distance.csv", index=False)


# In[39]:


# UMAP of compositions, colored by robust coverage
X_all = df[elem_cols].to_numpy()
reducer = umap.UMAP(random_state=SEED, n_neighbors=30, min_dist=0.1, metric="manhattan")
Z = reducer.fit_transform(X_all)

Z_df = pd.DataFrame(Z, columns=["z1","z2"])
Z_df["set"] = "other"
Z_df.loc[idx_train, "set"] = "train"
Z_df.loc[idx_test,  "set"] = "test"
Z_df["covered_robust"] = np.nan
Z_df.loc[idx_test, "covered_robust"] = (y_test_mm >= L_robust_hi_mm_test).astype(int)

plt.figure(figsize=(6.0, 5.2))
sns.scatterplot(data=Z_df[Z_df["set"]=="train"], x="z1", y="z2", s=8, alpha=0.3, label="train")
sns.scatterplot(data=Z_df[Z_df["set"]=="test"], x="z1", y="z2", hue="covered_robust", palette="Set1", s=24, alpha=0.9)
plt.title("UMAP of compositions (test colored by robust coverage)")
plt.legend(frameon=False)
savefig_both(FIGDIR / "hi_tau_umap_coverage.png")

Z_df.to_csv(SRC_DIR / "hi_tau_umap_embedding.csv", index=False)


# In[40]:


# Conformal lower bounds
assert "qhat_cal"   in globals() and "qhat_test"   in globals(),   "Run the τ-quantile fit first."
assert "idx_train"  in globals() and "idx_cal"     in globals() and "idx_test" in globals(), "Make splits first."
assert "X_cal"      in globals() and "X_test"      in globals(),   "Build features first."
assert "ALPHA"      in globals(), "Set ALPHA (e.g., 0.10 for 90% coverage)."

family_train = df.loc[idx_train, "family"].to_numpy()
family_cal = df.loc[idx_cal,   "family"].to_numpy()
family_test = df.loc[idx_test,  "family"].to_numpy()

# One-sided conformal scores for LOWER bounds (log space)
def one_sided_scores(y_true_log, q_pred_log):
    y_true_log = np.asarray(y_true_log, float).ravel()
    q_pred_log = np.asarray(q_pred_log, float).ravel()
    return np.maximum(0.0, q_pred_log - y_true_log)

S_cal = one_sided_scores(y_cal, qhat_cal)

def _safe_wq(values, q, w=None):
    v = np.asarray(values, float)
    if w is None:
        v = v[np.isfinite(v)]
        return np.quantile(v, q) if v.size else np.nan
    w = np.asarray(w, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    return weighted_quantile(v, q, sample_weight=w) if v.size else np.nan

# Marginal CP (distribution-free)
q_marginal     = _safe_wq(S_cal, 1 - ALPHA)
L_marginal_log = qhat_test - q_marginal
L_marginal_mm  = np.exp(L_marginal_log)

# Group-conditional CP (family-wise) with shrinkage toward global
M_SHRINK = int(globals().get("MIN_GROUP_N", 30))
global_q = _safe_wq(S_cal, 1 - ALPHA)

L_group_mm = np.empty_like(L_marginal_mm)
uniq_test_fams = np.unique(family_test)

fam2_scores = {}
for f in np.unique(family_cal):
    m = (family_cal == f)
    fam2_scores[f] = S_cal[m] if m.any() else np.array([], dtype=float)

for f in uniq_test_fams:
    S_f = fam2_scores.get(f, np.array([], dtype=float))
    n_f = S_f.size
    if n_f == 0:
        q_f = global_q
    else:
        q_f_emp = _safe_wq(S_f, 1 - ALPHA)
        lam = n_f / (n_f + M_SHRINK)
        q_f  = lam * q_f_emp + (1 - lam) * global_q
    m_test = (family_test == f)
    L_group_mm[m_test] = np.exp(qhat_test[m_test] - q_f)

# Covariate-shift-aware CP via density-ratio weighting (cal -> test)
X_cal_cls = X_cal.values
X_test_cls = X_test.values
X_cls = np.vstack([X_cal_cls, X_test_cls])
y_cls = np.hstack([np.zeros(len(X_cal_cls), dtype=int),
                   np.ones(len(X_test_cls), dtype=int)])

cls = LogisticRegression(
    penalty="l2", solver="liblinear",
    class_weight="balanced", max_iter=2000, random_state=SEED
)
cls.fit(X_cls, y_cls)

# Probabilities for calibration points
p_test_on_cal = cls.predict_proba(X_cal_cls)[:, 1]
p_cal_on_cal = 1.0 - p_test_on_cal

# Prior correction: π_cal/π_test (see odds identity)
prior_ratio = len(X_cal_cls) / max(1, len(X_test_cls))
w_shift = (p_test_on_cal / np.maximum(p_cal_on_cal, 1e-9)) * prior_ratio
w_shift = np.clip(w_shift, 0.0, 1e6)

q_weighted = _safe_wq(S_cal, 1 - ALPHA, w_shift)
L_weighted_mm  = np.exp(qhat_test - q_weighted)

# Metrics: coverage & sharpness (with simple uncertainty bars)
def coverage(y_true_log, L_mm):
    y_mm = np.exp(np.asarray(y_true_log, float))
    return float(np.mean(y_mm >= L_mm))

def sharpness(qhat_log, L_mm):
    return float(np.median(np.exp(qhat_log) - L_mm))

def wilson_ci(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return p, max(0.0, center - half), min(1.0, center + half)

cov_m = coverage(y_test, L_marginal_mm)
cov_g = coverage(y_test, L_group_mm)
cov_w = coverage(y_test, L_weighted_mm)

k_m, k_g, k_w = int(np.sum(y_test_mm >= L_marginal_mm)), int(np.sum(y_test_mm >= L_group_mm)), int(np.sum(y_test_mm >= L_weighted_mm))
n_t = len(y_test_mm)
cov_m_ci = wilson_ci(k_m, n_t)
cov_g_ci = wilson_ci(k_g, n_t)
cov_w_ci = wilson_ci(k_w, n_t)

shp_m = sharpness(qhat_test, L_marginal_mm)
shp_g = sharpness(qhat_test, L_group_mm)
shp_w = sharpness(qhat_test, L_weighted_mm)

print(f"Coverage@{1-ALPHA:.2f}  "
      f"[marginal={cov_m:.3f} (CI {cov_m_ci[1]:.3f}-{cov_m_ci[2]:.3f}), "
      f"group={cov_g:.3f} (CI {cov_g_ci[1]:.3f}-{cov_g_ci[2]:.3f}), "
      f"shift-weighted={cov_w:.3f} (CI {cov_w_ci[1]:.3f}-{cov_w_ci[2]:.3f})]")

print(f"Sharpness gap median (mm): "
      f"marginal={shp_m:.3f}, group={shp_g:.3f}, shift-weighted={shp_w:.3f}")


# In[41]:


# Tolerance-robust conformal lower bounds (± at.% drift)
X_cal_elem  = df.loc[idx_cal,  elem_cols].to_numpy(float)
X_test_elem = df.loc[idx_test, elem_cols].to_numpy(float)

def _row_normalize(X):
    s = X.sum(axis=1, keepdims=True)
    return X / np.where(s > 0, s, 1.0)

X_cal_elem  = _row_normalize(X_cal_elem)
X_test_elem = _row_normalize(X_test_elem)

# Canonical robust score (lower) helper
def robust_min_logq_batch(q_model, X_elem, eps, K, rng=None, chunk=4096):
    """
    For each x in X_elem, jitter K times (L1-ball radius eps), predict log-quantiles,
    and return the per-sample minimum (worst-case) log q_τ.
    Uses chunked batching to keep memory and prediction time in check.
    """
    if rng is None:
        rng = np.random.default_rng(SEED + DRIFT_SEED)
    X_elem = np.asarray(X_elem, float)
    n, d = X_elem.shape
    qmin = np.full(n, np.inf, dtype=float)

    buf_X = []
    buf_id = []
    def _flush():
        nonlocal qmin, buf_X, buf_id
        if not buf_X: return
        Xj = np.vstack(buf_X)
        feats = make_features_from_compositions(Xj)
        preds = np.asarray(q_model.predict(feats), float)
        start = 0
        for sid, count in buf_id:
            block = preds[start:start+count]
            qmin[sid] = min(qmin[sid], float(np.nanmin(block)))
            start += count
        buf_X.clear()
        buf_id.clear()

    for i, x in enumerate(X_elem):
        rng_i = np.random.default_rng(SEED + DRIFT_SEED)
        Xj = jitter_in_L1_ball_simplex(x, eps=float(eps), K=int(K), rng=rng_i)
        Xj = _row_normalize(Xj)
        buf_X.append(Xj)
        buf_id.append((i, K))
        if sum(bc for _, bc in buf_id) >= chunk:
            _flush()
    _flush()

    bad = ~np.isfinite(qmin)
    if np.any(bad):
        feats0 = make_features_from_compositions(X_elem[bad])
        q0 = np.asarray(q_model.predict(feats0), float)
        qmin[bad] = q0
    return qmin

# Robust calibration quantile on CAL
assert "q_robust_hi" in globals() and np.isfinite(q_robust_hi), \
    "Run Cell 28 first: q_robust_hi must already be defined."
assert "Q_ROBUST_HI_FROZEN" in globals() and np.isclose(float(q_robust_hi), Q_ROBUST_HI_FROZEN), \
    "q_robust_hi was modified after Cell 28."

S_cal_rob = np.maximum(0.0, np.nan_to_num(np.asarray(S_cal_hi_rob, float), nan=0.0))
print(f"[CP] Using frozen q_robust_hi = {q_robust_hi:.6f} (set in Cell 28)")

# Robust TEST bounds (mm)
qmin_test_log = robust_min_logq_batch(
    cat_qt_hi, X_test_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES,
    rng=np.random.default_rng(SEED + DRIFT_SEED), chunk=4096
)
L_robust_mm = np.exp(qmin_test_log - q_robust_hi)

if "qhat_test_hi" not in globals():
    qhat_test_hi = cat_qt_hi.predict(X_test)
qhat_test_hi_mm = np.exp(qhat_test_hi)

# Marginal lower bound for side-by-side reporting
S_cal_marg = np.maximum(0.0, cat_qt_hi.predict(X_cal) - y_cal)
q_marginal_hi = conformal_qhat(S_cal_marg, ALPHA)
L_marginal_mm = np.exp(qhat_test_hi - q_marginal_hi)

# Diagnostics + uncertainty bars
y_test_mm = np.exp(y_test)
qhat_test_hi_mm = np.exp(qhat_test_hi)

cov_r = float(np.mean(y_test_mm >= L_robust_mm))
gap_r = qhat_test_hi_mm - L_robust_mm
shp_r = float(np.median(gap_r))
cov_m = float(np.mean(y_test_mm >= L_marginal_mm))

def wilson_ci_counts(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), max(0.0, center - half), min(1.0, center + half)

k = int(np.sum(y_test_mm >= L_robust_mm))
n = int(len(y_test_mm))
p_hat, lo, hi = wilson_ci_counts(k, n)

print(f"Robust coverage@{1-ALPHA:.2f} = {cov_r:.3f} "
      f"(Wilson 95% CI {lo:.3f}–{hi:.3f}) | robust sharpness median gap = {shp_r:.3f} mm")

print(f"Marginal coverage@{1-ALPHA:.2f} = {cov_m:.3f}")

SRC_DIR = OUTDIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "index": idx_test,
    "family": df.loc[idx_test, "family"].to_numpy(),
    "signature": df.loc[idx_test, "signature"].to_numpy(),
    "true_mm": y_test_mm,
    "qhat_hi_mm": qhat_test_hi_mm,
    "L_robust_mm": L_robust_mm,
    "covered_robust": (y_test_mm >= L_robust_mm).astype(int),
    "gap_mm": gap_r
}).to_csv(SRC_DIR / "robust_bounds_test.csv", index=False)

pd.DataFrame({
    "S_robust": S_cal_rob,
    "alpha": ALPHA,
    "q_robust_hi": q_robust_hi,
    "eps_used": ROBUST_EPS,
    "K_jitter": ROBUST_SAMPLES
}).to_csv(SRC_DIR / "robust_calibration_scores.csv", index=False)

summary = {
    "tau_high": float(QT_TAU_HIGH) if "QT_TAU_HIGH" in globals() else None,
    "alpha": float(ALPHA),
    "epsilon": float(ROBUST_EPS),
    "K_jitter": int(ROBUST_SAMPLES),
    "q_robust_hi": float(q_robust_hi),
    "coverage_test": {
        "robust": float(cov_r),
        **({"marginal": float(cov_m)} if "L_marginal_mm" in globals() else {})
    },
    "sharpness_mm_median": float(shp_r),
    "n_test": int(len(y_test_mm))
}
(OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
with open(OUTDIR / "reports" / "robust_cp_summary.json", "w") as f:
    json.dump(summary, f, indent=2)


# Tail discovery metrics (with CIs, lift, AP)
sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

_rng_pk = np.random.default_rng(SEED + 606)
_rng_bs = np.random.default_rng(SEED + 607)

if "L_marginal_mm" not in globals():
    S_cal_marg = np.maximum(0.0, (cat_qt_hi.predict(X_cal) - y_cal))
    q_marginal_hi = conformal_qhat(S_cal_marg, ALPHA)
    qhat_test_hi  = cat_qt_hi.predict(X_test) if "qhat_test_hi" not in globals() else qhat_test_hi
    L_marginal_mm = np.exp(qhat_test_hi - q_marginal_hi)

FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

def _finite_pair(L, y):
    L = np.asarray(L, float); y = np.asarray(y, float)
    m = np.isfinite(L) & np.isfinite(y)
    return L[m], y[m]

def _rank_order(L, second=None):
    """Deterministic tie-break: primary=L desc, then second desc, then index asc."""
    idx = np.arange(len(L))
    if second is None:
        return np.lexsort((idx, -L))
    return np.lexsort((idx, -np.asarray(second, float), -np.asarray(L, float)))

def precision_at_k_curve(L_mm, y_mm, thresh, ks=(5,10,20,50,100)):
    Lm, Ym = _finite_pair(L_mm, y_mm)
    if Lm.size == 0:
        return np.array(ks, int), np.zeros(len(ks), float), 0.0, 0, 0
    order = np.argsort(-Lm)  # primary rank only (lower-bound score)
    P = int(np.sum(Ym >= thresh)); N = int(len(Ym))
    vals = []
    for k in ks:
        k_eff = min(k, N)
        if k_eff == 0:
            vals.append(0.0); continue
        sel = order[:k_eff]
        vals.append(float(np.mean(Ym[sel] >= thresh)))
    baseline = P / N if N > 0 else 0.0
    return np.array(ks, int), np.array(vals, float), baseline, P, N

def precision_at_k_bootstrap(L_mm, y_mm, thresh, ks=(5,10,20,50,100), B=1000, rng=_rng_bs):
    """Nonparametric bootstrap CIs for P@k (resample pairs). Returns lo/hi arrays."""
    Lm, Ym = _finite_pair(L_mm, y_mm)
    N = len(Lm)
    if N == 0:
        z = np.zeros(len(ks), float)
        return z, z
    boot = np.empty((B, len(ks)), float)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        Lb, Yb = Lm[idx], Ym[idx]
        order = np.argsort(-Lb)
        for j, k in enumerate(ks):
            k_eff = min(k, N)
            if k_eff == 0:
                boot[b, j] = 0.0
            else:
                sel = order[:k_eff]
                boot[b, j] = float(np.mean(Yb[sel] >= thresh))
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return lo, hi

def average_precision_score_safe(L_mm, y_mm, thresh):
    """AP over the ranking induced by L (score), for binary label (y>=thresh)."""
    Lm, Ym = _finite_pair(L_mm, y_mm)
    if len(Lm) == 0:
        return np.nan
    pos = (Ym >= thresh).astype(int)
    if pos.sum() == 0 or pos.sum() == len(pos):
        return np.nan
    return float(average_precision_score(pos, Lm))

methods = []
methods.append(("Marginal", L_marginal_mm))
if "L_group_mm" in globals():
    methods.append(("Group-cond.", L_group_mm))
if "L_weighted_mm" in globals():
    methods.append(("Shift-weighted", L_weighted_mm))
if "L_robust_mm" in globals():
    robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"
    methods.append((robust_label, L_robust_mm))

if not methods:
    raise RuntimeError("No lower-bound arrays available to compute tail discovery metrics.")

KS = (5, 10, 20, 50, 100)
N_BOOT_Pk = 1000

all_rows = []
ap_rows  = []

for Dstar in THRESHOLDS_MM:
    baseline = float(np.mean(y_test_mm >= Dstar))
    print(f"\n=== Precision@k for ≥{Dstar:.0f} mm (baseline prevalence = {baseline:.2f}) ===")

    lines = []
    for name, L in methods:
        ks, vals, base, P, N = precision_at_k_curve(L, y_test_mm, Dstar, ks=KS)
        lo, hi = precision_at_k_bootstrap(L, y_test_mm, Dstar, ks=KS, B=N_BOOT_Pk, rng=_rng_bs)
        ceiling = np.array([min(1.0, P/max(1,k)) for k in ks], float)
        lift = np.divide(vals, base, out=np.full_like(vals, np.nan), where=base>0)

        row_str = "  " + name.ljust(16) + " : " + "  ".join([f"P@{k}={v:.2f}" for k, v in zip(ks, vals)])
        lines.append(row_str)

        for j, k in enumerate(ks):
            all_rows.append({
                "threshold_mm": float(Dstar),
                "method": name,
                "k": int(k),
                "precision": float(vals[j]),
                "prec_lo95": float(lo[j]),
                "prec_hi95": float(hi[j]),
                "lift": float(lift[j]) if np.isfinite(lift[j]) else np.nan,
                "baseline": float(base),
                "ceiling": float(ceiling[j]),
                "positives_P": int(P),
                "N_test": int(N),
            })

        ap = average_precision_score_safe(L, y_test_mm, Dstar)
        ap_rows.append({"threshold_mm": float(Dstar), "method": name, "AP": ap})

    print("\n".join(lines))

    # Plot P@k
    plt.figure(figsize=(6.3, 4.0))
    plt.axhline(baseline, ls="--", lw=1, label=f"Baseline={baseline:.2f}")
    for name, L in methods:
        ks, vals, base, P, N = precision_at_k_curve(L, y_test_mm, Dstar, ks=KS)
        lo, hi = precision_at_k_bootstrap(L, y_test_mm, Dstar, ks=KS, B=N_BOOT_Pk, rng=_rng_bs)
        plt.plot(ks, vals, marker="o", label=name)
        yerr = np.vstack([vals - lo, hi - vals])
        plt.errorbar(ks, vals, yerr=yerr, fmt="none", capsize=4, lw=1)
        ceiling = [min(1.0, P/max(1,k)) for k in ks]
        plt.plot(ks, ceiling, linestyle=":", marker="x", ms=4, label=None, alpha=0.7)
    plt.ylim(0, 1)
    plt.xlabel("k"); plt.ylabel(f"Precision@k (≥ {Dstar:.0f} mm)")
    plt.title(f"Tail discovery (≥ {Dstar:.0f} mm)")
    plt.legend(frameon=False, loc="lower right")
    path = FIGDIR / f"fig_precision_k_{int(Dstar)}mm.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

df_pk  = pd.DataFrame(all_rows)
df_ap  = pd.DataFrame(ap_rows)
df_pk.to_csv(SRC_DIR / "precision_at_k_all_methods.csv", index=False)
df_ap.to_csv(SRC_DIR / "average_precision_all_methods.csv", index=False)

print("\nSaved curves:", [str(FIGDIR / f"fig_precision_k_{int(D)}mm.png") for D in THRESHOLDS_MM])
print("Saved CSVs:", SRC_DIR / "precision_at_k_all_methods.csv", "and", SRC_DIR / "average_precision_all_methods.csv")


# In[43]:


if "embed_allowed_to_full" not in globals():
    def embed_allowed_to_full(X_allowed):
        X_allowed = np.atleast_2d(np.asarray(X_allowed, float))
        d_full = len(elem_cols)
        X_full = np.zeros((X_allowed.shape[0], d_full), dtype=float)
        X_full[:, allowed_idx] = X_allowed
        rs = X_full.sum(axis=1, keepdims=True)
        X_full = X_full / np.where(rs > 0, rs, 1.0)
        return X_full

assert "sample_drift_neighborhood" in globals(), \
    "Run the canonical drift-sampler cell (In[4]) first."

def _min_jittered_qlog_for_row(x_allowed, *, eps, K, rngseed=None):
    rng = np.random.default_rng(globals().get("SEED_LOCAL", SEED) if rngseed is None else rngseed)
    x_allowed = np.asarray(x_allowed, float).ravel()

    if eps <= 0.0 or int(K) <= 1:
        x_full = embed_allowed_to_full(x_allowed[None, :])
        feats  = _make_feats(x_full)
        return float(np.asarray(cat_qt_hi.predict(feats), float).ravel()[0])

    Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
    Xj_full    = embed_allowed_to_full(Xj_allowed)
    feats      = _make_feats(Xj_full)
    qj         = np.asarray(cat_qt_hi.predict(feats), float).ravel()

    if qj.size == 0 or not np.isfinite(qj).any():
        x_full = embed_allowed_to_full(x_allowed[None, :])
        feats  = _make_feats(x_full)
        qj     = np.asarray(cat_qt_hi.predict(feats), float).ravel()

    return float(np.nanmin(qj))


# In[44]:


# Coverage bars for high-τ quantile (CAL vs TEST)
reports_dir = OUTDIR / "reports"
summary_paths = [
    reports_dir / "quantile_diag_summary.json",
    reports_dir / "metrics" / "summary_training.json",
    reports_dir / "robust_summary.json",
]

tau = 0.95
cov_cal = None
cov_test = None

for p in summary_paths:
    if p.exists():
        try:
            with open(p, "r") as f:
                js = json.load(f)
            tau = float(js.get("tau",
                      js.get("quantile_tau",
                      js.get("tau_high", tau))))
            cov_cal  = (cov_cal  or js.get("cal_coverage")
                                   or js.get("cal", {}).get("coverage")
                                   or js.get("pre_cp", {}).get("obs_tau_cal"))
            cov_test = (cov_test or js.get("test_coverage")
                                   or js.get("test", {}).get("coverage")
                                   or js.get("pre_cp", {}).get("obs_tau_test"))
        except Exception:
            pass

if cov_cal is None or cov_test is None:
    try:
        if 'qhat_cal_hi' in globals() and 'qhat_test_hi' in globals():
            _ycal = np.asarray(y_cal).ravel()
            _ytes = np.asarray(y_test).ravel()
            cov_cal  = float(np.mean(_ycal <= np.asarray(qhat_cal_hi).ravel()))
            cov_test = float(np.mean(_ytes <= np.asarray(qhat_test_hi).ravel()))
            tau = float(globals().get('QT_TAU_HIGH', tau))
        else:
            if 'X_cal' in globals() and 'X_test' in globals():
                qhat_cal_hi  = np.asarray(cat_qt_hi.predict(X_cal)).ravel()
                qhat_test_hi = np.asarray(cat_qt_hi.predict(X_test)).ravel()
            else:
                fe_cal  = make_features_from_compositions(df.loc[idx_cal,  elem_cols].to_numpy())
                fe_test = make_features_from_compositions(df.loc[idx_test, elem_cols].to_numpy())
                qhat_cal_hi  = np.asarray(cat_qt_hi.predict(fe_cal)).ravel()
                qhat_test_hi = np.asarray(cat_qt_hi.predict(fe_test)).ravel()
            _ycal = np.asarray(y_log)[df.index.get_indexer_for(idx_cal)]
            _ytes = np.asarray(y_log)[df.index.get_indexer_for(idx_test)]
            cov_cal  = float(np.mean(_ycal <= qhat_cal_hi))
            cov_test = float(np.mean(_ytes <= qhat_test_hi))
            tau = float(globals().get('QT_TAU_HIGH', tau))
    except Exception as e:
        print("[Coverage bars] Fallback compute failed:", e)

if cov_cal is None or cov_test is None:
    print("[Coverage bars] Could not find/compute coverage; skipping.")
else:
    out_json = reports_dir / "quantile_diag_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"tau": tau, "cal_coverage": cov_cal, "test_coverage": cov_test}, f, indent=2)

    target = tau
    data = pd.DataFrame({
        "split": ["Calibration", "Test"],
        "coverage": [cov_cal, cov_test],
        "target": [target, target],
    })

    plt.figure(figsize=(4.6, 3.4))
    plt.bar(data["split"], data["coverage"], width=0.6)
    plt.axhline(target, ls="--", lw=1.2)
    plt.ylim(0, 1)
    plt.ylabel("Empirical coverage  P(Y ≤ ŷτ)")
    plt.title(f"High-quantile calibration (τ={tau:.2f})")
    for i, v in enumerate(data["coverage"]):
        plt.text(i, min(v + 0.01, 0.99), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    outp = reports_dir / "fig_coverage_cal_vs_test.png"
    plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved coverage bars:", outp, "| and summary:", out_json)


# In[45]:


if 'allowed_idx' not in globals():
    allowed_idx = list(range(len(elem_cols)))

if 'project_simplex_fraction' not in globals():
    def project_simplex_fraction(x):
        x = np.asarray(x, float)
        x = np.maximum(x, 0.0)
        s = x.sum()
        return x/s if s > 0 else np.ones_like(x)/len(x)

if 'embed_allowed_to_full' not in globals():
    def embed_allowed_to_full(X_allowed):
        X_allowed = np.atleast_2d(X_allowed).astype(float)
        K = X_allowed.shape[0]
        X_full = np.zeros((K, len(elem_cols)), dtype=float)
        X_full[:, allowed_idx] = X_allowed
        row_sums = X_full.sum(axis=1, keepdims=True)
        return np.divide(X_full, np.where(row_sums > 0, row_sums, 1.0))

assert "sample_drift_neighborhood" in globals(), \
    "Run the canonical drift-sampler cell (In[4]) first."


# In[46]:


# Mondrian (stratified) calibration: by family and novelty bins
print("[Calib] Starting Mondrian (stratified) calibration by family & novelty...")

# Preconditions
for _name in ["df","elem_cols","y_log","cat_qt_hi","idx_cal"]:
    if _name not in globals():
        raise RuntimeError(f"Missing required global `{_name}` for Mondrian calibration.")
ALPHA = globals().get("ALPHA", 0.15)

# Local wrappers
def _make_feats(X_full):
    return make_features_from_compositions(X_full)

def _sig_family_argmax(X):
    fam_idx = np.argmax(X, axis=1)
    return np.array(elem_cols, dtype=object)[fam_idx].astype(str)

X_cal_full = df.loc[idx_cal, elem_cols].to_numpy(float, copy=False)
y_cal_log  = np.asarray(y_log)[df.index.get_indexer_for(idx_cal)]
fe_cal     = _make_feats(X_cal_full)
qhat_cal   = np.asarray(cat_qt_hi.predict(fe_cal), float)

# Marginal one-sided scores
S_marg = np.maximum(0.0, qhat_cal - y_cal_log)
q_marg_all = conformal_qhat(S_marg, ALPHA)

# Novelty (L1 at.% to TRAIN)
idx_train = globals().get("idx_train", df.index.difference(idx_cal))
X_train_full = df.loc[idx_train, elem_cols].to_numpy(float, copy=False)
nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(X_train_full)
nov_L1_atpct_cal = nn.kneighbors(X_cal_full, 1, return_distance=True)[0].ravel() * 100.0
nov_bins = pd.cut(nov_L1_atpct_cal, bins=[-0.01,0.5,1.0,2.0,np.inf], labels=["≤0.5","0.5–1.0","1–2",">2 at.%"])

# Families
family_cal = _sig_family_argmax(X_cal_full)

# Robust (jittered) scores on CAL
allowed_idx = list(range(len(elem_cols)))
EPS_robust  = float(globals().get("ROBUST_EPS", 0.01))
K_robust    = int(globals().get("ROBUST_SAMPLES", 64))
SEED_LOCAL  = int(globals().get("SEED", 123456))
BATCH       = int(globals().get("MONDRIAN_BATCH", 64))  # you can tune; stays in-memory

X_cal_allowed = X_cal_full[:, allowed_idx]
n_cal, d_allowed = X_cal_allowed.shape

print(f"[Calib] robust settings: CAL={n_cal}, d_allowed={d_allowed}, K={K_robust}, eps={EPS_robust}, batch={BATCH}")

qmin_log_cal = np.empty(n_cal, dtype=float)

def _jitters_one(x_allowed, eps, K, seed):
    rng = np.random.default_rng(seed)
    return jitter_allowed_simplex(x_allowed, eps=eps, K=K, rng=rng)

# process in batches to reuse a single predict() call per batch
processed = 0
for start in range(0, n_cal, BATCH):
    stop = min(start + BATCH, n_cal)
    Xb = X_cal_allowed[start:stop] 
    b  = Xb.shape[0]

    # build jitters per row, stack to (b, K, d_allowed)
    jitters = []
    for i in range(b):
        row_seed = SEED_LOCAL + DRIFT_SEED
        Xj = _jitters_one(Xb[i], eps=EPS_robust, K=K_robust, seed=row_seed)
        jitters.append(Xj)
    Xj_allowed = np.stack(jitters, axis=0)

    # flatten to (b*K, d_allowed) → embed full → features → predict
    Xj_allowed_flat = Xj_allowed.reshape(-1, Xj_allowed.shape[-1])
    Xj_full         = embed_allowed_to_full(Xj_allowed_flat)
    feats           = _make_feats(Xj_full)
    qj_log_flat     = np.asarray(cat_qt_hi.predict(feats), float)
    qj_log          = qj_log_flat.reshape(b, K_robust)

    # min over jitters per row
    qmin_log_cal[start:stop] = qj_log.min(axis=1)

    processed = stop
    if (start // BATCH) % 10 == 0 or processed == n_cal:
        print(f"[Calib] robust jitters: {processed}/{n_cal} rows processed")

# robust one-sided residuals
S_rob = np.maximum(0.0, qmin_log_cal - y_cal_log)

# Per-group quantiles
def _per_group_quantile(values, groups, alpha):
    values = np.asarray(values)
    groups = pd.Series(groups).astype(str).values
    out = {}
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.any():
            out[g] = float(np.quantile(values[mask], 1.0 - alpha))
    return out

q_marg_by_fam = _per_group_quantile(S_marg, family_cal, ALPHA)
q_rob_by_fam  = _per_group_quantile(S_rob,  family_cal, ALPHA)
q_marg_by_nov = _per_group_quantile(S_marg, nov_bins,    ALPHA)
q_rob_by_nov  = _per_group_quantile(S_rob,  nov_bins,    ALPHA)

# Save artifacts
out = (OUTDIR if 'OUTDIR' in globals() else Path("./outputs")) / "reports"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{"scheme":"family","group":k,"q_marg":q_marg_by_fam.get(k,np.nan),"q_rob":q_rob_by_fam.get(k,np.nan)}
              for k in sorted(set(q_marg_by_fam)|set(q_rob_by_fam))]).to_csv(out/"calibration_mondrian_by_family.csv", index=False)
pd.DataFrame([{"scheme":"novelty","group":k,"q_marg":q_marg_by_nov.get(k,np.nan),"q_rob":q_rob_by_nov.get(k,np.nan)}
              for k in sorted(set(q_marg_by_nov)|set(q_rob_by_nov))]).to_csv(out/"calibration_mondrian_by_novelty.csv", index=False)


# Coverage vs novelty bin at the chosen EPS
if 'idx_test' in globals():
    X_test_full = df.loc[idx_test, elem_cols].to_numpy(float, copy=False)
    y_test_log  = np.asarray(y_log)[df.index.get_indexer_for(idx_test)]
    nov_test    = nn.kneighbors(X_test_full, 1, return_distance=True)[0].ravel() * 100.0

    nov_labels = ["≤0.5","0.5–1.0","1–2",">2 at.%"]
    nov_test_bins = pd.Series(
        pd.cut(nov_test, bins=[-0.01, 0.5, 1.0, 2.0, np.inf], labels=nov_labels),
        index=np.arange(len(nov_test))
    ).astype(str)

    X_test_allowed = X_test_full[:, allowed_idx]
    qmin_log_test = np.array([
        _min_jittered_qlog_for_row(X_test_allowed[i], eps=EPS_robust, K=K_robust, rngseed=SEED+200+i)
        for i in range(len(X_test_allowed))
    ], float)

    cover = []
    for lab in nov_labels:
        mask = (nov_test_bins.values == lab)
        if not np.any(mask):
            continue
        qsub = q_rob_by_nov.get(lab, q_marg_all)
        L_mm = np.exp(qmin_log_test[mask] - qsub)
        cov  = float(np.mean(np.exp(y_test_log[mask]) >= L_mm))
        cover.append({"nov_bin": lab, "coverage": cov, "n": int(mask.sum())})

    df_cov = pd.DataFrame(cover)
    df_cov.to_csv(out/"coverage_by_novelty_test.csv", index=False)

    plt.figure(figsize=(4.6,3.2))
    plt.bar(df_cov["nov_bin"], df_cov["coverage"], width=0.7)
    plt.axhline(1-ALPHA, ls="--", lw=1.0, color="k")
    plt.ylim(0,1); plt.ylabel("Empirical coverage"); plt.xlabel("Novelty bin (at.% L1)")
    plt.title("TEST coverage by novelty (Mondrian robust)")
    plt.tight_layout()
    plt.savefig(out/"fig_coverage_by_novelty_test.png", dpi=300, bbox_inches="tight")
    plt.close()


# In[47]:


# Inverse design — tree surrogate + two-stage robustness
_required = ["df","elem_cols","SEED","ROBUST_EPS","ROBUST_SAMPLES","cat_qt_hi","q_robust_hi","make_features_from_compositions"]
for _k in _required:
    if _k not in globals():
        raise RuntimeError(f"Missing required global: `{_k}`")

if "et" not in globals():
    et = None

if 'DSTAR_PRED_MM' not in globals():
    DSTAR_PRED_MM = 5.0

# dataset compositions matrix
if '_elem_mat_all' not in globals():
    _elem_mat_all = df.loc[:, elem_cols].to_numpy(dtype=float, copy=False)
if 'idx_train' in globals():
    _elem_mat_train = df.loc[idx_train, elem_cols].to_numpy(dtype=float, copy=False)
else:
    _elem_mat_train = _elem_mat_all

# Stage-1 vs Stage-2 robustness settings
STAGE1_ROBUST_SAMPLES = int(ROBUST_SAMPLES)
STAGE1_MC_REPS        = 1
STAGE2_ROBUST_SAMPLES = int(ROBUST_SAMPLES)
STAGE2_MC_REPS        = 5
STAGE2_RESCORE_TOP    = 400

# Choose your allowed element list (order matters)
allowed_elems = ["Be", "Cu", "Fe", "Ti", "Zr", "Ni", "Al", "Co", "La", "Ag"]

allowed_elems_present = [e for e in allowed_elems if e in elem_cols]
missing = [e for e in allowed_elems if e not in elem_cols]
if missing:
    print("[BO] Warning: not in dataset and ignored:", missing)
assert len(allowed_elems_present) >= 2, "Need ≥2 allowed elements found in dataset."

allowed_idx = [elem_cols.index(e) for e in allowed_elems_present]
d_allowed = len(allowed_idx)

# helpers (simplex ops & jitter)
def project_simplex_fraction(x):
    x = np.asarray(x, float)
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)

def embed_allowed_to_full(X_allowed):
    X_allowed = np.atleast_2d(X_allowed).astype(float)
    K = X_allowed.shape[0]
    X_full = np.zeros((K, len(elem_cols)), dtype=float)
    X_full[:, allowed_idx] = X_allowed
    row_sums = X_full.sum(axis=1, keepdims=True)
    return np.divide(X_full, np.where(row_sums > 0, row_sums, 1.0))

assert "sample_drift_neighborhood" in globals(), \
    "Run the canonical drift-sampler cell (In[4]) first."

# Certified objective with caching + CRN
def _hash_x(x, nd=6):
    return hashlib.sha1(np.round(np.asarray(x, float), nd).tobytes()).hexdigest()

class CertifiedObjective:
    def __init__(self, model, q_cal_robust, eps=ROBUST_EPS, K=ROBUST_SAMPLES,
                 crn_seed=SEED+999, mc_reps=1, allowed_idx=None):
        self.model = model
        self.qc = float(q_cal_robust)
        self.eps = float(eps)
        self.K = int(K)
        self.mc_reps = int(mc_reps)
        self.allowed_idx = allowed_idx
        self.crn = np.random.default_rng(crn_seed)
        self.crn_dirs = [int(SEED + DRIFT_SEED + r) for r in range(max(1, self.mc_reps))]
        self._cache = {}

    def _eval_once(self, x_allowed, seed):
        rng_local = np.random.default_rng(seed)
        Xj_allowed = jitter_allowed_simplex(x_allowed, eps=self.eps, K=self.K, rng=rng_local)
        X_full = embed_allowed_to_full(Xj_allowed)
        feats = make_features_from_compositions(X_full)
        qj = np.asarray(self.model.predict(feats), float)  # log-quantiles
        return float(np.exp(np.min(qj) - self.qc))

    def __call__(self, x_allowed):
        key = _hash_x(x_allowed)
        if key in self._cache:
            return self._cache[key]
        seeds = list(self.crn_dirs[:max(1, self.mc_reps)])
        vals = [self._eval_once(x_allowed, s) for s in seeds]
        out = {
            "L_robust_mm": float(np.mean(vals)),
            "L_robust_mm_se": float(np.std(vals, ddof=1)/np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "n_mc": int(len(vals)),
        }
        self._cache[key] = out
        return out

# ε-sensitivity & jitter visualization helpers
def robust_sweep_for_row(row, eps_list=(0.000, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200),
                         model=cat_qt_hi, q_cal=q_robust_hi,
                         k_stage2=STAGE2_ROBUST_SAMPLES, mc_reps=STAGE2_MC_REPS,
                         rng_seed=SEED+7):
    """
    For one candidate row (with frac_* columns), certify L_robust_mm across an ε grid.
    Returns a small DataFrame (len=|eps_list|).
    """
    x_allowed = row[[f"frac_{e}" for e in allowed_elems_present]].to_numpy(float)
    out = []
    for eps in eps_list:
        obj_tmp = CertifiedObjective(model=model, q_cal_robust=q_cal,
                                     eps=float(eps), K=int(k_stage2),
                                     crn_seed=rng_seed, mc_reps=int(mc_reps),
                                     allowed_idx=allowed_idx)
        v = obj_tmp(x_allowed)
        out.append({"eps": float(eps), **v})
    return pd.DataFrame(out)

def plot_jitter_cloud(x_allowed, eps=ROBUST_EPS, K=1000, model=cat_qt_hi,
                      q_cal=q_robust_hi, fname="fig_jitter_cloud.png"):
    """
    Visualize the distribution of high-τ predictions over jittered compositions
    and overlay the certified lower bound L_robust for that ε.
    """
    rng = np.random.default_rng(SEED+303)
    Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
    Xj_full = embed_allowed_to_full(Xj_allowed)
    feats = make_features_from_compositions(Xj_full)
    qj_log = np.asarray(model.predict(feats), float)
    qj_mm  = np.exp(qj_log)
    L_mm   = float(np.exp(np.min(qj_log) - q_cal))

    plt.figure(figsize=(6.0, 3.6))
    plt.scatter(np.arange(len(qj_mm)), qj_mm, s=6, alpha=0.35)
    plt.axhline(L_mm, ls="--", lw=2, label=f"L_robust@ε={eps:.3f} → {L_mm:.2f} mm")
    plt.xlabel("jitter index"); plt.ylabel("qτ prediction (mm)")
    plt.title("Jitter cloud under L1 composition drift")
    plt.legend(frameon=False); plt.tight_layout()
    fp = OUTDIR / "reports" / fname
    fp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
    return fp

# parameterization: R^d → simplex via softmax
def _softmax_to_simplex(u):
    u = np.asarray(u, float)
    v = np.exp(u - u.max())
    s = v.sum()
    return (v / s) if s > 0 else np.ones_like(v)/len(v)

NOVELTY_L1_ATPCT = 10.0

def propose_candidates_BO(
    obj: CertifiedObjective,
    n_calls=1000, n_random_starts=40, xi=0.02, random_state=SEED,
    diversity_batch=30, diversity_min_L1_atpct=2.0,
    novelty_min_L1_atpct=NOVELTY_L1_ATPCT,
    novelty_reference="train",
    hard_caps=None, max_elements=None,
    stage2_rescore_top=STAGE2_RESCORE_TOP,
    stage2_K=STAGE2_ROBUST_SAMPLES,
    stage2_mc_reps=STAGE2_MC_REPS,
    pred_gate_mm=30.00,
    pred_gate_quantile=None,
    backend="forest",
):
    d = len(allowed_idx)
    space = [Real(-6.0, 6.0) for _ in range(d)]
    
    def _apply_optional_constraints(x_allowed):
        x_allowed = x_allowed.copy()
        if hard_caps:
            caps_vec  = np.array([hard_caps.get(e, None) for e in allowed_elems_present], dtype=object)
            caps_frac = np.array([np.inf if c is None else float(c)/100.0 for c in caps_vec])
            x_allowed = np.minimum(x_allowed, np.where(np.isfinite(caps_frac), caps_frac, 1.0))
            s = x_allowed.sum()
            if s > 0:
                x_allowed /= s
        if (max_elements is not None) and (max_elements < d):
            order = np.argsort(x_allowed)
            kill = order[:max(0, d - max_elements)]
            x_allowed[kill] = 0.0
            s = x_allowed.sum()
            if s > 0:
                x_allowed /= s
        return x_allowed
    
    # CALPHAD screens
    CALPHAD_SCREEN_CSV = OUTDIR / "source_data" / "calphad_screens.csv"
    
    _ATOMIC_RADIUS = { "Zr":160, "Ti":147, "Cu":128, "Ni":124, "Pd":137, "Pt":139, "Al":143,
                       "La":187, "Fe":126, "Co":125, "Cr":128, "Mn":127, "Y":180, "Nb":146,
                       "Ta":146, "Si":111, "B":87, "P":98 }  # pm
    _VALENCE_E = { "Zr":4, "Ti":4, "Cu":1, "Ni":10, "Pd":10, "Pt":10, "Al":3, "La":3, "Fe":8,
                   "Co":9, "Cr":6, "Mn":7, "Y":3, "Nb":5, "Ta":5, "Si":4, "B":3, "P":5 }
    
    def _signature_from_x(x, elems):
        frac = (np.asarray(x) * 200.0).round() / 2.0
        pairs = sorted([(e, float(fr)) for e, fr in zip(elems, frac) if fr > 0.0],
                       key=lambda t: (-t[1], t[0]))
        return ";".join([f"{e}:{fr:.1f}" for e, fr in pairs])
    
    def _vec_and_delta(x, elems):
        x = np.asarray(x, float)
        x = x/np.clip(x.sum(), 1e-12, None)
        radii = np.array([_ATOMIC_RADIUS.get(e, np.nan) for e in elems], float)
        vec   = np.array([_VALENCE_E.get(e, np.nan)      for e in elems], float)
        ok = np.isfinite(radii) & np.isfinite(vec) & (x > 0)
        if not ok.any():
            return np.nan, np.nan
        x = x[ok]; radii = radii[ok]; vec = vec[ok]
        r_bar = (x * radii).sum()
        delta = np.sqrt((x * (1 - (radii/r_bar))**2).sum()) * 100.0
        VEC = float((x * vec).sum())
        return VEC, float(delta)
    
    _CAL_SCREENS = None
    if CALPHAD_SCREEN_CSV.exists():
        _CAL_SCREENS = pd.read_csv(CALPHAD_SCREEN_CSV)
        _CAL_SCREENS["signature"] = _CAL_SCREENS["signature"].astype(str)
    
    def is_feasible_composition(x_allowed: np.ndarray):
        if _CAL_SCREENS is not None:
            sig = _signature_from_x(x_allowed, allowed_elems_present)
            row = _CAL_SCREENS.loc[_CAL_SCREENS["signature"] == sig]
            if len(row) == 1:
                return bool(int(row["ok"].values[0]))
        VEC, delta = _vec_and_delta(x_allowed, allowed_elems_present)

        if np.isfinite(delta) and (delta < 6.0):
            return False
        if np.isfinite(VEC) and not (7.0 <= VEC <= 8.5):
            return False
        return True

    def f(u):
        x = _softmax_to_simplex(u)
        x = _apply_optional_constraints(x)
    
        # physics/feasibility screen
        feas = is_feasible_composition(x)
        if (feas is False) or (isinstance(feas, (int, float)) and feas < 0):
            return +1e9
    
        L = obj(x)["L_robust_mm"] 
        X_full = embed_allowed_to_full(x[None, :])
        feats  = make_features_from_compositions(X_full)
        if ("cat_qt_grid" in globals()) and (0.98 in cat_qt_grid):
            pred_max_mm = float(np.exp(cat_qt_grid[0.98].predict(feats))[0])
        else:
            pred_max_mm = float(np.exp(cat_qt_hi.predict(feats))[0])
    
        B = 18.0
        lam = 0.25
        bonus = max(0.0, pred_max_mm - B)
        return -(L + lam * bonus)

    # Stage 0: BO (ET surrogate)
    if backend == "forest":
        res = forest_minimize(
            f, space,
            n_calls=n_calls, n_random_starts=n_random_starts,
            acq_func="EI", xi=xi, base_estimator="ET",
            random_state=random_state
        )
    elif backend == "gp":
        res = gp_minimize(
            f, space,
            n_calls=n_calls, n_random_starts=n_random_starts,
            acq_func="EI", xi=xi, random_state=random_state,
            n_restarts_optimizer=5, noise=1e-6
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    globals()["bo_res"] = res

    _trace = pd.DataFrame({
        "iter": np.arange(len(res.func_vals)),
        "best_so_far": np.maximum.accumulate(-res.func_vals),
        "acq": "EI",
        "xi": float(xi),
        "n_calls": int(n_calls),
        "n_random_starts": int(n_random_starts),
        "backend": str(backend),
    })
    (OUTDIR / "source_data").mkdir(parents=True, exist_ok=True)
    _trace.to_csv(OUTDIR / "source_data" / f"bo_trace_{backend}.csv", index=False)

    # Stage 1: evaluate all tried points (cheap robustness)
    eval_X_allowed = np.array([_apply_optional_constraints(_softmax_to_simplex(u)) for u in res.x_iters])
    vals = [obj(x) for x in eval_X_allowed]
    Ls  = np.array([v["L_robust_mm"]    for v in vals])
    SEs = np.array([v["L_robust_mm_se"] for v in vals])

    X_full = embed_allowed_to_full(eval_X_allowed)
    feats0 = make_features_from_compositions(X_full)

    # model predictions (mm)
    pred_point_mm = (np.exp(et.predict(feats0)) if et is not None else np.full(len(eval_X_allowed), np.nan))
    pred_qtau_mm  = np.exp(cat_qt_hi.predict(feats0))  # high-τ log → mm

    # "max possible" optimistic predictor (τ=0.99)
    if ("cat_qt_grid" in globals()) and isinstance(cat_qt_grid, dict) and (0.99 in cat_qt_grid):
        q99_log     = np.asarray(cat_qt_grid[0.99].predict(feats0), float)
        pred_max_mm = np.exp(q99_log)
    else:
        pred_max_mm = pred_qtau_mm

    # novelty vs requested reference (at.% L1 distance)
    ref_mat = {
        "train": _elem_mat_train,
        "all":   _elem_mat_all,
        None:    None
    }.get(novelty_reference, _elem_mat_train)

    if ref_mat is not None and getattr(ref_mat, "size", 0):
        nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(ref_mat)
        min_L1_atpct = nn.kneighbors(X_full, n_neighbors=1, return_distance=True)[0].ravel() * 50.0
    else:
        min_L1_atpct = np.full(len(eval_X_allowed), np.inf)

    # Assemble pool_all = ALL tried points
    pool_all = pd.DataFrame(eval_X_allowed, columns=[f"frac_{e}" for e in allowed_elems_present])
    for j, e in enumerate(allowed_elems_present):
        pool_all[f"atpct_{e}"] = 100.0 * pool_all[f"frac_{e}"]
    pool_all["L_robust_mm"]    = Ls
    pool_all["L_robust_mm_se"] = SEs
    pool_all["pred_point_mm"]  = pred_point_mm
    pool_all["pred_qtau_mm"]   = pred_qtau_mm
    pool_all["pred_max_mm"]    = pred_max_mm
    pool_all["min_L1_to_ref_atpct"] = min_L1_atpct
    pool_all["novelty_ref"] = novelty_reference

    designed_dir = OUTDIR / "data" / "designed"
    designed_dir.mkdir(parents=True, exist_ok=True)
    pool_all.to_csv(designed_dir / f"bo_tried_all_{backend}.csv", index=False)

    # Exceedance snapshot BEFORE filtering
    try:
        BENCHMARK_MM = 20.0
    except Exception:
        try:
            src = pd.read_csv(OUTDIR / "source_data" / "label_values.csv")
            BENCHMARK_MM = float(pd.to_numeric(src["value_mm"], errors="coerce").max())
        except Exception:
            BENCHMARK_MM = 20.0
    
    # mark predicted exceeders using the optimistic predictor already in pool_all
    pool_all["beats_benchmark_pred"] = (pool_all["pred_max_mm"] >= float(BENCHMARK_MM)).astype(int)
   
    (designed_dir).mkdir(parents=True, exist_ok=True)
    pool_all.assign(BENCHMARK_MM=float(BENCHMARK_MM)).to_csv(
        designed_dir / f"bo_pool_pre_filter_with_exceedance_{backend}.csv", index=False
    )

    n0 = len(pool_all)
    if pred_gate_quantile is not None:
        if pred_gate_mm is not None:
            raise ValueError("Set pred_gate_mm OR pred_gate_quantile, not both.")
        pred_gate_mm = float(np.quantile(pred_qtau_mm, pred_gate_quantile))
        print(f"[BO] Adaptive qτ gate at {pred_gate_quantile:.2f}-quantile: {pred_gate_mm:.2f} mm")
    elif pred_gate_mm is not None:
        print(f"[BO] Fixed qτ gate: {pred_gate_mm:.2f} mm (manuscript screening step)")
    mask_pred = np.ones(n0, bool) if pred_gate_mm is None else (pool_all["pred_qtau_mm"].values >= float(pred_gate_mm))
    mask_nov  = np.ones(n0, bool) if novelty_min_L1_atpct is None else (pool_all["min_L1_to_ref_atpct"].values >= float(novelty_min_L1_atpct))

    print(f"[BO prefilter] n0={n0}, pass_pred={int(mask_pred.sum())}, pass_novelty={int(mask_nov.sum())}, "
          f"pass_both={int((mask_pred & mask_nov).sum())}")
    print(f"  qτ(mm) stats:   min={np.nanmin(pred_qtau_mm):.2f}, p50={np.nanmedian(pred_qtau_mm):.2f}, p90={np.nanpercentile(pred_qtau_mm,90):.2f}")    
    print(f"  L_robust(mm):   max={np.nanmax(Ls):.2f}, p90={np.nanpercentile(Ls,90):.2f}, p50={np.nanmedian(Ls):.2f}")

    mask_both = mask_pred & mask_nov
    pool_prefilter = pool_all.loc[mask_both].copy().sort_values("L_robust_mm", ascending=False).reset_index(drop=True)
    pool_prefilter.to_csv(designed_dir / f"bo_prefiltered_{backend}.csv", index=False)

    if len(pool_prefilter) == 0:
        print(f"[BO prefilter] 0 survivors → relaxing qτ gate once (pred_gate_mm={pred_gate_mm}).")
        pool_prefilter = pool_all.loc[mask_nov].copy().sort_values("L_robust_mm", ascending=False).reset_index(drop=True)
        if len(pool_prefilter) == 0:
            return pd.DataFrame({"note": ["No BO candidates passed; consider lowering novelty_min_L1_atpct or increasing n_calls."]})

    # Stage 2: high-robustness rescoring
    obj_hi = CertifiedObjective(
        model=obj.model, q_cal_robust=obj.qc,
        eps=obj.eps, K=int(stage2_K),
        crn_seed=SEED+7777, mc_reps=int(stage2_mc_reps),
        allowed_idx=obj.allowed_idx
    )

    # (A) Certificates for ALL predicted exceeders (pre-thinning)
    #BENCHMARK_MM = float(np.nanmax(np.exp(y_log))) if 'y_log' in globals() else 5.0
    BENCHMARK_MM = 20.0

    exceed_prefilter = pool_prefilter.loc[pool_prefilter["pred_max_mm"] >= BENCHMARK_MM].copy()
    if len(exceed_prefilter):
        X_allowed_ex = exceed_prefilter[[f"frac_{e}" for e in allowed_elems_present]].to_numpy()
        vals_ex = [obj_hi(x) for x in X_allowed_ex]  # robust certification
        exceed_prefilter["L_robust_mm"]    = [v["L_robust_mm"]    for v in vals_ex]
        exceed_prefilter["L_robust_mm_se"] = [v["L_robust_mm_se"] for v in vals_ex]
        exceed_prefilter["L_robust_mm_lo"] = exceed_prefilter["L_robust_mm"] - 1.96*exceed_prefilter["L_robust_mm_se"]
        exceed_prefilter["L_robust_mm_hi"] = exceed_prefilter["L_robust_mm"] + 1.96*exceed_prefilter["L_robust_mm_se"]
        exceed_prefilter.to_csv(designed_dir / f"advanced_candidates_pred_ge_{int(BENCHMARK_MM)}mm_all_prethin_{backend}.csv", index=False)
        print(f"[Exceeders] Benchmark={BENCHMARK_MM:.1f} mm | max(pred_max_mm)={pool_all['pred_max_mm'].max():.2f} mm | n_prefilter_exceed={len(exceed_prefilter)}")
    else:
        print(f"[Exceeders] No prefiltered exceeders ≥ {BENCHMARK_MM:.1f} mm (predicted).")

    # (B) Re-score top-M by certificate (for final ordering)
    M = int(min(stage2_rescore_top, len(pool_prefilter)))
    if M > 0:
        X_allowed_top = pool_prefilter[[f"frac_{e}" for e in allowed_elems_present]].to_numpy()[:M]
        vals_hi = [obj_hi(x) for x in X_allowed_top]
        Ls_hi  = np.array([v["L_robust_mm"]    for v in vals_hi])
        SEs_hi = np.array([v["L_robust_mm_se"] for v in vals_hi])
        pool_prefilter.loc[:M-1, "L_robust_mm"]    = Ls_hi
        pool_prefilter.loc[:M-1, "L_robust_mm_se"] = SEs_hi
        pool_prefilter = pool_prefilter.sort_values("L_robust_mm", ascending=False).reset_index(drop=True)

    # (C) Diversity thinning in at.% L1 (on re-scored ordering)
    X_atpct = pool_prefilter[[f"atpct_{e}" for e in allowed_elems_present]].to_numpy()
    selected = []
    for i in range(len(pool_prefilter)):
        if not selected:
            selected.append(i); continue
        if len(selected) >= diversity_batch:
            break
        dmin = 0.5 * pairwise_distances(X_atpct[i:i+1], X_atpct[selected],
                                        metric='manhattan').min()
        if dmin >= diversity_min_L1_atpct:
            selected.append(i)

    pool_selected = pool_prefilter.iloc[selected].reset_index(drop=True)
    pool_prefilter["backend"] = str(backend)
    pool_selected["backend"]  = str(backend)

    return pool_selected

# Stage-1 objective (cheap robustness for BO)
cert_obj_stage1 = CertifiedObjective(
    model=cat_qt_hi, q_cal_robust=q_robust_hi,
    eps=ROBUST_EPS, K=STAGE1_ROBUST_SAMPLES,
    crn_seed=SEED+4242, mc_reps=STAGE1_MC_REPS, allowed_idx=allowed_idx
)

# Run BO with trees
t0 = time.time()
bo_df = propose_candidates_BO(
    cert_obj_stage1,
    n_calls=1000,
    n_random_starts=max(100, 5*len(allowed_idx)),
    xi=0.02, random_state=SEED,
    diversity_batch=30, diversity_min_L1_atpct=2.0,
    novelty_min_L1_atpct=NOVELTY_L1_ATPCT,
    novelty_reference="train",
    hard_caps=None, max_elements=None,
    stage2_rescore_top=STAGE2_RESCORE_TOP,
    stage2_K=STAGE2_ROBUST_SAMPLES,
    stage2_mc_reps=STAGE2_MC_REPS,
    pred_gate_mm=30.00,
    pred_gate_quantile=None,
    backend="forest"
)
t1 = time.time()

# Run BO with GPR
t2 = time.time()
bo_df_gp = propose_candidates_BO(
    cert_obj_stage1,
    n_calls=1000,
    n_random_starts=max(100, 5*len(allowed_idx)),
    xi=0.02, random_state=SEED,
    diversity_batch=30, diversity_min_L1_atpct=2.0,
    novelty_min_L1_atpct=NOVELTY_L1_ATPCT,
    novelty_reference="train",
    hard_caps=None, max_elements=None,
    stage2_rescore_top=STAGE2_RESCORE_TOP,
    stage2_K=STAGE2_ROBUST_SAMPLES,
    stage2_mc_reps=STAGE2_MC_REPS,
    pred_gate_mm=30.00,
    pred_gate_quantile=None,
    backend="gp"
)
t3 = time.time()

_out = OUTDIR / "data" / "designed"; _out.mkdir(parents=True, exist_ok=True)
bo_df["backend"] = "forest"; bo_df_gp["backend"] = "gp"
bo_df.to_csv(_out / "advanced_bo_pool_forest.csv", index=False)
bo_df_gp.to_csv(_out / "advanced_bo_pool_gp.csv", index=False)

def _best_L(df): 
    return float(np.nanmax(df["L_robust_mm"])) if len(df) else float("nan")

summary = pd.DataFrame([
    {"backend":"forest","best_L_robust_mm":_best_L(bo_df),   "wallclock_s": round(t1 - t0, 2), "n": len(bo_df)},
    {"backend":"gp",    "best_L_robust_mm":_best_L(bo_df_gp),"wallclock_s": round(t3 - t2, 2), "n": len(bo_df_gp)},
])
summary.to_csv(OUTDIR/"reports"/"bo_backend_summary.csv", index=False)

plt.figure(figsize=(4.2,3.0))
plt.bar(summary["backend"], summary["best_L_robust_mm"])
plt.ylabel("Best certified $L_{robust}$ (mm)")
plt.title("BO backends (same budget)")
plt.tight_layout()
plt.savefig(OUTDIR/"reports"/"fig_bo_backend_bestL.png", dpi=300, bbox_inches="tight")
plt.close()

if {"L_robust_mm", "L_robust_mm_se"}.issubset(bo_df.columns):
    bo_df["L_robust_mm_lo"] = bo_df["L_robust_mm"] - 1.96 * bo_df["L_robust_mm_se"]
    bo_df["L_robust_mm_hi"] = bo_df["L_robust_mm"] + 1.96 * bo_df["L_robust_mm_se"]

# mark novelty vs entire dataset
ROUND_TOL_ATPCT = 0.5

def _sig_from_atpct_row(row, elems, tol=ROUND_TOL_ATPCT):
    parts = []
    for e in elems:
        v = float(row.get(f"atpct_{e}", 0.0))
        vr = round(v / tol) * tol
        parts.append(f"{e}:{vr:.1f}")
    return "|".join(parts)

_known_sigs = set()
for _, r in df.iterrows():
    vals = []
    for e in allowed_elems_present:
        if f"atpct_{e}" in r:
            vals.append(float(r[f"atpct_{e}"]))
        elif f"at_{e}" in r:
            vals.append(float(r[f"at_{e}"]))
        elif e in r:
            vals.append(100.0 * float(r[e]))
        else:
            vals.append(0.0)
    tmp = {f"atpct_{e}": v for e, v in zip(allowed_elems_present, vals)}
    _known_sigs.add(_sig_from_atpct_row(tmp, allowed_elems_present, ROUND_TOL_ATPCT))

bo_df["signature_round"] = bo_df.apply(lambda r: _sig_from_atpct_row(r, allowed_elems_present, ROUND_TOL_ATPCT), axis=1)
bo_df["is_novel_vs_all"] = ~bo_df["signature_round"].isin(_known_sigs)
bo_df_novel = bo_df.loc[bo_df["is_novel_vs_all"]].copy()

# Exceeders
#BENCHMARK_MM = float(np.nanmax(np.exp(y_log)))
BENCHMARK_MM = 20.0

pref_csv = OUTDIR / "data" / "designed" / f"advanced_candidates_pred_ge_{int(BENCHMARK_MM)}mm_all_prethin.csv"
if pref_csv.exists():
    ex_pre = pd.read_csv(pref_csv)

    def _sig_row(row, elems, tol=ROUND_TOL_ATPCT):
        parts = []
        for e in elems:
            v = float(row.get(f"atpct_{e}", 0.0))
            vr = round(v / tol) * tol
            parts.append(f"{e}:{vr:.1f}")
        return "|".join(parts)

    ex_pre["signature_round"] = ex_pre.apply(lambda r: _sig_row(r, allowed_elems_present, ROUND_TOL_ATPCT), axis=1)
    ex_pre["is_novel_vs_all"] = ~ex_pre["signature_round"].isin(_known_sigs)

    outdir = OUTDIR / "data" / "designed"
    ex_pre.sort_values("pred_max_mm", ascending=False).to_csv(
        outdir / f"advanced_candidates_pred_ge_{int(BENCHMARK_MM)}mm_all.csv", index=False
    )
    ex_pre.loc[ex_pre["is_novel_vs_all"]].sort_values("pred_max_mm", ascending=False).to_csv(
        outdir / f"advanced_candidates_pred_ge_{int(BENCHMARK_MM)}mm_novel.csv", index=False
    )
    print(f"[Exceeders/FINAL] n_all={len(ex_pre)}, n_novel={(ex_pre['is_novel_vs_all']).sum()}")
else:
    print(f"[Exceeders/FINAL] No pre-thinning exceeder file found at {pref_csv}")

# ε-sensitivity exports
_pool_for_eps = bo_df

# 1) ε-sensitivity curves for top-30 by certificate
_topM = min(30, len(_pool_for_eps))
rows = []
for i in range(_topM):
    r = _pool_for_eps.iloc[i]
    df_i = robust_sweep_for_row(r, eps_list=(0.000, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200))
    df_i.insert(0, "rank", i+1)
    for e in allowed_elems_present:
        df_i[f"atpct_{e}"] = r.get(f"atpct_{e}", 0.0)
    rows.append(df_i)
df_eps = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
(_rep := OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
df_eps.to_csv(_rep / "eps_sensitivity_top30_forest.csv", index=False)
print("Saved:", _rep / "eps_sensitivity_top30.csv")

# 2) Jitter cloud plot for the top-1 candidate at your ROBUST_EPS
if len(_pool_for_eps) > 0:
    x_allowed_top = _pool_for_eps[[f"frac_{e}" for e in allowed_elems_present]].to_numpy(float)[0]
    fig_path = plot_jitter_cloud(x_allowed_top, eps=ROBUST_EPS, K=1500,
                                 fname="fig_jitter_cloud_top1.png")
    print("Saved:", fig_path)

sens_csv = OUTDIR / "reports" / "eps_sensitivity_top30.csv"
if sens_csv.exists():
    s = pd.read_csv(sens_csv)
    g = (s.groupby("eps")["L_robust_mm"]
           .agg(["mean","median","min","max"])
           .reset_index())
    g.to_csv(OUTDIR/"reports"/"eps_sensitivity_summary.csv", index=False)
    plt.figure(figsize=(4.5,3.2))
    plt.plot(g["eps"], g["mean"], marker="o", label="mean")
    plt.plot(g["eps"], g["median"], marker="o", label="median")
    plt.fill_between(g["eps"], g["min"], g["max"], alpha=0.25, label="min–max")
    plt.xlabel("ε (L1 fraction)"); plt.ylabel("$L_{robust}$ (mm)")
    plt.title("Certificate vs ε (top-30)")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(OUTDIR/"reports"/"fig_eps_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()

def _dump_jitter_cloud_csv(x_allowed, eps, K, model, q_cal, path_csv):
    rng = np.random.default_rng(SEED+303)
    Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
    Xj_full = embed_allowed_to_full(Xj_allowed)
    feats = make_features_from_compositions(Xj_full)
    qj_log = np.asarray(model.predict(feats), float)
    qj_mm  = np.exp(qj_log)
    L_mm   = float(np.exp(np.min(qj_log) - q_cal))
    df = pd.DataFrame({"idx":np.arange(K), "q_log":qj_log, "q_mm":qj_mm})
    for j, e in enumerate(allowed_elems_present):
        df[f"frac_{e}"] = Xj_allowed[:, j]
        df[f"atpct_{e}"] = 100.0 * Xj_full[:, elem_cols.index(e)]
    df["eps"] = float(eps); df["L_robust_mm"] = L_mm
    df.to_csv(path_csv, index=False)

_dump_jitter_cloud_csv(x_allowed_top, eps=ROBUST_EPS, K=1500, model=cat_qt_hi,
                       q_cal=q_robust_hi, path_csv=OUTDIR/"reports"/"jitter_cloud_top1.csv")


# Final selected exceeders with certified CIs (ALL + NOVEL)
# BENCHMARK_MM = float(np.nanmax(np.exp(y_log)))
BENCHMARK_MM = 20.0

ex_sel = bo_df.loc[bo_df.get("pred_qtau_mm", np.nan) >= BENCHMARK_MM].copy()

if "L_robust_mm_lo" not in ex_sel.columns and "L_robust_mm_se" in ex_sel.columns:
    ex_sel["L_robust_mm_lo"] = ex_sel["L_robust_mm"] - 1.96*ex_sel["L_robust_mm_se"]
    ex_sel["L_robust_mm_hi"] = ex_sel["L_robust_mm"] + 1.96*ex_sel["L_robust_mm_se"]

_out = OUTDIR / "data" / "designed"; _out.mkdir(parents=True, exist_ok=True)
ex_sel.sort_values("pred_qtau_mm", ascending=False).to_csv(_out / "selected_exceeders_all.csv", index=False)
if "is_novel_vs_all" in ex_sel.columns:
    ex_sel.loc[ex_sel["is_novel_vs_all"]].sort_values("pred_qtau_mm", ascending=False)\
        .to_csv(_out / "selected_exceeders_novel.csv", index=False)

# multi-threshold evaluation (ALL vs NOVEL)
THRESHOLDS_EVAL = [2.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
USE_CONSERVATIVE_CI = True
BASE_COL = "L_robust_mm_lo" if (USE_CONSERVATIVE_CI and "L_robust_mm_lo" in bo_df.columns) else "L_robust_mm"

def eval_and_save(pool: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for Dstar in THRESHOLDS_EVAL:
        flag = f"pass_ge_{int(Dstar)}mm"
        pool[flag] = (pool[BASE_COL] >= float(Dstar)).astype(int)
        passed = pool.loc[pool[flag] == 1].sort_values("L_robust_mm", ascending=False)
        outdir = OUTDIR / "data" / "designed"
        outdir.mkdir(parents=True, exist_ok=True)
        passed.to_csv(outdir / f"advanced_pour_list_{tag}_ge_{int(Dstar)}mm.csv", index=False)
        rows.append({"Dstar_mm": float(Dstar), f"n_pass_{tag}": int(len(passed))})
    df_sum = pd.DataFrame(rows)
    (OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(OUTDIR / "reports" / f"bo_multithreshold_summary_{tag}.csv", index=False)
    return df_sum

summary_all   = eval_and_save(bo_df.copy(),       "all")
summary_novel = eval_and_save(bo_df_novel.copy(), "novel")

# diagnostic plot: #candidates passing vs D* (all vs novel)
plt.figure(figsize=(6, 3.6))
plt.plot(summary_all["Dstar_mm"],   summary_all["n_pass_all"],   marker="o", label="All BO")
plt.plot(summary_novel["Dstar_mm"], summary_novel["n_pass_novel"], marker="o", label="Novel-only")
plt.xlabel("Target D* (mm)")
plt.ylabel("# candidates passing")
plt.title(f"BO pool passing counts (base={BASE_COL})")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTDIR / "reports" / "fig_bo_pass_counts_vs_threshold.png", dpi=300, bbox_inches="tight")
plt.close()

DSTAR = 5.0
bo_df["cert_pass"] = (bo_df["L_robust_mm"] >= DSTAR).astype(int)
pour_list = bo_df[bo_df["cert_pass"] == 1].copy()

outdir = OUTDIR / "data" / "designed"
outdir.mkdir(parents=True, exist_ok=True)
bo_df.to_csv(outdir / "advanced_bo_pool_et.csv", index=False)
bo_df_gp.to_csv(_out / "advanced_bo_pool_gp.csv", index=False)
pour_list.to_csv(outdir / f"advanced_pour_list_all_ge_{int(DSTAR)}mm.csv", index=False)

print(f"[Advanced BO] Proposed {len(bo_df)} total; {int(bo_df['is_novel_vs_all'].sum())} novel vs full dataset.")
print("Saved:")
print(" -", OUTDIR / "reports" / "bo_multithreshold_summary_all.csv")
print(" -", OUTDIR / "reports" / "bo_multithreshold_summary_novel.csv")
print(" -", OUTDIR / "reports" / "fig_bo_pass_counts_vs_threshold.png")

display_cols = [*(f"atpct_{e}" for e in allowed_elems_present),
                "L_robust_mm","L_robust_mm_se","L_robust_mm_lo","L_robust_mm_hi",
                "pred_qtau_mm","pred_point_mm","min_L1_to_ref_atpct","novelty_ref","cert_pass"]

try:
    display(bo_df.head(10)[display_cols])
except Exception:
    print(bo_df.head(10)[display_cols])


# In[48]:


# Exceeders decomposition (qmin_log, q_sub, L_cert_mm) per backend
def _qmin_log_from_row(row, K=STAGE2_ROBUST_SAMPLES, eps=ROBUST_EPS, seed=SEED+9090):
    rng = np.random.default_rng(seed)
    x_allowed = row[[f"frac_{e}" for e in allowed_elems_present]].to_numpy(float)
    Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
    Xj_full    = embed_allowed_to_full(Xj_allowed)
    feats      = make_features_from_compositions(Xj_full)
    qj_log     = np.asarray(cat_qt_hi.predict(feats), float)
    return float(np.min(qj_log))

q_sub = float(q_robust_hi)

designed_dir = OUTDIR / "data" / "designed"
prethin_files = sorted(designed_dir.glob("advanced_candidates_pred_ge_*mm_all_prethin_*.csv"))
for f in prethin_files:
    tag = f.stem.split("_prethin_")[-1]
    ex = pd.read_csv(f)
    if len(ex) == 0:
        continue
    qmins = []
    for _, r in ex.iterrows():
        qmins.append(_qmin_log_from_row(r))
    ex["qmin_log"]   = qmins
    ex["q_sub"]      = q_sub
    ex["L_cert_mm"]  = np.exp(ex["qmin_log"] - ex["q_sub"])
    ex.to_csv(designed_dir / f"advanced_candidates_exceeders_with_decomposition_{tag}.csv", index=False)
    print(f"[Exceeders decomp] Wrote:", designed_dir / f"advanced_candidates_exceeders_with_decomposition_{tag}.csv")


# In[49]:


# Prospective validation: join measured casts against predictions & certificate
try:
    reg = OUTDIR.parent / "prospective" / "measured_casts.csv"
    if reg.exists():
        pool = pd.read_csv(OUTDIR/"data"/"designed"/"advanced_bo_pool.csv")
        ex   = None
        try:
            ex = pd.read_csv(OUTDIR/"data"/"designed"/"advanced_candidates_exceeders_with_decomposition.csv")
        except Exception:
            pass

        # ensure signature present
        if "signature_round" not in pool.columns:
            ROUND_TOL_ATPCT = 0.5
            def _sig_row(row, elems, tol=ROUND_TOL_ATPCT):
                parts=[]; 
                for e in elems:
                    v = float(row.get(f"atpct_{e}", 0.0))
                    vr = round(v/tol)*tol
                    parts.append(f"{e}:{vr:.1f}")
                return "|".join(parts)
            pool["signature_round"] = pool.apply(lambda r: _sig_row(r, allowed_elems_present), axis=1)
        dfm = pd.read_csv(reg)

        keep_cols = ["signature_round","L_robust_mm","L_robust_mm_lo","L_robust_mm_hi","pred_max_mm","pred_qtau_mm"]
        for c in keep_cols:
            if c not in pool.columns:
                pool[c] = np.nan
        joined = dfm.merge(pool[["signature_round"]+keep_cols], on="signature_round", how="left")

        joined.to_csv(OUTDIR/"reports"/"prospective_joined.csv", index=False)

        # plot: measured vs optimistic vs certified
        plt.figure(figsize=(5.8,3.6))
        i = np.arange(len(joined))
        w = 0.28
        plt.bar(i- w, joined["pred_max_mm"], width=w, label="Optimistic (q̂ high)")
        plt.bar(i    , joined["L_robust_mm"], width=w, label="Certified lower bound")
        plt.bar(i+ w , joined["measured_Dmax_mm"], width=w, label="Measured")
        if "L_robust_mm_lo" in joined and "L_robust_mm_hi" in joined:
            y = joined["L_robust_mm"].values
            lo = joined["L_robust_mm_lo"].values
            hi = joined["L_robust_mm_hi"].values
            plt.errorbar(i, y, yerr=[y-lo, hi-y], fmt="none", capsize=4, lw=1, ecolor="k")
        plt.xticks(i, [s[:30]+"…" if len(s)>30 else s for s in joined["signature_round"]], rotation=45, ha="right")
        plt.ylabel("Diameter (mm)"); plt.title("Prospective validation")
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(OUTDIR/"reports"/"fig_prospective_validation.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved prospective validation artifacts.")
except Exception as e:
    print("[Prospective join] Skipped:", e)


# In[50]:


# BLOCK 2: Baselines & Ablations
reports_dir  = OUTDIR / "reports"
designed_dir = OUTDIR / "data" / "designed"
reports_dir.mkdir(parents=True, exist_ok=True)
designed_dir.mkdir(parents=True, exist_ok=True)

# 2.1 BO objective comparison: mean-optimized vs certified-optimized
print("[Ablation 2.1] Running BO with mean objective (ET) for comparison...")

def propose_candidates_BO_mean(
    n_calls=1000,
    n_random_starts=None, xi=0.02, random_state=SEED,
    diversity_batch=30, diversity_min_L1_atpct=2.0,
    novelty_min_L1_atpct=NOVELTY_L1_ATPCT, novelty_reference="train",
    stage2_rescore_top=STAGE2_RESCORE_TOP,
    stage2_K=STAGE2_ROBUST_SAMPLES,
    stage2_mc_reps=STAGE2_MC_REPS,
    pred_gate_mm=30.00, pred_gate_quantile=None
):
    """Same structure as propose_candidates_BO, but objective = *mean* ET prediction (mm).
    Final ranking still reported by robust certificate for apples-to-apples comparison."""
    d = len(allowed_idx)
    space = [Real(-6.0, 6.0) for _ in range(d)]
    if n_random_starts is None:
        n_random_starts = max(100, 5*d)

    def _apply_optional_constraints(x_allowed):
        x_allowed = x_allowed.copy()
        return x_allowed  # keep simple here; add caps if you used them upstream

    def _softmax_to_simplex(u):
        u = np.asarray(u, float)
        v = np.exp(u - u.max()); s = v.sum()
        return (v/s) if s>0 else np.ones_like(v)/len(v)

    def f(u):
        x = _softmax_to_simplex(u)
        x = _apply_optional_constraints(x)
        # mean-like objective: ET prediction (mm)
        X_full = embed_allowed_to_full([x])
        feats  = make_features_from_compositions(X_full)
        if et is None:
            return 0.0
        pred_log = float(et.predict(feats)[0])
        return -float(np.exp(pred_log))  # maximize mm

    # Warm-start from top historical alloys (within allowed elements)
    x0, y0 = [], []
    try:
        # Take top-N by observed Dmax (y_log on log-mm), filtered to allowed elements
        N_SEEDS = 25
        # Build per-row allowed-subspace fractions
        X_hist_full = df.loc[:, elem_cols].to_numpy(float)
        X_hist_sub  = X_hist_full[:, allowed_idx]
        # Normalize rows to simplex of allowed subset
        rs = X_hist_sub.sum(axis=1, keepdims=True)
        ok = (rs[:,0] > 1e-9)
        X_hist_sub = np.divide(X_hist_sub[ok], np.where(rs[ok] > 0, rs[ok], 1.0))
        y_hist = np.asarray(y_log)[ok]  # log-mm

        take = np.argsort(-y_hist)[:N_SEEDS]
        for x_allowed in X_hist_sub[take]:
            u = np.log(np.clip(x_allowed, 1e-12, None))
            u = u - np.mean(u)
            x0.append(u.tolist())
            try:
                y0_val = -obj(x_allowed)["L_robust_mm"]
            except Exception:
                y0_val = None
            y0.append(y0_val)
        # prune any None
        if any(v is None for v in y0):
            x0 = [a for a,b in zip(x0,y0) if b is not None]
            y0 = [b for b in y0 if b is not None]
        print(f"[BO] Warm-start with {len(x0)} seeds from historical top Dmax.")
    except Exception as e:
        print("[BO] Warm-start skipped:", e)
        x0, y0 = None, None

    t0 = time.time()
    res = forest_minimize(
        f, space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        acq_func="EI", xi=xi, base_estimator="ET",
        random_state=random_state,
        x0=x0 if x0 else None, y0=y0 if x0 else None
    )
    t1 = time.time()
    # Save trace
    _trace = pd.DataFrame({
        "iter": np.arange(len(res.func_vals)),
        "best_so_far": np.maximum.accumulate(-np.array(res.func_vals)),
        "acq": "EI","xi": float(xi),"n_calls": int(n_calls),
        "n_random_starts": int(n_random_starts),
        "backend": "forest","objective":"mean","wallclock_s": (t1-t0)
    })
    _trace.to_csv(OUTDIR/"source_data"/"bo_trace_mean.csv", index=False)

    # Evaluate tried points with *robust* certificate for fair comparison
    eval_X_allowed = np.array([_apply_optional_constraints(_softmax_to_simplex(u)) for u in res.x_iters])
    X_full         = embed_allowed_to_full(eval_X_allowed)
    feats0         = make_features_from_compositions(X_full)
    pred_point_mm  = (np.exp(et.predict(feats0)) if et is not None else np.full(len(eval_X_allowed), np.nan))
    pred_qtau_mm   = np.exp(cat_qt_hi.predict(feats0))  # optimistic tail (log->mm)

    # novelty vs reference
    ref_mat = {"train": _elem_mat_train, "all": _elem_mat_all, None: None}.get(novelty_reference, _elem_mat_train)
    if ref_mat is not None and getattr(ref_mat, "size", 0):
        nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(ref_mat)
        min_L1_atpct = nn.kneighbors(X_full, 1, return_distance=True)[0].ravel() * 100.0
    else:
        min_L1_atpct = np.full(len(eval_X_allowed), np.inf)

    # Stage-1 robust scores (cheap)
    obj = CertifiedObjective(model=cat_qt_hi, q_cal_robust=q_robust_hi,
                             eps=ROBUST_EPS, K=STAGE1_ROBUST_SAMPLES,
                             crn_seed=SEED+2424, mc_reps=STAGE1_MC_REPS, allowed_idx=allowed_idx)
    vals = [obj(x) for x in eval_X_allowed]
    Ls  = np.array([v["L_robust_mm"]    for v in vals])
    SEs = np.array([v["L_robust_mm_se"] for v in vals])

    pool = pd.DataFrame(eval_X_allowed, columns=[f"frac_{e}" for e in allowed_elems_present])
    for j,e in enumerate(allowed_elems_present):
        pool[f"atpct_{e}"] = 100.0*pool[f"frac_{e}"]
    pool["L_robust_mm"]    = Ls;  pool["L_robust_mm_se"] = SEs
    pool["pred_point_mm"]  = pred_point_mm
    pool["pred_qtau_mm"]   = pred_qtau_mm
    pool["min_L1_to_ref_atpct"] = min_L1_atpct
    pool["novelty_ref"]    = novelty_reference
    pool["backend"]        = "ET"
    pool["objective"]      = "mean"

    # optional prediction gate/novelty gate to be consistent with certified run
    if pred_gate_quantile is not None:
        qval = float(np.quantile(pred_qtau_mm, pred_gate_quantile))
        pred_gate_mm = qval
    mask_pred = np.ones(len(pool), bool) if pred_gate_mm is None else (pool["pred_qtau_mm"].values >= float(pred_gate_mm))
    mask_nov  = np.ones(len(pool), bool) if novelty_min_L1_atpct is None else (pool["min_L1_to_ref_atpct"].values >= float(novelty_min_L1_atpct))
    pool = pool.loc[mask_pred & mask_nov].copy().sort_values("L_robust_mm", ascending=False).reset_index(drop=True)

    # Stage-2 robust rescoring top-M (for stability & SEs)
    M = int(min(stage2_rescore_top, len(pool)))
    if M>0:
        obj_hi = CertifiedObjective(model=cat_qt_hi, q_cal_robust=q_robust_hi,
                                    eps=ROBUST_EPS, K=int(stage2_K),
                                    crn_seed=SEED+7878, mc_reps=int(stage2_mc_reps), allowed_idx=allowed_idx)
        X_top = pool[[f"frac_{e}" for e in allowed_elems_present]].to_numpy()[:M]
        vals_hi = [obj_hi(x) for x in X_top]
        pool.loc[:M-1,"L_robust_mm"]    = [v["L_robust_mm"]    for v in vals_hi]
        pool.loc[:M-1,"L_robust_mm_se"] = [v["L_robust_mm_se"] for v in vals_hi]
        pool = pool.sort_values("L_robust_mm", ascending=False).reset_index(drop=True)

    pool.to_csv(designed_dir/"advanced_bo_pool_mean.csv", index=False)
    print("[Ablation 2.1] Saved mean-objective pool:", designed_dir/"advanced_bo_pool_mean.csv")
    return pool, _trace

bo_df_mean, trace_mean = propose_candidates_BO_mean(
    n_calls=1000, n_random_starts=max(100, 5*len(allowed_idx)),
    xi=0.02, random_state=SEED, diversity_batch=30, diversity_min_L1_atpct=2.0,
    novelty_min_L1_atpct=NOVELTY_L1_ATPCT, novelty_reference="train",
    stage2_rescore_top=STAGE2_RESCORE_TOP,
    stage2_K=STAGE2_ROBUST_SAMPLES, stage2_mc_reps=STAGE2_MC_REPS,
    pred_gate_mm=30.0, pred_gate_quantile=None
)

# Overlay best-so-far curves if you also saved the certified trace earlier as bo_trace.csv
try:
    t_cert = pd.read_csv(OUTDIR/"source_data"/"bo_trace.csv")
    t_mean = pd.read_csv(OUTDIR/"source_data"/"bo_trace_mean.csv")
    plt.figure(figsize=(4.2,3.1))
    plt.plot(t_cert["iter"], t_cert["best_so_far"], label="certified objective")
    plt.plot(t_mean["iter"], t_mean["best_so_far"], label="mean objective")
    plt.xlabel("BO iteration"); plt.ylabel("Best-so-far (mm-based score)")
    plt.title("Search dynamics"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(reports_dir/"fig_bo_dynamics_overlay.png", dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", reports_dir/"fig_bo_dynamics_overlay.png")
except Exception as e:
    print("[Ablation 2.1] Overlay skipped:", e)

# 2.2 Uncalibrated vs calibrated (marginal) vs certified (robust)
print("[Ablation 2.2] Comparing optimistic vs marginal-calibrated vs robust-certified...")

# 2.2a: compute marginal conformal subtraction on CAL (non-robust), if not available
try:
    q_marginal_sub = float(q_marg_mondrian_all)
except Exception:
    X_cal_full = df.loc[idx_cal, elem_cols].to_numpy(float, copy=False)
    y_cal_log  = np.asarray(y_log)[df.index.get_indexer_for(idx_cal)]
    fe_cal     = make_features_from_compositions(X_cal_full)
    qhat_cal   = np.asarray(cat_qt_hi.predict(fe_cal), float)
    S_marg     = np.maximum(0.0, qhat_cal - y_cal_log)
    q_marginal_sub = float(np.quantile(S_marg, 1.0 - ALPHA))

# 2.2b: choose the evaluation pool:
pool_path_pref = designed_dir / "bo_prefiltered.csv"
pool_path_final = designed_dir / "advanced_bo_pool_et.csv"
if pool_path_pref.exists():
    base = pd.read_csv(pool_path_pref)
else:
    base = pd.read_csv(pool_path_final)

if "pred_max_mm" not in base.columns:
    base["pred_max_mm"] = base.get("pred_qtau_mm", np.nan) 

if "qhat_log" not in base.columns:
    # recompute qhat on-the-fly
    X_full = embed_allowed_to_full(base[[f"frac_{e}" for e in allowed_elems_present]].to_numpy(float))
    feats  = make_features_from_compositions(X_full)
    base["qhat_log"] = np.asarray(cat_qt_hi.predict(feats), float)
base["L_marg_mm"] = np.exp(base["qhat_log"] - q_marginal_sub)

# Robust-certified already present as L_robust_mm
THRS = [2,4,5,7,10,15,20]
rows = []
for tag,valcol in [("optimistic","pred_max_mm"), ("marginal","L_marg_mm"), ("robust","L_robust_mm")]:
    best = float(np.nanmax(base[valcol]))
    for D in THRS:
        n_pass = int(np.sum(base[valcol] >= float(D)))
        rows.append({"variant":tag, "best_mm":best, "Dstar":int(D), "n_pass":n_pass})
df_ablate = pd.DataFrame(rows)
df_ablate.to_csv(reports_dir/"ablation_uncal_vs_marginal_vs_robust.csv", index=False)

# Small figure: best value and pass-counts
plt.figure(figsize=(4.8,3.2))
ax = plt.gca()
x0 = np.arange(len(THRS))
for i,(tag,grp) in enumerate(df_ablate.groupby("variant")):
    y = grp.sort_values("Dstar")["n_pass"].to_numpy()
    ax.plot(THRS, y, marker="o", label=tag)
ax.set_xlabel("Decision threshold $D^*$ (mm)"); ax.set_ylabel("# candidates ≥ $D^*$")
plt.savefig(reports_dir/"fig_ablation_passcounts_variants.png", dpi=300, bbox_inches="tight"); plt.close()
print("Saved:", reports_dir/"ablation_uncal_vs_marginal_vs_robust.csv", "and fig_ablation_passcounts_variants.png")

# 2.3 Gate & element-constraint ablations
print("[Ablation 2.3] Gate and element-constraint ablations...")

dfp = pd.read_csv(pool_path_final)

active_thresh = 0.5
at_cols = [f"atpct_{e}" for e in allowed_elems_present if f"atpct_{e}" in dfp.columns]
if len(at_cols) != len(allowed_elems_present):
    # derive from frac_* if atpct_* absent
    for e in allowed_elems_present:
        col = f"atpct_{e}"
        if col not in dfp.columns and f"frac_{e}" in dfp.columns:
            dfp[col] = 100.0*dfp[f"frac_{e}"].astype(float)
    at_cols = [f"atpct_{e}" for e in allowed_elems_present]

dfp["n_active_elems"] = (dfp[at_cols] > active_thresh).sum(axis=1)

def _gate_counts(df, pred_gate_mm=30.00, pred_gate_quantile=None, max_elems=None):
    g = df.copy()
    if pred_gate_quantile is not None:
        qv = float(np.quantile(g["pred_qtau_mm"], pred_gate_quantile))
        pred_gate_mm = qv
    mask_pred = np.ones(len(g), bool) if pred_gate_mm is None else (g["pred_qtau_mm"] >= float(pred_gate_mm))
    mask_elem = np.ones(len(g), bool) if max_elems is None else (g["n_active_elems"] <= int(max_elems))
    gg = g.loc[mask_pred & mask_elem]
    return {
        "n_survive": int(len(gg)),
        "best_L_robust": float(gg["L_robust_mm"].max()) if len(gg) else float("nan"),
        "pred_gate_mm": None if pred_gate_mm is None else float(pred_gate_mm),
        "pred_gate_quantile": None if pred_gate_quantile is None else float(pred_gate_quantile),
        "max_elems": None if max_elems is None else int(max_elems)
    }

configs = [
    {"pred_gate_mm":20.0, "pred_gate_quantile":None, "max_elems":None},
    {"pred_gate_mm":None, "pred_gate_quantile":0.70, "max_elems":None},
    {"pred_gate_mm":None, "pred_gate_quantile":None, "max_elems":None},
    {"pred_gate_mm":20.0, "pred_gate_quantile":None, "max_elems":4},
    {"pred_gate_mm":None, "pred_gate_quantile":0.70, "max_elems":4},
]

rows=[]
for cfg in configs:
    rows.append(_gate_counts(dfp, **cfg))
df_gate = pd.DataFrame(rows)
df_gate.to_csv(reports_dir/"ablation_gates_and_element_caps.csv", index=False)

plt.figure(figsize=(5.2,3.0))
plt.bar(range(len(df_gate)), df_gate["n_survive"])
plt.xticks(range(len(df_gate)), [
    "fixed20", "adaptive70", "off",
    "fixed20\n≤4 elems", "adaptive70\n≤4 elems"
])
plt.ylabel("# survivors in pool"); plt.title("Gate & element-constraint ablation")
plt.tight_layout(); plt.savefig(reports_dir/"fig_ablation_gates_element_caps.png", dpi=300, bbox_inches="tight"); plt.close()
print("Saved:", reports_dir/"ablation_gates_and_element_caps.csv", "and fig_ablation_gates_element_caps.png")


# In[51]:


# Baseline comparison inset: mean-optimized vs certified-optimized
designed_dir = OUTDIR / "data" / "designed"
reports_dir  = OUTDIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

pool_csv = designed_dir / "advanced_bo_pool_et.csv"
pour_csv = designed_dir / f"advanced_pour_list_all_ge_{int(DSTAR) if 'DSTAR' in globals() else 15}mm.csv"

dfp = pd.read_csv(pool_csv)

# Robust metric and "mean-like" predictor
if "L_robust_mm" not in dfp.columns:
    raise RuntimeError("advanced_bo_pool_et.csv missing L_robust_mm")
if "pred_point_mm" not in dfp.columns:
    # Graceful fallback: use high-quantile as the 'optimistic' chooser
    dfp["pred_point_mm"] = dfp.get("pred_qtau_mm", np.nan)

K = 20 
by_mean = (dfp.dropna(subset=["pred_point_mm"])
             .sort_values("pred_point_mm", ascending=False)
             .head(K))
best_cert_from_mean = by_mean["L_robust_mm"].max()

by_cert = dfp.sort_values("L_robust_mm", ascending=False).head(K)
best_cert_from_cert = by_cert["L_robust_mm"].max()

vals = [best_cert_from_mean, best_cert_from_cert]
labels = ["Choose by mean", "Choose by certificate"]

plt.figure(figsize=(3.4, 3.2))
bars = plt.bar(labels, vals)
plt.ylabel("Best certified $L_{robust}$ (mm)")
for b,v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.title(f"Top-{K} selection rule vs certified outcome")
plt.tight_layout()
outp = reports_dir / "fig_baseline_mean_vs_cert.png"
plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close()
print("Saved baseline inset:", outp)


# In[52]:


# Stacked at.% bars for top candidates (horizontal)
designed_dir = OUTDIR / "data" / "designed"
reports_dir  = OUTDIR / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)

headline_dstar = int(DSTAR) if 'DSTAR' in globals() else 15
pl_path = designed_dir / f"advanced_pour_list_all_ge_{headline_dstar}mm.csv"
if not pl_path.exists():
    pool_csv = designed_dir / "advanced_bo_pool_et.csv"
    dfp = pd.read_csv(pool_csv).sort_values("L_robust_mm", ascending=False).head(20)
else:
    dfp = pd.read_csv(pl_path).sort_values("L_robust_mm", ascending=False).head(20)

# Ensure at.% columns exist
elem_cols_atpct = []
for e in allowed_elems_present:
    col = f"atpct_{e}"
    if col not in dfp.columns:
        frac_col = f"frac_{e}"
        if frac_col in dfp.columns:
            dfp[col] = 100.0 * dfp[frac_col].astype(float)
        else:
            dfp[col] = 0.0
    elem_cols_atpct.append(col)

labels = [f"cand_{i+1}" for i in range(len(dfp))]
Y = dfp[elem_cols_atpct].to_numpy(float)  # shape: [N, E]

fig, ax = plt.subplots(figsize=(7.5, 5.2))
left = np.zeros(len(dfp))
for j, e in enumerate(allowed_elems_present):
    ax.barh(range(len(dfp)), Y[:, j], left=left, label=e)
    left += Y[:, j]

ax.set_yticks(range(len(dfp)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Composition (at.%)")
ax.set_title(f"Top designs — stacked composition (at.%) @ D*≥{headline_dstar} mm")
ax.legend(ncols=3, fontsize=8, frameon=False)
plt.tight_layout()
outp = reports_dir / "fig_stacked_atpct_top_designs.png"
plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close()
print("Saved stacked at.% figure:", outp)


# In[53]:


# Validation figure: measured vs optimistic vs certified
exper_dir = OUTDIR / "data" / "experiments"
exper_dir.mkdir(parents=True, exist_ok=True)
exper_csv = exper_dir / "cast_results.csv"

if not exper_csv.exists():
    print(f"[Validation] No experiments file at {exper_csv}; skipping.")
else:
    def _sig_from_atpct_row(row, elems, tol=ROUND_TOL_ATPCT if 'ROUND_TOL_ATPCT' in globals() else 0.5):
        parts = []
        for e in elems:
            v = float(row.get(f"atpct_{e}", 0.0))
            vr = round(v / tol) * tol
            parts.append(f"{e}:{vr:.1f}")
        return "|".join(parts)

    pool = pd.read_csv(OUTDIR / "data" / "designed" / "advanced_bo_pool_et.csv")

    if "pred_max_mm" not in pool.columns:
        # Fall back to high-τ prediction if q0.99 wasn't available
        pool["pred_max_mm"] = pool.get("pred_qtau_mm", np.nan)

    if "signature_round" not in pool.columns:
        pool["signature_round"] = pool.apply(lambda r: _sig_from_atpct_row(r, allowed_elems_present), axis=1)

    ex = pd.read_csv(exper_csv)
    if "signature_round" not in ex.columns:
        have_cols = [c for c in ex.columns if c.startswith("atpct_")]
        if not have_cols:
            raise RuntimeError("cast_results.csv must have `signature_round` or at least some `atpct_*` columns.")
        ex["signature_round"] = ex.apply(lambda r: _sig_from_atpct_row(r, allowed_elems_present), axis=1)

    pool_max = pool.sort_values("L_robust_mm", ascending=False).drop_duplicates("signature_round", keep="first")
    M = (ex.merge(pool_max[["signature_round","L_robust_mm","L_robust_mm_se","pred_max_mm"]],
                  on="signature_round", how="left")
           .rename(columns={"L_robust_mm":"cert_mm", "L_robust_mm_se":"cert_se", "pred_max_mm":"optim_mm"}))

    # Panel (a): grouped bars per alloy
    ids = M.get("alloy_id", M["signature_round"]).astype(str).tolist()
    x = np.arange(len(M))
    w = 0.28

    plt.figure(figsize=(max(6.0, 0.5*len(M)), 3.8))
    plt.bar(x - w, M["optim_mm"], width=w, label="Optimistic")
    plt.bar(x,      M["cert_mm"],  width=w, yerr=1.96*M["cert_se"].fillna(0.0).to_numpy(), capsize=2, label="Certified")
    plt.bar(x + w,  M["measured_dmax_mm"], width=w, label="Measured")
    plt.xticks(x, [f"{i+1}" for i in range(len(M))])
    plt.xlabel("Cast alloy index")
    plt.ylabel("Diameter (mm)")
    plt.title("Measured vs optimistic vs certified")
    plt.legend(frameon=False, ncols=3)
    plt.tight_layout()
    outp_a = OUTDIR / "reports" / "fig_validation_grouped_bars.png"
    plt.savefig(outp_a, dpi=300, bbox_inches="tight"); plt.close()

    # Panel (b): scatter measured vs predictions
    plt.figure(figsize=(5.0, 4.2))
    plt.scatter(M["optim_mm"], M["measured_dmax_mm"], s=30, alpha=0.6, label="Optimistic")
    plt.errorbar(M["cert_mm"], M["measured_dmax_mm"],
                 xerr=1.96*M["cert_se"].fillna(0.0).to_numpy(),
                 fmt="o", ms=4, alpha=0.8, label="Certified")
    lim = [0, max( np.nanmax(M[["optim_mm","cert_mm","measured_dmax_mm"]].to_numpy()), 1.0 )*1.05]
    plt.plot(lim, lim, ls="--", lw=1.0, color="k")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Prediction (mm)")
    plt.ylabel("Measured $D_{max}$ (mm)")
    plt.title("Out-of-sample validation")
    plt.legend(frameon=False)
    plt.tight_layout()
    outp_b = OUTDIR / "reports" / "fig_validation_scatter.png"
    plt.savefig(outp_b, dpi=300, bbox_inches="tight"); plt.close()

    print("Saved validation figures:", outp_a, "and", outp_b)


# In[54]:


# Pareto map: novelty (x) vs certified performance (y)
designed_dir = OUTDIR / "data" / "designed"
reports_dir  = OUTDIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Pick a pool file that exists (forest/et naming)
for _name in ["advanced_bo_pool_forest.csv", "advanced_bo_pool_et.csv", "advanced_bo_pool.csv"]:
    _p = designed_dir / _name
    if _p.exists():
        pool_path = _p
        break
else:
    raise FileNotFoundError("No BO pool file found in 'data/designed/'. Expected advanced_bo_pool_forest.csv or *_et.csv.")

dfp = pd.read_csv(pool_path).copy()

# Required columns
if "min_L1_to_ref_atpct" not in dfp.columns or "L_robust_mm" not in dfp.columns:
    raise ValueError("Pareto plot needs 'min_L1_to_ref_atpct' and 'L_robust_mm' in the BO pool CSV.")

# Pull arrays (with safe fallbacks)
x   = pd.to_numeric(dfp["min_L1_to_ref_atpct"], errors="coerce").to_numpy()
y   = pd.to_numeric(dfp["L_robust_mm"],         errors="coerce").to_numpy()
yse = pd.to_numeric(dfp.get("L_robust_mm_se", pd.Series(0.0, index=dfp.index)), errors="coerce").fillna(0.0).to_numpy()
is_novel = dfp.get("is_novel_vs_all", pd.Series(False, index=dfp.index)).astype(bool).to_numpy()

# Keep only rows with finite x,y
mask_valid = np.isfinite(x) & np.isfinite(y)
x, y, yse, is_novel = x[mask_valid], y[mask_valid], yse[mask_valid], is_novel[mask_valid]

# Compute non-dominated front for maximizing both x and y
order = np.argsort(-x)  # sort by decreasing novelty
front_idx_local = []
best_y = -np.inf
for i in order:
    if y[i] > best_y + 1e-12:
        front_idx_local.append(i)
        best_y = y[i]
front_idx_local = np.array(sorted(front_idx_local, key=lambda i: x[i]))  # increasing x along the front

# Boolean mask for front (same length as filtered arrays)
mask_front = np.zeros_like(x, dtype=bool)
mask_front[front_idx_local] = True

# --- Plot ---
plt.figure(figsize=(6.8, 4.6))
# interior points
plt.scatter(x[~mask_front], y[~mask_front], s=14, alpha=0.30, label="Pool (interior)", zorder=1)
# front points: NOVEL vs known
plt.scatter(x[mask_front & is_novel],   y[mask_front & is_novel],   s=46, edgecolor="k", linewidths=0.6,
            label="Pareto front (NOVEL)", zorder=3)
plt.scatter(x[mask_front & ~is_novel],  y[mask_front & ~is_novel],  s=46, facecolor="none", edgecolor="k",
            linewidths=0.6, label="Pareto front (known)", zorder=3)

# error bars on front (±1.96·SE)
for i in front_idx_local:
    if yse[i] > 0:
        plt.errorbar(x[i], y[i], yerr=1.96*yse[i], fmt="none", ecolor="k", alpha=0.5, lw=0.6, zorder=2)

# connect the front
plt.plot(x[front_idx_local], y[front_idx_local], lw=1.4, alpha=0.7, zorder=2)

plt.xlabel("Novelty to training set (at.% L1)")
plt.ylabel("Certified robust diameter  $L_{robust}$  (mm)")
plt.title("Novel yet robust: Pareto front in novelty–certificate space")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(reports_dir / "fig_pareto_novelty_vs_cert.png", dpi=300, bbox_inches="tight")
plt.close()

# Write the Pareto-front table, using original dataframe rows
orig_idx = np.flatnonzero(mask_valid)[front_idx_local]
dfp.iloc[orig_idx].sort_values(
    ["min_L1_to_ref_atpct","L_robust_mm"], ascending=[False, False]
).to_csv(designed_dir / "pareto_front_novelty_vs_cert.csv", index=False)


# In[55]:


# === One-shot CSV exporter for the Pareto plot (robust: recomputes everything) ===
from pathlib import Path
import numpy as np
import pandas as pd

def export_pareto_csvs(dfp: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Coerce required columns
    x = pd.to_numeric(dfp["min_L1_to_ref_atpct"], errors="coerce").to_numpy()
    y = pd.to_numeric(dfp["L_robust_mm"],         errors="coerce").to_numpy()
    yse = pd.to_numeric(dfp.get("L_robust_mm_se", 0.0), errors="coerce").fillna(0.0).to_numpy()

    # Novelty flag optional
    if "is_novel_vs_all" in dfp.columns:
        is_novel = dfp["is_novel_vs_all"].astype(bool).to_numpy()
    else:
        is_novel = np.zeros_like(x, dtype=bool)

    # Keep finite pairs (aligns all arrays)
    mask_valid = np.isfinite(x) & np.isfinite(y)
    x, y, yse, is_novel = x[mask_valid], y[mask_valid], yse[mask_valid], is_novel[mask_valid]

    # Non-dominated front (maximize both x and y)
    order = np.argsort(-x)  # decreasing novelty
    front_idx_local = []
    best_y = -np.inf
    for i in order:
        if y[i] > best_y + 1e-12:
            front_idx_local.append(i)
            best_y = y[i]
    front_idx_local = np.array(sorted(front_idx_local, key=lambda i: x[i]), dtype=int)

    mask_front = np.zeros_like(x, dtype=bool)
    mask_front[front_idx_local] = True

    # Master (all points) table with series label
    series = np.where(mask_front &  is_novel, "Pareto front (NOVEL)",
              np.where(mask_front & ~is_novel, "Pareto front (known)", "Pool (interior)"))

    all_df = pd.DataFrame({
        "x_novelty_atpct_L1": x,
        "y_L_robust_mm": y,
        "y_se": yse,
        "is_front": mask_front.astype(int),
        "is_novel": is_novel.astype(int),
        "series": series,
    })
    all_df.to_csv(outdir / "pareto_novelty_vs_cert_all_points.csv", index=False)

    # Convenience splits
    pd.DataFrame({
        "x_novelty_atpct_L1": x[~mask_front],
        "y_L_robust_mm": y[~mask_front],
    }).to_csv(outdir / "pareto_pool_interior.csv", index=False)

    pd.DataFrame({
        "x_novelty_atpct_L1": x[mask_front & ~is_novel],
        "y_L_robust_mm": y[mask_front & ~is_novel],
        "y_err_95ci": 1.96 * yse[mask_front & ~is_novel],
    }).to_csv(outdir / "pareto_front_known.csv", index=False)

    pd.DataFrame({
        "x_novelty_atpct_L1": x[mask_front &  is_novel],
        "y_L_robust_mm": y[mask_front &  is_novel],
        "y_err_95ci": 1.96 * yse[mask_front &  is_novel],
    }).to_csv(outdir / "pareto_front_novel.csv", index=False)

    # Ordered line along the front
    pd.DataFrame({
        "x_novelty_atpct_L1": x[front_idx_local],
        "y_L_robust_mm": y[front_idx_local],
    }).to_csv(outdir / "pareto_front_line.csv", index=False)

    print("Wrote CSVs to", outdir)

# Call it (reuses your existing dfp and OUTDIR)
export_pareto_csvs(dfp, OUTDIR / "source_data")


# In[56]:


# Multi-τ quantiles + tail slope
TAUS = [0.80, 0.90, 0.95, 0.99]
MODELS_DIR = OUTDIR / "models"
SRC_DIR    = OUTDIR / "source_data"
FIGDIR     = OUTDIR / "reports" / "figures"
for p in [MODELS_DIR, SRC_DIR, FIGDIR]: p.mkdir(parents=True, exist_ok=True)

def _fit_or_load_catboost_quantile(tau, base_params=None, tag=None):
    model_path = MODELS_DIR / f"cat_qt_tau{tau:.2f}.cbm"
    if model_path.exists():
        m = CatBoostRegressor()
        m.load_model(str(model_path))
        return m

    params = (cat_qt_hi.get_params().copy() if base_params is None else base_params.copy())
    params["loss_function"] = f"Quantile:alpha={tau}"
    params["eval_metric"]   = f"Quantile:alpha={tau}"
    params.pop("verbose", None); params["verbose"] = False
    params["random_seed"] = SEED
    m = CatBoostRegressor(**params)

    if 'w_train' in globals():
        m.fit(X_train, y_train, sample_weight=w_train)
    else:
        m.fit(X_train, y_train)
    m.save_model(str(model_path))
    return m

# Prepare models for each τ
cat_qt_grid = {}
for t in TAUS:
    cat_qt_grid[t] = _fit_or_load_catboost_quantile(t)

def _ensure_candidate_frame():
    if 'bo_df' in globals() and len(bo_df):
        df_cand = bo_df.copy()

        if not any(c.startswith("atpct_") for c in df_cand.columns):
            for e in allowed_elems_present:
                if f"frac_{e}" in df_cand.columns:
                    df_cand[f"atpct_{e}"] = 100.0 * df_cand[f"frac_{e}"]
        return df_cand
    elif 'design_df_all' in globals() and len(design_df_all):
        return design_df_all.copy()
    else:
        raise RuntimeError("No candidate DataFrame (bo_df/design_df_all) found.")
    
cand_df = _ensure_candidate_frame()

if {"L_robust_mm", "L_robust_mm_se"}.issubset(cand_df.columns):
    cand_df["L_robust_mm_lo"] = cand_df["L_robust_mm"] - 1.96*cand_df["L_robust_mm_se"]
    cand_df["L_robust_mm_hi"] = cand_df["L_robust_mm"] + 1.96*cand_df["L_robust_mm_se"]

# Build feature matrix for all candidates
def _cand_allowed_matrix(df_cand):
    rows = []
    for _, r in df_cand.iterrows():
        v = np.zeros(len(allowed_elems_present), float)
        for j, e in enumerate(allowed_elems_present):
            if e in r:                 v[j] = float(r[e])
            elif f"frac_{e}" in r:     v[j] = float(r[f"frac_{e}"])
            elif f"atpct_{e}" in r:    v[j] = float(r[f"atpct_{e}"]) / 100.0
            else:                      v[j] = 0.0
        s = v.sum();  v = v / s if s > 0 else np.ones_like(v)/len(v)
        rows.append(v)
    return np.vstack(rows)

X_allowed_cand = _cand_allowed_matrix(cand_df)
X_full_cand    = embed_allowed_to_full(X_allowed_cand)
feats_cand     = make_features_from_compositions(X_full_cand)

# Predict multi-τ (in mm) and add tail slope Δq = q0.99 - q0.90
for t in TAUS:
    pred_log = np.asarray(cat_qt_grid[t].predict(feats_cand), float)
    cand_df[f"q{int(round(100*t))}_mm"] = np.exp(pred_log)

if 0.99 in cat_qt_grid and 0.90 in cat_qt_grid:
    cand_df["tail_slope_mm"] = cand_df["q99_mm"] - cand_df["q90_mm"]

cand_df.to_csv(SRC_DIR / "candidates_with_multi_tau.csv", index=False)


# In[57]:


# Sensitivity tornado (±1 at.% moves), robust & cached
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

# Safety: required globals
for _k in ["allowed_elems_present", "elem_cols", "cat_qt_hi", "q_robust_hi"]:
    if _k not in globals():
        raise RuntimeError(f"Missing required global: `{_k}`")

# Fallbacks if helpers were not defined
if "embed_allowed_to_full" not in globals():
    def embed_allowed_to_full(X_allowed):
        X_allowed = np.atleast_2d(X_allowed).astype(float)
        X_full = np.zeros((X_allowed.shape[0], len(elem_cols)), float)
        allowed_idx = [elem_cols.index(e) for e in allowed_elems_present]
        X_full[:, allowed_idx] = X_allowed
        rs = X_full.sum(axis=1, keepdims=True)
        return np.divide(X_full, np.where(rs > 0, rs, 1.0))

assert "sample_drift_neighborhood" in globals(), \
    "Run the canonical drift-sampler cell (In[4]) first."

def _cand_allowed_matrix(df_rows):
    """
    Convert rows to allowed-elements FRACTIONS (sum=1).
    Accepts columns: 'frac_E', 'atpct_E', 'at_E' (at.%), or 'E' (fraction).
    """
    mats = []
    for _, row in df_rows.iterrows():
        vec = []
        for e in allowed_elems_present:
            if f"frac_{e}" in row:
                v = float(row[f"frac_{e}"])
            elif f"atpct_{e}" in row:
                v = float(row[f"atpct_{e}"]) / 100.0
            elif f"at_{e}" in row:
                v = float(row[f"at_{e}"]) / 100.0
            elif e in row:
                v = float(row[e])  # assume already fraction
            else:
                v = 0.0
            vec.append(v)
        vec = np.asarray(vec, float)
        s = vec.sum()
        mats.append(vec / s if s > 0 else np.ones_like(vec)/len(vec))
    return np.vstack(mats) if mats else np.zeros((0, len(allowed_elems_present)), float)

# Small cache to speed up repeated evals during tornado
_eval_cache = {}
def _key_from_x(x, eps, K, nd=4):
    return (tuple(np.round(100.0*np.asarray(x,float), nd)), float(eps), int(K))

def eval_robust_L_allowed(x_allowed, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
    assert np.isclose(float(q_robust_hi), Q_ROBUST_HI_FROZEN), \
        "q_robust_hi drifted from its Cell-28 value."
    if rng is None:
        rng = np.random.default_rng(SEED + DRIFT_SEED)
    x_allowed = np.asarray(x_allowed, float)
    s = x_allowed.sum()
    x_allowed = x_allowed / s if s > 0 else np.ones_like(x_allowed)/len(x_allowed)
    key = _key_from_x(x_allowed, eps, K)
    if key in _eval_cache:
        return _eval_cache[key]
    Xj_allowed = sample_drift_neighborhood(x_allowed, eps=eps, K=K, rng=rng)
    Xj_full = embed_allowed_to_full(Xj_allowed)
    feats = make_features_from_compositions(Xj_full)
    qj = np.asarray(cat_qt_hi.predict(feats), float)  # log-quantiles
    L = float(np.exp(np.min(qj) - q_robust_hi))
    _eval_cache[key] = L
    return L

def tornado_data_for_candidate(x_allowed, step_atpct=1.0):
    """
    Directional sensitivity of L_robust to single-pair, mass-conserving
    transfers (manuscript Fig. 9 and Table S7).

    One row PER ORDERED PAIR (donor -> acceptor). The donor and acceptor
    identities are recorded, so Fig. 9 labels and Table S7 recipes are
    reproducible directly from the returned CSV.
    """
    step = float(step_atpct) / 100.0
    x0 = np.asarray(x_allowed, float)
    s = x0.sum()
    x0 = x0 / s if s > 0 else np.ones_like(x0) / len(x0)

    base = eval_robust_L_allowed(x0)
    d = len(x0)
    recs = []

    for i in range(d):                       # donor
        if x0[i] <= 1e-12:
            continue
        delta = min(step, float(x0[i]))      # clipped if donor holds less than step
        for j in range(d):                   # acceptor
            if i == j or x0[j] <= 1e-12:     # supp(x) only
                continue
            x = x0.copy()
            x[i] -= delta
            x[j] += delta
            x = np.clip(x, 0.0, None)
            t = x.sum()
            x = x / (t if t > 0 else 1.0)
            recs.append({
                "donor":       allowed_elems_present[i],
                "acceptor":    allowed_elems_present[j],
                "transfer":    f"{allowed_elems_present[i]}→{allowed_elems_present[j]}",
                "delta_atpct": 100.0 * delta,
                "L_robust_mm": eval_robust_L_allowed(x),
                "base_L_mm":   base,
            })

    df_tor = pd.DataFrame(recs)
    if df_tor.empty:
        return pd.DataFrame(columns=["donor", "acceptor", "transfer", "delta_atpct",
                                     "L_robust_mm", "base_L_mm", "dL_robust_mm"])
    df_tor["dL_robust_mm"] = df_tor["L_robust_mm"] - df_tor["base_L_mm"]
    return df_tor.sort_values("dL_robust_mm").reset_index(drop=True)


def plot_tornado(df_tor, title, save_path, top_n=14):
    """Horizontal bars, one per donor->acceptor transfer, ranked by dL_robust."""
    if df_tor.empty:
        return
    half = max(1, top_n // 2)
    sub = pd.concat([df_tor.head(half), df_tor.tail(half)])
    sub = sub.drop_duplicates("transfer").sort_values("dL_robust_mm")
    y = np.arange(len(sub))
    vals = sub["dL_robust_mm"].to_numpy()

    plt.figure(figsize=(6.6, max(3.0, 0.34 * len(sub))))
    plt.barh(y, vals, color=np.where(vals < 0, "#c0392b", "#2980b9"), alpha=0.85)
    plt.axvline(0, color="k", lw=1)
    plt.yticks(y, sub["transfer"].tolist())
    plt.xlabel(r"$\Delta L_{\mathrm{robust}}$ (mm)")
    plt.title(title)
    path = Path(save_path)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

# Choose candidates: prefer your BO pool, else cand_df, else design_df
if 'bo_df' in globals() and len(bo_df) > 0:
    cand_source = bo_df.copy()
elif 'cand_df' in globals() and len(cand_df) > 0:
    cand_source = cand_df.copy()
elif 'design_df' in globals() and len(design_df) > 0:
    cand_source = design_df.copy()
else:
    raise RuntimeError("No candidate table found (bo_df/cand_df/design_df).")

# Ensure certified bound exists for ranking (compute if needed)
if "L_robust_mm" not in cand_source.columns:
    X_allowed = _cand_allowed_matrix(cand_source)
    Lvals = [eval_robust_L_allowed(x) for x in X_allowed]
    cand_source["L_robust_mm"] = Lvals

# Generate tornado for top-N by L_robust
N_TOP_TORNADO = 20
cand_source = cand_source.sort_values("L_robust_mm", ascending=False).reset_index(drop=True)

for i in range(min(N_TOP_TORNADO, len(cand_source))):
    x_allowed = _cand_allowed_matrix(cand_source.iloc[[i]])[0]
    df_tor = tornado_data_for_candidate(x_allowed, step_atpct=1.0)
    # Save CSV + figure
    csv_path = SRC_DIR / f"tornado_candidate_{i+1}.csv"
    fig_path = FIGDIR / f"tornado_candidate_{i+1}.png"
    df_tor.to_csv(csv_path, index=False)
    title = (f"Directional drift sensitivity (1.0 at.% donor→acceptor transfer) — "
             f"Candidate #{i+1} (L₀={df_tor['base_L_mm'].iloc[0]:.1f} mm)")
    plot_tornado(df_tor, title, fig_path)

print(f"Saved {min(N_TOP_TORNADO, len(cand_source))} tornado CSVs to {SRC_DIR} and figures to {FIGDIR}.")


# In[58]:


# Risk-controlled shortlist & diversity
try:
    display
except NameError:
    def display(x): 
        print(x.head(10) if hasattr(x, "head") else x)

OUT_DESIGN = OUTDIR / "data" / "designed"
OUT_DESIGN.mkdir(parents=True, exist_ok=True)

# Choose the best candidate pool available
if 'bo_df' in globals() and isinstance(bo_df, pd.DataFrame) and len(bo_df) > 0:
    design_df = bo_df.copy()
elif 'design_df_cert_gt' in globals() and isinstance(design_df_cert_gt, pd.DataFrame) and len(design_df_cert_gt) > 0:
    design_df = design_df_cert_gt.copy()
elif 'design_df_pred' in globals() and isinstance(design_df_pred, pd.DataFrame) and len(design_df_pred) > 0:
    design_df = design_df_pred.copy()
else:
    design_df = design_df_all.copy()

# Add 95% CI bands if present
if {"L_robust_mm", "L_robust_mm_se"}.issubset(design_df.columns):
    design_df["L_robust_mm_lo"] = design_df["L_robust_mm"] - 1.96*design_df["L_robust_mm_se"]
    design_df["L_robust_mm_hi"] = design_df["L_robust_mm"] + 1.96*design_df["L_robust_mm_se"]

# Utilities: composition cols, signatures, shortlisting, diversity
def _infer_comp_cols(df, prefer_allowed=None):
    """
    Infer composition columns in df.
    Priority: atpct_* (percent), then frac_* (fractions), then plain element names.
    Restrict to prefer_allowed (list of symbols) when provided and present.
    """
    cols = df.columns.tolist()
    atpct = [c for c in cols if c.startswith("atpct_")]
    frac  = [c for c in cols if c.startswith("frac_")]
    plain = [c for c in cols if c in elem_cols]

    def _restrict(cs):
        if prefer_allowed is None: 
            return cs
        base = []
        for e in prefer_allowed:
            if f"atpct_{e}" in cs: base.append(f"atpct_{e}")
            elif f"frac_{e}" in cs: base.append(f"frac_{e}")
            elif e in cs:          base.append(e)
        return [c for c in base if c in df.columns]

    for candidate in (_restrict(atpct), _restrict(frac), _restrict(plain), atpct, frac, plain):
        if len(candidate) >= 2:
            varied = [c for c in candidate if df[c].astype(float).var() > 0]
            if len(varied) >= 2:
                return varied

    raise ValueError("Could not infer at least two varying composition columns.")

def _coerce_to_atpct(df, comp_cols):
    """
    Return a (K, d) numpy array of compositions in at.% for distance/diversity.
    Handles atpct_*, frac_*, or plain element names (0..1 fractions assumed).
    """
    X = df[comp_cols].to_numpy(float)
    if np.nanmax(X) <= 1.05:
        X = 100.0 * X
    return X

def _composition_signature_row(row, comp_cols, decimals=1):
    """Signature string rounded to `decimals` at.%. Avoids near-duplicate recipes."""
    vals = []
    for c in comp_cols:
        v = float(row[c])
        if c.startswith("frac_") or (c in elem_cols and v <= 1.05):
            v = 100.0 * v
        vals.append(round(v, decimals))
    return ";".join(f"{v:.{decimals}f}" for v in vals)

def deduplicate_nearby(df, prefer_allowed=None, decimals=1):
    """
    Drop near-duplicates by 0.1 at.% grid signature while keeping the
    highest L_robust candidate in each equivalence class.
    """
    comp_cols = _infer_comp_cols(df, prefer_allowed=prefer_allowed)
    key = "L_robust_mm"
    use_key = "L_robust_mm_lo" if "L_robust_mm_lo" in df.columns else key
    df = df.copy()
    df["__signature__"] = df.apply(lambda r: _composition_signature_row(r, comp_cols, decimals=decimals), axis=1)
    df = (df.sort_values(use_key, ascending=False)
            .drop_duplicates(subset="__signature__", keep="first")
            .drop(columns="__signature__"))
    return df

def shortlist(df, Dstar, use_ci_lower=True, min_novelty_atpct=None, min_tail_slope_mm=None):
    """
    Risk-controlled shortlist: certificate ≥ D* (uses CI-lower if available),
    with optional novelty and tail-slope gates.
    """
    df = df.copy()
    key = "L_robust_mm_lo" if (use_ci_lower and "L_robust_mm_lo" in df.columns) else "L_robust_mm"
    m = np.isfinite(df[key]) & (df[key] >= float(Dstar))
    if (min_novelty_atpct is not None) and ("min_L1_to_dataset_atpct" in df.columns):
        m &= (df["min_L1_to_dataset_atpct"].astype(float) >= float(min_novelty_atpct))
    if (min_tail_slope_mm is not None) and ("tail_slope_mm" in df.columns):
        m &= (df["tail_slope_mm"].astype(float) >= float(min_tail_slope_mm))
    return df.loc[m].copy()

def farthest_point_diversity(df_cand, k=15, score_weight=0.10, prefer_allowed=None, metric="l2", seed=None):
    """
    Farthest-point sampling in composition space, with a small bias toward higher certificates.
    Uses at.% geometry; auto-detects columns. `score_weight` blends certificate into the criterion.
    """
    if len(df_cand) <= k:
        return df_cand.copy()

    rng = np.random.default_rng(SEED if seed is None else seed)
    df = df_cand.copy()
    comp_cols = _infer_comp_cols(df, prefer_allowed=prefer_allowed)
    Xp = _coerce_to_atpct(df, comp_cols)
    mu = np.nanmean(Xp, axis=0, keepdims=True)
    sd = np.nanstd(Xp, axis=0, keepdims=True) + 1e-12
    Xn = (Xp - mu) / sd

    skey = "L_robust_mm_lo" if "L_robust_mm_lo" in df.columns else "L_robust_mm"
    s = df[skey].to_numpy(float)
    s = (s - s.mean()) / (s.std() + 1e-12)

    sel = [int(np.nanargmax(s))]
    cand_idx = list(range(len(df)))

    def _dist(a, B):
        if metric == "l1":   return np.sum(np.abs(B - a), axis=1)
        else:                return np.linalg.norm(B - a, axis=1)

    while len(sel) < k and len(sel) < len(df):
        dmins = []
        for i in cand_idx:
            if i in sel:
                dmins.append(-np.inf); continue
            a = Xn[i]; B = Xn[sel]
            dmin = float(np.min(_dist(a, B)))
            dmins.append(dmin + score_weight * s[i])
        j = int(np.nanargmax(dmins))
        if j in sel:
            break
        sel.append(j)

    return df.iloc[sel].copy()

# restrict to allowed elements for diversity geometry
try:
    prefer_allowed = allowed_elems_present
except NameError:
    prefer_allowed = None

# De-duplicate near-identical compositions (0.1 at.% grid)
design_df = deduplicate_nearby(design_df, prefer_allowed=prefer_allowed, decimals=1)

# Run shortlists across thresholds & export
SUMMARY = {"thresholds": [], "counts": {}, "files": []}

for Dstar in THRESHOLDS_MM:
    cand = shortlist(
        design_df, Dstar,
        use_ci_lower=True,
        min_novelty_atpct=2.0,
        min_tail_slope_mm=None
    )

    diverse = farthest_point_diversity(
        cand, k=15, score_weight=0.10,
        prefer_allowed=prefer_allowed, metric="l2", seed=SEED
    )

    print(f"\nD* = {Dstar:.0f} mm → {len(cand)} pass (risk-controlled); showing a diverse top 15:")
    display(diverse.sort_values("L_robust_mm", ascending=False).head(15))

    base = f"shortlist_Dstar_{int(Dstar)}mm"
    f_all = OUT_DESIGN / f"{base}_all.csv"
    f_top = OUT_DESIGN / f"{base}_diverse_top15.csv"
    cand.to_csv(f_all, index=False)
    diverse.to_csv(f_top, index=False)

    SUMMARY["thresholds"].append(float(Dstar))
    SUMMARY["counts"][str(int(Dstar))] = {"pass_all": int(len(cand)), "diverse_top": int(len(diverse))}
    SUMMARY["files"] += [str(f_all), str(f_top)]

# Also shortlist at dataset max (certified beyond known)
dmax_max_obs = float(df[dmax_col].max())
cand_max = shortlist(design_df, dmax_max_obs, use_ci_lower=True, min_novelty_atpct=2.0)
diverse_max = farthest_point_diversity(cand_max, k=15, score_weight=0.10, prefer_allowed=prefer_allowed, metric="l2")

print(f"\nD* = dataset max ({dmax_max_obs:.1f} mm) → {len(cand_max)} pass; diverse top 15:")
display(diverse_max.sort_values("L_robust_mm", ascending=False).head(15))

f_all_max = OUT_DESIGN / "shortlist_Dstar_dataset_max_all.csv"
f_top_max = OUT_DESIGN / "shortlist_Dstar_dataset_max_diverse_top15.csv"
cand_max.to_csv(f_all_max, index=False)
diverse_max.to_csv(f_top_max, index=False)
SUMMARY["counts"]["dataset_max"] = {"pass_all": int(len(cand_max)), "diverse_top": int(len(diverse_max))}
SUMMARY["files"] += [str(f_all_max), str(f_top_max)]

# Provenance summary JSON
summary_path = OUT_DESIGN / "shortlist_summary.json"
with open(summary_path, "w") as f:
    json.dump(SUMMARY, f, indent=2)

print("\nSaved shortlist artifacts to:", OUT_DESIGN)
print("Summary JSON:", summary_path)


# In[59]:


if 'RAD' not in globals():
    raise RuntimeError("RAD (atomic radii dict) is missing. Define RAD before Step 10.")
if 'VEC' not in globals():
    raise RuntimeError("VEC (valence electron count dict) is missing. Define VEC before Step 10.")

# Define delta_size_mismatch if it's not in scope
if 'delta_size_mismatch' not in globals():
    def delta_size_mismatch(row: pd.Series):
        """
        δ = 100 * sqrt( Σ_i w_i * (1 - r_i / r̄)^2 ), with weights w_i = c_i / Σ c_i.
        Expects 'row' as composition FRACTIONS (sum≈1) indexed by element symbols.
        Uses global RAD dict.
        """
        c = row.values.astype(float)
        r = np.array([RAD.get(el, np.nan) for el in row.index], dtype=float)
        mask = (~np.isnan(c)) & (~np.isnan(r))
        c, r = c[mask], r[mask]
        if c.size == 0 or c.sum() <= 0.0:
            return np.nan
        w = c / c.sum()
        rbar = float((w * r).sum())
        return 100.0 * np.sqrt(float((w * ((1.0 - r / rbar) ** 2)).sum()))


# In[60]:


# Certified minimal-edit recourse 
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
for p in (FIGDIR, SRC_DIR): p.mkdir(parents=True, exist_ok=True)

def delta_size_for_comp_full(x_full):
    """δ-size mismatch on FULL vector (expects fractions)."""
    row = pd.Series(x_full, index=elem_cols)
    return delta_size_mismatch(row)

def vec_for_comp_full(x_full):
    """VEC (weighted by fractions) on FULL vector, ignoring elements without VEC."""
    s = float(np.sum(x_full)); 
    if s <= 0: return np.nan
    vec_vals = np.array([VEC.get(el, np.nan) for el in elem_cols], dtype=float)
    mask = np.isfinite(vec_vals) & (np.asarray(x_full, float) > 0)
    if not np.any(mask): return np.nan
    w = np.asarray(x_full, float)[mask]; w /= w.sum()
    return float(np.dot(w, vec_vals[mask]))

# CRN-evaluated robust certificate
def eval_robust_L_allowed_crn(x_allowed, base_seed=SEED+707, eps=ROBUST_EPS, K=ROBUST_SAMPLES):
    """
    Deterministic (CRN) robust certificate: same jitter sequence for every evaluation,
    which drastically reduces noise during search.
    """
    rng_local = np.random.default_rng(int(base_seed))
    return eval_robust_L_allowed(x_allowed, eps=eps, K=K, rng=rng_local)

def eval_robust_L_allowed_mc(x_allowed, n_reps=5, base_seed=SEED+9001, eps=ROBUST_EPS, K=ROBUST_SAMPLES):
    """Mean±SE over independent jitter seeds for the final certificate."""
    seeds = [int(base_seed + 7919*i) for i in range(n_reps)]
    vals = [eval_robust_L_allowed(x_allowed, eps=eps, K=K, rng=np.random.default_rng(s)) for s in seeds]
    mu = float(np.mean(vals)); se = float(np.std(vals, ddof=1)/np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mu, se, np.asarray(vals, float)

def _apply_caps_and_maxels(x_allowed, hard_caps=None, max_elements=None):
    """
    Enforce per-element max at.% (hard_caps={'Ni': 30, ...}) and/or max distinct elements.
    """
    x = np.asarray(x_allowed, float).copy()
    if hard_caps:
        caps = np.array([np.inf if hard_caps.get(e) is None else float(hard_caps[e])/100.0
                         for e in allowed_elems_present], float)
        x = np.minimum(x, caps)
    s = x.sum(); x = x/s if s > 0 else np.ones_like(x)/len(x)
    if (max_elements is not None) and (max_elements < len(x)):
        order = np.argsort(x) 
        kill  = order[:len(x)-max_elements]
        x[kill] = 0.0
        s = x.sum(); x = x/s if s > 0 else np.ones_like(x)/len(x)
    return x

def _extract_allowed_fractions_from_row(row):
    """Robustly read allowed-element fractions from a row (supports atpct_*, frac_*, or plain element columns)."""
    v = np.zeros(len(allowed_elems_present), float)
    for j, e in enumerate(allowed_elems_present):
        if f"atpct_{e}" in row:   val = float(row[f"atpct_{e}"]) / 100.0
        elif f"frac_{e}" in row:  val = float(row[f"frac_{e}"])
        elif e in row:            # could be fraction or at.% if >1
            val = float(row[e]);  val = val/100.0 if val > 1.05 else val
        else:
            val = 0.0
        v[j] = val
    s = v.sum()
    return v/s if s > 0 else np.ones_like(v)/len(v)

def recourse_min_edit_allowed(
    x0_allowed,
    Dstar,
    schedule=(0.03, 0.02, 0.01, 0.005),
    patience=500,
    hard_caps=None,
    max_elements=None,
    crn_seed=SEED+707,
    tol_improve=1e-6,
):
    """
    Find the *smallest* composition edit (L1 in fractions) that achieves L_robust ≥ Dstar,
    using a greedy best-improvement local search with CRN for stability.
    Returns: result dict with before/after comps, certificate, path dataframe.
    """
    x = project_simplex_fraction(np.asarray(x0_allowed, float))
    x = _apply_caps_and_maxels(x, hard_caps, max_elements)
    base_seed = int(crn_seed)

    def _L(xa):
        return eval_robust_L_allowed_crn(xa, base_seed=base_seed)

    L0 = _L(x)
    path = [{
        "step": 0, "move": "init", "i": -1, "j": -1, "delta_atpct": 0.0,
        "cum_edit_atpct": 0.0, "L_robust_mm": L0
    }]

    if L0 >= Dstar:
        mu, se, vals = eval_robust_L_allowed_mc(x, n_reps=7)
        return {
            "x_before": x0_allowed, "x_after": x, "edit_L1_frac": 0.0,
            "L_init_mm": L0, "L_final_mm": mu, "L_final_se_mm": se, "L_mc_vals": vals,
            "path": pd.DataFrame(path)
        }

    total_edit = 0.0
    for step in schedule:
        no_improve_count = 0
        while no_improve_count < patience:
            best_gain = 0.0
            best_pair = None
            best_x    = None
            for i in range(len(x)):
                for j in range(len(x)):
                    if i == j or x[j] < step: 
                        continue
                    x_new = x.copy()
                    x_new[i] += step
                    x_new[j] -= step
                    x_new = np.clip(x_new, 0.0, None)
                    x_new = project_simplex_fraction(x_new)
                    x_new = _apply_caps_and_maxels(x_new, hard_caps, max_elements)
                    L_new = _L(x_new)
                    gain  = L_new - path[-1]["L_robust_mm"]
                    if gain > best_gain + tol_improve:
                        best_gain, best_pair, best_x, best_L = gain, (i, j), x_new, L_new

            if best_pair is None:
                break

            i, j = best_pair
            x = best_x
            total_edit += step * 2.0 * 0.5

            path.append({
                "step": len(path), "move": f"{allowed_elems_present[j]}→{allowed_elems_present[i]}",
                "i": int(i), "j": int(j), "delta_atpct": float(step*100.0),
                "cum_edit_atpct": float(total_edit*100.0),
                "L_robust_mm": float(best_L)
            })
            if best_L >= Dstar:
                mu, se, vals = eval_robust_L_allowed_mc(x, n_reps=7)
                return {
                    "x_before": x0_allowed, "x_after": x, "edit_L1_frac": float(np.sum(np.abs(x - x0_allowed))),
                    "L_init_mm": L0, "L_final_mm": mu, "L_final_se_mm": se, "L_mc_vals": vals,
                    "path": pd.DataFrame(path)
                }
            no_improve_count = 0 
            
    mu, se, vals = eval_robust_L_allowed_mc(x, n_reps=7)
    return {
        "x_before": x0_allowed, "x_after": x, "edit_L1_frac": float(np.sum(np.abs(x - x0_allowed))),
        "L_init_mm": L0, "L_final_mm": mu, "L_final_se_mm": se, "L_mc_vals": vals,
        "path": pd.DataFrame(path), "note": "Target not reached within schedule/patience."
    }

if 'design_df_cert_gt' in globals() and isinstance(design_df_cert_gt, pd.DataFrame) and len(design_df_cert_gt) > 0:
    design_seed_df = design_df_cert_gt
elif 'bo_df' in globals() and isinstance(bo_df, pd.DataFrame) and len(bo_df) > 0:
    design_seed_df = bo_df
else:
    design_seed_df = design_df_all

seed_row = design_seed_df.sort_values("L_robust_mm", ascending=False).iloc[0]
x0_allowed = _extract_allowed_fractions_from_row(seed_row)

x0_full   = embed_allowed_to_full(x0_allowed.reshape(1, -1))[0]
delta0    = delta_size_for_comp_full(x0_full)
vec0      = vec_for_comp_full(x0_full)

dmax_max_obs = float(df[dmax_col].max())
targets = list(THRESHOLDS_MM) + [dmax_max_obs]

recourse_summaries = []
for Dstar in targets:
    res = recourse_min_edit_allowed(
        x0_allowed, Dstar,
        schedule=(0.03, 0.02, 0.01, 0.005), patience=500,
        hard_caps=None, max_elements=None, crn_seed=SEED+707
    )
    x1_allowed = np.asarray(res["x_after"], float)
    x1_full    = embed_allowed_to_full(x1_allowed.reshape(1, -1))[0]
    delta1     = delta_size_for_comp_full(x1_full)
    vec1       = vec_for_comp_full(x1_full)

    print(f"\nTarget D* = {Dstar:.1f} mm")
    print(f"  L_robust init = {res['L_init_mm']:.2f} mm → final = {res['L_final_mm']:.2f} ± {1.96*res['L_final_se_mm']:.2f} mm (95% MC)")
    print(f"  Edit L1 = {res['edit_L1_frac']*100:.1f} at.% total change")
    print(f"  δ-size: {delta0:.2f} → {delta1:.2f} | VEC: {vec0:.2f} → {vec1:.2f}")
    if "note" in res: print("  Note:", res["note"])

    tag = f"Dstar_{int(round(Dstar))}mm"
    path_df = res["path"].copy()
    path_df.to_csv(SRC_DIR / f"recourse_path_{tag}.csv", index=False)
    
    before = {f"before_{e}": 100.0 * float(x0_allowed[i]) for i, e in enumerate(allowed_elems_present)}
    after  = {f"after_{e}":  100.0 * float(x1_allowed[i]) for i, e in enumerate(allowed_elems_present)}
    
    row = {
        **before,
        **after,
        "L_init_mm": float(res["L_init_mm"]),
        "L_final_mm": float(res["L_final_mm"]),
        "L_final_se_mm": float(res["L_final_se_mm"]),
        "edit_L1_atpct": 100.0 * float(res["edit_L1_frac"]),
        "delta0": float(delta0),
        "delta1": float(delta1),
        "vec0": float(vec0),
        "vec1": float(vec1),
    }

pd.DataFrame([row]).to_csv(SRC_DIR / f"recourse_before_after_{tag}.csv", index=False)

# Visualizations
sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

def _saveboth(path_png):
    path_png = Path(path_png)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

# 1) Path curve: L_robust vs cumulative edit for the highest target
if len(targets):
    Dstar = targets[-1]
    path_df = pd.read_csv(SRC_DIR / f"recourse_path_Dstar_{int(round(Dstar))}mm.csv")
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(path_df["cum_edit_atpct"], path_df["L_robust_mm"], marker="o")
    plt.xlabel("Cumulative edit (at.%)")
    plt.ylabel("Certified L_robust (mm)")
    plt.title(f"Recourse trajectory to D*={int(round(Dstar))} mm")
    _saveboth(FIGDIR / f"recourse_path_curve_Dstar_{int(round(Dstar))}mm.png")

# 2) Waterfall of element changes (before vs after) for the highest target
def _waterfall_before_after(tag):
    df_ba = pd.read_csv(SRC_DIR / f"recourse_before_after_{tag}.csv")
    # discover which elements actually exist in the file
    elems_in_file = sorted(
        {c.replace("before_","") for c in df_ba.columns if c.startswith("before_")}
        & {c.replace("after_","")  for c in df_ba.columns if c.startswith("after_")}
    )
    # keep the preferred order where possible
    elems = [e for e in allowed_elems_present if e in elems_in_file] or elems_in_file

    before = np.array([float(df_ba[f"before_{e}"].iloc[0]) for e in elems])
    after  = np.array([float(df_ba[f"after_{e}"].iloc[0])  for e in elems])
    delta  = after - before
    order  = np.argsort(-np.abs(delta))

    plt.figure(figsize=(7.2, max(3.6, 0.28*len(elems))))
    y = np.arange(len(order))
    plt.barh(y, delta[order])
    plt.yticks(y, [elems[i] for i in order])
    plt.axvline(0, lw=1, color="k")
    plt.xlabel("Δ at.% (after − before)")
    plt.title(f"Element changes from seed → recourse ({tag})")
    _saveboth(FIGDIR / f"recourse_waterfall_{tag}.png")

if len(targets):
    _waterfall_before_after(f"Dstar_{int(round(targets[-1]))}mm")


# In[61]:


# Reproducibility & artifacts
ART_DATA   = OUTDIR / "data" / "processed"
ART_MODELS = OUTDIR / "models"
ART_REP    = OUTDIR / "reports"
ART_SRC    = OUTDIR / "source_data"
for p in (ART_DATA, ART_MODELS, ART_REP, ART_SRC): p.mkdir(parents=True, exist_ok=True)

def _now_utc_iso():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float(x):
    try: 
        return float(x)
    except Exception:
        return None

def _wilson_ci_counts(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), max(0.0, center - half), min(1.0, center + half)

# Freeze & hash the cleaned dataset
base_csv = ART_DATA / "base.csv"
df.drop(columns=["signature"], errors="ignore").to_csv(base_csv, index=False)
base_hash = _sha256_of_file(base_csv)

schema = {
    "elem_cols": list(elem_cols),
    "X_COLUMNS": list(X_COLUMNS) if "X_COLUMNS" in globals() else None,
    "FAM_LEVELS": list(FAM_LEVELS) if "FAM_LEVELS" in globals() else None,
    "target_column": dmax_col,
    "n_rows": int(len(df)),
}
with open(ART_DATA / "schema.json", "w") as f:
    json.dump(schema, f, indent=2)

# Save split indices & training weights
splits = {
    "train_idx": list(map(int, idx_train)),
    "cal_idx"  : list(map(int, idx_cal)),
    "test_idx" : list(map(int, idx_test)),
}
with open(ART_DATA / "splits.json", "w") as f:
    json.dump(splits, f, indent=2)

if "w_train" in globals():
    pd.DataFrame({"idx": idx_train, "w_train": np.asarray(w_train, float)}).to_csv(
        ART_DATA / "train_weights.csv", index=False
    )

# Save model params & manifest
model_manifest = {
    "timestamp_utc": _now_utc_iso(),
    "seed": int(SEED),
    "cv_folds": int(CV_FOLDS) if "CV_FOLDS" in globals() else None,
    "alpha": _safe_float(ALPHA),
    "robust_eps_L1_fraction": _safe_float(ROBUST_EPS),
    "robust_samples_K": int(ROBUST_SAMPLES) if "ROBUST_SAMPLES" in globals() else None,
    "qt_tau_train": _safe_float(QT_TAU) if "QT_TAU" in globals() else None,
    "qt_tau_high": _safe_float(QT_TAU_HIGH) if "QT_TAU_HIGH" in globals() else None,
    "models": {}
}

if "et" in globals():
    model_manifest["models"]["ExtraTrees"] = {
        "params": et.get_params(deep=False),
        "path": str((ART_MODELS / "et_point.pkl").resolve()) if (ART_MODELS / "et_point.pkl").exists() else None
    }
if "cat_qt" in globals():
    model_manifest["models"]["CatBoost_tau"] = {
        "tau": _safe_float(QT_TAU) if "QT_TAU" in globals() else None,
        "params": cat_qt.get_params()
    }
if "cat_qt_hi" in globals():
    path_hi = ART_MODELS / f"cat_qt_tau{float(QT_TAU_HIGH):.2f}.cbm" if "QT_TAU_HIGH" in globals() else None
    model_manifest["models"]["CatBoost_tau_high"] = {
        "tau": _safe_float(QT_TAU_HIGH),
        "params": cat_qt_hi.get_params(),
        "path": str(path_hi.resolve()) if (path_hi and path_hi.exists()) else None
    }

with open(ART_MODELS / "model_manifest.json", "w") as f:
    json.dump(model_manifest, f, indent=2)

if "qhat_test_hi" in globals():
    qhat_for_sharpness_mm = np.exp(qhat_test_hi)
    tau_used_for_sharpness = float(QT_TAU_HIGH)
elif "qhat_test" in globals():
    qhat_for_sharpness_mm = np.exp(qhat_test)
    tau_used_for_sharpness = float(QT_TAU)
else:
    qhat_for_sharpness_mm = np.full_like(y_test_mm, np.nan)
    tau_used_for_sharpness = None

bounds = {}
for name in ("L_marginal_mm", "L_group_mm", "L_weighted_mm", "L_robust_mm",
             "L_marginal_hi_mm", "L_robust_hi_mm", "L_robust_hi_mm_test"):
    if name in globals():
        key = "robust_hi" if name in ("L_robust_hi_mm", "L_robust_hi_mm_test") else name.replace("_mm", "")
        bounds[key] = globals()[name]

per_rows = {
    "index": idx_test,
    "family": df.loc[idx_test, "family"].to_numpy(),
    "signature": df.loc[idx_test, "signature"].to_numpy(),
    "true_mm": y_test_mm,
    "qhat_mm_for_sharpness": qhat_for_sharpness_mm
}
for k, v in bounds.items():
    per_rows[f"{k}_mm"] = np.asarray(v, float)
    per_rows[f"{k}_covered"] = (y_test_mm >= np.asarray(v, float)).astype(int)

per_df = pd.DataFrame(per_rows)
per_df.to_csv(ART_DATA / "test_per_sample_all_bounds.csv", index=False)

# Calibration artifacts (scores & quantiles)
cal_art = {}
if "S_cal_hi" in globals():
    cal_art["S_cal_marginal_hi"] = np.asarray(S_cal_hi, float).tolist()
if "S_cal_hi_rob" in globals():
    cal_art["S_cal_robust_hi"] = np.asarray(S_cal_hi_rob, float).tolist()
if "q_marginal_hi" in globals():
    cal_art["q_marginal_hi"] = _safe_float(q_marginal_hi)
if "q_robust_hi" in globals():
    cal_art["q_robust_hi"] = _safe_float(q_robust_hi)

with open(ART_REP / "calibration_artifacts.json", "w") as f:
    json.dump(cal_art, f, indent=2)

# Metrics card (coverage, sharpness, losses, CIs)
coverage = {}
coverage_ci = {}
sharp_median = {}
for k, v in bounds.items():
    v = np.asarray(v, float)
    cov = float(np.mean(y_test_mm >= v))
    ksucc = int(np.sum(y_test_mm >= v))
    p_hat, lo, hi = _wilson_ci_counts(ksucc, int(len(y_test_mm)))
    coverage[k] = cov
    coverage_ci[k] = {"wilson_lo": lo, "wilson_hi": hi}
    sharp_median[k] = float(np.median(qhat_for_sharpness_mm - v))

pinball = None
if "qhat_test_hi" in globals():
    pinball = float(mean_pinball_loss(y_test, qhat_test_hi, alpha=QT_TAU_HIGH))
elif "qhat_test" in globals():
    pinball = float(mean_pinball_loss(y_test, qhat_test, alpha=QT_TAU))

# Log-scale errors (ET) and mm-scale errors
et_metrics = {}
if "yhat_test" in globals():
    et_metrics["log"] = {
        "R2": float(r2_score(y_test, yhat_test)),
        "MAE": float(mean_absolute_error(y_test, yhat_test)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, yhat_test))),
    }
    et_metrics["mm"] = {
        "MAE": float(mean_absolute_error(np.exp(y_test), np.exp(yhat_test))),
        "RMSE": float(np.sqrt(mean_squared_error(np.exp(y_test), np.exp(yhat_test)))),
    }

metrics = {
    "created_utc": _now_utc_iso(),
    "dataset": {"base_csv": str(base_csv.resolve()), "sha256": base_hash},
    "n_test": int(len(y_test_mm)),
    "alpha": float(ALPHA),
    "robust_eps_L1_fraction": float(ROBUST_EPS),
    "robust_samples_K": int(ROBUST_SAMPLES) if "ROBUST_SAMPLES" in globals() else None,
    "tau_used_for_sharpness": tau_used_for_sharpness,
    "coverage": coverage,
    "coverage_wilson95": coverage_ci,
    "sharpness_median_gap_mm": sharp_median,
    "pinball_loss_used_tau": pinball,
    "et_metrics": et_metrics
}
with open(ART_REP / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Environment fingerprint
env = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": {
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "scikit-learn": __import__("sklearn").__version__,
        "catboost": __import__("catboost").__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "seaborn": __import__("seaborn").__version__,
        "skopt": __import__("skopt").__version__,
    },
    "seed": int(SEED),
}
with open(ART_REP / "environment.json", "w") as f:
    json.dump(env, f, indent=2)


manifest = {
    "data": {
        "base_csv": str(base_csv.resolve()),
        "schema": str((ART_DATA / "schema.json").resolve()),
        "splits": str((ART_DATA / "splits.json").resolve()),
        "test_per_sample": str((ART_DATA / "test_per_sample_all_bounds.csv").resolve()),
        "train_weights": str((ART_DATA / "train_weights.csv").resolve()) if (ART_DATA / "train_weights.csv").exists() else None,
    },
    "models": str((ART_MODELS / "model_manifest.json").resolve()),
    "reports": {
        "metrics": str((ART_REP / "metrics.json").resolve()),
        "calibration_artifacts": str((ART_REP / "calibration_artifacts.json").resolve()),
        "environment": str((ART_REP / "environment.json").resolve()),
    },
    "source_data_dir": str(ART_SRC.resolve())
}
with open(ART_REP / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

readme_lines = [
    "# Reproducibility bundle",
    "",
    f"- Created (UTC): {metrics['created_utc']}",
    f"- Dataset SHA256: `{base_hash}`",
    f"- Alpha (conformal): {ALPHA:.3f}, Robust ε (L1): {ROBUST_EPS:.4f}, K={ROBUST_SAMPLES}",
    f"- τ used for sharpness: {tau_used_for_sharpness}",
    "",
    "## Key files",
    f"- Data schema: `data/processed/schema.json`",
    f"- Splits: `data/processed/splits.json`",
    f"- Test per-sample (all bounds): `data/processed/test_per_sample_all_bounds.csv`",
    f"- Metrics card: `reports/metrics.json`",
    f"- Calibration artifacts: `reports/calibration_artifacts.json`",
    f"- Environment: `reports/environment.json`",
    f"- Model manifest: `models/model_manifest.json`",
    "",
    "## Notes",
    "- All composition-dependent features are generated with a frozen schema (`X_COLUMNS`).",
    "- Coverage is reported with Wilson 95% intervals.",
    "- Robust certificates use the high-τ model and q_(1-α) from robust calibration.",
]
with open(ART_REP / "README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))

print("Saved cleaned data, schema, splits, per-sample test CSV, metrics, calibration artifacts, environment, and manifest under:", OUTDIR)


# In[62]:


# Artifact & provenance hardening (models, env, manifests, hashes)
MODELDIR = OUTDIR / "models"
REPDIR   = OUTDIR / "reports"
HPODIR   = OUTDIR / "reports" / "hpo"
FIGDIR   = OUTDIR / "reports" / "figures"
SRCDIR   = OUTDIR / "source_data"
for p in (MODELDIR, REPDIR, HPODIR, FIGDIR, SRCDIR): p.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> dict:
    try:
        h = hashlib.sha256()
        n = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk); n += len(chunk)
        return {"sha256": h.hexdigest(), "n_bytes": int(n)}
    except Exception as e:
        return {"sha256": None, "n_bytes": None, "error": str(e)}

def _save_sklearn_model(model, name, acc):
    try:
        path = MODELDIR / f"{name}.joblib"
        joblib.dump(model, path)
        h = _sha256(path)
        acc.append({"name": name, "type": "sklearn", "path": str(path.resolve()), **h})
    except Exception as e:
        print(f"[warn] could not save {name}: {e}")

def _save_catboost_model(model, name, acc):
    try:
        path = MODELDIR / f"{name}.cbm"
        if hasattr(model, "save_model"):
            model.save_model(str(path))
        else:
            joblib.dump(model, path.with_suffix(".joblib"))
            path = path.with_suffix(".joblib")
        h = _sha256(path)
        # record params if available
        params = {}
        try: params = model.get_params()
        except Exception: pass
        # try to infer τ from params
        tau_val = None
        try:
            lf = str(params.get("loss_function", ""))
            if "Quantile:alpha=" in lf:
                tau_val = float(lf.split("Quantile:alpha=")[1])
        except Exception:
            pass
        acc.append({"name": name, "type": "catboost", "tau": tau_val,
                    "path": str(path.resolve()), "params": params, **h})
    except Exception as e:
        print(f"[warn] could not save {name}: {e}")

# save models (point + quantile)
saved_models = []

if "rf" in globals() and rf is not None:
    _save_sklearn_model(rf, "rf_point", saved_models)

if "et" in globals() and et is not None:
    _save_sklearn_model(et, "extratrees_point", saved_models)

# quantile models
if "cat_qt_hi" in globals() and cat_qt_hi is not None:
    _save_catboost_model(cat_qt_hi, "catboost_qtau_high", saved_models)
if "cat_qt" in globals() and cat_qt is not None:
    _save_catboost_model(cat_qt, "catboost_qtau_main", saved_models)

# multi-τ grid
if "cat_qt_grid" in globals():
    for t, m in cat_qt_grid.items():
        if m is None: continue
        tag = f"catboost_qtau_{float(t):.2f}"
        _save_catboost_model(m, tag, saved_models)

(Path(REPDIR / "models_manifest.json")).write_text(json.dumps({
    "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
    "items": saved_models
}, indent=2))

# index HPO best-param files
hpo_index = {}
for fn in [
    "et_bayes_best_params.json",
    "rf_bayes_best_params.json",
    "gbm_tau_bayes_best_params.json",
    "gbm_hi_tau_bayes_best_params.json",
    "catboost_tau_bayes_best_params.json",
    "catboost_hi_tau_bayes_best_params.json",
]:
    p = HPODIR / fn
    if p.exists():
        try:
            hpo_index[fn] = json.loads(p.read_text())
        except Exception:
            hpo_index[fn] = f"[unreadable file at {p}]"
(REPDIR / "hpo_best_params_index.json").write_text(json.dumps(hpo_index, indent=2))

# Environment capture
try:
    freeze_txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (REPDIR / "env_requirements.txt").write_text(freeze_txt)
except Exception as e:
    (REPDIR / "env_requirements.txt").write_text(f"# pip freeze failed: {e}\n")

try:
    env_yaml = subprocess.check_output(["conda", "env", "export", "--no-builds"], text=True)
    (REPDIR / "conda_env.yaml").write_text(env_yaml)
except Exception:
    pass

env_card = {
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "env_threads": {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    },
    "packages": {
        "numpy": __import__("numpy").__version__,
        "pandas": __import__("pandas").__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "catboost": __import__("catboost").__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "seaborn": __import__("seaborn").__version__,
        "skopt": __import__("skopt").__version__,
    },
}
(REPDIR / "environment_card.json").write_text(json.dumps(env_card, indent=2))

git_info = {"in_repo": False}
try:
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    git_info = {
        "in_repo": True,
        "root": root,
        "head": head,
        "branch": branch,
        "dirty": len(status.strip()) > 0
    }
except Exception:
    pass
(REPDIR / "git_info.json").write_text(json.dumps(git_info, indent=2))

def _maybe(v): 
    return v if v is not None else None

run_cfg = {
    "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
    "seed": int(SEED),
    "alpha": float(ALPHA),
    "robust_eps_fraction_L1": float(ROBUST_EPS),
    "robust_samples": int(ROBUST_SAMPLES) if "ROBUST_SAMPLES" in globals() else None,
    "qt_tau": float(QT_TAU) if "QT_TAU" in globals() else None,
    "qt_tau_high": float(QT_TAU_HIGH) if "QT_TAU_HIGH" in globals() else None,
    "thresholds_mm": [float(x) for x in THRESHOLDS_MM] if "THRESHOLDS_MM" in globals() else None,
    "min_group_n": int(MIN_GROUP_N) if "MIN_GROUP_N" in globals() else None,
    "family_min_test": int(FAMILY_MIN_TEST) if "FAMILY_MIN_TEST" in globals() else None,
    "dmax_col": dmax_col,
    "elem_cols": list(elem_cols),
    "allowed_elems_subset": list(allowed_elems_present) if "allowed_elems_present" in globals() else None,
    "novelty_L1_atpct": float(NOVELTY_L1_ATPCT) if "NOVELTY_L1_ATPCT" in globals() else None,
    "bo_used": bool("bo_df" in globals()),
    "bo_params": {
        "n_calls": _maybe(1000),
        "n_random_starts": _maybe(50),
        "xi": _maybe(0.02),
        "diversity_batch": _maybe(30),
        "diversity_min_L1_atpct": _maybe(2.0),
        "hard_caps": _maybe(hard_caps) if "hard_caps" in globals() else None,
        "max_elements": _maybe(max_elements) if "max_elements" in globals() else None,
    },
    "split_files": {
        "random": str((OUTDIR / "splits_random.json").resolve()) if (OUTDIR / "splits_random.json").exists() else None,
        "family_out": str((OUTDIR / "splits_family_out.json").resolve()) if (OUTDIR / "splits_family_out.json").exists() else None,
    },
    "models_manifest": str((REPDIR / "models_manifest.json").resolve()),
    "hpo_best_params_index": str((REPDIR / "hpo_best_params_index.json").resolve()),
    "fig_dir": str(FIGDIR.resolve()),
    "src_data_dir": str(SRCDIR.resolve()),
}
(REPDIR / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

data_hash = {"raw_csv": globals().get("RAW_CSV", None), "exists": False, "sha256": None, "n_bytes": None}
try:
    raw_path = Path(RAW_CSV)
    if raw_path.exists():
        info = _sha256(raw_path)
        data_hash.update({"exists": True, **info})
except Exception as e:
    data_hash.update({"error": str(e)})
(REPDIR / "data_hash.json").write_text(json.dumps(data_hash, indent=2))

def _dir_manifest(root: Path, exts=None):
    out = []
    if not root.exists(): return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and (exts is None or p.suffix.lower() in exts):
            h = _sha256(p)
            out.append({"file": str(p.resolve()), **h})
    return out

fig_manifest = _dir_manifest(FIGDIR, exts={".png", ".pdf", ".svg"})
(REPDIR / "figures_manifest.json").write_text(json.dumps(fig_manifest, indent=2))

src_manifest = _dir_manifest(SRCDIR, exts={".csv", ".tsv", ".parquet"})
(REPDIR / "source_data_manifest.json").write_text(json.dumps(src_manifest, indent=2))

print("Models saved:", MODELDIR)
print("HPO index:", REPDIR / "hpo_best_params_index.json")
print("Environment:", REPDIR / "environment_card.json", "| pip freeze:", REPDIR / "env_requirements.txt")
print("Git info:", REPDIR / "git_info.json")
print("Run config:", REPDIR / "run_config.json")
print("Data hash:", REPDIR / "data_hash.json")
print("Figures manifest:", REPDIR / "figures_manifest.json")
print("Source-data manifest:", REPDIR / "source_data_manifest.json")


# In[63]:


# Figure helpers
try:
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

FIGDIR = OUTDIR / "reports" / "figures"
SRCDIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

def set_theme_ncs(font_scale=1.2):
    """
    One-time global styling for crisp, consistent, journal-ready figures.
    """
    if _HAS_SNS:
        sns.set_theme(context="paper", style="whitegrid", font_scale=font_scale)
    mpl.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titlepad": 8.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "figure.autolayout": False,
    })

set_theme_ncs()

def bootstrap_ci_indices(fn_on_indices, n, B=2000, seed=None):
    """
    Nonparametric bootstrap CI for a statistic that is computed from
    index-resampled data. `fn_on_indices` must accept a 1D index array.
    Returns (est, lo, hi) with 95% percentile CI.
    """
    if seed is None: seed = int(globals().get("SEED", 0))
    rng = np.random.default_rng(seed)
    if n <= 0 or B <= 0: 
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(n)
    stats = np.empty(B, float)
    for b in range(B):
        res = rng.choice(idx, size=n, replace=True)
        stats[b] = float(fn_on_indices(res))
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def bootstrap_ci_stat(values, stat_fn=np.nanmean, B=2000, seed=None):
    """
    Bootstrap CI for a 1D array-like of values with a scalar stat_fn (e.g., np.nanmean).
    """
    x = np.asarray(values, float)
    n = x.size
    return bootstrap_ci_indices(lambda res: stat_fn(x[res]), n=n, B=B, seed=seed)

def bootstrap_ci_pair(y_true, y_pred, metric_fn, B=2000, seed=None):
    """
    Bootstrap CI for paired data metric_fn(y_true, y_pred). E.g., MAE, RMSE, pinball loss (with partial).
    """
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    n = yt.size
    return bootstrap_ci_indices(lambda res: metric_fn(yt[res], yp[res]), n=n, B=B, seed=seed)

def wilson_ci(k, n, z=1.96):
    """
    Wilson CI for binomial coverage; returns (p_hat, lo, hi).
    """
    if n <= 0: 
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z/denom) * math.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))

def _to_path(p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def savefig_bundle(basepath, src_df=None, meta=None, formats=("png","pdf")):
    """
    Save current figure to FIGDIR with multiple formats, and optionally write
    (i) a source-data CSV and (ii) a sidecar JSON with figure metadata.
      - basepath: relative file stem inside FIGDIR, e.g., "parity/et_parity_mm"
      - src_df:   pandas DataFrame to save alongside (goes to SRCDIR with same stem)
      - meta:     dict of metadata (caption, panel id, seed, tau, etc.)
    """
    basepath = str(basepath).rstrip("/").rstrip("\\")
    fig_stem = _to_path(FIGDIR / basepath)
    for ext in formats:
        plt.savefig(f"{fig_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()

    if src_df is not None:
        src_stem = _to_path(SRCDIR / (Path(basepath).name + "_source"))
        src_path = src_stem.with_suffix(".csv")
        try:
            src_df.to_csv(src_path, index=False)
        except Exception as e:
            try:
                pd.DataFrame(src_df).to_csv(src_path, index=False)
            except Exception as ee:
                print(f"[warn] could not write source CSV for {basepath}: {e or ee}")

    card = {
        "path_png": str((fig_stem.with_suffix(".png")).resolve()),
        "path_pdf": str((fig_stem.with_suffix(".pdf")).resolve()),
        "source_csv": str((SRCDIR / (Path(basepath).name + "_source.csv")).resolve()) if src_df is not None else None,
        "meta": meta or {},
    }
    _to_path(fig_stem.with_suffix(".json")).write_text(json.dumps(card, indent=2))

def figsave(path_png):
    """
    Backward-compatible thin wrapper (PNG/PDF). Prefer savefig_bundle for richer output.
    """
    path_png = _to_path(path_png)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    try:
        plt.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close()

def parity_guides(ax, xymin=None, xymax=None, ls="--", lw=1.0, alpha=0.8):
    """
    Add y=x parity guide, optionally forcing square limits.
    """
    if xymin is None or xymax is None:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xymin = min(x0, y0); xymax = max(x1, y1)
    ax.plot([xymin, xymax], [xymin, xymax], ls=ls, lw=lw, alpha=alpha, color="k")
    ax.set_xlim(xymin, xymax); ax.set_ylim(xymin, xymax)
    ax.set_aspect("equal", "box")

def annotate_metrics(ax, lines, loc="lower right", fontsize=9):
    """
    Add a small text box with metrics (list of strings). Example:
      annotate_metrics(ax, [f"RMSE={rmse:.2f} mm", f"MAE={mae:.2f} mm"])
    """
    txt = "\n".join(lines)
    ax.text(0.99, 0.01, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fontsize, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

def label_panel(ax, label="a", x=0.0, y=1.02, **kw):
    """
    Panel label 'a', 'b', ... for multi-panel figures.
    """
    ax.text(x, y, f"{label})", transform=ax.transAxes, weight="bold", **({"fontsize": 12} | kw))

def stamp_info(ax, text=None):
    """
    Optional subtle stamp with run info (seed, tau, eps) in the lower-left corner.
    """
    if text is None:
        parts = []
        if "SEED" in globals(): parts.append(f"seed={SEED}")
        if "QT_TAU_HIGH" in globals(): parts.append(f"τ={QT_TAU_HIGH:.2f}")
        if "ROBUST_EPS" in globals(): parts.append(f"ε(L1)={ROBUST_EPS}")
        text = " | ".join(parts)
    if text:
        ax.text(0.01, 0.01, text, transform=ax.transAxes, fontsize=8, color="0.3", ha="left", va="bottom")

def safe_minmax(*arrays):
    """
    Robust min/max across arrays with NaN handling.
    """
    vals = np.hstack([np.asarray(a, float).ravel() for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: 
        return (0.0, 1.0)
    return float(vals.min()), float(vals.max())

def ensure_finite(df, fill=0.0):
    """
    Return a copy with non-finite replaced by `fill`.
    """
    out = df.copy()
    for c in out.columns:
        v = np.asarray(out[c], float)
        v[~np.isfinite(v)] = fill
        out[c] = v
    return out

METHOD_COLORS = {
    "Marginal": "#1f77b4",
    "Group-cond.": "#ff7f0e",
    "Shift-weighted": "#2ca02c",
    "Robust": "#d62728",
    "Robust (±)": "#d62728",
}

print("Figure helpers ready:",
      f"\n  FIGDIR = {FIGDIR}",
      f"\n  SRCDIR = {SRCDIR}",
      "\n  Use savefig_bundle('subdir/figure_stem', src_df=..., meta=...).")


# In[64]:


# Reliability figures
robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"

bounds = {}
if "L_marginal_mm" in globals():  bounds["Marginal"]        = np.asarray(L_marginal_mm,  float)
if "L_group_mm"    in globals():  bounds["Group-cond."]     = np.asarray(L_group_mm,     float)
if "L_weighted_mm" in globals():  bounds["Shift-weighted"]  = np.asarray(L_weighted_mm,  float)

_rob = globals().get("L_robust_mm",
       globals().get("L_robust_hi_mm",
       globals().get("L_robust_hi_mm_test", None)))
if _rob is not None:
    bounds[robust_label] = np.asarray(_rob, float)

if not bounds:
    raise RuntimeError("No bound arrays found in scope for reliability plots.")

labels, means, lows, highs = [], [], [], []
n = len(y_test_mm)

for name, L in bounds.items():
    L_arr = np.asarray(L, float)

    def cov_fn(idx_resampled):
        idx = np.asarray(idx_resampled, int)
        m = np.isfinite(y_test_mm[idx]) & np.isfinite(L_arr[idx])
        if not np.any(m):
            return np.nan
        return float(np.mean(y_test_mm[idx][m] >= L_arr[idx][m]))

    m, lo, hi = bootstrap_ci_indices(cov_fn, n=n, B=2000, seed=SEED+123)
    labels.append(name); means.append(m); lows.append(lo); highs.append(hi)

cov_df = pd.DataFrame({
    "method": labels,
    "coverage_mean": means,
    "ci_lo": lows,
    "ci_hi": highs,
    "target": 1.0 - ALPHA,
    "n_test": n
})

rows = []
test_index = np.array(idx_test) if "idx_test" in globals() else np.arange(n)
for name, L in bounds.items():
    L_arr = np.asarray(L, float)
    mfin = np.isfinite(y_test_mm) & np.isfinite(L_arr)
    rows.append(pd.DataFrame({
        "index":   test_index[mfin],
        "method":  name,
        "y_true_mm": y_test_mm[mfin],
        "L_mm":      L_arr[mfin],
        "covered":  (y_test_mm[mfin] >= L_arr[mfin]).astype(int)
    }))
per_sample_df = pd.concat(rows, ignore_index=True)

fig, ax = plt.subplots(figsize=(6.2, 3.9))
x = np.arange(len(labels))
bars = ax.bar(
    x, means,
    color=[METHOD_COLORS.get(lbl, "#5a5a5a") for lbl in labels],
    edgecolor="black", linewidth=0.6
)
err_low  = np.maximum(0.0, np.array(means) - np.array(lows))
err_high = np.maximum(0.0, np.array(highs) - np.array(means))
ax.errorbar(x, means, yerr=[err_low, err_high], fmt="none", capsize=4, ecolor="black", lw=1)
ax.axhline(1-ALPHA, ls="--", lw=1, color="k", label=f"Target {1-ALPHA:.2f}")

ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Coverage (fraction ≥ L)")
ax.set_title(f"Coverage@{int((1-ALPHA)*100)}% (test split)")
stamp_info(ax)
ax.legend(loc="lower right", frameon=False)

savefig_bundle(
    "reliability_coverage",
    src_df=cov_df,
    meta={
        "panel": "a",
        "alpha": float(ALPHA),
        "robust_eps_L1": float(ROBUST_EPS),
        "n_test": int(n),
        "note": "Bootstrap percentile CI (B=2000)."
    }
)

per_sample_df.to_csv(SRCDIR / "reliability_per_sample_long.csv", index=False)

qhat_for_sharp = (np.exp(qhat_test_hi) if "qhat_test_hi" in globals()
                  else np.exp(qhat_test))

gap_rows = []
for name, L in bounds.items():
    L_arr = np.asarray(L, float)
    mfin = np.isfinite(qhat_for_sharp) & np.isfinite(L_arr)
    gap = qhat_for_sharp[mfin] - L_arr[mfin]
    gap_rows.append(pd.DataFrame({
        "method": name,
        "gap_mm": gap
    }))
gaps_long = pd.concat(gap_rows, ignore_index=True)

# Boxplot
fig, ax = plt.subplots(figsize=(6.4, 3.9))
methods_order = list(bounds.keys())
data_by_method = [gaps_long.loc[gaps_long["method"]==m, "gap_mm"].values for m in methods_order]
bp = ax.boxplot(
    data_by_method, labels=methods_order, showfliers=False, patch_artist=True
)
for patch, m in zip(bp['boxes'], methods_order):
    patch.set_facecolor(METHOD_COLORS.get(m, "#8c8c8c"))
    patch.set_edgecolor("black"); patch.set_linewidth(0.6)

ax.set_ylabel("Gap (mm) = exp(q̂) − L")
ax.set_title("Sharpness of lower bounds (smaller is tighter)")
stamp_info(ax)

for med_line in bp["medians"]:
    x_med, y_med = np.mean(med_line.get_xdata()), np.mean(med_line.get_ydata())
    ax.text(x_med, y_med, f"{y_med:.2f}", ha="center", va="bottom", fontsize=8)

savefig_bundle(
    "reliability_sharpness",
    src_df=gaps_long,
    meta={
        "panel": "b",
        "tau_for_qhat": float(globals().get("QT_TAU_HIGH", globals().get("QT_TAU", np.nan))),
        "note": "Boxplots of exp(q̂)−L across methods; medians labeled."
    }
)

print("Saved:")
print("  - figures:", FIGDIR / "reliability_coverage.png", "and", FIGDIR / "reliability_sharpness.png")
print("  - sources:", SRCDIR / "reliability_coverage_source.csv",
      "and", SRCDIR / "reliability_sharpness_source.csv",
      "plus per-sample table:", SRCDIR / "reliability_per_sample_long.csv")


# In[65]:


# Tail discovery curves (Precision@k, with CI, baseline, ceiling)
if "bootstrap_ci_indices" not in globals():
    def bootstrap_ci_indices(fn, n, B=1000, seed=SEED+202):
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        vals = np.empty(B, float)
        for b in range(B):
            res = rng.choice(idx, size=n, replace=True)
            vals[b] = float(fn(res))
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

if "savefig_bundle" not in globals():
    def savefig_bundle(stem, src_df=None, meta=None):
        path_png = FIGDIR / f"{stem}.png"
        path_pdf = FIGDIR / f"{stem}.pdf"
        plt.tight_layout()
        plt.savefig(path_png, dpi=300, bbox_inches="tight")
        try:
            plt.savefig(path_pdf, bbox_inches="tight")
        except Exception:
            pass
        plt.close()
        if src_df is not None:
            src_df.to_csv(SRCDIR / f"{stem}_source.csv", index=False)
        if meta is not None:
            with open(SRCDIR / f"{stem}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

if "METHOD_COLORS" not in globals():
    METHOD_COLORS = {
        "Marginal": "#1f77b4",
        "Group-cond.": "#ff7f0e",
        "Shift-weighted": "#2ca02c",
    }

robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"
bounds = {}
if "L_marginal_mm" in globals():  bounds["Marginal"]        = np.asarray(L_marginal_mm,  float)
if "L_group_mm"    in globals():  bounds["Group-cond."]     = np.asarray(L_group_mm,     float)
if "L_weighted_mm" in globals():  bounds["Shift-weighted"]  = np.asarray(L_weighted_mm,  float)
_rob = globals().get("L_robust_mm",
       globals().get("L_robust_hi_mm",
       globals().get("L_robust_hi_mm_test", None)))
if _rob is not None:
    bounds[robust_label] = np.asarray(_rob, float)
    METHOD_COLORS.setdefault(robust_label, "#d62728")

if not bounds:
    raise RuntimeError("No lower-bound arrays found for tail discovery curves.")

N = len(y_test_mm)
KS = (5, 10, 20, 50, 100)

def precision_at_k(L_vec, y_vec, thresh, ks=KS):
    L = np.asarray(L_vec, float); y = np.asarray(y_vec, float)
    m = np.isfinite(L) & np.isfinite(y)
    if not np.any(m): return np.zeros(len(ks), float)
    Lm, Ym = L[m], y[m]
    order = np.argsort(-Lm)
    out = []
    for k in ks:
        k_eff = min(k, len(order))
        if k_eff == 0:
            out.append(0.0); continue
        sel = order[:k_eff]
        out.append(float(np.mean(Ym[sel] >= thresh)))
    return np.array(out, float)

def precision_at_k_on_indices(idx_resampled, L_vec, y_vec, thresh, ks=KS):
    idx = np.asarray(idx_resampled, int)
    L = np.asarray(L_vec, float)[idx]
    y = np.asarray(y_vec, float)[idx]
    m = np.isfinite(L) & np.isfinite(y)
    if not np.any(m): return np.zeros(len(ks), float)
    Lm, Ym = L[m], y[m]
    order = np.argsort(-Lm)
    vals = []
    for k in ks:
        k_eff = min(k, len(order))
        if k_eff == 0:
            vals.append(0.0); continue
        sel = order[:k_eff]
        vals.append(float(np.mean(Ym[sel] >= thresh)))
    return np.array(vals, float)

rows_all = []

for Dstar in THRESHOLDS_MM:
    m_base = np.isfinite(y_test_mm)
    base = float(np.mean(y_test_mm[m_base] >= Dstar)) if np.any(m_base) else 0.0
    P = int(np.sum(y_test_mm[m_base] >= Dstar)) if np.any(m_base) else 0
    ceiling = np.array([min(1.0, P / k) if k > 0 else 0.0 for k in KS], float)

    df_curves = []
    for name, L in bounds.items():
        point = precision_at_k(L, y_test_mm, Dstar, ks=KS)

        lo = np.zeros_like(point); hi = np.zeros_like(point)
        for j, k in enumerate(KS):
            def fn(res_idx):
                return precision_at_k_on_indices(res_idx, L, y_test_mm, Dstar, ks=(k,))[0]
            m, lo_j, hi_j = bootstrap_ci_indices(fn, n=N, B=1000, seed=SEED+int(100*Dstar)+j)
            lo[j], hi[j] = lo_j, hi_j

        df_m = pd.DataFrame({
            "threshold_mm": float(Dstar),
            "method": name,
            "k": np.array(KS, int),
            "precision": point,
            "prec_lo": lo,
            "prec_hi": hi,
            "baseline": base,
            "ceiling": ceiling,
            "P": P,
            "N": N,
            "enrichment": (point / base) if base > 0 else np.full_like(point, np.nan),
            "enrich_lo": (lo / base) if base > 0 else np.full_like(lo, np.nan),
            "enrich_hi": (hi / base) if base > 0 else np.full_like(hi, np.nan),
        })
        df_curves.append(df_m)
        rows_all.append(df_m)

    df_curves = pd.concat(df_curves, ignore_index=True)

    # Plot Precision@k with CI ribbons
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.axhline(base, ls="--", lw=1, color="k", label=f"Baseline={base:.2f}")
    ax.plot(KS, ceiling, linestyle=":", lw=1.2, color="k", label="Ceiling")

    for name in bounds.keys():
        sub = df_curves[df_curves["method"] == name]
        col = METHOD_COLORS.get(name, None)
        ax.plot(sub["k"], sub["precision"], marker="o", label=name, color=col)
        ax.fill_between(sub["k"], sub["prec_lo"], sub["prec_hi"], alpha=0.15, color=col)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel(f"Precision@k (≥ {int(Dstar)} mm)")
    ax.set_title(f"Tail discovery (≥ {int(Dstar)} mm)")
    stamp_info(ax) if "stamp_info" in globals() else None
    ax.legend(frameon=False, loc="lower right")

    savefig_bundle(
        f"tail_precision_k_{int(Dstar)}mm",
        src_df=df_curves,
        meta={
            "threshold_mm": float(Dstar),
            "alpha": float(ALPHA),
            "robust_eps_L1": float(ROBUST_EPS),
            "ks": list(map(int, KS)),
            "bootstrap_B": 1000
        }
    )

    # Plot Enrichment Factor (lift)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for name in bounds.keys():
        sub = df_curves[df_curves["method"] == name]
        col = METHOD_COLORS.get(name, None)
        ax.plot(sub["k"], sub["enrichment"], marker="o", label=name, color=col)
        ax.fill_between(sub["k"], sub["enrich_lo"], sub["enrich_hi"], alpha=0.15, color=col)
    ax.axhline(1.0, ls="--", lw=1, color="k", label="Random = 1.0")
    ax.set_xlabel("k")
    ax.set_ylabel(f"Enrichment@k (≥ {int(Dstar)} mm)")
    ax.set_title(f"Tail discovery enrichment (≥ {int(Dstar)} mm)")
    stamp_info(ax) if "stamp_info" in globals() else None
    ax.legend(frameon=False, loc="upper right")

    savefig_bundle(
        f"tail_enrichment_k_{int(Dstar)}mm",
        src_df=df_curves,
        meta={
            "threshold_mm": float(Dstar),
            "alpha": float(ALPHA),
            "robust_eps_L1": float(ROBUST_EPS),
            "ks": list(map(int, KS)),
            "bootstrap_B": 1000,
            "note": "Enrichment = Precision / Baseline"
        }
    )

df_all = pd.concat(rows_all, ignore_index=True)
df_all.to_csv(SRCDIR / "tail_discovery_all_thresholds.csv", index=False)

print("Saved tail discovery figures to:", FIGDIR)
print("Saved per-threshold sources and tail_discovery_all_thresholds.csv to:", SRCDIR)


# In[66]:


# Stress curves: coverage & sharpness vs tolerance ε
FIGDIR = (OUTDIR / "reports" / "figures"); FIGDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "source_data").mkdir(parents=True, exist_ok=True)

if 'X_test' not in globals(): X_test = X.iloc[idx_test]
if 'X_cal'  not in globals(): X_cal  = X.iloc[idx_cal]
if 'y_cal'  not in globals(): y_cal  = y_log[idx_cal]
if 'y_test' not in globals(): y_test = y_log[idx_test]

# Master epsilon list (must match write+read)
EPS_LIST = [0.000, 0.005, 0.010, 0.020, 0.030]

def eps_tag(eps: float) -> str:
    return f"{int(round(1000*float(eps))):03d}"

# Compose matrices in element space (fractions)
X_test_elem_full = df.loc[idx_test, elem_cols].to_numpy(float)
X_test_elem_full /= np.where(X_test_elem_full.sum(axis=1, keepdims=True) > 0,
                             X_test_elem_full.sum(axis=1, keepdims=True), 1.0)
X_cal_elem = df.loc[idx_cal, elem_cols].to_numpy(float)
X_cal_elem /= np.where(X_cal_elem.sum(axis=1, keepdims=True) > 0,
                       X_cal_elem.sum(axis=1, keepdims=True), 1.0)

# High-τ predictions on CAL/TEST features (already-built feature matrix X_*)
if "qhat_test_hi" not in globals(): qhat_test_hi = cat_qt_hi.predict(X_test)
if "qhat_cal_hi"  not in globals(): qhat_cal_hi  = cat_qt_hi.predict(X_cal)

qhat_test_hi = np.asarray(qhat_test_hi, float)       # log scale
qhat_cal_hi  = np.asarray(qhat_cal_hi,  float)       # log scale
S_cal_hi     = np.maximum(0.0, qhat_cal_hi - y_cal)  # marginal one-sided residuals (log)
q_marginal_hi = conformal_qhat(S_cal_hi, ALPHA)

# Subset TEST for stress curves (speed)
N_eval = min(500, len(idx_test))
_eval_rng = np.random.default_rng(SEED + 1212)
eval_subset = _eval_rng.choice(len(idx_test), size=N_eval, replace=False)
X_test_sub_full   = X_test_elem_full[eval_subset]
y_test_sub_log    = y_test[eval_subset]
y_test_sub_mm     = np.exp(y_test_sub_log)
qhat_test_hi_sub  = qhat_test_hi[eval_subset]
qhat_test_hi_sub_mm = np.exp(qhat_test_hi_sub)

# Robust min helper (use allowed-subset jitter + embed back to full)
def _robust_min_logq(q_model, X_elem_full, eps, K, rng, chunk=2048):
    out = np.full(len(X_elem_full), np.inf, float)
    for i, x_full in enumerate(X_elem_full):
        x_allowed = x_full[allowed_idx]
        Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
        Xj_full    = embed_allowed_to_full(Xj_allowed)
        feats      = make_features_from_compositions(Xj_full)
        qj_log     = np.asarray(q_model.predict(feats), float)
        out[i]     = float(np.min(qj_log))
    return out

# Compute per-ε coverage & sharpness on the TEST subset and WRITE per-ε CSVs
records = []
for eps in EPS_LIST:
    if eps == 0.0:
        q_eps = q_marginal_hi
        L_eps = np.exp(qhat_test_hi_sub - q_eps)
    else:
        rng_cal = np.random.default_rng(SEED + int(10000 * (1 + 1000*eps)))
        S_cal_eps = robust_scores_lower_dispatch(
            y_cal,         # log
            cat_qt_hi,     # model
            X_cal_elem,    # fractions (full)
            eps=float(eps), K=ROBUST_SAMPLES, rng=rng_cal
        )
        S_cal_eps = np.maximum(0.0, np.nan_to_num(S_cal_eps, nan=0.0))
        q_eps = weighted_quantile(S_eps, 1 - ALPHA)

        rng_te = np.random.default_rng(SEED + int(20000 * (1 + 1000*eps)))
        qmin_sub_log = _robust_min_logq(cat_qt_hi, X_test_sub_full, eps=float(eps),
                                        K=ROBUST_SAMPLES, rng=rng_te)
        L_eps = np.exp(qmin_sub_log - q_eps)

    m = np.isfinite(L_eps) & np.isfinite(y_test_sub_mm) & np.isfinite(qhat_test_hi_sub_mm)
    n_eff = int(np.sum(m))
    if n_eff > 0:
        # bootstrap CIs
        def cov_fn(res_idx):
            idx = np.asarray(res_idx, int)
            return np.mean(y_test_sub_mm[m][idx] >= L_eps[m][idx])
        cov_mean, cov_lo, cov_hi = bootstrap_ci_indices(cov_fn, n=n_eff, B=1000, seed=SEED + 4242 + int(1e5*eps))

        gaps_vec = (qhat_test_hi_sub_mm[m] - L_eps[m])
        def gap_fn(res_idx):
            idx = np.asarray(res_idx, int)
            return float(np.median(gaps_vec[idx]))
        gap_med, gap_lo, gap_hi = bootstrap_ci_indices(gap_fn, n=n_eff, B=1000, seed=SEED + 5252 + int(1e5*eps))
    else:
        cov_mean = cov_lo = cov_hi = np.nan
        gap_med  = gap_lo = gap_hi = np.nan

    records.append({
        "eps_L1_fraction": float(eps),
        "eps_atpct": float(100*eps),
        "coverage_mean": float(cov_mean),
        "coverage_lo": float(cov_lo),
        "coverage_hi": float(cov_hi),
        "gap_median_mm": float(gap_med),
        "gap_lo_mm": float(gap_lo),
        "gap_hi_mm": float(gap_hi),
        "n_eval": int(n_eff)
    })

    # write per-ε subset CSV with a stable tag
    tag = eps_tag(eps)
    pd.DataFrame({
        "idx_test_local": eval_subset[m],
        "true_mm": y_test_sub_mm[m],
        "qhat_hi_mm": qhat_test_hi_sub_mm[m],
        "L_mm": L_eps[m]
    }).to_csv(OUTDIR / "source_data" / f"stress_subset_eps_{tag}.csv", index=False)

df_stress = pd.DataFrame(records)

# ---- Conditional coverage by novelty/family (read the files we just wrote) ----
X_train_full = df.loc[idx_train, elem_cols].to_numpy(float) if 'idx_train' in globals() else df.loc[:, elem_cols].to_numpy(float)
nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(X_train_full)
nov_L1_atpct = nn.kneighbors(X_test_elem_full, 1, return_distance=True)[0].ravel() * 100.0
nov_bins = pd.cut(nov_L1_atpct, bins=[-0.01, 0.5, 1.0, 2.0, np.inf], labels=["≤0.5","0.5–1.0","1–2",">2 at.%"])
fam = np.array(elem_cols)[np.argmax(X_test_elem_full, axis=1)]
cond_df = pd.DataFrame({"idx_test": np.arange(len(X_test_elem_full)), "nov_bin": nov_bins, "family": fam})

records_cond = []
for eps in EPS_LIST:
    tag = eps_tag(eps)
    one = pd.read_csv(OUTDIR / "source_data" / f"stress_subset_eps_{tag}.csv")
    j = one.merge(cond_df, left_on="idx_test_local", right_on="idx_test", how="left")
    for key in ["nov_bin", "family"]:
        for grp, g in j.groupby(key):
            if g.empty: continue
            cov = float(np.mean(g["true_mm"] >= g["L_mm"]))
            n   = int(len(g))
            records_cond.append({"eps": float(eps), "grouping": key, "group": str(grp), "coverage": cov, "n": n})

df_cond = pd.DataFrame(records_cond)
df_cond.to_csv(OUTDIR/"reports"/"coverage_conditional_by_group.csv", index=False)

# Coverage vs ε (TEST subset) with 95% CI
plt.figure(figsize=(6.4, 4.0))
x = df_stress["eps_atpct"].values
y = df_stress["coverage_mean"].values
lo = df_stress["coverage_lo"].values
hi = df_stress["coverage_hi"].values
plt.plot(x, y, marker="o", label="Observed coverage")
plt.fill_between(x, lo, hi, alpha=0.15)
plt.axhline(1-ALPHA, ls="--", color="k", label=f"Target {(1-ALPHA):.2f}")
plt.xlabel("Tolerance ε (total L1, at.%)")
plt.ylabel("Coverage (subset)")
plt.ylim(0, 1.0)
plt.title("Robust coverage vs tolerance ε")
stamp_info(plt.gca()) if "stamp_info" in globals() else None
savefig_bundle("stress_coverage_vs_eps",
               src_df=df_stress[["eps_atpct","coverage_mean","coverage_lo","coverage_hi","n_eval"]],
               meta={"alpha": float(ALPHA), "robust_samples": int(ROBUST_SAMPLES)})

# Coverage by novelty bin vs ε
plt.figure(figsize=(5.0,3.4))
for grp, g in df_cond.query("grouping=='nov_bin'").groupby("group"):
    g2 = g.sort_values("eps")
    plt.plot(g2["eps"], g2["coverage"], marker="o", label=f"nov={grp}")
plt.axhline(1-ALPHA, linestyle="--")
plt.xlabel("ε (L1 fraction)"); plt.ylabel("Empirical coverage")
plt.title("Conditional coverage by novelty bin (TEST)")
plt.legend(frameon=False, ncol=2); plt.tight_layout()
plt.savefig(OUTDIR/"reports"/"fig_coverage_by_novelty.png", dpi=300, bbox_inches="tight")
plt.close()

SRC_DIR = OUTDIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)

dplot = (df_cond.query("grouping=='nov_bin'")[["group", "eps", "coverage"]]
         .sort_values(["group", "eps"])
         .copy())

dplot.to_csv(SRC_DIR / "coverage_by_novelty_tidy.csv", index=False)
def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

for grp, g in dplot.groupby("group"):
    g.to_csv(SRC_DIR / f"coverage_by_novelty_group_{_slug(grp)}.csv", index=False)

eps_union = sorted(dplot["eps"].unique())
pd.DataFrame({"eps": eps_union, "target": [1 - ALPHA] * len(eps_union)}).to_csv(
    SRC_DIR / "coverage_by_novelty_target_line.csv", index=False
)

print("Saved:",
      SRC_DIR / "coverage_by_novelty_tidy.csv",
      "per-group CSVs in", SRC_DIR,
      "and", SRC_DIR / "coverage_by_novelty_target_line.csv")

# Sharpness (median gap) vs ε with 95% CI
plt.figure(figsize=(6.4, 4.0))
g = df_stress["gap_median_mm"].values
glo = df_stress["gap_lo_mm"].values
ghi = df_stress["gap_hi_mm"].values
plt.plot(x, g, marker="o", label="Median gap")
plt.fill_between(x, glo, ghi, alpha=0.15)
plt.xlabel("Tolerance ε (total L1, at.%)")
plt.ylabel("Median gap (mm)")
plt.title("Robust sharpness vs tolerance ε")
stamp_info(plt.gca()) if "stamp_info" in globals() else None
savefig_bundle("stress_sharpness_vs_eps",
               src_df=df_stress[["eps_atpct","gap_median_mm","gap_lo_mm","gap_hi_mm","n_eval"]],
               meta={"alpha": float(ALPHA), "robust_samples": int(ROBUST_SAMPLES)})

print("Saved stress curves + CSVs to:", OUTDIR / "reports" / "figures", "and", OUTDIR / "source_data")


# In[67]:


# Design board export (CSV + uncertainty + novelty color + meta)
if 'bo_df' in globals() and isinstance(bo_df, pd.DataFrame) and len(bo_df):
    design_df_src = bo_df.copy()
elif 'design_df_gt' in globals() and isinstance(design_df_gt, pd.DataFrame) and len(design_df_gt):
    design_df_src = design_df_gt.copy()
elif 'design_df_all' in globals() and isinstance(design_df_all, pd.DataFrame) and len(design_df_all):
    design_df_src = design_df_all.copy()
else:
    design_df_src = design_df.copy()

BOARD_DIR = OUTDIR / "data" / "designed"
SRC_DIR   = OUTDIR / "source_data"
for p in (BOARD_DIR, SRC_DIR, FIGDIR):
    p.mkdir(parents=True, exist_ok=True)

def _value_as_fraction(v):
    """Accepts either fraction (0–1) or at.% (>1). Returns fraction."""
    try:
        x = float(v)
    except Exception:
        return 0.0
    if np.isfinite(x) and x > 1.5:
        return x / 100.0
    return x

def row_to_full_fraction(row):
    """Build FULL fraction vector (len(elem_cols)) robustly from any schema."""
    x_full = np.zeros(len(elem_cols), dtype=float)
    basis = (allowed_elems_present if 'allowed_elems_present' in globals() else elem_cols)
    seen = set()
    for el in basis:
        if el in row:
            val = _value_as_fraction(row[el])
        elif f"frac_{el}" in row:
            val = _value_as_fraction(row[f"frac_{el}"])
        elif f"atpct_{el}" in row:
            val = _value_as_fraction(row[f"atpct_{el}"])
        elif f"at_{el}" in row:
            val = _value_as_fraction(row[f"at_{el}"])
        else:
            val = 0.0
        x_full[elem_cols.index(el)] = val
        seen.add(el)

    for el in elem_cols:
        if el in seen:
            continue
        if el in row:
            x_full[elem_cols.index(el)] = _value_as_fraction(row[el])
        elif f"atpct_{el}" in row:
            x_full[elem_cols.index(el)] = _value_as_fraction(row[f"atpct_{el}"])
        elif f"at_{el}" in row:
            x_full[elem_cols.index(el)] = _value_as_fraction(row[f"at_{el}"])
    s = x_full.sum()
    if s > 0:
        x_full /= s
    return x_full

def add_phys_props(df_in):
    """Attach δ-size mismatch and VEC (fractions on FULL vector)."""
    df = df_in.copy()
    deltas, vecs = [], []
    for _, r in df.iterrows():
        x_full = row_to_full_fraction(r)
        deltas.append(delta_size_mismatch(pd.Series(x_full, index=elem_cols)))
        vec_arr = np.array([VEC.get(el, np.nan) for el in elem_cols], float)
        mask = np.isfinite(vec_arr) & (x_full > 0)
        if np.any(mask):
            w = x_full[mask] / x_full[mask].sum()
            vecs.append(float(np.dot(w, vec_arr[mask])))
        else:
            vecs.append(np.nan)
    df["delta_size"] = deltas
    df["VEC"] = vecs
    return df

def ensure_uncertainty(df):
    """Compute 95% CI for L_robust if SE present; else fill NaNs for CI columns."""
    df = df.copy()
    if "L_robust_mm_se" in df.columns:
        df["L_robust_mm_lo"] = df["L_robust_mm"] - 1.96 * df["L_robust_mm_se"]
        df["L_robust_mm_hi"] = df["L_robust_mm"] + 1.96 * df["L_robust_mm_se"]
    else:
        df["L_robust_mm_lo"] = np.nan
        df["L_robust_mm_hi"] = np.nan
    return df

def _build_allowed_matrix(df):
    """Return matrix of allowed-element fractions per row (rows sum to 1)."""
    X = []
    for _, r in df.iterrows():
        x_full = row_to_full_fraction(r)
        if 'allowed_idx' in globals():
            x = x_full[allowed_idx]
        elif 'allowed_elems_present' in globals():
            idxs = [elem_cols.index(e) for e in allowed_elems_present]
            x = x_full[idxs]
        else:
            x = x_full
        s = x.sum()
        X.append(x / (s if s > 0 else 1.0))
    return np.vstack(X)

def _dist(a, b, metric="l2"):
    if metric == "l1":
        return float(np.sum(np.abs(a - b)))
    return float(np.linalg.norm(a - b))  # l2

def _farthest_point_diversity_impl(df, k=15, metric="l2", seed=None):
    """Greedy farthest-point sampling on composition vectors."""
    if len(df) == 0:
        return df
    k = int(min(k, len(df)))
    X = _build_allowed_matrix(df)

    if "L_robust_mm" in df.columns and df["L_robust_mm"].notna().any():
        start_idx = int(df["L_robust_mm"].fillna(-np.inf).values.argmax())
    else:
        start_idx = 0

    selected = [start_idx]

    def _min_dist_to_selected(idx):
        return min(_dist(X[idx], X[j], metric=metric) for j in selected)

    while len(selected) < k:
        cand_idx = None
        best = -1.0
        for i in range(len(df)):
            if i in selected:
                continue
            d = _min_dist_to_selected(i)
            if d > best:
                best, cand_idx = d, i
        if cand_idx is None:
            break
        selected.append(cand_idx)

    return df.iloc[selected].copy()

if 'farthest_point_diversity' in globals() and callable(farthest_point_diversity):
    try:
        sig = inspect.signature(farthest_point_diversity)
        if 'scale' not in sig.parameters:
            _orig_fpd = farthest_point_diversity
            def farthest_point_diversity(df, k=15, scale=1.0, prefer_allowed=None, metric="l2", seed=None):
                return _orig_fpd(df, k=k)
    except Exception:
        def farthest_point_diversity(df, k=15, scale=1.0, prefer_allowed=None, metric="l2", seed=None):
            return _farthest_point_diversity_impl(df, k=k, metric=metric, seed=seed)
else:
    def farthest_point_diversity(df, k=15, scale=1.0, prefer_allowed=None, metric="l2", seed=None):
        return _farthest_point_diversity_impl(df, k=k, metric=metric, seed=seed)

def shortlist_with_metrics(design_df, Dstar, k=15):
    """
    Risk-controlled shortlist + diversity (L1 in composition space).
    Adds δ, VEC, keeps those with L_robust ≥ D*, then farthest-point thins to k.
    """
    if "L_robust_mm" not in design_df.columns:
        tmp = design_df.copy()
        Xa = []
        for _, r in tmp.iterrows():
            vec = []
            for el in (allowed_elems_present if 'allowed_elems_present' in globals() else elem_cols):
                if f"frac_{el}" in r: vec.append(_value_as_fraction(r[f"frac_{el}"]))
                elif f"atpct_{el}" in r: vec.append(_value_as_fraction(r[f"atpct_{el}"]))
                elif el in r: vec.append(_value_as_fraction(r[el]))
                else: vec.append(0.0)
            v = np.asarray(vec, float); s = v.sum(); v = v/s if s>0 else np.ones_like(v)/len(v)
            Xa.append(v)
        Xa = np.vstack(Xa)
        Lvals = [eval_robust_L_allowed(x) for x in Xa]
        design_df = design_df.copy()
        design_df["L_robust_mm"] = Lvals

    df = add_phys_props(design_df)
    df = ensure_uncertainty(df)

    cand = df.loc[np.isfinite(df["L_robust_mm"]) & (df["L_robust_mm"] >= float(Dstar))].copy()
    if len(cand) == 0:
        return cand

    prefer_allowed = allowed_elems_present if 'allowed_elems_present' in globals() else None
    diverse = farthest_point_diversity(cand, k=k, scale=5.0, prefer_allowed=prefer_allowed, metric="l2")
    return diverse.sort_values("L_robust_mm", ascending=False).reset_index(drop=True)

def _save_meta_json(stem, meta):
    with open(SRC_DIR / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def _save_fig_and_csv(stem, df_src_plot):
    png = FIGDIR / f"{stem}.png"
    pdf = FIGDIR / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    try: plt.savefig(pdf, bbox_inches="tight")
    except Exception: pass
    plt.close()
    df_src_plot.to_csv(SRC_DIR / f"{stem}.csv", index=False)

BOARD_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

has_tail = any(c.startswith("q") and c.endswith("_mm") for c in design_df_src.columns)
has_novel = "min_L1_to_dataset_atpct" in design_df_src.columns
has_cost  = "cost_usd_per_kg" in design_df_src.columns

for Dstar in BOARD_THRESHOLDS:
    board = shortlist_with_metrics(design_df_src, Dstar, k=15)

    if board.empty:
        print(f"No designs met D* ≥ {int(Dstar)} mm — skipping plot.")
        continue

    keep_cols = []
    if has_tail:
        keep_cols += [c for c in board.columns if c.startswith("q") and c.endswith("_mm")]
        if "tail_slope_mm" in board.columns:
            keep_cols.append("tail_slope_mm")
    if has_novel:
        keep_cols.append("min_L1_to_dataset_atpct")
    if has_cost:
        keep_cols.append("cost_usd_per_kg")

    out_csv = BOARD_DIR / f"design_shortlist_{int(Dstar)}mm.csv"
    cols_export = [c for c in board.columns if c.startswith("atpct_") or c in [
        "L_robust_mm","L_robust_mm_lo","L_robust_mm_hi","L_robust_mm_se",
        "pred_point_mm","pred_qtau_mm","delta_size","VEC", *keep_cols
    ]]

    if not any(c.startswith("atpct_") for c in cols_export):
        cols_export += [c for c in board.columns if c.startswith("frac_")]
    board.loc[:, cols_export].to_csv(out_csv, index=False)

    vals = board["L_robust_mm"].to_numpy(float)
    lo   = board["L_robust_mm_lo"].to_numpy(float)
    hi   = board["L_robust_mm_hi"].to_numpy(float)
    has_ci = np.any(np.isfinite(lo)) and np.any(np.isfinite(hi))

    x = np.arange(len(board))
    plt.figure(figsize=(8.6, 4.2))
    if has_novel:
        nov = board["min_L1_to_dataset_atpct"].to_numpy(float)
        nov_norm = None
        if np.any(np.isfinite(nov)):
            a, b = float(np.nanmin(nov)), float(np.nanmax(nov))
            nov_norm = (nov - a) / max(1e-9, (b - a))
        colors = plt.cm.viridis(nov_norm) if nov_norm is not None else None
    else:
        colors = None

    plt.bar(x, vals, color=colors)
    if has_ci:
        yerr = np.vstack([np.maximum(0, vals - lo), np.maximum(0, hi - vals)])
        plt.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="k", elinewidth=1, capsize=3)

    plt.axhline(Dstar, ls="--", color="k", lw=1, label=f"D* = {int(Dstar)} mm")
    try:
        dmax_max_obs = float(df[dmax_col].max())
        plt.axhline(dmax_max_obs, ls=":", color="grey", lw=1, label=f"Dataset max = {dmax_max_obs:.1f} mm")
    except Exception:
        pass

    labels = [f"C{i+1}" for i in range(len(board))]
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Certified lower bound L (mm)")
    title = f"Design board (diverse top-{min(15, len(board))}), D* ≥ {int(Dstar)} mm"
    plt.title(title)
    plt.legend(frameon=False, loc="upper left")

    if "tail_slope_mm" in board.columns:
        for i in range(min(3, len(board))):
            ts = float(board["tail_slope_mm"].iloc[i])
            plt.text(i, vals[i] + 0.02*max(1, np.nanmax(vals)), f"Δq={ts:.1f}", ha="center", va="bottom", fontsize=9)

    stem = f"fig_design_board_{int(Dstar)}mm"
    df_fig_src = pd.DataFrame({
        "rank": np.arange(1, len(board)+1, dtype=int),
        "label": labels,
        "L_robust_mm": vals,
        "L_robust_mm_lo": lo,
        "L_robust_mm_hi": hi,
        **({"novelty_atpct": board["min_L1_to_dataset_atpct"].to_numpy(float)} if has_novel else {})
    })

    if "savefig_bundle" in globals():
        savefig_bundle(stem, src_df=df_fig_src,
                       meta={"alpha": float(ALPHA), "eps_L1": float(ROBUST_EPS),
                             "robust_samples": int(ROBUST_SAMPLES),
                             "tau_high": float(QT_TAU_HIGH) if "QT_TAU_HIGH" in globals() else None})
    else:
        _save_fig_and_csv(stem, df_fig_src)

    print("Saved:", out_csv, "and", FIGDIR / f"{stem}.png")

    if has_novel:
        plt.figure(figsize=(4.8, 4.0))
        xnov = board["min_L1_to_dataset_atpct"].to_numpy(float)
        yL   = board["L_robust_mm"].to_numpy(float)
        if "tail_slope_mm" in board.columns:
            cval = board["tail_slope_mm"].to_numpy(float)
            sc = plt.scatter(xnov, yL, c=cval)
            cb = plt.colorbar(sc); cb.set_label("Δq = q99 − q90 (mm)")
        else:
            plt.scatter(xnov, yL)
        plt.xlabel("Novelty to dataset (L1 at.%)")
        plt.ylabel("Certified L (mm)")
        plt.title(f"L vs Novelty (D* ≥ {int(Dstar)} mm shortlist)")
        stem2 = f"fig_board_scatter_novelty_vs_L_{int(Dstar)}mm"
        df_fig_src2 = pd.DataFrame({
            "novelty_atpct": xnov,
            "L_robust_mm": yL,
            **({"tail_slope_mm": board["tail_slope_mm"].to_numpy(float)} if "tail_slope_mm" in board.columns else {})
        })
        if "savefig_bundle" in globals():
            savefig_bundle(stem2, src_df=df_fig_src2,
                           meta={"note": "board scatter", "Dstar": float(Dstar)})
        else:
            _save_fig_and_csv(stem2, df_fig_src2)
        print("Saved:", FIGDIR / f"{stem2}.png")


# In[68]:


# Paper/SI manifest (auto-discovery, hashes, PDFs, CSVs, env, config)
FIGDIR   = OUTDIR / "reports" / "figures"
SRC_DIR  = OUTDIR / "source_data"
BOARD_DIR= OUTDIR / "data" / "designed"
REPDIR   = OUTDIR / "reports"
MODELDIR = OUTDIR / "models"
for p in (FIGDIR, SRC_DIR, BOARD_DIR, REPDIR, MODELDIR):
    p.mkdir(parents=True, exist_ok=True)

def _sha256(path: Path, chunk=1024*1024):
    """SHA256 of a file; returns None if missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None

def _exists(p): return Path(p).exists()

def _existing(paths):
    """De-dupe & keep only present files, preserving order."""
    seen, out = set(), []
    for p in map(Path, paths):
        if p.exists():
            s = str(p.resolve())
            if s not in seen:
                out.append(Path(s)); seen.add(s)
    return out

def _glob_many(root: Path, patterns):
    """Return sorted unique paths under root that match any of the patterns."""
    out = []
    for pat in patterns:
        out.extend(root.glob(pat))
    out = sorted(set(map(lambda p: p.resolve(), out)), key=lambda p: str(p))
    return [Path(p) for p in out]

def _pair_csv_for_fig(fig_path: Path):
    """Try to find a CSV with the same stem in SRC_DIR."""
    cand = SRC_DIR / (fig_path.stem + ".csv")
    return cand if cand.exists() else None

def _report_files(paths, header):
    print(header)
    for p in paths:
        size_kb = p.stat().st_size / 1024.0 if p.exists() else 0.0
        print(f"- {p}  [{'OK' if p.exists() else 'MISSING'}{'' if not p.exists() else f', {size_kb:.1f} KB'}]")

try:
    _THR = [float(x) for x in THRESHOLDS_MM]
except Exception:
    _THR = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

# Figures (core)
figs_core = [
    FIGDIR / "fig_reliability_coverage.png",
    FIGDIR / "fig_reliability_sharpness.png",
    FIGDIR / "fig_robust_coverage_vs_eps.png",
    FIGDIR / "fig_robust_sharpness_vs_eps.png",
    FIGDIR / "hi_tau_quantile_calibration.png",
    FIGDIR / "hi_tau_coverage_marg_vs_rob.png",
    FIGDIR / "hi_tau_sharpness_box.png",
    FIGDIR / "hi_tau_reliability_by_L.png",
    FIGDIR / "hi_tau_per_family_coverage.png",
]

for D in _THR:
    figs_core += [
        FIGDIR / f"fig_precision_k_{int(D)}mm.png",
        FIGDIR / f"fig_design_board_{int(D)}mm.png",
        FIGDIR / f"fig_board_scatter_novelty_vs_L_{int(D)}mm.png",
    ]

# Figures (extras discovered by pattern)
figs_extra = list(chain(
    _glob_many(FIGDIR, [
        "tornado_candidate_*.png",
        "ternary_cert_region_*.png",
        "fig_label_distribution.png",
        "fig_label_cdf.png",
        "fig_cov_by_true_bin_marg_robust.png",
        "fig_L_vs_true_scatter.png",
        "fig_shift_weight_hist.png",
        "fig_worstcase_drop_hist.png",
        "fig_learning_curve_*.png",
        "fig_coverage_by_ood.png",
        "fig_ablation_familyout_coverage.png",
    ])
))

figs_extra += [
    FIGDIR/"fig_riskmap_coverage.png",
    FIGDIR/"fig_riskmap_sharpness.png",
    FIGDIR/"fig_lift_vs_novelty.png",
    FIGDIR/"fig_tail_slope_hist.png",
    FIGDIR/"fig_tail_slope_vs_q99.png",
    FIGDIR/"fig_manifold_map.png",
    FIGDIR/"fig_bo_trace.png",
    FIGDIR/"fig_bo_diversity.png",
    FIGDIR/"fig_calibration_size_coverage.png",
    FIGDIR/"fig_stability_vs_K.png",
    FIGDIR/"fig_quantile_reliability_by_family.png",
    FIGDIR/"fig_recourse_path_ternary.png",
    FIGDIR/"fig_pareto_cost_robust_novelty.png",
]
figs_all = _existing(figs_core + figs_extra)

# Source-data CSVs (known names)
src_core = [
    SRC_DIR / "hi_tau_quantile_calibration_bar.csv",
    SRC_DIR / "hi_tau_coverage_bars.csv",
    SRC_DIR / "hi_tau_sharpness_gaps.csv",
    SRC_DIR / "hi_tau_reliability_by_bin.csv",
    SRC_DIR / "hi_tau_coverage_by_family.csv",
    SRC_DIR / "hi_tau_test_bounds.csv",
    SRC_DIR / "robust_bounds_test.csv",
    SRC_DIR / "robust_calibration_scores.csv",
    SRC_DIR / "coverage_vs_alpha.csv",
    SRC_DIR / "coverage_vs_epsilon.csv",
    SRC_DIR / "precision_at_k_all_methods.csv",
    SRC_DIR / "candidates_with_multi_tau.csv",
]

for D in _THR:
    src_core.append(SRC_DIR / f"precision_at_k_{int(D)}mm.csv")

src_extra = list(chain(
    _glob_many(SRC_DIR, [
        "tornado_candidate_*.csv",
        "ternary_cert_region_*.csv",
        "label_values.csv",
        "family_counts.csv",
        "cov_by_true_bin.csv",
        "L_vs_true_points.csv",
        "shift_weights_cal.csv",
        "shift_weight_stats.csv",
        "worstcase_drop_by_eps.csv",
        "learning_curve_metrics.csv",
        "coverage_by_ood.csv",
        "ablation_familyout.csv",
        "coverage_vs_alpha.csv",
        "coverage_vs_epsilon.csv",
        "hi_tau_precision_at_k.csv",
        "hi_tau_reliability_by_bin.csv",
        "hi_tau_coverage_by_family.csv",
        "hi_tau_test_ranking_by_Lrob.csv",
        "hi_tau_test_bounds.csv",
        "hi_tau_precision_at_k.csv",
    ])
))

src_extra += [
    SRC_DIR / "risk_map_alpha_eps.csv",
    SRC_DIR / "robust_lift_vs_novelty.csv",
    SRC_DIR / "test_multi_tau_tail.csv",
    SRC_DIR / "composition_embedding.csv",
    SRC_DIR / "bo_trace.csv",
    SRC_DIR / "bo_trace_from_memory.csv",
    SRC_DIR / "bo_pairwise_distances.csv",
    SRC_DIR / "calibration_size_sensitivity.csv",
    SRC_DIR / "stability_vs_K.csv",
    SRC_DIR / "quantile_reliability_by_family.csv",
    SRC_DIR / "recourse_path.csv",
    SRC_DIR / "pareto_cost_robust_novelty.csv",
    SRC_DIR / "perm_precision_at_k.csv",
    SRC_DIR / "heaping_stress_summary.csv",
    SRC_DIR / "coverage_per_family_random_all_methods.csv",
    SRC_DIR / "family_out_per_family_coverage_wilson_all_methods.csv",
    SRC_DIR / "family_out_summary_bootstrap_means.csv",
]


si_shortlists = [BOARD_DIR / f"design_shortlist_{int(D)}mm.csv" for D in _THR]
design_artifacts = list(chain(
    si_shortlists,
    _glob_many(BOARD_DIR, [
        "advanced_bo_pool.csv",
        "advanced_pour_list_ge_*mm.csv",
        "designed_candidates_allowed_all.csv",
        "designed_candidates_allowed_pred_ge_*mm.csv",
        "designed_candidates_allowed_gt_dataset_max.csv",
        "shortlist_Dstar_*_all.csv",
        "shortlist_Dstar_*_diverse_top15.csv",
    ])
))

csv_from_figs = []
for f in figs_all:
    c = _pair_csv_for_fig(f)
    if c is not None:
        csv_from_figs.append(c)

src_all = _existing(src_core + src_extra + csv_from_figs)
design_all = _existing(design_artifacts)

# Config / metrics / env / HPO / models
cfg_files = _existing([
    REPDIR / "metrics.json",
    REPDIR / "robust_cp_summary.json",
    REPDIR / "run_config.json",
    REPDIR / "manifest.json",
    REPDIR / "hpo_best_params_index.json",
    REPDIR / "env_requirements.txt",
    REPDIR / "conda_env.yaml",
    REPDIR / "data_hash.json",
])

model_files = _glob_many(MODELDIR, ["*.cbm", "*.joblib"])

_report_files(figs_all,   "\nKey figures for manuscript (core + extras):")
_report_files(src_all,    "\nSource-data CSVs (including auto-paired from figures):")
_report_files(design_all, "\nShortlists & design artifacts (CSV):")
_report_files(cfg_files,  "\nRun config, metrics, environment, HPO, and data-hash files:")
_report_files(model_files,"\nSaved model binaries:")

def _file_entry(p: Path):
    return {
        "path": str(p),
        "exists": p.exists(),
        "bytes": p.stat().st_size if p.exists() else None,
        "sha256": _sha256(p) if p.exists() else None,
    }

manifest = {
    "figures":    [_file_entry(p) for p in figs_all],
    "source_data":[_file_entry(p) for p in src_all],
    "design_csvs":[_file_entry(p) for p in design_all],
    "models":     [_file_entry(p) for p in model_files],
    "config":     [_file_entry(p) for p in cfg_files],
    "params": {
        "alpha": float(ALPHA),
        "robust_eps_fraction_L1": float(ROBUST_EPS),
        "qt_tau": float(QT_TAU),
        "qt_tau_high": float(QT_TAU_HIGH) if 'QT_TAU_HIGH' in globals() else None,
        "thresholds_mm": [float(x) for x in _THR],
        "seed": int(SEED),
        "n_figures": len(figs_all),
        "n_source_csv": len(src_all),
        "n_design_csv": len(design_all),
        "n_models": len(model_files),
    }
}
with open(REPDIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nWrote manifest with hashes to:", REPDIR / "manifest.json")

def _missing_pdf_buddies(figs):
    miss = []
    for p in figs:
        buddy = p.with_suffix(".pdf")
        if not buddy.exists():
            miss.append(buddy)
    return miss

missing_pdfs = _missing_pdf_buddies(figs_all)
if missing_pdfs:
    print("\n[Note] Some figures are missing PDF companions (journals often prefer PDFs):")
    for p in missing_pdfs:
        print("  -", p)

total_bytes = sum(e["bytes"] or 0 for e in chain(manifest["figures"],
                                                 manifest["source_data"],
                                                 manifest["design_csvs"],
                                                 manifest["models"],
                                                 manifest["config"]))
print(f"\nSummary: {manifest['params']['n_figures']} figs, "
      f"{manifest['params']['n_source_csv']} src CSVs, "
      f"{manifest['params']['n_design_csv']} design CSVs, "
      f"{manifest['params']['n_models']} model files | total ~{total_bytes/1e6:.2f} MB")


# In[69]:


# Label landscape (hist/ECDF, tail markers, split-aware)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True); SRC_DIR.mkdir(parents=True, exist_ok=True)

def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

Dcol = dmax_col  # measured Dmax column (mm)
THR  = [float(x) for x in (THRESHOLDS_MM if 'THRESHOLDS_MM' in globals() else [6,8,10,12,15])]

def _mk_split_col(df):
    n = len(df)
    split = np.full(n, "all", dtype=object)
    # Only tag splits if available
    if 'idx_train' in globals(): split[np.asarray(idx_train, int)] = "train"
    if 'idx_cal'   in globals(): split[np.asarray(idx_cal,   int)] = "cal"
    if 'idx_test'  in globals(): split[np.asarray(idx_test,  int)] = "test"
    return split

vals = pd.to_numeric(df[Dcol], errors="coerce").astype(float)
mask = np.isfinite(vals)
labels_df = pd.DataFrame({
    "value_mm": vals.where(mask).dropna(),
    "split": pd.Series(_mk_split_col(df))[mask].values,
    "family": df.loc[mask, "family"].astype(str).values
})
labels_df.to_csv(SRC_DIR/"label_values.csv", index=False)

# Quantiles & tail prevalences (overall + by split)
QPS = [0.5, 0.9, 0.95, 0.99]
def _quantiles(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(100*p)}": np.nan for p in QPS}
    return {f"q{int(100*p)}": float(np.quantile(x, p)) for p in QPS}

def _tail_prev(x, thr):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.mean(x >= thr)) if x.size else np.nan

# overall
quant_all = {"split": "all", **_quantiles(labels_df["value_mm"])}
prev_rows = [{"split": "all", "threshold_mm": t, "prevalence": _tail_prev(labels_df["value_mm"], t)} for t in THR]
# per split
quant_rows = [quant_all]
for sp, g in labels_df.groupby("split", dropna=False):
    quant_rows.append({"split": sp, **_quantiles(g["value_mm"])})
    for t in THR:
        prev_rows.append({"split": sp, "threshold_mm": t, "prevalence": _tail_prev(g["value_mm"], t)})

pd.DataFrame(quant_rows).to_csv(SRC_DIR/"label_quantiles_by_split.csv", index=False)
pd.DataFrame(prev_rows).to_csv(SRC_DIR/"label_tail_prevalence_by_split.csv", index=False)

# Family counts overall + by split
fam_counts_all = (labels_df
    .groupby("family", dropna=False)
    .size().reset_index(name="n")
    .sort_values("n", ascending=False))
fam_counts_all.to_csv(SRC_DIR/"family_counts.csv", index=False)

fam_counts_split = (labels_df
    .groupby(["split","family"], dropna=False)
    .size().reset_index(name="n")
    .sort_values(["split","n"], ascending=[True, False]))
fam_counts_split.to_csv(SRC_DIR/"family_counts_by_split.csv", index=False)

# Figure A: Histogram (all) with tail markers
vals_all = labels_df["value_mm"].to_numpy()
plt.figure(figsize=(6.2, 3.9))
bins = max(20, int(np.sqrt(max(1, np.isfinite(vals_all).sum()))))
plt.hist(vals_all[np.isfinite(vals_all)], bins=bins, alpha=0.85)
for t in THR:
    plt.axvline(t, ls="--", lw=1, label=f"D*={int(t)} mm")
plt.xlabel("Dmax (mm)"); plt.ylabel("Count")
plt.title("Dmax distribution (all)")
plt.legend(frameon=False, ncol=min(1+len(THR)//3, 3))
_figsave(FIGDIR/"fig_label_distribution.png")

# log-x view helps show the heavy tail
plt.figure(figsize=(6.2, 3.9))
plt.hist(vals_all[np.isfinite(vals_all)], bins=bins, alpha=0.85)
for t in THR:
    plt.axvline(t, ls="--", lw=1)
plt.xscale("log")
plt.xlabel("Dmax (mm, log scale)"); plt.ylabel("Count")
plt.title("Dmax distribution (all, log-x)")
_figsave(FIGDIR/"fig_label_distribution_logx.png")

# Figure B: ECDF by split with tail markers
def _ecdf(x):
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    y = np.linspace(0,1,len(x), endpoint=True)
    return x, y

plt.figure(figsize=(6.6, 4.0))
colors = {"train":"#1f77b4","cal":"#ff7f0e","test":"#2ca02c","all":"#7f7f7f"}

# plot per split present
for sp in ["train","cal","test"]:
    xx = labels_df.loc[labels_df["split"]==sp, "value_mm"].to_numpy()
    if np.isfinite(xx).any():
        xs, ys = _ecdf(xx)
        plt.plot(xs, ys, label=f"{sp} (n={np.isfinite(xx).sum()})", lw=2, color=colors.get(sp, None))

# overall (light)
xs, ys = _ecdf(vals_all)
plt.plot(xs, ys, label=f"all (n={np.isfinite(vals_all).sum()})", lw=1.5, color=colors["all"], alpha=0.6)
for t in THR:
    plt.axvline(t, ls="--", lw=1, color="k", alpha=0.7)
plt.xlabel("Dmax (mm)"); plt.ylabel("ECDF")
plt.title("Dmax ECDF by split")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR/"fig_label_cdf.png")

# Figure C: Tail prevalence by split (bar)
prev_df = pd.DataFrame(prev_rows)
splits_present = prev_df["split"].unique().tolist()
nT = len(THR)
ncols = 3
nrows = int(np.ceil(nT / ncols))
plt.figure(figsize=(max(6.0, 3.0*ncols), max(3.2, 2.6*nrows)))
for i, t in enumerate(THR, 1):
    ax = plt.subplot(nrows, ncols, i)
    sub = prev_df[prev_df["threshold_mm"] == t]
    xs  = ["train","cal","test","all"]
    xs  = [s for s in xs if s in splits_present]
    ys  = [float(sub.loc[sub["split"]==s, "prevalence"].values[0]) if (sub["split"]==s).any() else np.nan for s in xs]
    ax.bar(xs, ys)
    ax.set_ylim(0,1); ax.axhline(float(sub.loc[sub["split"]=="all","prevalence"].values[0]) if (sub["split"]=="all").any() else 0, ls=":", lw=1)
    ax.set_title(f"≥ {int(t)} mm"); 
    if i % ncols == 1: ax.set_ylabel("Prevalence")
plt.suptitle("Tail prevalence by split")
plt.tight_layout(rect=[0,0,1,0.96])
_figsave(FIGDIR/"fig_label_tail_prevalence_by_split.png")

print("Saved:",
      FIGDIR/"fig_label_distribution.png",
      FIGDIR/"fig_label_distribution_logx.png",
      FIGDIR/"fig_label_cdf.png",
      FIGDIR/"fig_label_tail_prevalence_by_split.png")
print("Saved CSVs:",
      SRC_DIR/"label_values.csv",
      SRC_DIR/"label_quantiles_by_split.csv",
      SRC_DIR/"label_tail_prevalence_by_split.csv",
      SRC_DIR/"family_counts.csv",
      SRC_DIR/"family_counts_by_split.csv")


# In[70]:


#  Coverage vs TRUE Dmax bins (marginal & robust, with CIs, NaN-safe)
y_true_mm = np.exp(y_test)
robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)" if 'ROBUST_EPS' in globals() else "Robust"

methods = {}
if 'L_marginal_mm' in globals(): methods["Marginal"] = L_marginal_mm
if 'L_robust_mm'   in globals(): methods[robust_label] = L_robust_mm
if not methods:
    if 'L_marginal_hi_mm' in globals(): methods["Marginal"] = L_marginal_hi_mm
    if 'L_robust_hi_mm'   in globals(): methods[robust_label] = L_robust_hi_mm
    if 'L_robust_hi_mm_test' in globals() and robust_label not in methods:
        methods[robust_label] = L_robust_hi_mm_test

if not methods:
    raise RuntimeError("No lower-bound arrays found for Step 12.7. Provide L_marginal_mm and/or L_robust_mm.")

def wilson_ci(k, n, z=1.96):
    """Wilson interval for binomial proportion; returns (p_hat, lo, hi)."""
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))

def make_bins_qcut(x, q=[0, .2, .4, .6, .8, 1.0]):
    """Quantile bins with duplicates='drop' fallback; returns intervals + centers."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None, None
    try:
        binned = pd.qcut(x, q=q, duplicates="drop")
        edges = np.unique(np.concatenate(([x.min()], [iv.right for iv in binned.cat.categories])))
        # centers from intervals
        centers = np.array([0.5*(edges[i] + edges[i+1]) for i in range(len(edges)-1)])
        return edges, centers, binned
    except Exception:
        qs = np.unique(np.quantile(x, q))
        if len(qs) < 2:
            return None, None, None
        centers = np.array([0.5*(qs[i] + qs[i+1]) for i in range(len(qs)-1)])
        return qs, centers, None

edges, centers, binned_ref = make_bins_qcut(y_true_mm, q=[0, .2, .4, .6, .8, 1.0])
if edges is None or len(edges) < 2:
    raise RuntimeError("Unable to construct coverage bins from y_true_mm.")

# Compute coverage + Wilson CIs in each bin for each method
rows = []

for name, L in methods.items():
    L_arr = np.asarray(L, float)
    y_arr = np.asarray(y_true_mm, float)
    m_all = np.isfinite(L_arr) & np.isfinite(y_arr)
    if not np.any(m_all):
        continue

    b_idx = np.digitize(y_arr, edges[1:-1], right=True)

    for bi in range(len(edges)-1):
        in_bin = (b_idx == bi) & m_all
        n = int(np.sum(in_bin))
        if n == 0:
            continue
        k = int(np.sum(y_arr[in_bin] >= L_arr[in_bin]))
        p_hat, lo, hi = wilson_ci(k, n)
        rows.append({
            "method": name,
            "bin": int(bi),
            "bin_lo": float(edges[bi]),
            "bin_hi": float(edges[bi+1]),
            "bin_center": float(centers[bi]),
            "n": n,
            "k": k,
            "coverage": float(p_hat),
            "ci_lo": float(lo),
            "ci_hi": float(hi)
        })

cov_by_true = pd.DataFrame(rows).sort_values(["method","bin"]).reset_index(drop=True)
cov_by_true.to_csv(SRC_DIR/"cov_by_true_bin.csv", index=False)

# Plot with error bars & target line
plt.figure(figsize=(6.6, 4.0))
xvals = cov_by_true["bin_center"].unique()
xspan = (np.nanmax(xvals) - np.nanmin(xvals)) if len(xvals) > 1 else 1.0
dx = 0.012 * xspan

palette = {list(methods.keys())[0]: "#1f77b4"}
if len(methods) > 1:
    palette[list(methods.keys())[1]] = "#ff7f0e"

for j, (name, d) in enumerate(cov_by_true.groupby("method", sort=False)):
    xs = d["bin_center"].to_numpy()
    ys = d["coverage"].to_numpy()
    lo = d["ci_lo"].to_numpy()
    hi = d["ci_hi"].to_numpy()
    err_low  = np.clip(ys - lo, 0, 1)
    err_high = np.clip(hi - ys, 0, 1)
    xoff = xs + (j - 0.5*(len(methods)-1))*dx
    plt.errorbar(xoff, ys, yerr=[err_low, err_high], fmt="-o", capsize=4,
                 label=name, lw=2, ms=5, color=palette.get(name, None))

plt.axhline(1-ALPHA, ls="--", lw=1.2, color="k", label=f"Target {1-ALPHA:.2f}")
plt.xlabel("True Dmax (mm) — bin centers")
plt.ylabel("Coverage within bin")
plt.ylim(0, 1)
plt.title("Coverage vs. true Dmax (test split)")
plt.legend(frameon=False, loc="lower right")
try:
    figsave(FIGDIR/"fig_cov_by_true_bin_marg_robust.png")
except NameError:
    plt.tight_layout(); plt.savefig(FIGDIR/"fig_cov_by_true_bin_marg_robust.png", dpi=300, bbox_inches="tight"); plt.close()

print("Saved:",
      FIGDIR/"fig_cov_by_true_bin_marg_robust.png",
      "and", SRC_DIR/"cov_by_true_bin.csv")


# In[71]:


# L (robust) vs truth; mark uncovered
robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)" if 'ROBUST_EPS' in globals() else "Robust"

if 'L_robust_mm' in globals():
    Lr_full = np.asarray(L_robust_mm, float)
elif 'L_robust_hi_mm_test' in globals():
    Lr_full = np.asarray(L_robust_hi_mm_test, float)
elif 'L_robust_hi_mm' in globals():
    Lr_full = np.asarray(L_robust_hi_mm, float)
else:
    raise RuntimeError("No robust lower bound array found (L_robust_mm / L_robust_hi_mm(_test)).")

y_true_full = np.exp(y_test).astype(float)
if 'qhat_test_hi' in globals():
    qhat_for_sharp_full = np.exp(qhat_test_hi).astype(float)
else:
    qhat_for_sharp_full = np.exp(qhat_test).astype(float)

mfin = np.isfinite(y_true_full) & np.isfinite(Lr_full) & np.isfinite(qhat_for_sharp_full)
y_true = y_true_full[mfin]
Lr     = Lr_full[mfin]
qhat_for_sharp = qhat_for_sharp_full[mfin]

covered = y_true >= Lr
coverage = float(np.mean(covered)) if len(y_true) else float('nan')
gap_sharp_med = float(np.median(qhat_for_sharp - Lr)) if len(y_true) else float('nan')
viol_mag_med = float(np.median((Lr - y_true)[~covered])) if np.any(~covered) else 0.0  # mm

src_df = pd.DataFrame({
    "index": np.array(idx_test)[mfin] if 'idx_test' in globals() else np.arange(len(y_true)),
    "family": df.loc[idx_test, "family"].to_numpy()[mfin] if 'idx_test' in globals() else None,
    "signature": df.loc[idx_test, "signature"].to_numpy()[mfin] if 'idx_test' in globals() and "signature" in df.columns else None,
    "true_mm": y_true,
    "L_robust_mm": Lr,
    "qhat_mm_for_sharpness": qhat_for_sharp,
    "covered": covered.astype(int),
    "margin_mm": (y_true - Lr),
    "violation_mm": np.maximum(0.0, Lr - y_true)
})
src_df.to_csv(SRC_DIR / "L_vs_true_points.csv", index=False)

plt.figure(figsize=(5.6, 5.0))

# Plot covered vs uncovered with different alphas
plt.scatter(Lr[ covered], y_true[ covered], s=18, alpha=0.60, label="covered")
plt.scatter(Lr[~covered], y_true[~covered], s=18, alpha=0.85, label="uncovered")

if len(y_true):
    lo = float(min(np.min(Lr), np.min(y_true)))
    hi = float(max(np.max(Lr), np.max(y_true)))
    # small padding
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lims = [max(0.0, lo - pad), hi + pad]
    plt.plot(lims, lims, ls="--", lw=1, color="k")
    plt.xlim(lims); plt.ylim(lims)

plt.gca().set_aspect('equal', 'box')
plt.xlabel("Certified lower bound  $L$  (mm)")
plt.ylabel("True $D_{\\max}$ (mm)")
plt.title(f"{robust_label} lower bound vs. truth (test)")

txt = (f"Coverage = {coverage:.3f}\n"
       f"Median sharpness gap = {gap_sharp_med:.2f} mm\n"
       f"Median violation (misses) = {viol_mag_med:.2f} mm")
plt.gca().text(0.02, 0.99, txt, transform=plt.gca().transAxes,
               ha="left", va="top", fontsize=9,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))

plt.legend(frameon=False, loc="upper left")

try:
    figsave(FIGDIR / "fig_L_vs_true_scatter.png")
except NameError:
    p = FIGDIR / "fig_L_vs_true_scatter.png"
    plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

print("Saved figure:", FIGDIR / "fig_L_vs_true_scatter.png")
print("Saved source:", SRC_DIR / "L_vs_true_points.csv")


# In[72]:


# Shift-weighted CP diagnostics (enhanced)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True); SRC_DIR.mkdir(parents=True, exist_ok=True)
def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight")
        try:
            plt.savefig(Path(p).with_suffix(".pdf"), bbox_inches="tight")
        except Exception:
            pass
        plt.close()

# Fit a CAL vs TEST classifier (logistic) on features X
X_cal_cls  = X.iloc[idx_cal].to_numpy()
X_test_cls = X.iloc[idx_test].to_numpy()
X_cls = np.vstack([X_cal_cls, X_test_cls])
y_cls = np.hstack([np.zeros(len(X_cal_cls)), np.ones(len(X_test_cls))])  # 0=CAL, 1=TEST

clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=None, random_state=SEED)
clf.fit(X_cls, y_cls)

proba_all = clf.predict_proba(X_cls)[:, 1]
auc = float(roc_auc_score(y_cls, proba_all))

# Density-ratio weights on CAL:  w(x) = p_test(x)/p_cal(x)
n_cal, n_test = len(X_cal_cls), len(X_test_cls)
ratio_prior = n_cal / max(1, n_test)  # π_cal / π_test
p_test_on_cal = clf.predict_proba(X_cal_cls)[:, 1].astype(float)
p_cal_on_cal  = 1.0 - p_test_on_cal
w_raw = ratio_prior * (p_test_on_cal / np.clip(p_cal_on_cal, 1e-12, None))

# Sanitize weights (finite, nonnegative) and optionally clip very extreme tails for plotting
w = np.nan_to_num(w_raw, nan=0.0, posinf=1e6, neginf=0.0)
w = np.clip(w, 0.0, 1e12)

def ess(v):
    v = np.asarray(v, float)
    s1 = v.sum(); s2 = np.dot(v, v)
    return float((s1 * s1) / (s2 + 1e-12)) if s1 > 0 else 0.0

caps = [None, 2, 5, 10, 20, 50]
ess_rows = []
for c in caps:
    wc = np.minimum(w, c) if c is not None else w
    ess_rows.append({
        "cap": ("none" if c is None else float(c)),
        "ESS": ess(wc),
        "sum_w": float(wc.sum()),
        "mean_w": float(wc.mean()),
        "max_w": float(wc.max()),
        "p99_w": float(np.quantile(wc, 0.99)),
        "p999_w": float(np.quantile(wc, 0.999)) if len(wc) >= 1000 else float(np.quantile(wc, 0.99)),
        "n_cal": int(n_cal),
        "n_test": int(n_test),
        "AUC_cal_vs_test": auc
    })
ess_df = pd.DataFrame(ess_rows)

pct = np.percentile(w, [50, 90, 95, 99, 99.9])
tail_summary = {
    "w_median": float(pct[0]),
    "w_p90": float(pct[1]),
    "w_p95": float(pct[2]),
    "w_p99": float(pct[3]),
    "w_p999": float(pct[4]),
    "max_w": float(w.max()),
    "ESS_none": float(ess_df.loc[ess_df["cap"]=="none","ESS"].iloc[0]),
    "n_cal": n_cal,
    "n_test": n_test,
    "AUC_cal_vs_test": auc
}

pd.DataFrame({"w": w}).to_csv(SRC_DIR / "shift_weights_cal.csv", index=False)
ess_df.to_csv(SRC_DIR / "shift_weight_stats.csv", index=False)
pd.DataFrame([tail_summary]).to_csv(SRC_DIR / "shift_weight_tail_summary.csv", index=False)

# Plots
# (a) Linear-scale histogram of w with ESS annotation
plt.figure(figsize=(6.4, 3.8))
plt.hist(w, bins=50, alpha=0.9)
plt.xlabel("Density-ratio weight  $w(x)=p_{test}(x)/p_{cal}(x)$")
plt.ylabel("Count")
plt.title(f"Shift-weighted CP — weights (AUC={auc:.2f}, ESS={tail_summary['ESS_none']:.1f} / n_cal={n_cal})")
# vertical markers for heavy-tail landmarks
for qval, lab in zip([pct[2], pct[3]], ["p95", "p99"]):
    plt.axvline(qval, ls=":", lw=1, label=f"{lab}={qval:.2f}")
plt.legend(frameon=False, loc="upper right")
_figsave(FIGDIR / "fig_shift_weight_hist.png")

# (b) Log10-scale histogram (reveals tail structure)
logw = np.log10(np.clip(w, 1e-12, None))
plt.figure(figsize=(6.4, 3.8))
plt.hist(logw, bins=50, alpha=0.9)
plt.xlabel(r"$\log_{10}(w)$")
plt.ylabel("Count")
plt.title("Shift weights — log scale")
plt.axvline(np.log10(pct[3]), ls=":", lw=1, label=f"log10 p99 = {np.log10(pct[3]):.2f}")
plt.legend(frameon=False, loc="upper right")
_figsave(FIGDIR / "fig_shift_weight_loghist.png")

# (c) ESS vs truncation cap
plt.figure(figsize=(6.4, 3.8))
xs = [0 if c is None else float(c) for c in caps]  # 0 marks "none"
ys = ess_df["ESS"].values
plt.plot(xs, ys, marker="o")
plt.xlabel("Truncation cap  (0 = none)")
plt.ylabel("Effective sample size (ESS)")
plt.title("ESS vs. truncation of weights")
for x, yv in zip(xs, ys):
    plt.text(x, yv, f"{yv:.0f}", ha="center", va="bottom", fontsize=8)
_figsave(FIGDIR / "fig_shift_weight_ess_vs_cap.png")

print("Saved:",
      FIGDIR / "fig_shift_weight_hist.png",
      FIGDIR / "fig_shift_weight_loghist.png",
      FIGDIR / "fig_shift_weight_ess_vs_cap.png")
print("Saved CSV:",
      SRC_DIR / "shift_weights_cal.csv",
      SRC_DIR / "shift_weight_stats.csv",
      SRC_DIR / "shift_weight_tail_summary.csv")


# In[73]:


# Worst-case drop Δ under composition jitter
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True); SRC_DIR.mkdir(parents=True, exist_ok=True)

def _figsave(png_path):
    try:
        figsave(png_path)
    except NameError:
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        try:
            plt.savefig(Path(png_path).with_suffix(".pdf"), bbox_inches="tight")
        except Exception:
            pass
        plt.close()

# Choose a quantile model (prefer high-τ)
_q_model = cat_qt_hi 

# Calibration compositions (fractions) & base predictions
X_cal_elem = df.loc[idx_cal, elem_cols].to_numpy(float)
row_sums = X_cal_elem.sum(axis=1, keepdims=True)
X_cal_elem = np.divide(X_cal_elem, np.where(row_sums > 0, row_sums, 1.0))

feats_cal_base = make_features_from_compositions(X_cal_elem)
q_base_log = np.asarray(_q_model.predict(feats_cal_base), float)
q_base_mm  = np.exp(q_base_log)

def _min_logq_under_jitter(model, X_elem, eps, K, rng=None, chunk=4096):
    """
    For each x in X_elem: draw K jitters in an L1-ball (radius=eps), predict log-quantiles,
    return per-sample min (worst case). Chunked to avoid memory blow-ups.
    """
    if rng is None:
        rng = np.random.default_rng(SEED + 1010)
    n = X_elem.shape[0]
    out = np.full(n, np.inf, float)
    buf_X, buf_id, buf_count = [], [], 0

    def _flush():
        nonlocal out, buf_X, buf_id, buf_count
        if not buf_X: return
        Xj = np.vstack(buf_X)
        feats = make_features_from_compositions(Xj)
        preds = np.asarray(model.predict(feats), float)
        start = 0
        for sid, kcnt in buf_id:
            block = preds[start:start+kcnt]
            out[sid] = min(out[sid], float(np.nanmin(block)))
            start += kcnt
        buf_X.clear(); buf_id.clear(); buf_count = 0

    if eps <= 0.0 or K <= 1:
        feats0 = make_features_from_compositions(X_elem)
        return np.asarray(model.predict(feats0), float)

    for i, x in enumerate(X_elem):
        rng_i = np.random.default_rng(SEED + DRIFT_SEED)
        Xj = jitter_in_L1_ball_simplex(x, eps=float(eps), K=int(K), rng=rng_i)
        rs = Xj.sum(axis=1, keepdims=True)
        Xj = np.divide(Xj, np.where(rs > 0, rs, 1.0))
        buf_X.append(Xj); buf_id.append((i, K)); buf_count += K
        if buf_count >= chunk:
            _flush()
    _flush()

    bad = ~np.isfinite(out)
    if np.any(bad):
        feats0 = make_features_from_compositions(X_elem[bad])
        out[bad] = np.asarray(model.predict(feats0), float)
    return out

eps_list = [0.00, 0.01, 0.02]
rng_local = np.random.default_rng(SEED + 123)
drop_rows = []

for eps in eps_list:
    q_min_log = _min_logq_under_jitter(_q_model, X_cal_elem, eps=eps, K=ROBUST_SAMPLES, rng=rng_local)
    delta_log = q_base_log - q_min_log
    delta_mm  = q_base_mm - np.exp(q_min_log)

    # Clean numerics
    delta_log = np.nan_to_num(delta_log, nan=0.0, posinf=0.0, neginf=0.0)
    delta_mm  = np.nan_to_num(delta_mm,  nan=0.0, posinf=0.0, neginf=0.0)
    delta_log[delta_log < 0] = 0.0
    delta_mm[delta_mm   < 0] = 0.0

    drop_rows.append(pd.DataFrame({
        "index": df.index[idx_cal],
        "eps_fraction": eps,
        "delta_log": delta_log.astype(float),
        "delta_mm":  delta_mm.astype(float),
    }))

df_drop = pd.concat(drop_rows, ignore_index=True)
df_drop.to_csv(SRC_DIR / "worstcase_drop_by_eps.csv", index=False)

def _summ(d):
    return {
        "n": int(len(d)),
        "mean": float(np.mean(d)) if len(d) else np.nan,
        "median": float(np.median(d)) if len(d) else np.nan,
        "p90": float(np.quantile(d, 0.90)) if len(d) else np.nan,
        "p95": float(np.quantile(d, 0.95)) if len(d) else np.nan,
        "p99": float(np.quantile(d, 0.99)) if len(d) else np.nan,
        "max": float(np.max(d)) if len(d) else np.nan,
    }

sum_rows = []
for eps, g in df_drop.groupby("eps_fraction"):
    s_log = _summ(g["delta_log"].to_numpy())
    s_mm  = _summ(g["delta_mm"].to_numpy())
    sum_rows.append({
        "eps_fraction": float(eps),
        **{f"log_{k}": v for k, v in s_log.items()},
        **{f"mm_{k}":  v for k, v in s_mm.items()},
    })
pd.DataFrame(sum_rows).to_csv(SRC_DIR / "worstcase_drop_summary.csv", index=False)

# (1) Histograms in mm (stacked by ε)
plt.figure(figsize=(6.6, 3.8))
for eps in eps_list:
    mm_vals = df_drop.loc[df_drop["eps_fraction"] == eps, "delta_mm"].to_numpy()
    plt.hist(mm_vals, bins=40, alpha=0.55, label=f"ε={int(eps*100)} at.%")
plt.xlabel(r"Worst-case drop  $\Delta_{mm} = e^{q_\tau(x)} - e^{\min q_\tau(x')}$  (mm)")
plt.ylabel("Count")
plt.title("Worst-case mm drop under composition jitter (CAL)")
plt.legend(frameon=False)
_figsave(FIGDIR / "fig_worstcase_drop_hist_mm.png")

# (2) Violin plot (Δ_mm by ε)
plt.figure(figsize=(6.4, 3.8))
data = [df_drop.loc[df_drop["eps_fraction"] == eps, "delta_mm"].to_numpy() for eps in eps_list]
plt.violinplot(data, showextrema=True, showmeans=False)
plt.xticks(range(1, len(eps_list)+1), [f"{int(e*100)} at.%" for e in eps_list])
plt.ylabel("Worst-case drop Δ (mm)")
plt.title("Worst-case mm drop vs tolerance ε")
_figsave(FIGDIR / "fig_worstcase_drop_violin_mm.png")

# (3) CDFs (mm)
plt.figure(figsize=(6.4, 3.8))
for eps in eps_list:
    v = np.sort(df_drop.loc[df_drop["eps_fraction"] == eps, "delta_mm"].to_numpy())
    if len(v):
        y = np.linspace(0, 1, len(v), endpoint=True)
        plt.plot(v, y, label=f"ε={int(eps*100)} at.%")
plt.xlabel("Worst-case drop Δ (mm)")
plt.ylabel("CDF")
plt.title("CDF of worst-case mm drop by ε")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR / "fig_worstcase_drop_cdf_mm.png")

print("Saved:",
      FIGDIR / "fig_worstcase_drop_hist_mm.png,",
      FIGDIR / "fig_worstcase_drop_violin_mm.png,",
      FIGDIR / "fig_worstcase_drop_cdf_mm.png")
print("Saved CSV:",
      SRC_DIR / "worstcase_drop_by_eps.csv,",
      SRC_DIR / "worstcase_drop_summary.csv")


# In[76]:


# Back-compat wrapper to ensure we can set the seed for replicates
def _mk_quantile_estimator(tau, seed):
    """
    Returns a quantile regressor with the requested seed, even if an older
    make_quantile_estimator(tau) (no seed kwarg) is already defined in globals.
    """
    try:
        # Try the current API first
        return mk_quantile_estimator(tau, seed=seed)
    except TypeError as e:
        if "unexpected keyword argument 'seed'" not in str(e):
            raise  # some other error
        # Older API: build a model directly so we still control the seed
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                loss_function=f"Quantile:alpha={float(tau)}",
                eval_metric=f"Quantile:alpha={float(tau)}",
                random_seed=int(seed),
                verbose=False, allow_writing_files=False, thread_count=-1
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                loss="quantile", alpha=float(tau),
                n_estimators=600, learning_rate=0.05, max_depth=3,
                random_state=int(seed)
            )


# Learning curves with replicates: coverage, precision@20, sharpness  (fixed)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True); SRC_DIR.mkdir(parents=True, exist_ok=True)

def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

# Safe fallbacks / factories
if 'weighted_quantile' not in globals():
    def weighted_quantile(values, q, sample_weight=None):
        v = np.asarray(values, float)
        return float(np.quantile(v, q))

# Robust score (lower one-sided): S_i = max(0, min_j q̂τ(x'_ij) − y_i) in LOG scale
def _robust_L_for_test(model, X_elem, q_cal, eps, K, rng):
    """Certified L on TEST for a fitted model at tolerance eps."""
    L_mm = np.zeros(len(X_elem))
    for i, x in enumerate(X_elem):
        rng_i = np.random.default_rng(SEED + DRIFT_SEED)
        Xj = jitter_in_L1_ball_simplex(x, eps=float(eps), K=int(K), rng=rng_i)
        rs = Xj.sum(axis=1, keepdims=True)
        Xj = np.divide(Xj, np.where(rs > 0, rs, 1.0))
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), float)  # log-quantiles
        L_mm[i] = float(np.exp(np.min(qj) - q_cal))
    return L_mm

# ---- Metrics helpers ----
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
R_REP     = 7
DSTAR     = 5.0
TAU_USED  = (QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU)
ROB_LABEL = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"

def _coverage(y_mm, L_mm):
    m = np.isfinite(y_mm) & np.isfinite(L_mm)
    return float(np.mean(y_mm[m] >= L_mm[m])) if np.any(m) else np.nan

def _sharpness(qhat_mm, L_mm):
    m = np.isfinite(qhat_mm) & np.isfinite(L_mm)
    return float(np.median(qhat_mm[m] - L_mm[m])) if np.any(m) else np.nan

def _precision_at_k(L_mm, y_mm, thresh, k=20):
    L = np.asarray(L_mm, float); Y = np.asarray(y_mm, float)
    m = np.isfinite(L) & np.isfinite(Y)
    if not np.any(m): return np.nan
    order = np.argsort(-L[m])
    k_eff = min(k, order.size)
    if k_eff == 0: return np.nan
    sel = order[:k_eff]
    return float(np.mean(Y[m][sel] >= float(thresh)))

# ---- Data slices ----
X_train_df = X.iloc[idx_train]
X_cal_df   = X.iloc[idx_cal]
X_test_df  = X.iloc[idx_test]
y_train    = y_log[idx_train]
y_cal_log  = y_log[idx_cal]
y_test_log = y_log[idx_test]
y_test_mm  = np.exp(y_test_log)

X_cal_elem  = df.loc[idx_cal,  elem_cols].to_numpy(float)
X_test_elem = df.loc[idx_test, elem_cols].to_numpy(float)
def _row_norm(A):
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)
X_cal_elem  = _row_norm(X_cal_elem)
X_test_elem = _row_norm(X_test_elem)

# Learning curves
rows_rep = []
for frac in FRACTIONS:
    tr_idx_full = np.array(idx_train)
    n_sub = max(20, int(len(tr_idx_full) * frac))

    for r in range(R_REP):
        rng_rep = np.random.default_rng(SEED + 7700 + int(1000*frac) + r)
        tr_sub = rng_rep.choice(tr_idx_full, size=n_sub, replace=False)

        # fresh τ-quantile model for this replicate
        model = _mk_quantile_estimator(TAU_USED, seed=SEED + 123 + r)
        model.fit(X.iloc[tr_sub], y_log[tr_sub])

        # Marginal CP on CAL
        q_cal = model.predict(X_cal_df)
        S_cal = np.maximum(0.0, q_cal - y_cal_log)
        q_marg = weighted_quantile(S_cal, 1 - ALPHA)

        # Robust CP on CAL (one-sided lower)
        S_cal_rob = robust_scores_lower(
            y_true_log=y_cal_log, model=model, X_elem=X_cal_elem,
            eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_rep
        )
        S_cal_rob = np.maximum(0.0, np.nan_to_num(S_cal_rob, nan=0.0))
        q_rob = weighted_quantile(S_cal_rob, 1 - ALPHA)

        # TEST predictions and certified bounds
        q_te = model.predict(X_test_df)
        qhat_mm = np.exp(q_te)
        L_marg = np.exp(q_te - q_marg)
        L_rob  = _robust_L_for_test(model, X_test_elem, q_rob, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_rep)

        # Metrics
        cov_m = _coverage(y_test_mm, L_marg)
        cov_r = _coverage(y_test_mm, L_rob)
        shp_m = _sharpness(qhat_mm, L_marg)
        shp_r = _sharpness(qhat_mm, L_rob)
        p20   = _precision_at_k(L_rob, y_test_mm, DSTAR, k=20)

        rows_rep.append({
            "frac_train": float(frac),
            "rep": int(r),
            "n_train_sub": int(n_sub),
            "coverage_marginal": cov_m,
            "coverage_robust":   cov_r,
            "sharpness_marginal_mm": shp_m,
            "sharpness_robust_mm":   shp_r,
            "precision20_at_15mm":   p20
        })

df_lc_rep = pd.DataFrame(rows_rep)
df_lc_rep.to_csv(SRC_DIR / "learning_curve_metrics_replicates.csv", index=False)

# Aggregate (mean ± percentile CI over replicates)
def _mean_ci(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if x.size == 0: return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    lo   = float(np.percentile(x, 2.5))
    hi   = float(np.percentile(x, 97.5))
    return mean, lo, hi

sum_rows = []
for frac, g in df_lc_rep.groupby("frac_train"):
    cm_m, cm_lo, cm_hi = _mean_ci(g["coverage_marginal"])
    cr_m, cr_lo, cr_hi = _mean_ci(g["coverage_robust"])
    sm_m, sm_lo, sm_hi = _mean_ci(g["sharpness_marginal_mm"])
    sr_m, sr_lo, sr_hi = _mean_ci(g["sharpness_robust_mm"])
    p20_m, p20_lo, p20_hi = _mean_ci(g["precision20_at_15mm"])
    sum_rows.append({
        "frac_train": float(frac), "n_train_sub": int(g["n_train_sub"].iloc[0]),
        "coverage_marginal_mean": cm_m, "coverage_marginal_lo": cm_lo, "coverage_marginal_hi": cm_hi,
        "coverage_robust_mean":   cr_m, "coverage_robust_lo":   cr_lo, "coverage_robust_hi":   cr_hi,
        "sharpness_marginal_mm_mean": sm_m, "sharpness_marginal_mm_lo": sm_lo, "sharpness_marginal_mm_hi": sm_hi,
        "sharpness_robust_mm_mean":   sr_m, "sharpness_robust_mm_lo":   sr_lo, "sharpness_robust_mm_hi":   sr_hi,
        "precision20_at_15mm_mean":   p20_m, "precision20_at_15mm_lo":   p20_lo, "precision20_at_15mm_hi":   p20_hi,
    })

df_lc = pd.DataFrame(sum_rows).sort_values("frac_train")
df_lc.to_csv(SRC_DIR / "learning_curve_metrics_summary.csv", index=False)

# Plots

# Coverage
plt.figure(figsize=(6.6, 3.9))
x = df_lc["frac_train"].to_numpy()
plt.errorbar(x, df_lc["coverage_marginal_mean"],
             yerr=[df_lc["coverage_marginal_mean"]-df_lc["coverage_marginal_lo"],
                   df_lc["coverage_marginal_hi"]-df_lc["coverage_marginal_mean"]],
             marker="o", capsize=4, label="Marginal")
plt.errorbar(x, df_lc["coverage_robust_mean"],
             yerr=[df_lc["coverage_robust_mean"]-df_lc["coverage_robust_lo"],
                   df_lc["coverage_robust_hi"]-df_lc["coverage_robust_mean"]],
             marker="o", capsize=4, label=ROB_LABEL)
plt.axhline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
plt.xlabel("Fraction of training data used"); plt.ylabel("Coverage (test)")
plt.ylim(0, 1); plt.title("Learning curve — coverage (mean ± 95% CI)")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR / "fig_learning_curve_coverage.png")

# Precision @ 20 (≥30 mm)
plt.figure(figsize=(8, 6))
plt.errorbar(x, df_lc["precision20_at_15mm_mean"],
             yerr=[df_lc["precision20_at_15mm_mean"]-df_lc["precision20_at_15mm_lo"],
                   df_lc["precision20_at_15mm_hi"]-df_lc["precision20_at_15mm_mean"]],
             marker="o", capsize=4)
base = float(np.mean(np.exp(y_test_log) >= DSTAR))
plt.axhline(base, ls="--", lw=1, label=f"Baseline prevalence={base:.2f}")
plt.xlabel("Fraction of training data used"); plt.ylabel("Precision@20 (≥ 15 mm)")
plt.ylim(0, 1); plt.title("Learning curve — precision@20 (robust ranking, mean ± 95% CI)")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR / "fig_learning_curve_precision15.png")

# Sharpness
plt.figure(figsize=(8, 6))
plt.errorbar(x, df_lc["sharpness_marginal_mm_mean"],
             yerr=[df_lc["sharpness_marginal_mm_mean"]-df_lc["sharpness_marginal_mm_lo"],
                   df_lc["sharpness_marginal_mm_hi"]-df_lc["sharpness_marginal_mm_mean"]],
             marker="o", capsize=4, label="Marginal")
plt.errorbar(x, df_lc["sharpness_robust_mm_mean"],
             yerr=[df_lc["sharpness_robust_mm_mean"]-df_lc["sharpness_robust_mm_lo"],
                   df_lc["sharpness_robust_mm_hi"]-df_lc["sharpness_robust_mm_mean"]],
             marker="o", capsize=4, label=ROB_LABEL)
plt.xlabel("Fraction of training data used"); plt.ylabel("Median gap (mm)")
plt.title("Learning curve — sharpness (mean ± 95% CI)")
plt.legend(frameon=False, loc="upper right")
_figsave(FIGDIR / "fig_learning_curve_sharpness.png")

print("Saved summary:", SRC_DIR / "learning_curve_metrics_summary.csv")
print("Saved replicates:", SRC_DIR / "learning_curve_metrics_replicates.csv")
print("Saved figures:",
      FIGDIR / "fig_learning_curve_coverage.png,",
      FIGDIR / "fig_learning_curve_precision15.png,",
      FIGDIR / "fig_learning_curve_sharpness.png")


# In[80]:


# Coverage vs OOD (kNN distance in composition space), with CI + multiple metrics
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True); SRC_DIR.mkdir(parents=True, exist_ok=True)

def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

Lm = globals().get('L_marginal_mm', globals().get('L_marginal_hi_mm', None))
Lr = globals().get('L_robust_mm',  globals().get('L_robust_hi_mm_test', None))
if Lm is None or Lr is None:
    raise RuntimeError("Need L_marginal_mm (or _hi_) and L_robust_mm (or _hi_mm_test) defined before Step 12.12.")

def _row_norm(A):
    A = np.asarray(A, float)
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)

def _clr(X, eps=1e-8):
    """Centered log-ratio for compositional geometry; X rows ~ fractions."""
    X = np.asarray(X, float)
    X = _row_norm(X)
    X = np.clip(X, eps, 1.0)
    logX = np.log(X)
    gm = logX.mean(axis=1, keepdims=True)
    return logX - gm

def _knn_ood(train, test, k=5, metric="l2"):
    """Mean distance to k-NN in chosen space."""
    if metric == "clr":
        Ztr, Zte = _clr(train), _clr(test)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Ztr)
        d, _ = nn.kneighbors(Zte, n_neighbors=k, return_distance=True)
        return d.mean(axis=1)
    elif metric == "l1":
        nn = NearestNeighbors(n_neighbors=k, metric="manhattan").fit(train)
        d, _ = nn.kneighbors(test, n_neighbors=k, return_distance=True)
        return d.mean(axis=1)
    else:
        nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(train)
        d, _ = nn.kneighbors(test, n_neighbors=k, return_distance=True)
        return d.mean(axis=1)

def _wilson_ci(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))

X_train_elem = _row_norm(df.loc[idx_train, elem_cols].to_numpy(dtype=float))
X_test_elem  = _row_norm(df.loc[idx_test,  elem_cols].to_numpy(dtype=float))

OOD_METRICS = ["clr", "l1", "l2"]
K_LIST      = [3, 5, 10]

all_rows = []
for metric in OOD_METRICS:
    for k in K_LIST:
        ood = _knn_ood(X_train_elem, X_test_elem, k=k, metric=metric)

        edges = np.quantile(ood, np.linspace(0, 1, 6))
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([ood.min(), ood.max() + 1e-12])
        edges[0] -= 1e-12; edges[-1] += 1e-12

        bin_id = np.digitize(ood, edges[1:-1], right=True)
        for bi in range(edges.size - 1):
            m = (bin_id == bi)
            n = int(m.sum())
            if n == 0:
                continue

            km = int(np.sum(np.asarray(y_test_mm[m] >= np.asarray(Lm)[m], dtype=int)))
            kr = int(np.sum(np.asarray(y_test_mm[m] >= np.asarray(Lr)[m], dtype=int)))
            pm, pm_lo, pm_hi = _wilson_ci(km, n)
            pr, pr_lo, pr_hi = _wilson_ci(kr, n)

            gap_r = (np.asarray(Lr)[m] * 0 + np.exp(qhat_test_hi[m]) - np.asarray(Lr)[m]) \
                    if 'qhat_test_hi' in globals() else \
                    (np.exp(qhat_test[m]) - np.asarray(Lr)[m])
            gap_med = float(np.nanmedian(gap_r))
            all_rows.append({
                "metric": metric, "k": int(k),
                "bin": int(bi),
                "ood_lo": float(edges[bi]),
                "ood_hi": float(edges[bi+1]),
                "ood_center": float(0.5*(edges[bi] + edges[bi+1])),
                "n": n,
                "coverage_marginal": pm,  "coverage_marginal_lo": pm_lo, "coverage_marginal_hi": pm_hi,
                "coverage_robust":   pr,  "coverage_robust_lo":   pr_lo, "coverage_robust_hi":   pr_hi,
                "gap_robust_median_mm": gap_med
            })

df_ood = pd.DataFrame(all_rows)
df_ood.to_csv(SRC_DIR / "coverage_by_ood.csv", index=False)

# Plot: Coverage vs OOD with 95% CIs (for the primary metric/neighbor setting)
PRIMARY_METRIC = "clr"
PRIMARY_K = 5
dplot = df_ood[(df_ood["metric"] == PRIMARY_METRIC) & (df_ood["k"] == PRIMARY_K)].copy()
dplot = dplot.sort_values("ood_center")

plt.figure(figsize=(6.6, 3.9))

# Marginal
plt.errorbar(dplot["ood_center"], dplot["coverage_marginal"],
             yerr=[dplot["coverage_marginal"] - dplot["coverage_marginal_lo"],
                   dplot["coverage_marginal_hi"] - dplot["coverage_marginal"]],
             marker="o", capsize=4, label="Marginal")

# Robust
robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"
plt.errorbar(dplot["ood_center"], dplot["coverage_robust"],
             yerr=[dplot["coverage_robust"] - dplot["coverage_robust_lo"],
                   dplot["coverage_robust_hi"] - dplot["coverage_robust"]],
             marker="o", capsize=4, label=robust_label)

plt.axhline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
plt.xlabel(f"OOD score — mean {PRIMARY_K}-NN distance ({'Aitchison/CLR' if PRIMARY_METRIC=='clr' else PRIMARY_METRIC.upper()})")
plt.ylabel("Coverage"); plt.ylim(0, 1)
plt.title("Coverage vs OOD (test, with 95% Wilson CI)")
plt.legend(frameon=False, loc="lower left")
_figsave(FIGDIR / f"fig_coverage_by_ood_{PRIMARY_METRIC}_k{PRIMARY_K}.png")

ood_primary = _knn_ood(X_train_elem, X_test_elem, k=PRIMARY_K, metric=PRIMARY_METRIC)
q_mm = np.exp(qhat_test_hi) if 'qhat_test_hi' in globals() else np.exp(qhat_test)
gap_r_all = q_mm - np.asarray(Lr)

plt.figure(figsize=(6.6, 3.9))
plt.scatter(ood_primary, gap_r_all, s=12, alpha=0.35, label="candidates")
edges = np.quantile(ood_primary, np.linspace(0, 1, 9))
edges = np.unique(edges); edges[0]-=1e-12; edges[-1]+=1e-12
bin_id = np.digitize(ood_primary, edges[1:-1], right=True)
centers, medians = [], []
for bi in range(edges.size - 1):
    m = (bin_id == bi)
    if m.sum() == 0: continue
    centers.append(0.5*(edges[bi] + edges[bi+1]))
    medians.append(float(np.nanmedian(gap_r_all[m])))
plt.plot(centers, medians, marker="o", lw=2, label="binned median")
plt.xlabel(f"OOD score ({'Aitchison/CLR' if PRIMARY_METRIC=='clr' else PRIMARY_METRIC.upper()})")
plt.ylabel("Robust sharpness gap (mm)")
plt.title("Sharpness vs OOD (lower is better)")
plt.legend(frameon=False)
_figsave(FIGDIR / f"fig_gap_vs_ood_{PRIMARY_METRIC}_k{PRIMARY_K}.png")

print("Saved:",
      FIGDIR / f"fig_coverage_by_ood_{PRIMARY_METRIC}_k{PRIMARY_K}.png",
      "and", FIGDIR / f"fig_gap_vs_ood_{PRIMARY_METRIC}_k{PRIMARY_K}.png")
print("Saved CSV:", SRC_DIR / "coverage_by_ood.csv")


# In[81]:


# Minimal export for the Sharpness vs OOD scatter
n = len(ood_primary)

# Try to use your test IDs; fall back to 0..n-1 if lengths don't match
try:
    test_index = df.loc[idx_test].index
    ids = test_index.astype(str) if len(test_index) == n else pd.Index(range(n)).astype(str)
except Exception:
    ids = pd.Index(range(n)).astype(str)

pd.DataFrame({
    "id": ids,
    "ood": np.asarray(ood_primary, float),
    "gap_robust_mm": np.asarray(gap_r_all, float),
}).to_csv(SRC_DIR / "sharpness_vs_ood_scatter.csv", index=False)

print("Saved:", SRC_DIR / "sharpness_vs_ood_scatter.csv")


# In[82]:


# Family-out ablation: auto-compute if missing, then summary (coverage bars, deltas) + CSVs
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True); SRC_DIR.mkdir(parents=True, exist_ok=True)

def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

# Small safe fallbacks
if 'weighted_quantile' not in globals():
    def weighted_quantile(values, q, sample_weight=None):
        v = np.asarray(values, float)
        return float(np.quantile(v, q))

if 'mk_quantile_estimator' not in globals():
    def mk_quantile_estimator(tau, seed=None):
        try:
            return CatBoostRegressor(
                loss_function=f"Quantile:alpha={float(tau)}",
                eval_metric=f"Quantile:alpha={float(tau)}",
                random_seed=int(SEED if seed is None else seed),
                verbose=False, allow_writing_files=False, thread_count=-1
            )
        except Exception:
            return GradientBoostingRegressor(
                loss="quantile", alpha=float(tau),
                n_estimators=600, learning_rate=0.05, max_depth=3,
                random_state=int(SEED if seed is None else seed)
            )

# Simplex jitter in an L1-ball around x (fallback), K samples
def _jitter_in_L1_ball_simplex(x, eps, K, rng):
    return sample_drift_neighborhood(x, eps, K, rng, include_x=True)

# One-sided robust score for lower bound: S_i = max(0, min_j q̂τ(x'_ij) − y_i) in log-space
def _robust_L_for_test(model, X_elem, q_cal, eps, K, rng):
    L_mm = np.zeros(len(X_elem))
    for i, x in enumerate(X_elem):
        Xj = _jitter_in_L1_ball_simplex(x, eps=eps, K=K, rng=rng)
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), float)  # log-quantiles
        L_mm[i] = float(np.exp(np.min(qj) - q_cal))
    return L_mm

def _row_norm(A):
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)

# ---------- Auto-compute family_out_metrics.csv if missing ----------
fo_path = OUTDIR / "reports" / "family_out_metrics.csv"
need_compute = not fo_path.exists()

if need_compute:
    if "family" not in df.columns:
        raise RuntimeError("df has no 'family' column; cannot run family-out ablation.")

    tau_used = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU
    fams = list(pd.Series(df["family"]).dropna().unique())

    X_all = X
    y_all = y_log
    comp_all = df.loc[:, elem_cols].to_numpy(float)
    comp_all = _row_norm(comp_all)

    rows = []
    for fam in sorted(fams):
        idx_te = np.where(df["family"].values == fam)[0]
        idx_rest = np.setdiff1d(np.arange(len(df)), idx_te)

        # deterministic per-family split of "rest" into train/cal (80/20)
        fam_seed = (int(hashlib.sha1(str(fam).encode()).hexdigest(), 16) % 10**9) + SEED
        rng = np.random.default_rng(fam_seed)
        idx_shuf = idx_rest.copy()
        rng.shuffle(idx_shuf)
        n_tr = int(0.80 * len(idx_shuf))
        idx_tr = idx_shuf[:n_tr]
        idx_ca = idx_shuf[n_tr:]

        # train model
        model = _mk_quantile_estimator(TAU_USED, seed=SEED + 123 + r)
        model.fit(X_all.iloc[idx_tr], y_all[idx_tr])

        # marginal CP (calibration on idx_ca)
        q_ca = model.predict(X_all.iloc[idx_ca])
        S_ca = np.maximum(0.0, q_ca - y_all[idx_ca])
        q_marg = weighted_quantile(S_ca, 1 - ALPHA)

        # robust CP (calibration on idx_ca)
        X_ca_elem = comp_all[idx_ca]
        S_ca_rob = robust_scores_lower(
            y_true_log=y_all[idx_ca], model=model, X_elem=X_ca_elem,
            eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng
        )
        q_rob = weighted_quantile(S_ca_rob, 1 - ALPHA)

        # evaluate on held-out family (idx_te)
        y_te_mm = np.exp(y_all[idx_te])
        q_te = model.predict(X_all.iloc[idx_te])
        L_marg = np.exp(q_te - q_marg)

        X_te_elem = comp_all[idx_te]
        L_rob  = _robust_L_for_test(model, X_te_elem, q_cal=q_rob, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng)

        cov_m = float(np.mean(y_te_mm >= L_marg)) if len(y_te_mm) else np.nan
        cov_r = float(np.mean(y_te_mm >= L_rob))  if len(y_te_mm) else np.nan

        rows.append({
            "family": fam,
            "n_test": int(len(idx_te)),
            "cov_marginal": cov_m,
            "cov_robust":   cov_r
        })

    df_fo = pd.DataFrame(rows)
    df_fo.to_csv(fo_path, index=False)
    print(f"[family-out] Computed and saved: {fo_path}")
else:
    df_fo = pd.read_csv(fo_path)
    print(f"[family-out] Loaded: {fo_path}")

# Summary + figures (your original intent)
robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"
order = [
    ("cov_marginal", "Marginal"),
    ("cov_group", "Group-cond."),
    ("cov_weighted", "Shift-weighted"),
    ("cov_robust", robust_label),
]
present = [(col, lab) for col, lab in order if col in df_fo.columns]
if not present:
    raise RuntimeError("No coverage columns found in df_fo. Expected columns like 'cov_marginal', 'cov_robust', etc.")

w_col = "n_test" if "n_test" in df_fo.columns else ("n" if "n" in df_fo.columns else None)

# Bootstrap helper (paired across families)
def _bootstrap_mean(x, B=2000, rng=None):
    rng = np.random.default_rng(SEED+123) if rng is None else rng
    x = np.asarray(x, float)
    if x.size == 0: return np.nan, np.nan, np.nan
    idx = np.arange(x.size)
    stats = np.empty(B, float)
    for b in range(B):
        res = rng.choice(idx, size=x.size, replace=True)
        stats[b] = float(np.nanmean(x[res]))
    return float(np.nanmean(x)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

# Build summary table (per method)
rows = []
for col, lab in present:
    x = pd.to_numeric(df_fo[col], errors="coerce").values
    mean_, lo_, hi_ = _bootstrap_mean(x, B=2000)
    row = {"method": lab, "mean": mean_, "ci_lo": lo_, "ci_hi": hi_, "n_families": int(np.sum(np.isfinite(x)))}
    if w_col is not None:
        w = pd.to_numeric(df_fo[w_col], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        row["weighted_mean"] = float(np.average(x[m], weights=w[m])) if np.any(m) else np.nan
        row["total_n"] = int(np.sum(w[m])) if np.any(m) else int(np.sum(m))
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(SRC_DIR / "ablation_familyout_summary.csv", index=False)

keep_cols = ["family"] if "family" in df_fo.columns else []
keep_cols += [c for c, _ in present]
if w_col is not None: keep_cols += [w_col]
df_fo[keep_cols].to_csv(SRC_DIR / "ablation_familyout_per_family.csv", index=False)

# Mean coverage bars with 95% CIs + per-family dots
x = np.arange(len(summary))
means = summary["mean"].values
err_low  = means - summary["ci_lo"].values
err_high = summary["ci_hi"].values - means

plt.figure(figsize=(6.6, 3.9))
plt.bar(x, means, alpha=0.85)
plt.errorbar(x, means, yerr=[err_low, err_high], fmt="none", capsize=4, color="k", lw=1)

# Overlay the per-family points (jitter)
jitter = (np.random.default_rng(SEED+456).random(len(df_fo),) - 0.5) * 0.15
for i, (col, lab) in enumerate(present):
    vals = pd.to_numeric(df_fo[col], errors="coerce").values
    xs = np.full_like(vals, i, dtype=float) + jitter
    plt.scatter(xs, vals, s=14, alpha=0.45, zorder=3)

plt.axhline(1-ALPHA, ls="--", label=f"Target {1-ALPHA:.2f}")
plt.xticks(x, summary["method"], rotation=15)
plt.ylim(0, 1)
plt.ylabel("Mean family-out coverage")
ttl_weight = f" | weighted mean shown in CSV" if w_col is not None else ""
plt.title(f"Family-out coverage by CP variant{ttl_weight}")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR / "fig_ablation_familyout_coverage.png")

# Paired deltas: robust − marginal per family (if both present)
if any(c == "cov_marginal" for c, _ in present) and any(c == "cov_robust" for c, _ in present):
    dm = pd.to_numeric(df_fo["cov_marginal"], errors="coerce").values
    dr = pd.to_numeric(df_fo["cov_robust"],  errors="coerce").values
    mask = np.isfinite(dm) & np.isfinite(dr)
    delta = dr[mask] - dm[mask]

    # save per-family deltas
    pd.DataFrame({
        "family": (df_fo["family"][mask].values if "family" in df_fo.columns else np.arange(np.sum(mask))),
        "delta_robust_minus_marginal": delta
    }).to_csv(SRC_DIR / "ablation_familyout_paired_deltas.csv", index=False)

    # boxplot with mean±95% CI in title
    d_mean, d_lo, d_hi = _bootstrap_mean(delta, B=2000)
    plt.figure(figsize=(6.2, 3.6))
    plt.boxplot([delta], labels=[f"{robust_label} − Marginal"], showfliers=False)
    plt.axhline(0.0, color="k", lw=1, ls=":")
    plt.ylabel("Δ coverage (absolute)")
    plt.title(f"Paired family deltas: mean={d_mean:.3f} (95% CI {d_lo:.3f}–{d_hi:.3f})")
    _figsave(FIGDIR / "fig_ablation_familyout_deltas.png")
else:
    print("[info] Skipping paired deltas: need both 'cov_robust' and 'cov_marginal' in df_fo.")

print("Saved:",
      FIGDIR / "fig_ablation_familyout_coverage.png",
      "and (if available) fig_ablation_familyout_deltas.png")
print("Saved CSVs:",
      SRC_DIR / "ablation_familyout_summary.csv", ",",
      SRC_DIR / "ablation_familyout_per_family.csv",
      "and (if available) ablation_familyout_paired_deltas.csv")


# In[83]:


# Step 12.14 — α–ε risk map (fixed: use proper features + allowed-subset jitter/embedding)
FIGDIR = OUTDIR / "reports" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUTDIR / "source_data";       SRC_DIR.mkdir(parents=True, exist_ok=True)

alpha_grid = [0.05, 0.10, 0.20]
eps_grid   = [0.00, 0.01, 0.02]
rng_grid   = np.random.default_rng(SEED + 1400)

# ---- Safety fallback: robust_scores_lower if missing
if "robust_scores_lower" not in globals():
    def robust_scores_lower(y_cal_log, q_model, X_cal_elem_full, eps, K, rng):
        """Return robust one-sided residuals on CAL: S_i^rob = max(0, min_j q̂τ(x'_ij) - y_i)."""
        scores = np.empty(len(X_cal_elem_full), float)
        for i, x_full in enumerate(X_cal_elem_full):
            x_allowed = x_full[allowed_idx]                        # allowed-subset view
            Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
            Xj_full    = embed_allowed_to_full(Xj_allowed)         # back to full element space
            feats      = make_features_from_compositions(Xj_full)  # build model features
            qj_log     = np.asarray(q_model.predict(feats), float)
            scores[i]  = max(0.0, float(np.min(qj_log)) - float(y_cal_log[i]))
        return scores

# ---- TEST / CAL compositions (fractions) and features
X_test_elem = df.loc[idx_test, elem_cols].to_numpy(float)
X_test_elem /= np.where(X_test_elem.sum(axis=1, keepdims=True) > 0, X_test_elem.sum(axis=1, keepdims=True), 1.0)
fe_test = make_features_from_compositions(X_test_elem)

X_cal_elem = df.loc[idx_cal, elem_cols].to_numpy(float)
X_cal_elem /= np.where(X_cal_elem.sum(axis=1, keepdims=True) > 0, X_cal_elem.sum(axis=1, keepdims=True), 1.0)
fe_cal = make_features_from_compositions(X_cal_elem)

# Labels (log and mm)
y_test_log = y_log[idx_test] if 'y_log' in globals() else y_test  # ensure log(mm)
y_test_mm  = np.exp(y_test_log)

# High-τ predictions (log → mm) for gap metric
qhat_test_hi_log = np.asarray(cat_qt_hi.predict(fe_test), float)
qhat_hi_mm       = np.exp(qhat_test_hi_log)

def _robust_min_logq_test(q_model, X_elem_full, eps, K, rng):
    """min_j q̂τ(x'_j) (log-space) per TEST point, using allowed-subset jitter, then embed-to-full + features."""
    out = np.full(len(X_elem_full), np.inf, float)
    for i, x_full in enumerate(X_elem_full):
        x_allowed  = x_full[allowed_idx]
        Xj_allowed = jitter_allowed_simplex(x_allowed, eps=float(eps), K=int(K), rng=rng)
        Xj_full    = embed_allowed_to_full(Xj_allowed)
        feats      = make_features_from_compositions(Xj_full)
        qj_log     = np.asarray(q_model.predict(feats), float)
        out[i]     = float(np.min(qj_log))
    return out

def _robust_calibrate_and_eval(alpha, eps):
    """Return (coverage, median gap) at (alpha, eps) on TEST; calibration is split conformal on CAL."""
    # --- CAL side quantile for subtraction
    if eps == 0.0:
        q_ca_log = np.asarray(cat_qt_hi.predict(fe_cal), float)
        S = np.maximum(0.0, q_ca_log - np.asarray(y_log)[idx_cal])
    else:
        S = robust_scores_lower(np.asarray(y_log)[idx_cal], cat_qt_hi, X_cal_elem,
                                eps=float(eps), K=int(ROBUST_SAMPLES), rng=rng_grid)
    q_alpha = weighted_quantile(S, 1 - alpha) if "weighted_quantile" in globals() else np.quantile(S, 1 - alpha)

    # --- TEST side lower bound under eps
    if eps == 0.0:
        q_te_log = qhat_test_hi_log
        L = np.exp(q_te_log - q_alpha)
    else:
        rng_te = np.random.default_rng(SEED + int(20000 * (1 + 1000*eps)))
        qmin_log = _robust_min_logq_test(cat_qt_hi, X_test_elem, eps=float(eps),
                                         K=int(ROBUST_SAMPLES), rng=rng_te)
        L = np.exp(qmin_log - q_alpha)

    m  = np.isfinite(L) & np.isfinite(y_test_mm)
    cov = float(np.mean(y_test_mm[m] >= L[m])) if np.any(m) else np.nan
    gap = float(np.median(qhat_hi_mm[m] - L[m])) if np.any(m) else np.nan
    return cov, gap

# ---- Sweep α × ε
rows = []
for a in alpha_grid:
    for e in eps_grid:
        cov, gap = _robust_calibrate_and_eval(a, e)
        rows.append({"alpha": float(a), "eps_fraction": float(e), "coverage": cov, "median_gap_mm": gap})

df_risk = pd.DataFrame(rows)
df_risk.to_csv(SRC_DIR / "risk_map_alpha_eps.csv", index=False)

# ---- Heatmaps
pivot_cov = df_risk.pivot(index="alpha", columns="eps_fraction", values="coverage").sort_index(ascending=False)
pivot_gap = df_risk.pivot(index="alpha", columns="eps_fraction", values="median_gap_mm").sort_index(ascending=False)

plt.figure(figsize=(6,4))
plt.imshow(pivot_cov.values, aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(pivot_cov.columns)), [f"{100*e:.0f}" for e in pivot_cov.columns]); plt.xlabel("ε (at.% total L1)")
plt.yticks(range(len(pivot_cov.index)),   [f"{int((1-a)*100)}%" for a in pivot_cov.index]); plt.ylabel("Coverage target")
plt.title("Coverage heatmap vs α & ε"); plt.colorbar(label="Coverage")
plt.tight_layout(); plt.savefig(FIGDIR / "fig_riskmap_coverage.png", dpi=300, bbox_inches="tight"); plt.close()

plt.figure(figsize=(6,4))
plt.imshow(pivot_gap.values, aspect="auto")
plt.xticks(range(len(pivot_gap.columns)), [f"{100*e:.0f}" for e in pivot_gap.columns]); plt.xlabel("ε (at.% total L1)")
plt.yticks(range(len(pivot_gap.index)),   [f"{int((1-a)*100)}%" for a in pivot_gap.index]); plt.ylabel("Coverage target")
plt.title("Sharpness (median gap, mm) vs α & ε"); plt.colorbar(label="Median gap (mm)")
plt.tight_layout(); plt.savefig(FIGDIR / "fig_riskmap_sharpness.png", dpi=300, bbox_inches="tight"); plt.close()

print("Saved risk maps:", FIGDIR / "fig_riskmap_coverage.png", "and", FIGDIR / "fig_riskmap_sharpness.png")


# In[84]:


# lIFT VS. NOVELTY
elem = elem_cols

# Split matrices
X_tr = df.loc[idx_train, elem].to_numpy(float)
X_ca = df.loc[idx_cal,   elem].to_numpy(float)
X_te = df.loc[idx_test,  elem].to_numpy(float)

X_ref = np.vstack([X_tr, X_ca])

D = pairwise_distances(X_te, X_ref, metric="manhattan")
nov_te = D.min(axis=1) * 100.0

# Robust lift
Lr = np.asarray(L_robust_mm, float)
Lm = np.asarray(L_marginal_mm, float)
lift = Lr - Lm

m = np.isfinite(lift) & np.isfinite(nov_te)
df_lift = pd.DataFrame({"novelty_atpct": nov_te[m], "lift_mm": lift[m]})
df_lift.to_csv(SRC_DIR/"lift_vs_novelty_points.csv", index=False)


# In[85]:


B = 5
edges = np.quantile(df_lift.novelty_atpct, np.linspace(0,1,B+1))
edges[0] -= 1e-9  # include leftmost point
bins = pd.cut(df_lift.novelty_atpct, edges, include_lowest=True)

def bca_ci(x, B=2000):
    a = np.array(x)
    boot = np.array([np.mean(np.random.choice(a, size=len(a), replace=True)) for _ in range(B)])
    return np.percentile(boot, [2.5, 97.5])

rows=[]
for i, g in df_lift.groupby(bins):
    nov_lo, nov_hi = i.left, i.right
    nov_ctr = (nov_lo+nov_hi)/2
    mean_lift = g.lift_mm.mean()
    lo95, hi95 = bca_ci(g.lift_mm)
    rows.append(dict(bin=str(i), nov_lo=nov_lo, nov_hi=nov_hi, nov_center=nov_ctr,
                     n=len(g), mean_lift_mm=mean_lift, lo95=lo95, hi95=hi95))
df_bins = pd.DataFrame(rows)
df_bins.to_csv(SRC_DIR/"lift_vs_novelty_bins.csv", index=False)

# (Optional) trend stats for caption
from scipy.stats import spearmanr
rho, p = spearmanr(df_lift.novelty_atpct, df_lift.lift_mm)
print("Spearman rho =", rho, "p =", p)


# In[86]:


# Step 12.16 — Tail-shape atlas (multi-τ on TEST)
TAUS = [0.80, 0.90, 0.95, 0.99]
MODELS_DIR = OUTDIR / "models"; MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _fit_or_load_catboost_quantile(tau, base=None):
    path = MODELS_DIR / f"cat_qt_tau{tau:.2f}.cbm"
    if path.exists():
        m = CatBoostRegressor(); m.load_model(str(path)); return m
    params = (cat_qt_hi.get_params().copy() if 'cat_qt_hi' in globals() else {})
    params.update({"loss_function": f"Quantile:alpha={tau}", "eval_metric": f"Quantile:alpha={tau}",
                   "random_seed": SEED, "verbose": False})
    m = CatBoostRegressor(**params)
    m.fit(X.iloc[idx_train], y_log[idx_train])
    m.save_model(str(path)); return m

qt_grid = {}
for t in TAUS:
    qt_grid[t] = _fit_or_load_catboost_quantile(t)

fe = X.iloc[idx_test]
preds = {}
for t in TAUS:
    preds[t] = np.exp(np.asarray(qt_grid[t].predict(fe), float))

df_tail = pd.DataFrame({"q80_mm": preds[0.80], "q90_mm": preds[0.90],
                        "q95_mm": preds[0.95], "q99_mm": preds[0.99]})
df_tail["tail_slope_mm"] = df_tail["q99_mm"] - df_tail["q90_mm"]
df_tail.to_csv(SRC_DIR/"test_multi_tau_tail.csv", index=False)

plt.figure(figsize=(6.2,3.8))
plt.hist(df_tail["tail_slope_mm"], bins=40, alpha=0.85)
plt.xlabel("Δq (mm) = q₀.₉₈ − q₀.₉₀"); plt.ylabel("Count")
plt.title("Tail slope (heavy-tail indicator) on TEST")
figsave(FIGDIR/"fig_tail_slope_hist.png")

plt.figure(figsize=(6.2,3.8))
plt.scatter(df_tail["q99_mm"], df_tail["tail_slope_mm"], s=16, alpha=0.6)
plt.xlabel("q₀.₉₈ (mm)"); plt.ylabel("Δq (mm)")
plt.title("Heavy-tail atlas: Δq vs q₀.₉₈")
figsave(FIGDIR/"fig_tail_slope_vs_q99.png")


# In[87]:


# Step 12.17 — Composition manifold map with certifications
X_frac = df.loc[:, elem_cols].to_numpy(float)
mask_train = np.isin(np.arange(len(df)), idx_train)
mask_test  = np.isin(np.arange(len(df)), idx_test)

# Embed
try:
    
    emb = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="euclidean", random_state=SEED).fit_transform(X_frac)
except Exception:
    emb = PCA(n_components=2, random_state=SEED).fit_transform(X_frac)

df_emb = pd.DataFrame({"x": emb[:,0], "y": emb[:,1],
                       "split": np.where(mask_train, "train", np.where(mask_test, "test", "other"))})

# color TEST by L_robust
df_emb["L_robust_mm"] = np.nan
df_emb.loc[mask_test, "L_robust_mm"] = np.asarray(L_robust_mm, float)

# optional: overlay designed candidates if available
try:
    pool = bo_df if ('bo_df' in globals() and len(bo_df)) else design_df_all
    X_cand = embed_allowed_to_full(_cand_allowed_matrix(pool))
    try:
        emb_cand = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="euclidean", random_state=SEED).fit(X_frac).transform(X_cand)
    except Exception:
        emb_cand = PCA(n_components=2, random_state=SEED).fit(X_frac).transform(X_cand)
except Exception:
    emb_cand = None

df_emb.to_csv(SRC_DIR/"composition_embedding.csv", index=False)

plt.figure(figsize=(6.2,5.4))
mtrain = (df_emb["split"]=="train"); mtest=(df_emb["split"]=="test")
plt.scatter(df_emb.loc[mtrain,"x"], df_emb.loc[mtrain,"y"], s=6, alpha=0.25, label="train")
sc = plt.scatter(df_emb.loc[mtest,"x"], df_emb.loc[mtest,"y"], c=df_emb.loc[mtest,"L_robust_mm"], s=10, alpha=0.8, label="test")
if emb_cand is not None:
    plt.scatter(emb_cand[:,0], emb_cand[:,1], marker="*", s=80, edgecolor="k", facecolor="none", label="designed")
cb = plt.colorbar(sc); cb.set_label("L_robust (mm)")
plt.legend(frameon=False, loc="best")
plt.title("Composition manifold map")
figsave(FIGDIR/"fig_manifold_map.png")


# In[88]:


SRC_DIR = Path(SRC_DIR)
SRC_DIR.mkdir(parents=True, exist_ok=True)

# 1) Combined file (all points)
df_emb_out = df_emb.copy()
df_emb_out.insert(0, "row_id", np.arange(len(df_emb_out)))  # useful for back-referencing
df_emb_out.to_csv(SRC_DIR / "composition_embedding_all.csv", index=False, float_format="%.7g")

# 2) Train-only (scatter layer 1 in Origin)
df_train = df_emb_out.loc[df_emb_out["split"] == "train", ["row_id", "x", "y"]]
df_train.to_csv(SRC_DIR / "composition_embedding_train.csv", index=False, float_format="%.7g")

# 3) Test-only (scatter layer 2 in Origin; color-map by L_robust_mm)
df_test = df_emb_out.loc[df_emb_out["split"] == "test", ["row_id", "x", "y", "L_robust_mm"]].dropna(subset=["L_robust_mm"])
df_test.to_csv(SRC_DIR / "composition_embedding_test.csv", index=False, float_format="%.7g")

# (Optional) Store colorbar limits so you can match the matplotlib scale in Origin, if you want.
if len(df_test):
    cmin, cmax = float(df_test["L_robust_mm"].min()), float(df_test["L_robust_mm"].max())
    pd.DataFrame({"cmin": [cmin], "cmax": [cmax]}).to_csv(SRC_DIR / "composition_embedding_colorbar_limits.csv",
                                                          index=False, float_format="%.7g")

# 4) Designed candidates (scatter layer 3 in Origin; star marker)
if "emb_cand" in globals() and emb_cand is not None and len(emb_cand):
    df_cand = pd.DataFrame({"x": np.asarray(emb_cand)[:, 0], "y": np.asarray(emb_cand)[:, 1]})
    df_cand.to_csv(SRC_DIR / "composition_embedding_candidates.csv", index=False, float_format="%.7g")
else:
    # No candidates available; skip file
    pass


# In[89]:


# Step 12.18 — Tail-shape atlas (multi-τ on TEST)
TAUS = [0.80, 0.90, 0.95, 0.99]
MODELS_DIR = OUTDIR / "models"; MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _fit_or_load_catboost_quantile(tau, base=None):
    path = MODELS_DIR / f"cat_qt_tau{tau:.2f}.cbm"
    if path.exists():
        m = CatBoostRegressor(); m.load_model(str(path)); return m
    params = (cat_qt_hi.get_params().copy() if 'cat_qt_hi' in globals() else {})
    params.update({"loss_function": f"Quantile:alpha={tau}", "eval_metric": f"Quantile:alpha={tau}",
                   "random_seed": SEED, "verbose": False})
    m = CatBoostRegressor(**params)
    m.fit(X.iloc[idx_train], y_log[idx_train])
    m.save_model(str(path)); return m

qt_grid = {}
for t in TAUS:
    qt_grid[t] = _fit_or_load_catboost_quantile(t)

fe = X.iloc[idx_test]
preds = {}
for t in TAUS:
    preds[t] = np.exp(np.asarray(qt_grid[t].predict(fe), float))

df_tail = pd.DataFrame({"q80_mm": preds[0.80], "q90_mm": preds[0.90],
                        "q95_mm": preds[0.95], "q99_mm": preds[0.99]})
df_tail["tail_slope_mm"] = df_tail["q99_mm"] - df_tail["q90_mm"]
df_tail.to_csv(SRC_DIR/"test_multi_tau_tail.csv", index=False)

plt.figure(figsize=(6.2,3.8))
plt.hist(df_tail["tail_slope_mm"], bins=40, alpha=0.85)
plt.xlabel("Δq (mm) = q₀.₉₈ − q₀.₉₀"); plt.ylabel("Count")
plt.title("Tail slope (heavy-tail indicator) on TEST")
figsave(FIGDIR/"fig_tail_slope_hist.png")

plt.figure(figsize=(6.2,3.8))
plt.scatter(df_tail["q99_mm"], df_tail["tail_slope_mm"], s=16, alpha=0.6)
plt.xlabel("q₀.₉₈ (mm)"); plt.ylabel("Δq (mm)")
plt.title("Heavy-tail atlas: Δq vs q₀.₉₈")
figsave(FIGDIR/"fig_tail_slope_vs_q99.png")


# In[90]:


# Step 12.19 — Calibration size sensitivity
fracs = [0.25, 0.5, 0.75, 1.0]
rng_cz = np.random.default_rng(SEED+1219)
y_mm = np.exp(y_test)
qhat_mm = (np.exp(qhat_test_hi) if 'qhat_test_hi' in globals() else np.exp(qhat_test))

rows=[]
for f in fracs:
    ca = np.array(idx_cal)
    n = max(10, int(len(ca)*f))
    ca_sub = rng_cz.choice(ca, size=n, replace=False)
    X_ca_elem = df.loc[ca_sub, elem_cols].to_numpy()
    # marginal
    q_ca = (cat_qt_hi.predict(X.iloc[ca_sub]) if 'cat_qt_hi' in globals() else cat_qt.predict(X.iloc[ca_sub]))
    S = np.maximum(0.0, q_ca - y_log[ca_sub])
    q_m = weighted_quantile(S, 1-ALPHA) if 'weighted_quantile' in globals() else np.quantile(S, 1-ALPHA)
    Lm = np.exp((qhat_test_hi if 'qhat_test_hi' in globals() else qhat_test) - q_m)
    # robust
    mdl = cat_qt_hi if 'cat_qt_hi' in globals() else cat_qt
    S_r = robust_scores_lower(y_log[ca_sub], mdl, X_ca_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_cz)
    q_r = weighted_quantile(S_r, 1-ALPHA) if 'weighted_quantile' in globals() else np.quantile(S_r, 1-ALPHA)
    # L_rob for TEST
    Lr = np.asarray(L_robust_mm, float)
    cov_m = float(np.mean(y_mm >= Lm))
    cov_r = float(np.mean(y_mm >= Lr))
    gap_m = float(np.median(qhat_mm - Lm))
    gap_r = float(np.median(qhat_mm - Lr))
    rows.append({"frac_cal": f, "n_cal": n, "cov_marginal": cov_m, "cov_robust": cov_r,
                 "gap_median_marginal_mm": gap_m, "gap_median_robust_mm": gap_r})

df_calsize = pd.DataFrame(rows)
df_calsize.to_csv(SRC_DIR/"calibration_size_sensitivity.csv", index=False)

plt.figure(figsize=(6.2,3.8))
plt.plot(df_calsize["n_cal"], df_calsize["cov_marginal"], marker="o", label="Marginal")
plt.plot(df_calsize["n_cal"], df_calsize["cov_robust"], marker="o", label="Robust (ε)")
plt.axhline(1-ALPHA, ls="--"); plt.xlabel("|CAL|"); plt.ylabel("Coverage"); plt.ylim(0,1)
plt.title("Coverage vs calibration set size"); plt.legend(frameon=False)
figsave(FIGDIR/"fig_calibration_size_coverage.png")


# In[91]:


# Step 12.20 — Stability of L_robust vs K and RNG seed
K_list = [16, 32, 64, 128, 256, 512, 1024]
seeds   = [SEED+1, SEED+3, SEED+5, SEED+7, SEED+9]
N_eval  = min(100, len(idx_test))
rng_st  = np.random.default_rng(SEED+2020)
pick    = rng_st.choice(len(idx_test), size=N_eval, replace=False)

X_eval = df.loc[idx_test, elem_cols].to_numpy()[pick]

def _Lrob_for_K(K, seed):
    mdl = cat_qt_hi if 'cat_qt_hi' in globals() else cat_qt
    L = np.zeros(len(X_eval))
    for i, x in enumerate(X_eval):
        Xj = jitter_in_L1_ball_simplex(x, eps=ROBUST_EPS, K=K, rng=np.random.default_rng(seed))
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(mdl.predict(feats), float)
        L[i] = np.exp(np.min(qj) - q_robust_hi) if 'q_robust_hi' in globals() else np.exp(np.min(qj) - np.quantile(np.maximum(0.0, mdl.predict(X.iloc[idx_cal]) - y_log[idx_cal]), 1-ALPHA))
    return L

rows=[]
for K in K_list:
    vals = []
    for s in seeds:
        vals.append(_Lrob_for_K(K, s))
    V = np.vstack(vals)
    sd = V.std(axis=0, ddof=1)
    rows.append({"K": K, "median_sd_mm": float(np.median(sd)), "p90_sd_mm": float(np.percentile(sd, 90))})

df_stab = pd.DataFrame(rows)
df_stab.to_csv(SRC_DIR/"stability_vs_K.csv", index=False)

plt.figure(figsize=(6.2,3.8))
plt.plot(df_stab["K"], df_stab["median_sd_mm"], marker="o", label="median SD")
plt.plot(df_stab["K"], df_stab["p90_sd_mm"], marker="o", label="90th pct SD")
plt.xlabel("K (jitter samples)"); plt.ylabel("SD of L_robust across seeds (mm)")
plt.title("Stability of certificates vs K"); plt.legend(frameon=False)
figsave(FIGDIR/"fig_stability_vs_K.png")


# In[92]:


# Step 12.21 — Family-stratified quantile reliability
tau = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU
qhat_te = (qhat_test_hi if 'qhat_test_hi' in globals() else qhat_test)
y_te    = y_test
fam_te  = df.loc[idx_test, "family"].to_numpy() if "family" in df.columns else np.repeat("all", len(y_te))

rows=[]
for fam in pd.unique(fam_te):
    m = (fam_te == fam)
    if m.sum() < 15: 
        continue
    obs = float(np.mean(y_te[m] <= qhat_te[m]))
    rows.append({"family": fam, "n": int(m.sum()), "obs_prob": obs, "target_tau": float(tau)})
df_qrel = pd.DataFrame(rows).sort_values("n", ascending=False)
df_qrel.to_csv(SRC_DIR/"quantile_reliability_by_family.csv", index=False)

plt.figure(figsize=(8.2, max(3.6, 0.28*len(df_qrel))))
y = np.arange(len(df_qrel))
plt.barh(y, df_qrel["obs_prob"].values)
plt.yticks(y, [f"{f} (n={n})" for f,n in zip(df_qrel["family"], df_qrel["n"])])
plt.axvline(tau, ls="--", label=f"τ={tau:.2f}")
plt.xlabel("Observed P(Y ≤ q̂τ)"); plt.xlim(0,1); plt.title("Quantile reliability by family (test)")
plt.legend(frameon=False, loc="lower right")
figsave(FIGDIR/"fig_quantile_reliability_by_family.png")


# In[93]:


# Step 12.22 — Recourse path overlay on ternary (fixed)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
for p in (FIGDIR, SRC_DIR): p.mkdir(parents=True, exist_ok=True)

def _figsave(path):
    try: figsave(path)
    except NameError:
        plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

def _value_as_fraction(v):
    """Accept either fraction (0–1) or at.% (>1); return fraction."""
    try: x = float(v)
    except Exception: return 0.0
    return (x/100.0) if np.isfinite(x) and x > 1.5 else x

# Robust extractor for allowed-element fraction vector from a row
def _extract_allowed_fractions_from_row(row):
    v = np.zeros(len(allowed_elems_present), float)
    for j, e in enumerate(allowed_elems_present):
        if f"frac_{e}"   in row: val = _value_as_fraction(row[f"frac_{e}"])
        elif f"atpct_{e}" in row: val = _value_as_fraction(row[f"atpct_{e}"])
        elif f"at_{e}"   in row: val = _value_as_fraction(row[f"at_{e}"])
        elif e in row:            val = _value_as_fraction(row[e])
        else:                     val = 0.0
        v[j] = val
    s = v.sum()
    return v/s if s > 0 else np.ones_like(v)/len(v)

# Simplex projection & caps/nnz constraints (match earlier recourse block)
if "project_simplex_fraction" not in globals():
    def project_simplex_fraction(x):
        x = np.maximum(np.asarray(x, float), 0.0)
        s = x.sum()
        return x/s if s > 0 else np.ones_like(x)/len(x)

def _apply_caps_and_maxels(x_allowed, hard_caps=None, max_elements=None):
    x = np.asarray(x_allowed, float).copy()
    if hard_caps:
        caps = np.array([np.inf if hard_caps.get(e) is None else float(hard_caps[e])/100.0
                         for e in allowed_elems_present], float)
        x = np.minimum(x, caps)
    s = x.sum(); x = x/s if s > 0 else np.ones_like(x)/len(x)
    if (max_elements is not None) and (max_elements < len(x)):
        order = np.argsort(x); kill = order[:len(x)-max_elements]
        x[kill] = 0.0
        s = x.sum(); x = x/s if s > 0 else np.ones_like(x)/len(x)
    return x

# Rebuild the full sequence of compositions from the recorded path (uses i, j, delta_atpct)
def _rebuild_path_comps(path_df, x0_allowed, hard_caps=None, max_elements=None):
    xs = [np.asarray(x0_allowed, float)]
    for k in range(1, len(path_df)):  # row 0 is init
        i = int(path_df.loc[k, "i"]); j = int(path_df.loc[k, "j"])
        step_atpct = float(path_df.loc[k, "delta_atpct"])
        if i >= 0 and j >= 0 and step_atpct > 0:
            step = step_atpct / 100.0
            x = xs[-1].copy()
            x[i] += step; x[j] -= step
            x = np.clip(x, 0.0, None)
            x = project_simplex_fraction(x)
            x = _apply_caps_and_maxels(x, hard_caps, max_elements)
        else:
            x = xs[-1].copy()
        xs.append(x)
    return np.vstack(xs)

# --- choose seed design (best certified) ---
seed_df = (design_df_cert_gt if 'design_df_cert_gt' in globals() and isinstance(design_df_cert_gt, pd.DataFrame) and len(design_df_cert_gt)>0
           else design_df_all if 'design_df_all' in globals() and isinstance(design_df_all, pd.DataFrame) and len(design_df_all)>0
           else bo_df)
if seed_df is None or len(seed_df) == 0:
    raise RuntimeError("No candidate pool found (design_df_cert_gt / design_df_all / bo_df are all empty).")

row0 = seed_df.sort_values("L_robust_mm", ascending=False).iloc[0]
x0_allowed = _extract_allowed_fractions_from_row(row0)

# --- run recourse with the correct API (schedule/patience) ---
DSTAR = 5.0
res = recourse_min_edit_allowed(
    x0_allowed, DSTAR,
    schedule=(0.03, 0.02, 0.01, 0.005),
    patience=500, hard_caps=None, max_elements=None, crn_seed=SEED+707
)
path_df = res["path"].copy()

# add compositions along the path so CSV is self-contained
Xseq = _rebuild_path_comps(path_df, x0_allowed, hard_caps=None, max_elements=None)
for j, e in enumerate(allowed_elems_present):
    path_df[f"frac_{e}"] = Xseq[:, j]
path_df.to_csv(SRC_DIR / f"recourse_path_Dstar_{int(round(DSTAR))}mm.csv", index=False)

# --- ternary helpers on the top-3 elements of the *seed* composition ---
def _top3_names(frac_vec):
    order = np.argsort(frac_vec)[::-1][:3]
    return [allowed_elems_present[i] for i in order], order

names3, idx3 = _top3_names(x0_allowed)

def _bary_map(vec):  # map allowed vector to 3D bary of top3 (renormalize on those)
    v = np.asarray(vec, float)[idx3]
    s = v.sum()
    return (v / s) if s > 0 else np.ones(3)/3

def _bary_to_xy(a, b, c):
    # equilateral ternary coordinates (simple, consistent with many plotting libs)
    x = 0.5*(2*b + c)
    y = (np.sqrt(3)/2.0)*c
    return x, y

# project the whole path to ternary of those 3 elements
P = np.vstack([_bary_map(Xseq[t]) for t in range(Xseq.shape[0])])
xy = np.vstack([_bary_to_xy(p[0], p[1], p[2]) for p in P])

# --- plot with arrows ---
plt.figure(figsize=(6.0, 5.6))
plt.plot(xy[:,0], xy[:,1], "-o", ms=4)
for i in range(1, len(xy)):
    dx, dy = xy[i,0] - xy[i-1,0], xy[i,1] - xy[i-1,1]
    plt.arrow(xy[i-1,0], xy[i-1,1], dx, dy, length_includes_head=True,
              head_width=0.02, head_length=0.02, alpha=0.6)
plt.title(f"Recourse path on ternary: {names3[0]}–{names3[1]}–{names3[2]}\n"
          f"L₀={res['L_init_mm']:.1f} → L*={res['L_final_mm']:.1f}±{1.96*res['L_final_se_mm']:.1f} mm @ D*={int(DSTAR)}")
plt.axis("equal"); plt.axis("off")
_figsave(FIGDIR / "fig_recourse_path_ternary.png")

print("Saved:")
print(" - Path CSV with per-step compositions:", SRC_DIR / f"recourse_path_Dstar_{int(round(DSTAR))}mm.csv")
print(" - Ternary path figure:", FIGDIR / "fig_recourse_path_ternary.png")


# In[94]:


# Coverage per family (random test), multi-method with CIs, FDR, deltas
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
    return float(p), max(0.0, center - half), min(1.0, center + half)

try:
    from scipy.special import gammaln, logsumexp    
except Exception:
    gammaln = None
    logsumexp = None

def _log_choose_vec(n, k_arr):
    """Vectorized log nCk for scalar n and array-like k."""
    k = np.asarray(k_arr, dtype=float)
    if gammaln is not None:
        return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    # fallback: log-factorial via sum of logs (OK for moderate n)
    def _logfact(m):
        m = int(m)
        if m <= 1: return 0.0
        return float(np.sum(np.log(np.arange(2, m + 1))))
    ln_n = _logfact(n)
    # vectorize k
    ln_k   = np.array([_logfact(int(kk)) for kk in k], dtype=float)
    ln_nmk = np.array([_logfact(int(n - kk)) for kk in k], dtype=float)
    return ln_n - ln_k - ln_nmk

def binom_test_less_p(k, n, p0):
    """
    Exact one-sided binomial test: P[X <= k | X~Bin(n, p0)].
    Returns a probability in [0,1].
    """
    if n <= 0:
        return np.nan
    p0 = float(np.clip(p0, 1e-12, 1 - 1e-12))
    ks = np.arange(0, int(k) + 1, dtype=int)
    logpmf = _log_choose_vec(n, ks) + ks * np.log(p0) + (n - ks) * np.log1p(-p0)
    if logsumexp is not None:
        return float(np.exp(logsumexp(logpmf)))
    # fallback log-sum-exp
    m = float(np.max(logpmf))
    return float(np.exp(m) * np.sum(np.exp(logpmf - m)))

def bh_fdr(pvals, alpha=0.10):
    """Benjamini–Hochberg FDR correction; returns q-values array."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(m-1, -1, -1):
        q[i] = min(prev, ranked[i] * m / (i+1))
        prev = q[i]
    out = np.empty_like(q)
    out[order] = q
    return out

def bootstrap_delta_ci(hit_a, hit_b, B=2000, rng=None):
    """
    CI for difference in coverages (mean(hit_b) - mean(hit_a)) within a family.
    hit_* are 0/1 arrays of equal length.
    """
    rng = np.random.default_rng(SEED+131) if rng is None else rng
    hit_a = np.asarray(hit_a, int); hit_b = np.asarray(hit_b, int)
    n = len(hit_a)
    if n == 0: return np.nan, np.nan, np.nan
    idx = np.arange(n)
    stats = np.empty(B, float)
    for b in range(B):
        res = rng.choice(idx, size=n, replace=True)
        stats[b] = float(hit_b[res].mean() - hit_a[res].mean())
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

methods = []
if 'L_marginal_mm' in globals(): methods.append(('L_marginal',  L_marginal_mm,  'Marginal'))
if 'L_group_mm'    in globals(): methods.append(('L_group',     L_group_mm,     'Group-cond.'))
if 'L_weighted_mm' in globals(): methods.append(('L_weighted',  L_weighted_mm,  'Shift-weighted'))
if 'L_robust_mm'   in globals(): methods.append(('L_robust',    L_robust_mm,    f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"))

if not methods:
    raise RuntimeError("No bound arrays found. Expected at least one of L_marginal_mm, L_group_mm, L_weighted_mm, L_robust_mm.")

df_test = pd.DataFrame({
    "family": df.loc[idx_test, "family"].values,
    "y_mm":   np.exp(y_test),
})
for key, arr, _lab in methods:
    df_test[key] = np.asarray(arr, float)

# Per-family stats with Wilson CIs + undercoverage tests (vs target 1-ALPHA)
TARGET = 1.0 - ALPHA
MIN_FAM = 5

rows = []
long_rows = []
delta_rows = []

have_marg = any(k == 'L_marginal' for k,_,_ in methods)
have_robu = any(k == 'L_robust'   for k,_,_ in methods)

for fam, g in df_test.groupby("family"):
    n = len(g)
    if n < MIN_FAM:
        continue

    # Per-method coverage + Wilson CI + p-value under target
    pvals_under = []
    row = {"family": fam, "n": int(n)}
    hits_by_method = {}

    for key, _arr, lab in methods:
        hit = (g["y_mm"].values >= g[key].values).astype(int)
        k = int(hit.sum())
        p, lo, hi = wilson_ci(k, n)
        row[f"cov_{key}"] = p
        row[f"lo_{key}"]  = lo
        row[f"hi_{key}"]  = hi
        hits_by_method[key] = hit

        # exact one-sided p-value for undercoverage (p < TARGET)
        pval = binom_test_less_p(k, n, TARGET)
        row[f"p_under_{key}"] = pval
        pvals_under.append((key, pval))

        long_rows.append({
            "family": fam, "n": int(n),
            "method": lab, "key": key,
            "coverage": p, "ci_lo": lo, "ci_hi": hi,
            "gap_to_target": float(p - TARGET),
            "p_under": float(pval),
        })

    # robust−marginal paired delta
    if have_marg and have_robu:
        dm, dl, dh = bootstrap_delta_ci(hits_by_method["L_marginal"], hits_by_method["L_robust"])
        row["delta_cov_robust_minus_marginal"] = dm
        row["delta_ci_lo"] = dl
        row["delta_ci_hi"] = dh
        delta_rows.append({"family": fam, "n": int(n), "delta": dm, "ci_lo": dl, "ci_hi": dh})

    rows.append(row)

df_perfam = pd.DataFrame(rows).sort_values(["n","family"], ascending=[False, True]).reset_index(drop=True)
df_long   = pd.DataFrame(long_rows)
df_delta  = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame(columns=["family","n","delta","ci_lo","ci_hi"])

# BH-FDR (10%) per method on undercoverage p-values
for key, _arr, lab in methods:
    col = f"p_under_{key}"
    if col in df_perfam.columns:
        mask = np.isfinite(df_perfam[col].values)
        pvals = df_perfam.loc[mask, col].values
        qvals = bh_fdr(pvals, alpha=0.10)
        df_perfam.loc[mask, f"q_under_{key}"] = qvals
        mskL = (df_long["key"] == key)
        df_long.loc[mskL, "q_under"] = qvals

# Robust traffic-light flag vs target (using CI)
if 'cov_L_robust' in df_perfam.columns:
    # green: CI_lo >= target; yellow: mean>=target but CI_lo<target; red: mean<target
    flags = []
    for _, r in df_perfam.iterrows():
        p  = r["cov_L_robust"]; lo = r["lo_L_robust"]
        if np.isnan(p): flags.append("NA")
        elif lo >= TARGET: flags.append("green")
        elif p >= TARGET:  flags.append("yellow")
        else:              flags.append("red")
    df_perfam["robust_flag"] = flags

df_perfam.to_csv(SRC_DIR / "coverage_per_family_random_all_methods.csv", index=False)
df_long.to_csv(SRC_DIR / "coverage_per_family_random_long.csv", index=False)
if len(df_delta):
    df_delta.to_csv(SRC_DIR / "coverage_per_family_delta_robust_minus_marginal.csv", index=False)

# Figures
def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

# Figure A: robust coverage per family with Wilson CIs + traffic-light color
if 'cov_L_robust' in df_perfam.columns:
    plot_df = df_perfam.sort_values("n", ascending=False).copy()
    vals = plot_df["cov_L_robust"].values
    lows = plot_df["lo_L_robust"].values
    highs= plot_df["hi_L_robust"].values
    fams = plot_df["family"].values
    ns   = plot_df["n"].values
    colors = plot_df.get("robust_flag", pd.Series(["blue"]*len(plot_df))).replace({
        "green":"#2ca02c","yellow":"#ffbf00","red":"#d62728","NA":"#7f7f7f"
    }).values


    vals = np.asarray(vals, float)
    lows = np.asarray(lows, float)
    highs = np.asarray(highs, float)
    
    # Clip into [0,1] and enforce lo<=val<=hi
    vals  = np.clip(vals,  0.0, 1.0)
    lows  = np.minimum(np.clip(lows,  0.0, 1.0), vals)
    highs = np.maximum(np.clip(highs, 0.0, 1.0), vals)
    
    err_low  = vals - lows
    err_high = highs - vals
    
    # Replace any NaNs / tiny negatives with 0
    err_low  = np.where(np.isfinite(err_low)  & (err_low  >= 0), err_low,  0.0)
    err_high = np.where(np.isfinite(err_high) & (err_high >= 0), err_high, 0.0)
    
    y = np.arange(len(vals))
    plt.figure(figsize=(9, max(4.0, 0.30*len(vals))))
    plt.barh(y, vals, color=colors, alpha=0.9)
    err_low  = vals - lows
    err_high = highs - vals
    plt.errorbar(vals, y,
                 xerr=np.vstack([err_low, err_high]),
                 fmt="none", ecolor="k", elinewidth=0.8, capsize=3)    
    plt.axvline(TARGET, ls="--", label=f"Target {TARGET:.2f}", color="k")
    plt.yticks(y, [f"{f}  (n={n})" for f, n in zip(fams, ns)])
    plt.xlim(0, 1)
    plt.xlabel("Coverage (Robust)")
    plt.title("Per-family robust coverage (random test) with 95% Wilson CIs")
    plt.legend(frameon=False, loc="lower right")
    _figsave(FIGDIR / "fig_per_family_coverage_random.png")

# Figure B: paired deltas (robust − marginal) with bootstrap CIs
if len(df_delta):
    d2 = df_delta.sort_values("delta", ascending=False).reset_index(drop=True)
    y = np.arange(len(d2))
    plt.figure(figsize=(9, max(4.0, 0.30*len(d2))))
    plt.barh(y, d2["delta"], alpha=0.85)

    for i, (lo, hi, mu) in enumerate(zip(d2["ci_lo"], d2["ci_hi"], d2["delta"])):
        plt.plot([lo, hi], [i, i], color="k", lw=1.0)
    plt.axvline(0.0, ls="--", color="k")
    plt.yticks(y, [f"{f} (n={n})" for f, n in zip(d2["family"], d2["n"])])
    plt.xlabel("Δ coverage = Robust − Marginal")
    plt.title("Per-family coverage gain (bootstrap 95% CIs)")
    _figsave(FIGDIR / "fig_per_family_delta_robust_minus_marginal.png")

print("Saved tables:",
      SRC_DIR / "coverage_per_family_random_all_methods.csv", ",",
      SRC_DIR / "coverage_per_family_random_long.csv", ",",
      (SRC_DIR / "coverage_per_family_delta_robust_minus_marginal.csv" if len(df_delta) else "—"))
print("Saved figures:",
      FIGDIR / "fig_per_family_coverage_random.png",
      "and",
      (FIGDIR / "fig_per_family_delta_robust_minus_marginal.png" if len(df_delta) else "—"))


# In[95]:


# Family-out evaluation (CatBoost-quantile, CIs, FDR, calibration, OOD, shift diag)
_HAVE_SKOPT = True

req_vars = ["df","elem_cols","X","y_log","split_family_out","OUTDIR","SEED",
            "ALPHA","ROBUST_EPS","ROBUST_SAMPLES","MIN_GROUP_N","FAMILY_MIN_TEST",
            "jitter_in_L1_ball_simplex","make_features_from_compositions"]
for _rv in req_vars:
    if _rv not in globals():
        raise RuntimeError(f"Missing required global: '{_rv}'")

tau = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU

def wilson_ci(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = z * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
    return float(p), max(0.0, center - half), min(1.0, center + half)

def gammaln(z):
    z = np.asarray(z, dtype=float)
    return np.vectorize(_lgamma)(z) 

def _log_choose(n, k):
    n = float(n)
    k = np.asarray(k, dtype=float)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)

def binom_test_less_p(k, n, p0):
    k = int(k); n = int(n)
    if n <= 0: 
        return np.nan
    p0 = float(np.clip(p0, 1e-12, 1 - 1e-12))
    ks = np.arange(0, k + 1, dtype=int)
    logpmf = _log_choose(n, ks) + ks * np.log(p0) + (n - ks) * np.log1p(-p0)
    m = np.max(logpmf)
    return float(np.exp(m) * np.sum(np.exp(logpmf - m)))

def bh_fdr(pvals, alpha=0.10):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(m-1, -1, -1):
        q[i] = min(prev, ranked[i] * m / (i+1))
        prev = q[i]
    out = np.empty_like(q)
    out[order] = q
    return out

def robust_L_for_test(model, comp_elem, q_cal_robust, eps, K, rng):
    L_mm = np.zeros(len(comp_elem))
    for i, x in enumerate(comp_elem):
        Xj = jitter_in_L1_ball_simplex(x, eps=eps, K=K, rng=rng)
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), float)
        L_mm[i] = float(np.exp(np.min(qj) - q_cal_robust))
    return L_mm

# Per-family loop
results = []
long_rows = []
best_params_index = []

elem_mat = df.loc[:, elem_cols].to_numpy()
cv_local = cv if 'cv' in globals() else 3
TARGET = 1.0 - ALPHA

for fam_name, split in split_family_out.items():
    tr, ca, te = split["train"], split["cal"], split["test"]
    if len(te) < FAMILY_MIN_TEST:
        continue

    X_tr, y_tr = X.iloc[tr], y_log[tr]
    X_ca, y_ca = X.iloc[ca], y_log[ca]
    X_te, y_te = X.iloc[te], y_log[te]
    y_te_mm = np.exp(y_te)

    et_fo = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1, max_features = 1.0, max_depth=30, min_samples_split = 2, random_state=SEED, n_jobs=-1)
    try:
        et_fo.fit(X_tr, y_tr)
    except Exception:
        pass

    # CatBoost quantile (fold-specific) with light HPO (or fallback)
    cb_base = CatBoostRegressor(
        loss_function=f"Quantile:alpha={tau}",
        eval_metric=f"Quantile:alpha={tau}",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1
    )
    pin_scorer = make_scorer(mean_pinball_loss, greater_is_better=False, alpha=tau)

    fam_tr = df.loc[tr, "family"].values
    fam_counts_tr = pd.Series(fam_tr).value_counts()
    w_tr = 1.0 / np.sqrt(pd.Series(fam_tr).map(fam_counts_tr).values)

    cb_model = None
    best_params = {}
    if _HAVE_SKOPT:
        space = {
            'iterations'         : Integer(500, 1000),
            'depth'              : Integer(3, 10),
            'learning_rate'      : Real(1e-3, 2e-1, prior='log-uniform'),
            'l2_leaf_reg'        : Real(1e-3, 10.0, prior='log-uniform'),
            'bootstrap_type'     : Categorical(['Bayesian']),
            'bagging_temperature': Real(0.0, 1.0),
            'random_strength'    : Real(0.0, 1.0),
            'border_count'       : Integer(64, 255),
            'rsm'                : Real(0.6, 1.0),
        }
        cat_fo = BayesSearchCV(
            estimator=cb_base, search_spaces=space, n_iter=300, scoring=None,
            cv=cv_local, n_jobs=-1, random_state=SEED, verbose=0, refit=True, return_train_score=False
        )
        try:
            groups_tr = df.loc[tr, "signature"].values if "signature" in df.columns else None
            if groups_tr is not None:
                cat_fo.fit(X_tr, y_tr, groups=groups_tr, sample_weight=w_tr)
            else:
                cat_fo.fit(X_tr, y_tr, sample_weight=w_tr)
            cb_model = cat_fo.best_estimator_
            best_params = cat_fo.best_params_
        except TypeError:
            cat_fo.fit(X_tr, y_tr, sample_weight=w_tr)
            cb_model = cat_fo.best_estimator_
            best_params = cat_fo.best_params_

    if cb_model is None:
        cb_model = cb_base
        cb_model.fit(X_tr, y_tr, sample_weight=w_tr)

    best_params_index.append({"family_out": fam_name, "best_params": best_params})

    # Quantile predictions
    q_ca = cb_model.predict(X_ca)
    q_te = cb_model.predict(X_te)

    # τ-calibration diagnostics
    obs_tau_cal  = float(np.mean(y_ca <= q_ca))
    obs_tau_test = float(np.mean(y_te <= q_te))
    pin_cal = mean_pinball_loss(y_ca, q_ca, alpha=tau)
    pin_te  = mean_pinball_loss(y_te, q_te,  alpha=tau)

    # Marginal CP (lower-bound sign)
    S_ca = np.maximum(0.0, q_ca - y_ca)
    q_m  = weighted_quantile(S_ca, 1 - ALPHA)
    Lm   = np.exp(q_te - q_m)

    # Group-conditional CP (shrinkage if few cal points)
    fam_ca = df.loc[ca, "family"].values
    fam_te = df.loc[te, "family"].values
    Lg = np.empty_like(Lm)
    for f in np.unique(fam_te):
        S_f = S_ca[fam_ca == f]
        if S_f.size >= MIN_GROUP_N:
            q_f = weighted_quantile(S_f, 1 - ALPHA)
        elif S_f.size > 0:
            lam = S_f.size / (S_f.size + MIN_GROUP_N)
            q_small = weighted_quantile(S_f, 1 - ALPHA)
            q_f = lam*q_small + (1-lam)*q_m
        else:
            q_f = q_m
        Lg[fam_te == f] = np.exp(q_te[fam_te == f] - q_f)

    # Shift-weighted CP (cal vs test)
    X_cls = np.vstack([X_ca.values, X_te.values])
    y_cls = np.hstack([np.zeros(len(X_ca)), np.ones(len(X_te))])
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_cls, y_cls)
    prob = clf.predict_proba(X_cls)[:, 1]
    auc  = roc_auc_score(y_cls, prob)

    ptest_on_cal = clf.predict_proba(X_ca.values)[:, 1]
    pcal_on_cal  = 1.0 - ptest_on_cal
    w = np.clip(ptest_on_cal / np.maximum(pcal_on_cal, 1e-6), 0.0, 1e6)
    ess = (w.sum()**2) / np.sum(w**2)

    q_w = weighted_quantile(S_ca, 1 - ALPHA, sample_weight=w)
    Lw  = np.exp(q_te - q_w)

    # Robust CP (ε in FRACTIONS)
    X_ca_elem = elem_mat[ca]
    X_te_elem = elem_mat[te]
    rng_fold = np.random.default_rng(SEED + 17)
    S_ca_rob = robust_scores_lower(y_ca, cb_model, X_ca_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_fold)
    q_r = weighted_quantile(S_ca_rob, 1 - ALPHA)
    Lr = robust_L_for_test(cb_model, X_te_elem, q_r, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_fold)

    # Coverage + Wilson CIs + one-sided p-values vs target
    def _cov_ci_p(y_true_mm, L_mm):
        m = np.isfinite(y_true_mm) & np.isfinite(L_mm)
        if not np.any(m): return (np.nan, np.nan, np.nan, np.nan)
        hit = (y_true_mm[m] >= L_mm[m]).astype(int)
        k, n = int(hit.sum()), int(len(hit))
        p, lo, hi = wilson_ci(k, n)
        pval = binom_test_less_p(k, n, TARGET)  # under-coverage p-val
        return float(p), float(lo), float(hi), float(pval)

    cov_m, lo_m, hi_m, pv_m = _cov_ci_p(y_te_mm, Lm)
    cov_g, lo_g, hi_g, pv_g = _cov_ci_p(y_te_mm, Lg)
    cov_w, lo_w, hi_w, pv_w = _cov_ci_p(y_te_mm, Lw)
    cov_r, lo_r, hi_r, pv_r = _cov_ci_p(y_te_mm, Lr)

    # OOD distances: avg 5-NN in composition space (test vs train)
    nn = NearestNeighbors(n_neighbors=5, metric="minkowski", p=2).fit(elem_mat[tr])
    dists, _ = nn.kneighbors(elem_mat[te], n_neighbors=5, return_distance=True)
    ood_mean = float(np.mean(dists))
    ood_median = float(np.median(np.mean(dists, axis=1)))

    # Record per-family summary (keep legacy column names for compatibility)
    row = {
        "family_out": fam_name,
        "n_train": int(len(tr)), "n_cal": int(len(ca)), "n_test": int(len(te)),
        "cov_marginal": cov_m, "cov_group": cov_g, "cov_weighted": cov_w, "cov_robust": cov_r,

        "ci_lo_marginal": lo_m, "ci_hi_marginal": hi_m,
        "ci_lo_group":    lo_g, "ci_hi_group":    hi_g,
        "ci_lo_weighted": lo_w, "ci_hi_weighted": hi_w,
        "ci_lo_robust":   lo_r, "ci_hi_robust":   hi_r,

        "p_under_marginal": pv_m,
        "p_under_group":    pv_g,
        "p_under_weighted": pv_w,
        "p_under_robust":   pv_r,

        "tau_used": float(tau),
        "obs_tau_cal":  obs_tau_cal,
        "obs_tau_test": obs_tau_test,
        "pinball_cal":  pin_cal,
        "pinball_test": pin_te,

        "shift_auc_cal_test": float(auc),
        "shift_ess_cal": float(ess),

        "robust_eps": float(ROBUST_EPS),
        "robust_K":   int(ROBUST_SAMPLES),
        "q_robust_cal": float(q_r),

        "ood_knn5_mean": ood_mean,
        "ood_knn5_median": ood_median,

        "shp_m": float(np.median(np.exp(q_te) - Lm)),
        "shp_g": float(np.median(np.exp(q_te) - Lg)),
        "shp_w": float(np.median(np.exp(q_te) - Lw)),
        "shp_r": float(np.median(np.exp(q_te) - Lr)),
    }
    results.append(row)

    for meth, L, cov, lo, hi, pv in [
        ("Marginal", Lm, cov_m, lo_m, hi_m, pv_m),
        ("Group-cond.", Lg, cov_g, lo_g, hi_g, pv_g),
        ("Shift-weighted", Lw, cov_w, lo_w, hi_w, pv_w),
        (f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)", Lr, cov_r, lo_r, hi_r, pv_r),
    ]:
        long_rows.append({
            "family_out": fam_name, "n_test": int(len(te)),
            "method": meth, "coverage": cov, "ci_lo": lo, "ci_hi": hi,
            "p_under": pv, "target": TARGET,
            "tau_used": float(tau),
            "obs_tau_cal": obs_tau_cal, "obs_tau_test": obs_tau_test,
            "pinball_cal": pin_cal, "pinball_test": pin_te,
            "shift_auc_cal_test": float(auc), "shift_ess_cal": float(ess),
            "ood_knn5_mean": ood_mean, "ood_knn5_median": ood_median,
        })

df_fo = pd.DataFrame(results).sort_values("n_test", ascending=False).reset_index(drop=True)
df_fo_long = pd.DataFrame(long_rows)

# BH-FDR (10%) per method on under-coverage p-values; attach q-values to df_fo and df_fo_long
def _apply_fdr(df_wide, df_long, wide_col, long_label):
    if wide_col in df_wide.columns:
        pv = df_wide[wide_col].values
        mask = np.isfinite(pv)
        qv = np.full_like(pv, np.nan, dtype=float)
        if np.any(mask):
            qv[mask] = bh_fdr(pv[mask], alpha=0.10)
        df_wide["q_"+wide_col] = qv
        m = (df_long["method"] == long_label)
        df_long.loc[m, "q_under"] = df_wide["q_"+wide_col].values

_apply_fdr(df_fo, df_fo_long, "p_under_marginal", "Marginal")
_apply_fdr(df_fo, df_fo_long, "p_under_group", "Group-cond.")
_apply_fdr(df_fo, df_fo_long, "p_under_weighted", "Shift-weighted")
_apply_fdr(df_fo, df_fo_long, "p_under_robust", f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)")

(OUTDIR / "reports").mkdir(parents=True, exist_ok=True)
df_fo.to_csv(OUTDIR / "reports" / "family_out_metrics.csv", index=False)
df_fo_long.to_csv(OUTDIR / "reports" / "family_out_metrics_long.csv", index=False)

if _HAVE_SKOPT:
    with open(OUTDIR / "reports" / "family_out_best_params.json", "w") as f:
        json.dump(best_params_index, f, indent=2)

display(df_fo.head(10))
print("Saved:",
      OUTDIR / "reports" / "family_out_metrics.csv",
      "and", OUTDIR / "reports" / "family_out_metrics_long.csv",
      "(plus family_out_best_params.json if HPO was available)")


# In[96]:


# Family-out: per-family error bars + summary figures (enhanced, FDR, OOD, deltas)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

if 'df_fo' not in globals():
    df_fo = pd.read_csv(OUTDIR / "reports" / "family_out_metrics.csv")

robust_label = f"Drift-robust (ε = {ROBUST_EPS*100:.2f} at.% transferred mass)"

def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(p), float(lo), float(hi)

def bootstrap_mean(values, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    means = np.empty(B, float)
    for b in range(B):
        i = rng.integers(0, n, size=n)
        means[b] = float(vals[i].mean())
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

TARGET = 1.0 - ALPHA

cov_cols = [c for c in ["cov_marginal","cov_group","cov_weighted","cov_robust"] if c in df_fo.columns]
shp_cols = [c for c in ["shp_m","shp_g","shp_w","shp_r"] if c in df_fo.columns]
method_nice = {
    "cov_marginal": "Marginal",
    "cov_group": "Group-cond.",
    "cov_weighted": "Shift-weighted",
    "cov_robust": robust_label,
    "shp_m": "Marginal",
    "shp_g": "Group-cond.",
    "shp_w": "Shift-weighted",
    "shp_r": robust_label,
}

q_cols = {
    "cov_marginal": "q_p_under_marginal",
    "cov_group":    "q_p_under_group",
    "cov_weighted": "q_p_under_weighted",
    "cov_robust":   "q_p_under_robust",
}
have_q = any((qc in df_fo.columns) for qc in q_cols.values())

# A) Build per-family coverage with Wilson CIs for all methods
rows = []
for _, r in df_fo.iterrows():
    fam = r["family_out"]; n = int(r["n_test"])
    for cc in cov_cols:
        cov = float(r[cc])
        if np.isfinite(cov):
            k = int(np.round(cov * n))
            p, lo, hi = wilson_ci(k, n)
        else:
            p, lo, hi = np.nan, np.nan, np.nan
        qv = float(r[q_cols[cc]]) if (have_q and q_cols[cc] in df_fo.columns and np.isfinite(r[q_cols[cc]])) else np.nan
        under_sig = int(np.isfinite(qv) and (qv <= 0.10))
        rows.append({
            "family": fam, "n": n, "method_key": cc,
            "method": method_nice[cc], "coverage": p, "lo": lo, "hi": hi,
            "excess_vs_target": (p - TARGET) if np.isfinite(p) else np.nan,
            "q_under": qv, "under_sig_fdr10": under_sig,
            "ood_knn5_mean": float(r["ood_knn5_mean"]) if "ood_knn5_mean" in df_fo.columns else np.nan
        })

df_cov_wilson = pd.DataFrame(rows)

df_cov_wilson.to_csv(SRC_DIR / "family_out_per_family_coverage_wilson_all_methods.csv", index=False)

# B) Robust method per-family figure with FDR highlights; choose ordering
order_mode = "ood" if np.isfinite(df_cov_wilson["ood_knn5_mean"]).any() else "n"  # "ood" or "n"

df_rob = df_cov_wilson[df_cov_wilson["method_key"]=="cov_robust"].copy()
if order_mode == "ood":
    df_rob = df_rob.sort_values(["ood_knn5_mean","n","family"], ascending=[False, False, True])
else:
    df_rob = df_rob.sort_values(["n","family"], ascending=[False, True])

df_rob.assign(plot_rank=np.arange(1, len(df_rob)+1)).to_csv(
    SRC_DIR / "family_out_robust_coverage_plot_order.csv", index=False
)

# Coverage bars w/ Wilson CI; FDR under-coverage flagged with hatch
plt.figure(figsize=(9.2, max(3.8, 0.28*len(df_rob))))
y = np.arange(len(df_rob))
vals = df_rob["coverage"].values
err_low = vals - df_rob["lo"].values
err_high= df_rob["hi"].values - vals
colors = np.where(df_rob["under_sig_fdr10"].values == 1, "#d55e00", "#0072b2")  # red if under-cover at FDR10

plt.barh(y, vals, color=colors, alpha=0.95, edgecolor="black", linewidth=0.4)
plt.errorbar(vals, y, xerr=[err_low, err_high], fmt="none", capsize=3, color="black", linewidth=0.8)
plt.yticks(y, [
    (f"{f} (n={n}, OOD={m:.3f})" if order_mode=="ood" and np.isfinite(m) else f"{f} (n={n})")
    for f, n, m in zip(df_rob["family"], df_rob["n"], df_rob["ood_knn5_mean"])
])
plt.axvline(TARGET, ls="--", color="gray", label=f"Target {TARGET:.2f}")
plt.xlabel(f"Coverage — {robust_label}")
plt.xlim(0, 1)
plt.title(f"Family-out: per-family {robust_label} coverage (95% Wilson CI)\n"
          f"Red bars: under-coverage significant at FDR 10%")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_family_out_per_family_coverage_robust_wilson.png", dpi=300, bbox_inches="tight")
plt.close()

SRC_DIR = (FIGDIR / "source_data")
SRC_DIR.mkdir(parents=True, exist_ok=True)

if "ood_knn5_mean" in df_rob.columns:
    ood_col = df_rob["ood_knn5_mean"].values
else:
    ood_col = np.full(len(df_rob), np.nan)

labels = [
    (f"{f} (n={n}, OOD={m:.3f})" if ('order_mode' in globals() and order_mode == "ood" and np.isfinite(m))
     else f"{f} (n={n})")
    for f, n, m in zip(df_rob["family"], df_rob["n"], ood_col)
]

df_export = pd.DataFrame({
    "y_index": np.arange(len(df_rob)),
    "family": df_rob["family"].values,
    "label": labels,                  
    "n": df_rob["n"].astype(int).values,
    "ood_knn5_mean": ood_col,
    "coverage": vals,                 
    "lo": df_rob["lo"].values,        
    "hi": df_rob["hi"].values,        
    "err_low": err_low,               
    "err_high": err_high,             
    "under_sig_fdr10": df_rob["under_sig_fdr10"].astype(int).values,
    "fill_color_hex": colors,
})

csv_path = SRC_DIR / "family_out_per_family_coverage_robust_wilson_source.csv"
df_export.to_csv(csv_path, index=False)

# C) Excess coverage (coverage − target) per family (robust)
plt.figure(figsize=(9.2, max(3.8, 0.28*len(df_rob))))
exc = df_rob["excess_vs_target"].values
err_low_exc = err_low
err_high_exc= err_high
colors_exc = colors
plt.barh(y, exc, color=colors_exc, alpha=0.95, edgecolor="black", linewidth=0.4)
plt.errorbar(exc, y, xerr=[err_low_exc, err_high_exc], fmt="none", capsize=3, color="black", linewidth=0.8)
plt.axvline(0.0, ls="--", color="gray", label="Target")
plt.yticks(y, [f"{f}" for f in df_rob["family"]])
plt.xlabel("Excess coverage vs target")
plt.title(f"Family-out: excess coverage (coverage − {TARGET:.2f}), {robust_label}\n"
          f"Red bars: under-coverage significant at FDR 10%")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_family_out_excess_coverage_robust.png", dpi=300, bbox_inches="tight")
plt.close()

# D) Small-multiples: per-family coverage by method (keeps same order)
meth_order = [m for m in ["cov_marginal","cov_group","cov_weighted","cov_robust"] if m in cov_cols]
n_m = len(meth_order)
if n_m > 0:
    fam_order = df_rob["family"].tolist()
    fig, axes = plt.subplots(nrows=int(np.ceil(n_m/2)), ncols=2, figsize=(13, max(4.5, 0.28*len(fam_order))*int(np.ceil(n_m/2))))
    axes = np.ravel(axes) if n_m > 1 else [axes]
    for ax_i, cc in enumerate(meth_order):
        d = df_cov_wilson[df_cov_wilson["method_key"]==cc].copy()
        d = d.set_index("family").reindex(fam_order).dropna(subset=["coverage"]).reset_index()
        y = np.arange(len(d))
        vals = d["coverage"].values
        err_lo = vals - d["lo"].values
        err_hi = d["hi"].values - vals
        ax = axes[ax_i]
        ax.barh(y, vals, alpha=0.95)
        ax.errorbar(vals, y, xerr=[err_lo, err_hi], fmt="none", capsize=3, color="black", linewidth=0.8)
        ax.axvline(TARGET, ls="--", color="gray")
        ax.set_yticks(y)
        if ax_i % 2 == 0:
            ax.set_yticklabels(d["family"].tolist())
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0,1)
        ax.set_xlabel("Coverage")
        ax.set_title(method_nice[cc])

    for j in range(ax_i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_family_out_per_family_coverage_small_multiples.png", dpi=300, bbox_inches="tight")
    plt.close()

# E) Summary across families: coverage & sharpness (mean ± bootstrap)
summary_rows = []

for cc in cov_cols:
    m, lo, hi = bootstrap_mean(df_fo[cc].values, B=2000, seed=SEED+11)
    summary_rows.append({"metric":"Coverage", "method": method_nice[cc], "mean": m, "lo": lo, "hi": hi})

for sc in shp_cols:
    m, lo, hi = bootstrap_mean(df_fo[sc].values, B=2000, seed=SEED+17)
    summary_rows.append({"metric":"Sharpness gap (mm)", "method": method_nice[sc], "mean": m, "lo": lo, "hi": hi})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(SRC_DIR / "family_out_summary_bootstrap_means.csv", index=False)

order = ["Marginal","Group-cond.","Shift-weighted", robust_label]
df_cov_sum = df_summary[df_summary["metric"]=="Coverage"].copy()
df_cov_sum["method"] = pd.Categorical(df_cov_sum["method"], categories=order, ordered=True)
df_cov_sum = df_cov_sum.sort_values("method")

plt.figure(figsize=(6.6, 3.9))
x = np.arange(len(df_cov_sum))
plt.bar(x, df_cov_sum["mean"].values,
        yerr=[df_cov_sum["mean"]-df_cov_sum["lo"], df_cov_sum["hi"]-df_cov_sum["mean"]],
        capsize=4)
plt.axhline(TARGET, ls="--", label=f"Target {TARGET:.2f}")
plt.xticks(x, df_cov_sum["method"].tolist(), rotation=15)
plt.ylim(0, 1)
plt.ylabel("Mean coverage across families")
plt.title("Family-out summary: coverage (mean ± bootstrap 95% CI)")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_family_out_summary_coverage_bootstrap.png", dpi=300, bbox_inches="tight")
plt.close()

# Sharpness summary plot
df_shp_sum = df_summary[df_summary["metric"]=="Sharpness gap (mm)"].copy()
df_shp_sum["method"] = pd.Categorical(df_shp_sum["method"], categories=order, ordered=True)
df_shp_sum = df_shp_sum.sort_values("method")

plt.figure(figsize=(6.6, 3.9))
x = np.arange(len(df_shp_sum))
plt.bar(x, df_shp_sum["mean"].values,
        yerr=[df_shp_sum["mean"]-df_shp_sum["lo"], df_shp_sum["hi"]-df_shp_sum["mean"]],
        capsize=4)
plt.xticks(x, df_shp_sum["method"].tolist(), rotation=15)
plt.ylabel("Median gap (mm) — lower is better")
plt.title("Family-out summary: sharpness (mean ± bootstrap 95% CI)")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_family_out_summary_sharpness_bootstrap.png", dpi=300, bbox_inches="tight")
plt.close()

# F) Paired delta sharpness (Robust − Marginal) across families
if all(c in df_fo.columns for c in ["shp_r","shp_m"]):
    deltas = (df_fo["shp_r"] - df_fo["shp_m"]).values
    m, lo, hi = bootstrap_mean(deltas, B=4000, seed=SEED+29)
    pd.DataFrame({"delta_sharpness_rm_mean":[m],"lo":[lo],"hi":[hi]}).to_csv(
        SRC_DIR / "family_out_sharpness_delta_robust_minus_marginal.csv", index=False
    )
    plt.figure(figsize=(6.0,3.6))
    plt.axhline(0, ls="--", color="gray")
    plt.scatter(np.arange(len(deltas)), deltas, s=18, alpha=0.8)
    plt.hlines([m], 0, len(deltas)-1, colors="C0", linestyles="-", label=f"Mean Δ={m:.3f} (95% CI [{lo:.3f},{hi:.3f}])")
    plt.xlabel("Families"); plt.ylabel("Δ sharpness (Robust − Marginal, mm)")
    plt.title("Paired sharpness difference per family (negative is better for Robust)")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_family_out_sharpness_diff_robust_minus_marginal.png", dpi=300, bbox_inches="tight")
    plt.close()

# 1) Per-family scatter data (x vs Δ sharpness)
N = len(deltas)
points_cols = {
    "family_idx0": np.arange(N, dtype=int),
    "family_idx1": np.arange(1, N+1, dtype=int),
    "delta_sharpness_mm": deltas
}
if "family" in df_fo.columns:
    points_cols["family_label"] = df_fo["family"].astype(str).values
elif "fam" in df_fo.columns:
    points_cols["family_label"] = df_fo["fam"].astype(str).values

df_points = pd.DataFrame(points_cols)
df_points.to_csv(SRC_DIR / "family_out_sharpness_deltas_per_family.csv", index=False)

# 2) Summary stats (mean & CI)
pd.DataFrame({
    "mean_delta_mm": [m],
    "ci_lo_mm": [lo],
    "ci_hi_mm": [hi],
    "n_families": [N]
}).to_csv(SRC_DIR / "family_out_sharpness_deltas_summary.csv", index=False)

x0, x1 = 0, N-1
df_lines = pd.DataFrame({
    "x":      [x0, x1, x0, x1, x0, x1],
    "y":      [m,  m,  lo,  lo,  hi,  hi],
    "series": ["mean","mean","ci_lo","ci_lo","ci_hi","ci_hi"]
})
df_lines.to_csv(SRC_DIR / "family_out_sharpness_deltas_lines.csv", index=False)

print("Saved per-family CI table:", SRC_DIR / "family_out_per_family_coverage_wilson_all_methods.csv")
print("Saved robust plot order:", SRC_DIR / "family_out_robust_coverage_plot_order.csv")
print("Saved summary table:", SRC_DIR / "family_out_summary_bootstrap_means.csv")
print("Saved figures:")
print("-", FIGDIR / "fig_family_out_per_family_coverage_robust_wilson.png")
print("-", FIGDIR / "fig_family_out_excess_coverage_robust.png")
print("-", FIGDIR / "fig_family_out_per_family_coverage_small_multiples.png")
print("-", FIGDIR / "fig_family_out_summary_coverage_bootstrap.png")
print("-", FIGDIR / "fig_family_out_summary_sharpness_bootstrap.png")
print("-", FIGDIR / "fig_family_out_sharpness_diff_robust_minus_marginal.png")


# In[97]:


# add near other sklearn imports
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit

def _safe_family_split(groups, test_size=0.5, random_state=0, prefer_group_split=True):
    """
    Robust CAL/TEST split of indices given family labels.
    - If prefer_group_split=True: use GroupShuffleSplit to keep entire families in one split.
    - Else try StratifiedShuffleSplit on family labels; if any class has <2 samples or only 1 class, fall back to ShuffleSplit.
    Returns: (cal_idx, test_idx) as relative indices [0..n-1].
    """
    groups = np.asarray(groups).astype(str)
    n = len(groups)
    idx_all = np.arange(n)

    if prefer_group_split:
        # Need at least 2 unique groups to split
        if len(np.unique(groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            cal_rel, test_rel = next(gss.split(np.zeros(n), groups=groups))
            return cal_rel, test_rel
        # fall through to random split if only one family present

    # Try stratified; if any family has <2 or only one class → fallback
    vc = pd.Series(groups).value_counts()
    if (vc < 2).any() or len(vc) < 2:
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        cal_rel, test_rel = next(ss.split(np.zeros(n)))
        return cal_rel, test_rel

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    cal_rel, test_rel = next(sss.split(np.zeros(n), groups))
    return cal_rel, test_rel


# In[98]:


# Redundancy stress: dedup, stratified split, CP metrics, shift diagnostics, artifacts
def _wilson_ci(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * math.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), max(0.0, center - half), min(1.0, center + half)

if 'bootstrap_ci' not in globals():
    def bootstrap_ci(fn, n, B=1000, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        idx = np.arange(n)
        stats = np.empty(B, float)
        for b in range(B):
            res = rng.choice(idx, size=n, replace=True)
            stats[b] = float(fn(res))
        return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

if 'weighted_quantile' not in globals():
    def weighted_quantile(values, q, sample_weight=None):
        v = np.asarray(values, float)
        return float(np.quantile(v, q))

# Jitter & feature builders assumed from earlier steps:
SIG_ATPCT_DECIMALS = 2
if "signature" not in df.columns:
    atpct = (df[elem_cols].astype(float) * 100.0).round(SIG_ATPCT_DECIMALS)
    df = df.copy()
    df["signature"] = atpct.apply(lambda r: "|".join(f"{e}:{r[e]:.0{SIG_ATPCT_DECIMALS}f}" for e in elem_cols), axis=1)

# Map: each original row → its first occurrence signature row (kept) or dropped
dup_map = df[["signature","family"]].copy()
dup_map["orig_index"] = np.arange(len(df))
first_idx = dup_map.groupby("signature")["orig_index"].transform("min")
dup_map["kept_index"] = first_idx
dup_map["kept_flag"]  = (dup_map["orig_index"] == dup_map["kept_index"]).astype(int)

df_dedup = df.drop_duplicates(subset="signature", keep="first").reset_index(drop=True)
y_log_dedup = np.log(df_dedup[dmax_col].astype(float).values)
X_dedup     = build_features(df_dedup, elem_cols)

n_removed = int((dup_map["kept_flag"] == 0).sum())
print(f"[Dedup] {n_removed} duplicates dropped ({100.0*n_removed/len(df):.1f}% of rows).")

# 2) Family-stratified 80/10/10 split on deduped set
fam_labels = df_dedup["family"].astype(str).values

rel_tr_big, rel_hold = _safe_family_split(
    fam_labels, test_size=0.20, random_state=SEED, prefer_group_split=True
)
tr_big = rel_tr_big
hold   = rel_hold

fam_hold = fam_labels[hold]
rel_ca, rel_te = _safe_family_split(
    fam_hold, test_size=0.50, random_state=SEED+1, prefer_group_split=True
)

ca = hold[rel_ca]
te = hold[rel_te]
tr = tr_big

print(f"[Redundancy split] TR={len(tr)}  CA={len(ca)}  TE={len(te)}  "
      f"| fam(TR)={pd.Series(fam_labels[tr]).nunique()}  "
      f"fam(CA)={pd.Series(fam_labels[ca]).nunique()}  "
      f"fam(TE)={pd.Series(fam_labels[te]).nunique()}")

X_tr, y_tr = X_dedup.iloc[tr], y_log_dedup[tr]
X_ca, y_ca = X_dedup.iloc[ca], y_log_dedup[ca]
X_te, y_te = X_dedup.iloc[te], y_log_dedup[te]

# Fraction matrices for robust CP
X_ca_elem = df_dedup.loc[ca, elem_cols].to_numpy(float)
X_te_elem = df_dedup.loc[te, elem_cols].to_numpy(float)

# 3) Quantile model (prefer your trained CatBoost high-τ)
tau_used = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU
if 'cat_qt_hi' in globals() and cat_qt_hi is not None:
    _q_model = cat_qt_hi
elif 'cat_qt' in globals() and cat_qt is not None and abs(getattr(cat_qt, "get_params", lambda: {"loss_function": ""})().get("loss_function","").endswith(str(tau_used)) or True):
    _q_model = cat_qt
    _q_model.fit(X_tr, y_tr)

q_ca = np.asarray(_q_model.predict(X_ca), float)
q_te = np.asarray(_q_model.predict(X_te), float)

# 4) Marginal CP (lower-bound score = max(0, qτ − y))
S_ca = np.maximum(0.0, q_ca - y_ca)
q_m  = weighted_quantile(S_ca, 1 - ALPHA)
L_marg_mm = np.exp(q_te - q_m)

# 5) Robust CP at ε (fractions L1)
if 'robust_scores_lower' not in globals():
    def robust_scores_lower(y_true, model, X_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
        rng = np.random.default_rng(SEED) if rng is None else rng
        y_true = np.asarray(y_true, float)
        S = np.empty_like(y_true)
        for i, (y_i, x_i) in enumerate(zip(y_true, X_elem)):
            Xj = jitter_in_L1_ball_simplex(x_i, eps=eps, K=K, rng=rng)
            feats = make_features_from_compositions(Xj)
            qj = np.asarray(model.predict(feats), float)
            S[i] = max(0.0, float(np.min(qj)) - y_i)
        return S

rng_local = np.random.default_rng(SEED+404)
S_ca_rob = robust_scores_lower(y_ca, _q_model, X_ca_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_local)
q_rob    = weighted_quantile(S_ca_rob, 1 - ALPHA)

def _robust_L_for_test(model, X_elem, q_cal_rob, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
    rng = np.random.default_rng(SEED+405) if rng is None else rng
    L_mm = np.zeros(len(X_elem))
    for i, x in enumerate(X_elem):
        Xj = jitter_in_L1_ball_simplex(x, eps=eps, K=K, rng=rng)
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), float)
        L_mm[i] = float(np.exp(np.min(qj) - q_cal_rob))
    return L_mm

L_rob_mm = _robust_L_for_test(_q_model, X_te_elem, q_rob, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_local)

# 6) Metrics on the deduped test set (point, Wilson, bootstrap)
y_te_mm = np.exp(y_te)

def _cov_and_ci(L):
    hits = (y_te_mm >= L).astype(int)
    k, n = int(hits.sum()), int(len(hits))
    p, wlo, whi = _wilson_ci(k, n)
    # bootstrap CI
    def stat(res_idx):
        i = np.asarray(res_idx, int)
        return float((y_te_mm[i] >= L[i]).mean())
    bmean, blo, bhi = bootstrap_ci(stat, n, B=1000, rng=rng_local)
    return dict(point=float(p), wilson_lo=float(wlo), wilson_hi=float(whi),
                boot_lo=float(blo), boot_hi=float(bhi), n=n, k=k)

cov_m = _cov_and_ci(L_marg_mm)
cov_r = _cov_and_ci(L_rob_mm)

shp_m = float(np.median(np.exp(q_te) - L_marg_mm))
shp_r = float(np.median(np.exp(q_te) - L_rob_mm))

# Discovery metric: Precision@20 at ≥15 mm using robust ranking
DSTAR_DISC = 5.0
order = np.argsort(-L_rob_mm)
k = min(20, len(order))
p20 = float(np.mean(y_te_mm[order[:k]] >= DSTAR_DISC)) if k > 0 else np.nan

# 7) Shift diagnostics (CAL vs TEST): density-ratio weights & ESS, classifier AUC
clf = LogisticRegression(max_iter=1000, random_state=SEED)
X_cls = np.vstack([X_ca.values, X_te.values])
y_cls = np.hstack([np.zeros(len(X_ca)), np.ones(len(X_te))])
clf.fit(X_cls, y_cls)
proba_cal = clf.predict_proba(X_ca.values)[:, 1]
w = np.clip(proba_cal / np.maximum(1.0 - proba_cal, 1e-9), 0.0, 1e6)
ess = float((w.sum()**2) / np.sum(w**2))
auc = float(roc_auc_score(y_cls, clf.predict_proba(X_cls)[:,1]))

# 8) Save artifacts (metrics JSON, bounds CSV, mapping CSV, split JSON) + figures
REPDIR = OUTDIR / "reports"
REPDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUTDIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)
FIGDIR = OUTDIR / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

dedup_metrics = {
    "n_total": int(len(df)),
    "n_dedup": int(len(df_dedup)),
    "n_removed": n_removed,
    "frac_removed": float(n_removed / len(df)),
    "alpha": float(ALPHA),
    "tau_used": float(tau_used),
    "robust_eps_fraction_L1": float(ROBUST_EPS),
    "coverage": {
        "marginal": cov_m,
        "robust":   cov_r
    },
    "sharpness_median_gap_mm": {"marginal": shp_m, "robust": shp_r},
    "discovery": {"precision20_at_15mm": p20},
    "shift_diagnostics": {"ESS_cal_weights": ess, "AUC_cal_vs_test": auc},
}
with open(REPDIR/"redundancy_stress_metrics.json","w") as f:
    json.dump(dedup_metrics, f, indent=2)

pd.DataFrame({
    "true_mm": y_te_mm,
    "qhat_mm": np.exp(q_te),
    "L_marginal_mm": L_marg_mm,
    "L_robust_mm":   L_rob_mm,
    "covered_marginal": (y_te_mm >= L_marg_mm).astype(int),
    "covered_robust":   (y_te_mm >= L_rob_mm).astype(int),
}).to_csv(SRC_DIR/"redundancy_stress_test_bounds.csv", index=False)

dup_map.to_csv(SRC_DIR/"redundancy_dedup_mapping.csv", index=False)

with open(REPDIR/"redundancy_dedup_splits.json","w") as f:
    json.dump({"train": tr.tolist(), "cal": ca.tolist(), "test": te.tolist()}, f, indent=2)

# Figures
def _figsave(p):
    plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

# (i) Coverage bars with Wilson + bootstrap whiskers
plt.figure(figsize=(6.6,3.9))
labels = ["Marginal", f"Robust (±{int(ROBUST_EPS*100)} at.%)"]
pts = [cov_m["point"], cov_r["point"]]
wil_lo = [cov_m["point"]-cov_m["wilson_lo"], cov_r["point"]-cov_r["wilson_lo"]]
wil_hi = [cov_m["wilson_hi"]-cov_m["point"], cov_r["wilson_hi"]-cov_r["point"]]
x = np.arange(2)
plt.bar(x, pts, alpha=0.9, label="Point (Wilson CI)")
plt.errorbar(x, pts, yerr=[wil_lo, wil_hi], fmt="none", capsize=4, color="black")
plt.axhline(1-ALPHA, ls="--", color="gray", label=f"Target {1-ALPHA:.2f}")
plt.xticks(x, labels, rotation=15)
plt.ylim(0,1); plt.ylabel("Coverage"); plt.title("Redundancy stress — coverage on dedup TEST")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR/"fig_redundancy_stress_coverage.png")

# (ii) Sharpness box for exp(q̂) − L
gap_m = np.exp(q_te) - L_marg_mm
gap_r = np.exp(q_te) - L_rob_mm
plt.figure(figsize=(6.6,3.9))
plt.boxplot([gap_m, gap_r], labels=labels, showfliers=False)
plt.ylabel("Gap (mm) — lower is better"); plt.title("Redundancy stress — sharpness")
_figsave(FIGDIR/"fig_redundancy_stress_sharpness.png")

print("Redundancy stress ✓  Saved:")
print("-", REPDIR / "redundancy_stress_metrics.json")
print("-", SRC_DIR / "redundancy_stress_test_bounds.csv")
print("-", SRC_DIR / "redundancy_dedup_mapping.csv")
print("-", REPDIR / "redundancy_dedup_splits.json")
print("-", FIGDIR / "fig_redundancy_stress_coverage.png")
print("-", FIGDIR / "fig_redundancy_stress_sharpness.png")





from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit

if "_safe_family_split" not in globals():
    def _safe_family_split(groups, test_size=0.5, random_state=0, prefer_group_split=True):
        """
        Robust CAL/TEST split given family labels.
        - If possible, keep entire families together (GroupShuffleSplit).
        - Otherwise try stratified by family.
        - If that still isn't feasible (singletons or one family), fall back to ShuffleSplit.
        Returns: (idx_cal_relative, idx_test_relative) in [0..n-1].
        """
        groups = np.asarray(groups).astype(str)
        n = len(groups)

        if prefer_group_split and len(np.unique(groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            cal_rel, test_rel = next(gss.split(np.zeros(n), groups=groups))
            return cal_rel, test_rel

        # Try stratified if every class has at least 2 samples and ≥2 classes exist
        vc = pd.Series(groups).value_counts()
        if (vc >= 2).all() and len(vc) >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            cal_rel, test_rel = next(sss.split(np.zeros(n), groups))
            return cal_rel, test_rel

        # Fallback: plain random split (no stratification)
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        cal_rel, test_rel = next(ss.split(np.zeros(n)))
        return cal_rel, test_rel





# Heaping stress: round labels at several granularities and re-evaluate CP
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
for p in (FIGDIR, SRC_DIR): p.mkdir(parents=True, exist_ok=True)

def _wilson_ci(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
    return float(p), max(0.0, center - half), min(1.0, center + half)

if 'weighted_quantile' not in globals():
    def weighted_quantile(values, q, sample_weight=None):
        v = np.asarray(values, float)
        return float(np.quantile(v, q))

if 'bootstrap_ci' not in globals():
    def bootstrap_ci(fn, n, B=1000, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        idx = np.arange(n)
        stats = np.empty(B, float)
        for b in range(B):
            res = rng.choice(idx, size=n, replace=True)
            stats[b] = float(fn(res))
        return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

if 'robust_scores_lower' not in globals():
    def robust_scores_lower(y_true, model, X_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
        rng = np.random.default_rng(SEED) if rng is None else rng
        y_true = np.asarray(y_true, float)
        S = np.empty_like(y_true)
        for i, (y_i, x_i) in enumerate(zip(y_true, X_elem)):
            Xj = jitter_in_L1_ball_simplex(x_i, eps=eps, K=K, rng=rng)
            feats = make_features_from_compositions(Xj)
            qj = np.asarray(model.predict(feats), float)
            S[i] = max(0.0, float(np.min(qj)) - y_i)
        return S

def _robust_L_for_test(model, X_elem, q_cal_rob, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
    rng = np.random.default_rng(SEED+505) if rng is None else rng
    L_mm = np.zeros(len(X_elem))
    for i, x in enumerate(X_elem):
        Xj = jitter_in_L1_ball_simplex(x, eps=eps, K=K, rng=rng)
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), float)
        L_mm[i] = float(np.exp(np.min(qj) - q_cal_rob))
    return L_mm

def stress_round_and_eval(df_in, elem_cols, dmax_col, round_to_mm=0.0, use_hi_tau=True):
    """
    Rounds labels to 'round_to_mm' and re-trains a τ-quantile model on the rounded labels.
    Uses family-stratified 80/10/10 split; evaluates marginal & robust CP with CIs.
    Returns a dict of metrics + small diagnostics.
    """
    rng_local = np.random.default_rng(SEED + int(1e4 * round_to_mm + 7))

    df_s = df_in.copy()
    d_raw = pd.to_numeric(df_s[dmax_col], errors="coerce").astype(float)
    if round_to_mm > 0:
        d_heaped = np.round(d_raw / round_to_mm) * round_to_mm
    else:
        d_heaped = d_raw.copy()
    df_s[dmax_col] = d_heaped

    df_s = df_s[~df_s[dmax_col].isna() & (df_s[dmax_col] > 0)].reset_index(drop=True)

    y_log_s = np.log(df_s[dmax_col].values)
    X_s     = build_features(df_s, elem_cols)

    fam = df_s["family"].astype(str).values
    
    # 80/20 TRAIN vs HOLD — keep families together if possible
    rel_tr_big, rel_hold = _safe_family_split(
        fam, test_size=0.20, random_state=SEED, prefer_group_split=True
    )
    tr_big = rel_tr_big
    hold   = rel_hold
    
    # 50/50 of HOLD → CAL vs TEST — again keep families together if possible
    fam_hold = fam[hold]
    rel_ca, rel_te = _safe_family_split(
        fam_hold, test_size=0.50, random_state=SEED+1, prefer_group_split=True
    )
    
    # absolute indices
    ca = hold[rel_ca]
    te = hold[rel_te]
    tr = tr_big
    
    # sanity log
    print(f"[Heaping stress] TR={len(tr)}  CA={len(ca)}  TE={len(te)} | "
          f"fam(TR)={pd.Series(fam[tr]).nunique()}  fam(CA)={pd.Series(fam[ca]).nunique()}  fam(TE)={pd.Series(fam[te]).nunique()}")

    X_tr, y_tr = X_s.iloc[tr], y_log_s[tr]
    X_ca, y_ca = X_s.iloc[ca], y_log_s[ca]
    X_te, y_te = X_s.iloc[te], y_log_s[te]

    X_ca_elem = df_s.loc[ca, elem_cols].to_numpy(float)
    X_te_elem = df_s.loc[te, elem_cols].to_numpy(float)

    tau_used = QT_TAU_HIGH if (use_hi_tau and 'QT_TAU_HIGH' in globals()) else QT_TAU
    cat_qt_hi.fit(X_tr, y_tr)

    q_ca = np.asarray(cat_qt_hi.predict(X_ca), float)
    q_te = np.asarray(cat_qt_hi.predict(X_te), float)

    obs_tau_cal  = float(np.mean(y_ca <= q_ca))
    obs_tau_test = float(np.mean(y_te <= q_te))

    S_ca = np.maximum(0.0, q_ca - y_ca)
    q_m  = weighted_quantile(S_ca, 1 - ALPHA)
    Lm   = np.exp(q_te - q_m)

    S_ca_rob = robust_scores_lower(y_ca, cat_qt_hi, X_ca_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_local)
    q_r      = weighted_quantile(S_ca_rob, 1 - ALPHA)
    Lr       = _robust_L_for_test(cat_qt_hi, X_te_elem, q_r, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_local)

    y_te_mm = np.exp(y_te)

    def _cov_and_ci(L):
        hits = (y_te_mm >= L).astype(int)
        k, n = int(hits.sum()), int(len(hits))
        p, lo, hi = _wilson_ci(k, n)
        def stat(res_idx):
            i = np.asarray(res_idx, int)
            return float((y_te_mm[i] >= L[i]).mean())
        bmean, blo, bhi = bootstrap_ci(stat, n, B=1000, rng=rng_local)
        return dict(point=float(p), wilson_lo=float(lo), wilson_hi=float(hi),
                    boot_lo=float(blo), boot_hi=float(bhi), n=n, k=k)

    cov_m = _cov_and_ci(Lm)
    cov_r = _cov_and_ci(Lr)
    shp_m = float(np.median(np.exp(q_te) - Lm))
    shp_r = float(np.median(np.exp(q_te) - Lr))

    return {
        "round_to_mm": float(round_to_mm),
        "tau": float(tau_used),
        "alpha": float(ALPHA),
        "eps": float(ROBUST_EPS),
        "n_total": int(len(df_s)),
        "coverage_marginal": cov_m,
        "coverage_robust":   cov_r,
        "sharpness_median_mm_marginal": shp_m,
        "sharpness_median_mm_robust":   shp_r,
        "obs_tau_cal":  obs_tau_cal,
        "obs_tau_test": obs_tau_test,
    }

ROUNDING_GRID_MM = [0.0, 0.5, 1.0, 2.0]
results = []
for rmm in ROUNDING_GRID_MM:
    res = stress_round_and_eval(df, elem_cols, dmax_col, round_to_mm=rmm, use_hi_tau=True)
    results.append(res)

rows = []
for r in results:
    rows.append({
        "round_to_mm": r["round_to_mm"],
        "tau": r["tau"], "alpha": r["alpha"], "eps": r["eps"],
        "n_total": r["n_total"],
        "cov_m_point": r["coverage_marginal"]["point"],
        "cov_m_wil_lo": r["coverage_marginal"]["wilson_lo"],
        "cov_m_wil_hi": r["coverage_marginal"]["wilson_hi"],
        "cov_m_boot_lo": r["coverage_marginal"]["boot_lo"],
        "cov_m_boot_hi": r["coverage_marginal"]["boot_hi"],
        "cov_r_point": r["coverage_robust"]["point"],
        "cov_r_wil_lo": r["coverage_robust"]["wilson_lo"],
        "cov_r_wil_hi": r["coverage_robust"]["wilson_hi"],
        "cov_r_boot_lo": r["coverage_robust"]["boot_lo"],
        "cov_r_boot_hi": r["coverage_robust"]["boot_hi"],
        "sharp_m_med_mm": r["sharpness_median_mm_marginal"],
        "sharp_r_med_mm": r["sharpness_median_mm_robust"],
        "obs_tau_cal": r["obs_tau_cal"],
        "obs_tau_test": r["obs_tau_test"],
    })
df_heap = pd.DataFrame(rows).sort_values("round_to_mm").reset_index(drop=True)
df_heap.to_csv(SRC_DIR / "heaping_stress_metrics.csv", index=False)

x = df_heap["round_to_mm"].values
plt.figure(figsize=(6.6,3.9))
plt.errorbar(x, df_heap["cov_m_point"], 
             yerr=[df_heap["cov_m_point"]-df_heap["cov_m_boot_lo"],
                   df_heap["cov_m_boot_hi"]-df_heap["cov_m_point"]],
             marker="o", capsize=4, label="Marginal")
plt.errorbar(x, df_heap["cov_r_point"],
             yerr=[df_heap["cov_r_point"]-df_heap["cov_r_boot_lo"],
                   df_heap["cov_r_boot_hi"]-df_heap["cov_r_point"]],
             marker="o", capsize=4, label=f"Robust (±{int(ROBUST_EPS*100)} at.%)")
plt.axhline(1-ALPHA, ls="--", color="gray", label=f"Target {1-ALPHA:.2f}")
plt.xlabel("Label rounding (mm)")
plt.ylabel("Coverage (test)")
plt.ylim(0,1)
plt.title("Heaping stress — coverage vs rounding")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_heaping_coverage_vs_rounding.png", dpi=300, bbox_inches="tight")
plt.close()

# Sharpness vs rounding (median gap)
plt.figure(figsize=(6.6,3.9))
plt.plot(x, df_heap["sharp_m_med_mm"], marker="o", label="Marginal")
plt.plot(x, df_heap["sharp_r_med_mm"], marker="o", label=f"Robust (±{int(ROBUST_EPS*100)} at.%)")
plt.xlabel("Label rounding (mm)")
plt.ylabel("Median gap (mm) — lower is better")
plt.title("Heaping stress — sharpness vs rounding")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_heaping_sharpness_vs_rounding.png", dpi=300, bbox_inches="tight")
plt.close()

# Quantile calibration panel on CAL/TEST (OBS P(Y≤q̂τ))
plt.figure(figsize=(6.6,3.9))
plt.plot(x, df_heap["obs_tau_cal"], marker="o", label="CAL")
plt.plot(x, df_heap["obs_tau_test"], marker="o", label="TEST")
target_tau = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU
plt.axhline(target_tau, ls="--", color="gray", label=f"Target τ={target_tau:.2f}")
plt.ylim(0,1); plt.xlabel("Label rounding (mm)"); plt.ylabel("Observed P(Y ≤ q̂τ)")
plt.title("Heaping stress — quantile calibration vs rounding")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_heaping_quantile_calibration_vs_rounding.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print("-", SRC_DIR / "heaping_stress_metrics.csv")
print("-", FIGDIR / "fig_heaping_coverage_vs_rounding.png")
print("-", FIGDIR / "fig_heaping_sharpness_vs_rounding.png")
print("-", FIGDIR / "fig_heaping_quantile_calibration_vs_rounding.png")





# Heaping stress: plots + CSV export
if 'figsave' not in globals():
    def figsave(path):
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

robust_tag = f"Robust (±{int(ROBUST_EPS*100)} at.%)"

def _call_stress_round(df, elem_cols, dmax_col, rmm):
    """
    Call stress_round_and_eval regardless of its signature/schema:
    - If using the enhanced version: returns dict with nested coverage dicts, calibration, etc.
    - If using the light version: returns dict with simple scalars.
    """
    try:
        res = stress_round_and_eval(df, elem_cols, dmax_col, round_to_mm=rmm, use_hi_tau=True)
    except TypeError:
        res = stress_round_and_eval(df, elem_cols, dmax_col, round_to=rmm, use_hi_tau=True)
    return res

def _flatten_res(res, rmm):
    """
    Normalize result dict to a tidy, columnar row with:
    round_to_mm, cov_m_point + CI, cov_r_point + CI, sharpness (median mm), calibration, n_total.
    Works for both enhanced and light versions.
    """
    row = {"round_to_mm": float(rmm)}
    if isinstance(res.get("coverage_marginal"), dict):
        cm = res["coverage_marginal"]; cr = res["coverage_robust"]
        row.update({
            "tau": res.get("tau", np.nan),
            "alpha": res.get("alpha", np.nan),
            "eps": res.get("eps", np.nan),
            "n_total": res.get("n_total", np.nan),

            "cov_m_point": cm.get("point", np.nan),
            "cov_m_wil_lo": cm.get("wilson_lo", np.nan),
            "cov_m_wil_hi": cm.get("wilson_hi", np.nan),
            "cov_m_boot_lo": cm.get("boot_lo", np.nan),
            "cov_m_boot_hi": cm.get("boot_hi", np.nan),

            "cov_r_point": cr.get("point", np.nan),
            "cov_r_wil_lo": cr.get("wilson_lo", np.nan),
            "cov_r_wil_hi": cr.get("wilson_hi", np.nan),
            "cov_r_boot_lo": cr.get("boot_lo", np.nan),
            "cov_r_boot_hi": cr.get("boot_hi", np.nan),

            "sharp_m_med_mm": res.get("sharpness_median_mm_marginal", np.nan),
            "sharp_r_med_mm": res.get("sharpness_median_mm_robust", np.nan),

            "obs_tau_cal":  res.get("obs_tau_cal", np.nan),
            "obs_tau_test": res.get("obs_tau_test", np.nan),
        })
    else:
        row.update({
            "tau": np.nan, "alpha": float(ALPHA), "eps": float(ROBUST_EPS),
            "n_total": np.nan,

            "cov_m_point": float(res.get("coverage_marginal", np.nan)),
            "cov_m_wil_lo": np.nan, "cov_m_wil_hi": np.nan,
            "cov_m_boot_lo": np.nan, "cov_m_boot_hi": np.nan,

            "cov_r_point": float(res.get("coverage_robust", np.nan)),
            "cov_r_wil_lo": np.nan, "cov_r_wil_hi": np.nan,
            "cov_r_boot_lo": np.nan, "cov_r_boot_hi": np.nan,

            "sharp_m_med_mm": float(res.get("sharpness_marginal_mm", np.nan)),
            "sharp_r_med_mm": float(res.get("sharpness_robust_mm", np.nan)),

            "obs_tau_cal":  np.nan,
            "obs_tau_test": np.nan,
        })
    return row

ROUND_GRID = [0.0, 0.5, 1.0, 2.0]
rows = []
for rmm in ROUND_GRID:
    res = _call_stress_round(df, elem_cols, dmax_col, rmm)
    rows.append(_flatten_res(res, rmm))

df_heaping = pd.DataFrame(rows).sort_values("round_to_mm").reset_index(drop=True)
heaping_csv = SRC_DIR / "heaping_stress_summary.csv"
df_heaping.to_csv(heaping_csv, index=False)

# Plot 1: Coverage vs rounding (with bootstrap CIs if available)
x = df_heaping["round_to_mm"].values

plt.figure(figsize=(6.8, 4.0))

# Marginal
y_m = df_heaping["cov_m_point"].values
yl_m = df_heaping["cov_m_boot_lo"].values
yh_m = df_heaping["cov_m_boot_hi"].values
has_ci_m = np.isfinite(yl_m).all() and np.isfinite(yh_m).all()
if has_ci_m:
    yerr_m = np.vstack([y_m - yl_m, yh_m - y_m])
    plt.errorbar(x, y_m, yerr=yerr_m, marker="o", capsize=4, label="Marginal")
else:
    plt.plot(x, y_m, marker="o", label="Marginal")

# Robust
y_r = df_heaping["cov_r_point"].values
yl_r = df_heaping["cov_r_boot_lo"].values
yh_r = df_heaping["cov_r_boot_hi"].values
has_ci_r = np.isfinite(yl_r).all() and np.isfinite(yh_r).all()
if has_ci_r:
    yerr_r = np.vstack([y_r - yl_r, yh_r - y_r])
    plt.errorbar(x, y_r, yerr=yerr_r, marker="o", capsize=4, label=robust_tag)
else:
    plt.plot(x, y_r, marker="o", label=robust_tag)

plt.axhline(1-ALPHA, ls="--", color="gray", label=f"Target {1-ALPHA:.2f}")
plt.xlabel("Label rounding (mm)")
plt.ylabel("Coverage (test)")
plt.ylim(0, 1)
plt.title("Heaping stress — coverage vs rounding")
plt.legend(frameon=False, loc="lower right")
figsave(FIGDIR / "fig_heaping_coverage_vs_round.png")

# Plot 2: Sharpness vs rounding (median gap)
plt.figure(figsize=(6.8, 4.0))
plt.plot(x, df_heaping["sharp_m_med_mm"], marker="o", label="Marginal")
plt.plot(x, df_heaping["sharp_r_med_mm"], marker="o", label=robust_tag)
plt.xlabel("Label rounding (mm)")
plt.ylabel("Median gap (mm) — lower is better")
plt.title("Heaping stress — sharpness vs rounding")
plt.legend(frameon=False)
figsave(FIGDIR / "fig_heaping_sharpness_vs_round.png")

# Plot 3: Quantile calibration vs rounding
if "obs_tau_cal" in df_heaping.columns and "obs_tau_test" in df_heaping.columns:
    plt.figure(figsize=(6.8, 4.0))
    plt.plot(x, df_heaping["obs_tau_cal"], marker="o", label="CAL")
    plt.plot(x, df_heaping["obs_tau_test"], marker="o", label="TEST")
    target_tau = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU
    plt.axhline(target_tau, ls="--", color="gray", label=f"Target τ={target_tau:.2f}")
    plt.ylim(0, 1)
    plt.xlabel("Label rounding (mm)")
    plt.ylabel("Observed P(Y ≤ q̂τ)")
    plt.title("Heaping stress — quantile calibration vs rounding")
    plt.legend(frameon=False, loc="lower right")
    figsave(FIGDIR / "fig_heaping_quantile_calibration_vs_round.png")

print("Saved CSV:", heaping_csv)
print("Saved figures:",
      FIGDIR / "fig_heaping_coverage_vs_round.png",
      ",",
      FIGDIR / "fig_heaping_sharpness_vs_round.png",
      "and (if available) fig_heaping_quantile_calibration_vs_round.png")





# Permutation sanity test (marginal + robust, CIs, precision@k, Spearman ρ)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

if 'figsave' not in globals():
    def figsave(path):
        plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

def wilson_ci_counts(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k/n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), max(0.0, center - half), min(1.0, center + half)

def precision_at_k_curve(L_mm, y_mm, thresh, ks=(5,10,20,50,100)):
    order = np.argsort(-np.asarray(L_mm, float))
    y_mm = np.asarray(y_mm, float)
    vals = []
    for k in ks:
        sel = order[:min(k, len(order))]
        vals.append(float(np.mean(y_mm[sel] >= thresh)))
    return np.array(vals), np.array(ks)

def make_quantile_estimator(tau):
    try:
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={tau}",
            eval_metric=f"Quantile:alpha={tau}",
            random_seed=SEED, verbose=False,
            allow_writing_files=False, thread_count=-1
        )
    except Exception:
        return GradientBoostingRegressor(
            loss="quantile", alpha=tau,
            n_estimators=200, learning_rate=0.485, max_depth=5,
            random_state=SEED
        )

# robust score fallback (lower-bound sign, on log-scale)
if 'robust_scores_lower' not in globals():
    def robust_scores_lower(y_true_log, q_model, comp_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
        if rng is None: rng = np.random.default_rng(SEED)
        y_true_log = np.asarray(y_true_log, float)
        S = np.empty_like(y_true_log)
        for i, (y_i, x_i) in enumerate(zip(y_true_log, comp_elem)):
            Xj = jitter_in_L1_ball_simplex(x_i, eps=eps, K=K, rng=rng)
            feats = make_features_from_compositions(Xj)
            qj = np.asarray(q_model.predict(feats), float)
            S[i] = max(0.0, float(np.min(qj)) - y_i)
        return S

tr, ca, te = split_random["train"], split_random["cal"], split_random["test"]

# compositions for CAL/TEST (fractions)
X_cal_elem = df.loc[ca, elem_cols].to_numpy(dtype=float)
X_test_elem = df.loc[te, elem_cols].to_numpy(dtype=float)

# Optional speed control for robust evaluation on TEST
N_EVAL_ROB = min(500, len(te)) 
rng_perm = np.random.default_rng(SEED + 606)
te_sub = np.asarray(te)[rng_perm.choice(len(te), size=N_EVAL_ROB, replace=False)] if N_EVAL_ROB < len(te) else np.asarray(te)
X_test_elem_sub = df.loc[te_sub, elem_cols].to_numpy(dtype=float)

y_perm = y_log.copy()
rng_perm.shuffle(y_perm)

tau_used = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU
q_model_perm = make_quantile_estimator(tau_used)
q_model_perm.fit(X.iloc[tr], y_perm[tr])

# predictions
q_ca = q_model_perm.predict(X.iloc[ca])
q_te = q_model_perm.predict(X.iloc[te])
q_te_sub = q_model_perm.predict(X.iloc[te_sub])

# marginal CP on permuted world (correct one-sided score for LOWER bound)
S_ca_m = np.maximum(0.0, q_ca - y_perm[ca])
q_m = weighted_quantile(S_ca_m, 1 - ALPHA) if 'weighted_quantile' in globals() else float(np.quantile(S_ca_m, 1 - ALPHA))
L_perm_marg = np.exp(q_te - q_m)
L_perm_marg_sub = np.exp(q_te_sub - q_m)

# robust CP on permuted world (ε tolerance on compositions)
S_ca_r = robust_scores_lower_dispatch(y_perm[ca], q_model_perm, X_cal_elem, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_perm)
q_r = weighted_quantile(S_ca_r, 1 - ALPHA) if 'weighted_quantile' in globals() else float(np.quantile(S_ca_r, 1 - ALPHA))

def robust_L_for_test(model, comp_elem, q_cal_rob, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=None):
    if rng is None: rng = np.random.default_rng(SEED + 707)
    L_mm = np.zeros(len(comp_elem))
    for i, x in enumerate(comp_elem):
        Xj = jitter_in_L1_ball_simplex(x, eps=eps, K=K, rng=rng)
        feats = make_features_from_compositions(Xj)
        qj = np.asarray(model.predict(feats), float)
        L_mm[i] = float(np.exp(np.min(qj) - q_cal_rob))
    return L_mm

L_perm_rob_sub = robust_L_for_test(q_model_perm, X_test_elem_sub, q_r, eps=ROBUST_EPS, K=ROBUST_SAMPLES, rng=rng_perm)

y_true_test_mm   = np.exp(y_log[te])
y_true_test_sub_mm = np.exp(y_log[te_sub])
q_te_mm          = np.exp(q_te)
q_te_sub_mm      = np.exp(q_te_sub)

# coverage + Wilson CI
k_m = int(np.sum(y_true_test_mm >= L_perm_marg)); n_m = int(len(y_true_test_mm))
cov_m_point, cov_m_lo, cov_m_hi = wilson_ci_counts(k_m, n_m)

k_r = int(np.sum(y_true_test_sub_mm >= L_perm_rob_sub)); n_r = int(len(y_true_test_sub_mm))
cov_r_point, cov_r_lo, cov_r_hi = wilson_ci_counts(k_r, n_r)

# sharpness (median gap)
sharp_m_med = float(np.median(q_te_mm      - L_perm_marg))
sharp_r_med = float(np.median(q_te_sub_mm  - L_perm_rob_sub))

# rank correlation (should be ~0 under permutation)
try:
    rho_m, p_m = spearmanr(L_perm_marg,     y_true_test_mm)
    rho_r, p_r = spearmanr(L_perm_rob_sub,  y_true_test_sub_mm)
except Exception:
    r1 = pd.Series(L_perm_marg).rank().to_numpy()
    r2 = pd.Series(y_true_test_mm).rank().to_numpy()
    rho_m = float(np.corrcoef(r1, r2)[0,1]); p_m = np.nan
    r1 = pd.Series(L_perm_rob_sub).rank().to_numpy()
    r2 = pd.Series(y_true_test_sub_mm).rank().to_numpy()
    rho_r = float(np.corrcoef(r1, r2)[0,1]); p_r = np.nan

print(f"[Permutation sanity] τ={tau_used:.2f}, α={ALPHA:.2f}, ε={ROBUST_EPS:.2f} (eval robust on n={n_r} subsample)")
print(f"  Marginal: coverage={cov_m_point:.3f}  (95% CI {cov_m_lo:.3f}–{cov_m_hi:.3f}), sharpness(med)={sharp_m_med:.2f} mm,  ρ={rho_m:.3f}")
print(f"  Robust  : coverage={cov_r_point:.3f}  (95% CI {cov_r_lo:.3f}–{cov_r_hi:.3f}), sharpness(med)={sharp_r_med:.2f} mm,  ρ={rho_r:.3f}")

# precision@k curves (perm) for both methods vs baseline prevalence
rows = []
for Dstar in THRESHOLDS_MM:
    ks = np.array([5,10,20,50,100])
    base_full = float(np.mean(y_true_test_mm >= Dstar))
    base_sub  = float(np.mean(y_true_test_sub_mm >= Dstar))

    # marginal (full test)
    vals_m, ks_m = precision_at_k_curve(L_perm_marg, y_true_test_mm, Dstar, ks=tuple(ks))
    
    # robust (subset)
    vals_r, ks_r = precision_at_k_curve(L_perm_rob_sub, y_true_test_sub_mm, Dstar, ks=tuple(ks))

    plt.figure(figsize=(6.2,4.0))
    plt.axhline(base_full, ls="--", lw=1, label=f"Baseline (prevalence={base_full:.2f})")
    plt.plot(ks_m, vals_m, marker="o", label="Permutation (Marginal)")
    plt.plot(ks_r, vals_r, marker="s", label=f"Permutation (Robust, n={n_r})")
    plt.xlabel("k"); plt.ylabel(f"Precision@k (≥ {int(Dstar)} mm)")
    plt.ylim(0,1); plt.title(f"Permutation test: precision@k (≥ {int(Dstar)} mm)")
    plt.legend(frameon=False, loc="lower right")
    figsave(FIGDIR / f"fig_perm_precision_k_{int(Dstar)}mm.png")

    for k, v in zip(ks_m, vals_m):
        rows.append({"method":"marginal", "Dstar_mm": float(Dstar), "k": int(k), "precision": float(v), "baseline": base_full})
    for k, v in zip(ks_r, vals_r):
        rows.append({"method":"robust",   "Dstar_mm": float(Dstar), "k": int(k), "precision": float(v), "baseline": base_sub})

pd.DataFrame(rows).to_csv(SRC_DIR / "perm_precision_at_k_both.csv", index=False)

# Scatter: L_perm vs TRUE (marginal & robust subset)
plt.figure(figsize=(5.6,5.2))
plt.scatter(L_perm_marg,    y_true_test_mm,     s=16, alpha=0.6, label="Marginal")
plt.scatter(L_perm_rob_sub, y_true_test_sub_mm, s=16, alpha=0.6, label="Robust (subset)")
lims=[min(np.min(L_perm_marg), np.min(y_true_test_mm)), max(np.max(L_perm_marg), np.max(y_true_test_mm))]
plt.plot(lims, lims, ls="--", lw=1); plt.xlim(lims); plt.ylim(lims); plt.gca().set_aspect('equal','box')
plt.xlabel("Permutation certified lower bound L (mm)")
plt.ylabel("True Dmax (mm)")
plt.title("Permutation sanity: L vs truth (should show no enrichment)")
plt.legend(frameon=False, loc="upper left")
figsave(FIGDIR / "fig_perm_L_vs_true_scatter.png")

perm_summary = {
    "tau": float(tau_used),
    "alpha": float(ALPHA),
    "robust_eps": float(ROBUST_EPS),
    "n_test_full": int(len(te)),
    "n_test_subset_robust": int(n_r),
    "coverage": {
        "marginal": {"point": cov_m_point, "wilson_lo": cov_m_lo, "wilson_hi": cov_m_hi},
        "robust":   {"point": cov_r_point, "wilson_lo": cov_r_lo, "wilson_hi": cov_r_hi},
    },
    "sharpness_median_gap_mm": {"marginal": sharp_m_med, "robust": sharp_r_med},
    "spearman": {
        "marginal": {"rho": float(rho_m), "p": None if isinstance(p_m,float) and np.isnan(p_m) else float(p_m)},
        "robust":   {"rho": float(rho_r), "p": None if isinstance(p_r,float) and np.isnan(p_r) else float(p_r)},
    },
    "thresholds_mm": [float(x) for x in THRESHOLDS_MM],
}
with open(OUTDIR / "reports" / "perm_sanity_summary.json", "w") as f:
    json.dump(perm_summary, f, indent=2)

print("Saved figures:",
      [str(FIGDIR / f"fig_perm_precision_k_{int(D)}mm.png") for D in THRESHOLDS_MM] +
      [str(FIGDIR / "fig_perm_L_vs_true_scatter.png")])
print("Saved CSV:", SRC_DIR / "perm_precision_at_k_both.csv")
print("Saved JSON:", OUTDIR / "reports" / "perm_sanity_summary.json")





# Quantile parity & residual diagnostics (conditional + family CIs + heteroskedasticity + PIT)
FIGDIR = OUTDIR / "reports" / "figures"
SRC_DIR = OUTDIR / "source_data"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

def _figsave(p):
    try:
        figsave(p)
    except NameError:
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

def wilson_ci_counts(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    p = k/n
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z/denom) * np.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    return float(p), max(0.0, center - half), min(1.0, center + half)

try:
    make_quantile_estimator
except NameError:
    def make_quantile_estimator(tau):
        try:
            return CatBoostRegressor(
                loss_function=f"Quantile:alpha={tau}",
                eval_metric=f"Quantile:alpha={tau}",
                random_seed=SEED, verbose=False,
                allow_writing_files=False, thread_count=-1
            )
        except Exception:
            return GradientBoostingRegressor(
                loss="quantile", alpha=tau, n_estimators=200, learning_rate=0.485, max_depth=5,
                random_state=SEED
            )

tr, ca, te = split_random["train"], split_random["cal"], split_random["test"]
tau_used = QT_TAU_HIGH if 'QT_TAU_HIGH' in globals() else QT_TAU

q_model = make_quantile_estimator(tau_used)
q_model.fit(X.iloc[tr], y_log[tr])

q_te_log = np.asarray(q_model.predict(X.iloc[te]), float)
y_te_log = y_log[te].astype(float)
q_te_mm  = np.exp(q_te_log)
y_te_mm  = np.exp(y_te_log)

# 1) Parity (mm)
plt.figure(figsize=(6.2, 6.2))
plt.scatter(y_te_mm, q_te_mm, s=18, alpha=0.6)
mx = float(np.nanmax([np.nanmax(y_te_mm), np.nanmax(q_te_mm)]))
plt.plot([0, mx], [0, mx], ls="--", lw=1)
plt.xlabel("True Dmax (mm)")
plt.ylabel(f"Predicted $\\hat q_\\tau$ (mm), $\\tau={tau_used:.2f}$")
plt.title("Quantile parity (test)")
_figsave(FIGDIR / "fig_quantile_parity_mm.png")

# --- Export data for Origin (CSV) ---
y_arr = np.asarray(y_te_mm, dtype=float).ravel()
q_arr = np.asarray(q_te_mm, dtype=float).ravel()

df_parity = pd.DataFrame({
    "True_Dmax_mm": y_arr,
    "Predicted_qtau_mm": q_arr,
})
df_parity = df_parity.replace([np.inf, -np.inf], np.nan).dropna()
df_parity["Residual_mm"] = df_parity["Predicted_qtau_mm"] - df_parity["True_Dmax_mm"]
df_parity["Abs_Error_mm"] = df_parity["Residual_mm"].abs()
df_parity["Tau"] = float(tau_used)

if "SRC_DIR" in globals():
    export_dir = Path(SRC_DIR)
elif "FIGDIR" in globals():
    export_dir = Path(FIGDIR).parent / "source_data"
else:
    export_dir = Path.cwd() / "source_data"

export_dir.mkdir(parents=True, exist_ok=True)

csv_scatter = export_dir / f"quantile_parity_mm_tau_{float(tau_used):.2f}.csv"
df_parity.to_csv(csv_scatter, index=False)
mx = float(np.nanmax([np.nanmax(y_arr), np.nanmax(q_arr)]))
df_diag = pd.DataFrame({"x_mm": [0.0, mx], "y_mm": [0.0, mx]})
csv_diag = export_dir / f"quantile_parity_diag_tau_{float(tau_used):.2f}.csv"
df_diag.to_csv(csv_diag, index=False)
print(f"Saved:\n- {csv_scatter}\n- {csv_diag} (optional y=x line)")

# 2) Residuals r_τ = y − q̂τ (log space) & heteroskedasticity
r_te = y_te_log - q_te_log
plt.figure(figsize=(7.0, 4.8))
plt.scatter(q_te_log, r_te, s=12, alpha=0.5)
plt.axhline(0, ls="--", lw=1)
plt.xlabel("Predicted log $\\hat q_\\tau$")
plt.ylabel("Residual $r_\\tau = y - \\hat q_\\tau$")
plt.title(f"Quantile residuals vs prediction (τ={tau_used:.2f})")
_figsave(FIGDIR / "fig_quantile_residual_vs_pred_log.png")

# Heteroskedasticity: |r| vs prediction with running-median line
abs_r = np.abs(r_te)

qbins = np.quantile(q_te_log, np.linspace(0, 1, 16))
bid = np.digitize(q_te_log, qbins[1:-1], right=True)
med_x, med_absr, ns = [], [], []
for b in range(len(qbins)-1):
    m = (bid == b)
    if m.sum() == 0: continue
    med_x.append(float(np.median(q_te_log[m])))
    med_absr.append(float(np.median(abs_r[m])))
    ns.append(int(m.sum()))

plt.figure(figsize=(7.0, 4.2))
plt.scatter(q_te_log, abs_r, s=10, alpha=0.25, label="points")
plt.plot(med_x, med_absr, marker="o", lw=2, label="running median |residual|")
plt.xlabel("Predicted log $\\hat q_\\tau$")
plt.ylabel("|Residual|")
plt.title("Heteroskedasticity check")
plt.legend(frameon=False)
_figsave(FIGDIR / "fig_quantile_absres_vs_pred_log.png")

pd.DataFrame({
    "qhat_log_bin_center": med_x,
    "median_abs_residual": med_absr,
    "n_in_bin": ns
}).to_csv(SRC_DIR / "quantile_residual_running_median.csv", index=False)


if 'SRC_DIR' in globals():
    _SRC = Path(SRC_DIR)
else:
    if 'FIGDIR' in globals():
        _SRC = Path(FIGDIR).parent / "source_data"
    else:
        _SRC = Path("source_data")
_SRC.mkdir(parents=True, exist_ok=True)

_tau_tag = f"{tau_used:.2f}"
_mask = np.isfinite(q_te_log) & np.isfinite(y_te_log)
_df_scatter = pd.DataFrame({
    "predicted_log_qhat_tau": np.asarray(q_te_log)[_mask],
    "residual_r_tau":        np.asarray(r_te)[_mask],
    "abs_residual":          np.asarray(np.abs(r_te))[_mask],
    "true_log_y":            np.asarray(y_te_log)[_mask],
})
_scatter_path = _SRC / f"quantile_residuals_vs_pred_log_tau{_tau_tag}.csv"
_df_scatter.to_csv(_scatter_path, index=False)

qbins = np.quantile(q_te_log, np.linspace(0, 1, 16))
bid = np.digitize(q_te_log, qbins[1:-1], right=True)

med_x, med_absr, n_in_bin, bin_lo, bin_hi = [], [], [], [], []
for b in range(len(qbins)-1):
    m = (bid == b)
    if m.sum() == 0:
        continue
    med_x.append(float(np.median(q_te_log[m])))
    med_absr.append(float(np.median(np.abs(r_te[m]))))
    n_in_bin.append(int(m.sum()))
    bin_lo.append(float(qbins[b]))
    bin_hi.append(float(qbins[b+1]))

_df_median = pd.DataFrame({
    "bin_low":  bin_lo,
    "bin_high": bin_hi,
    "median_predicted_log_qhat_tau": med_x,
    "median_abs_residual": med_absr,
    "n_points": n_in_bin,
})
_median_path = _SRC / f"heteroskedasticity_running_median_log_tau{_tau_tag}.csv"
_df_median.to_csv(_median_path, index=False)

print(f"[Origin export] Wrote:\n  - {_scatter_path}\n  - {_median_path}")


# 3) Conditional calibration: P(Y ≤ q̂τ | q̂τ in bin)
hit = (y_te_log <= q_te_log).astype(int)
obs_global, lo_g, hi_g = wilson_ci_counts(int(hit.sum()), len(hit))

rows = []
qbins_c = np.quantile(q_te_mm, np.linspace(0, 1, 11))
bid_c = np.digitize(q_te_mm, qbins_c[1:-1], right=True)
for b in range(len(qbins_c)-1):
    m = (bid_c == b)
    if m.sum() == 0: continue
    k = int(hit[m].sum()); n = int(m.sum())
    p, lo, hi = wilson_ci_counts(k, n)
    rows.append({
        "bin": b, "n": n,
        "qhat_mm_lo": float(qbins_c[b]),
        "qhat_mm_hi": float(qbins_c[b+1]),
        "qhat_mm_center": float(0.5*(qbins_c[b] + qbins_c[b+1])),
        "observed_P_hit": p, "ci_lo": lo, "ci_hi": hi,
        "target_tau": float(tau_used)
    })
df_cond_cal = pd.DataFrame(rows)
df_cond_cal.to_csv(SRC_DIR / "quantile_conditional_calibration_bins.csv", index=False)

plt.figure(figsize=(6.6, 4.0))
plt.errorbar(df_cond_cal["qhat_mm_center"], df_cond_cal["observed_P_hit"],
             yerr=[df_cond_cal["observed_P_hit"]-df_cond_cal["ci_lo"],
                   df_cond_cal["ci_hi"]-df_cond_cal["observed_P_hit"]],
             fmt="o-", capsize=3, lw=1.5)
plt.axhline(tau_used, ls="--", label=f"Target τ={tau_used:.2f}")
plt.xlabel("Predicted $\\hat q_\\tau$ (mm) — bin centers")
plt.ylabel("Observed P(Y ≤ $\\hat q_\\tau$)")
plt.ylim(0, 1); plt.title("Conditional quantile calibration (test)")
plt.legend(frameon=False, loc="lower right")
_figsave(FIGDIR / "fig_quantile_conditional_calibration.png")

# 4) Family-wise calibration with Wilson CIs (only families with >=5 test samples)
fam_te = df.iloc[te]["family"].values if "family" in df.columns else np.repeat("all", len(te))
rows = []
for fam, idx in pd.Series(range(len(te)), index=fam_te).groupby(level=0):
    ii = idx.values
    if len(ii) < 5: 
        continue
    k = int(np.sum(hit[ii] == 1)); n = int(len(ii))
    p, lo, hi = wilson_ci_counts(k, n)
    rows.append({"family": fam, "n": n, "coverage_obs": p, "ci_lo": lo, "ci_hi": hi, "target_tau": float(tau_used)})
df_fam_cal = pd.DataFrame(rows).sort_values(["n","family"], ascending=[False, True]).reset_index(drop=True)
df_fam_cal.to_csv(SRC_DIR / "quantile_family_calibration_wilson.csv", index=False)

if len(df_fam_cal):
    df_fc = df_fam_cal.copy()
    df_fc = df_fc.replace([np.inf, -np.inf], np.nan).dropna(subset=["coverage_obs","ci_lo","ci_hi"])
    lo = np.minimum(df_fc["ci_lo"].values, df_fc["ci_hi"].values)
    hi = np.maximum(df_fc["ci_lo"].values, df_fc["ci_hi"].values)
    p  = df_fc["coverage_obs"].values
    left_err  = np.clip(p - lo, 0.0, None)
    right_err = np.clip(hi - p, 0.0, None)

    p_plot = np.clip(p, 0.0, 1.0)
    plt.figure(figsize=(8.2, max(3.6, 0.28*len(df_fc))))
    y = np.arange(len(df_fc))
    plt.barh(
        y, p_plot,
        xerr=[left_err, right_err],
        capsize=3, alpha=0.9
    )
    plt.yticks(y, [f"{f} (n={n})" for f, n in zip(df_fc["family"], df_fc["n"])])
    plt.axvline(tau_used, ls="--", label=f"Target τ={tau_used:.2f}")
    plt.xlabel("Observed P(Y ≤ $\\hat q_\\tau$)")
    plt.xlim(0, 1)
    plt.title("Quantile calibration by family (95% Wilson CI)")
    plt.legend(frameon=False, loc="lower right")
    _figsave(FIGDIR / "fig_quantile_calibration_by_family.png")


# 5) PIT-style curve across τ if you trained multiple τ models
if 'cat_qt_grid' in globals() and isinstance(cat_qt_grid, dict) and len(cat_qt_grid) >= 3:
    taus = sorted(cat_qt_grid.keys())
    rows = []
    for t in taus:
        q_t_log = np.asarray(cat_qt_grid[t].predict(X.iloc[te]), float)
        hit_t = (y_te_log <= q_t_log).astype(int)
        k, n = int(hit_t.sum()), int(len(hit_t))
        p, lo, hi = wilson_ci_counts(k, n)
        rows.append({"tau": float(t), "observed": p, "ci_lo": lo, "ci_hi": hi})
    df_pit = pd.DataFrame(rows).sort_values("tau")
    df_pit.to_csv(SRC_DIR / "quantile_pit_curve.csv", index=False)
    
    df_pit_origin_wide = (
        df_pit.assign(ideal=df_pit["tau"])
              .rename(columns={
                  "tau": "Nominal_tau",
                  "observed": "Observed",
                  "ci_lo": "CI_Lo",
                  "ci_hi": "CI_Hi",
                  "ideal": "Ideal",
              })
              .loc[:, ["Nominal_tau", "Observed", "CI_Lo", "CI_Hi", "Ideal"]]
    )
    df_pit_origin_wide.to_csv(SRC_DIR / "quantile_pit_curve_for_origin_wide.csv", index=False)    
    df_pit_origin_long = df_pit_origin_wide.melt(
        id_vars=["Nominal_tau"],
        value_vars=["Observed", "CI_Lo", "CI_Hi", "Ideal"],
        var_name="Curve",
        value_name="Value",
    )
    df_pit_origin_long.to_csv(SRC_DIR / "quantile_pit_curve_for_origin_long.csv", index=False)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(df_pit["tau"], df_pit["observed"], marker="o", label="Observed")
    plt.fill_between(df_pit["tau"], df_pit["ci_lo"], df_pit["ci_hi"], alpha=0.2, label="95% CI")
    plt.plot([0,1], [0,1], ls="--", lw=1, label="Ideal")
    plt.xlabel("Nominal τ"); plt.ylabel("Observed P(Y ≤ $\\hat q_\\tau$)")
    plt.title("PIT-style calibration across quantiles (test)")
    plt.xlim(0,1); plt.ylim(0,1); plt.legend(frameon=False, loc="best")
    _figsave(FIGDIR / "fig_quantile_pit_curve.png")

summary = {
    "tau_used": float(tau_used),
    "global_obs_hit_rate": float(obs_global),
    "global_wilson_lo": float(lo_g),
    "global_wilson_hi": float(hi_g),
    "n_test": int(len(te))
}
with open(OUTDIR / "reports" / "quantile_diag_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved figures:",
      FIGDIR / "fig_quantile_parity_mm.png",
      FIGDIR / "fig_quantile_residual_vs_pred_log.png",
      FIGDIR / "fig_quantile_absres_vs_pred_log.png",
      FIGDIR / "fig_quantile_conditional_calibration.png",
      (FIGDIR / "fig_quantile_calibration_by_family.png" if len(df_fam_cal) else "(no family plot)"),
      (FIGDIR / "fig_quantile_pit_curve.png" if 'cat_qt_grid' in globals() and len(cat_qt_grid)>=3 else "(no PIT)"))
print("Saved CSVs:",
      SRC_DIR / "quantile_residual_running_median.csv",
      SRC_DIR / "quantile_conditional_calibration_bins.csv",
      SRC_DIR / "quantile_family_calibration_wilson.csv",
      (SRC_DIR / "quantile_pit_curve.csv" if 'cat_qt_grid' in globals() and len(cat_qt_grid)>=3 else "(no PIT CSV)"))
print("Saved JSON:", OUTDIR / "reports" / "quantile_diag_summary.json")
