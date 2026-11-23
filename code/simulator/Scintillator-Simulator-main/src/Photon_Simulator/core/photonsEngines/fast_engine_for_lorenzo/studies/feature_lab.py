"""Feature lab: systematic search over timing-feature blocks and combinations
for single-fiber position reconstruction (v mainly; u reported for honesty).

Blocks (all from ABSOLUTE arrival times, t=0 at particle hit):
  count      : [N]
  abs        : abs-moments (N, mean, std, skew, t0, q05, q50, q90)
  gemini7    : N + shifted-gamma MLE (k, theta, t0) + t_(1), q10, q50
  early      : order statistics of the FIRST photons - t_(1),t_(2),t_(3),
               t_(5),t_(10), mean/std of first 10 (cleanest time-of-flight)
  quant      : fine absolute quantiles q01..q99 (9 values)
  kde16      : gaussian KDE evaluated on a 16-point grid over 0-40 ns
  kdeshape   : KDE descriptors - peak time, peak height, rise time
               (10%->peak), FWHM, tail fraction >20 ns, log-tail slope
  logwin     : photon fractions in log-spaced windows 0-1-2-4-8-16-32-64 ns + N

Combos = concatenations; ENSEMBLE = mean of per-block model predictions
(abs, gemini7, kde16, early, logwin) - late fusion, robust at small n.

Model: HistGradientBoosting, 5-fold CV z-MSE (1.0 = predict-mean).
Effective light yield emulated by Poisson-thinning the saved photons.

Run:  .venv-srso/Scripts/python.exe studies/feature_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))
import common  # noqa: E402
from common import set_seed  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

RAW = STUDY / "raw_photons.npz"
LY_SIM = 10000.0
LY_EFF = [10000]        # focus: full light yield only
KDE_GRID = np.linspace(0.0, 40.0, 16)
WIN_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64])


# ---------------- feature blocks ----------------
def f_count(t):
    return [len(t)]


def f_abs(t):
    if len(t) < 3:
        return [len(t)] + [0] * 7
    return [len(t), t.mean(), t.std(), float(stats.skew(t)), float(t.min()),
            *np.quantile(t, [0.05, 0.5, 0.9])]


def f_gemini7(t):
    if len(t) < 5 or t.std() == 0:
        return [len(t), 0, 0, 0, 0, 0, 0]
    try:
        a, loc, sc = stats.gamma.fit(t)
    except Exception:
        a, loc, sc = 0.0, float(t.min()), 0.0
    return [len(t), float(a), float(sc), float(loc), float(t.min()),
            *np.quantile(t, [0.1, 0.5])]


def f_early(t):
    if len(t) < 10:
        return [0] * 7
    s = np.sort(t)
    return [s[0], s[1], s[2], s[4], s[9], s[:10].mean(), s[:10].std()]


def f_quant(t):
    if len(t) < 3:
        return [0] * 9
    return list(np.quantile(t, [0.01, 0.05, 0.1, 0.25, 0.5,
                                0.75, 0.9, 0.95, 0.99]))


def f_kde16(t):
    if len(t) < 5 or t.std() == 0:
        return [0.0] * len(KDE_GRID)
    try:
        return list(stats.gaussian_kde(t)(KDE_GRID))
    except Exception:
        return [0.0] * len(KDE_GRID)


def f_kdeshape(t):
    if len(t) < 5 or t.std() == 0:
        return [0.0] * 6
    try:
        grid = np.linspace(0, 60, 240)
        d = stats.gaussian_kde(t)(grid)
    except Exception:
        return [0.0] * 6
    ip = int(np.argmax(d))
    peak_t, peak_h = float(grid[ip]), float(d[ip])
    # rise time: last crossing of 10% peak BEFORE the peak -> peak
    pre = np.where(d[:ip + 1] < 0.1 * peak_h)[0]
    rise = peak_t - float(grid[pre[-1]]) if len(pre) else peak_t
    half = np.where(d >= 0.5 * peak_h)[0]
    fwhm = float(grid[half[-1]] - grid[half[0]]) if len(half) else 0.0
    tail_frac = float((t > 20.0).mean())
    tail = t[t > 10.0]
    if len(tail) > 10:
        h, e = np.histogram(tail, bins=12)
        c = (e[:-1] + e[1:]) / 2
        m = h > 0
        slope = float(np.polyfit(c[m], np.log(h[m]), 1)[0]) if m.sum() > 3 else 0.0
    else:
        slope = 0.0
    return [peak_t, peak_h, rise, fwhm, tail_frac, slope]


def f_logwin(t):
    n = max(1, len(t))
    h, _ = np.histogram(t, bins=WIN_EDGES)
    return [len(t)] + list(h / n)


def f_abs13(t):
    """abs + kurtosis, quartiles, max (13 stats)."""
    if len(t) < 3:
        return [len(t)] + [0] * 12
    return [len(t), t.mean(), t.std(), float(stats.skew(t)),
            float(stats.kurtosis(t)), float(t.min()), float(t.max()),
            *np.quantile(t, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95])]


def f_abs21(t):
    """Rich absolute-time statistics (21): moments incl. kurtosis, extremes,
    fine quantiles, spread measures (IQR, decile span, MAD), log-mean."""
    if len(t) < 3:
        return [len(t)] + [0] * 20
    q = np.quantile(t, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    mad = float(np.median(np.abs(t - np.median(t))))
    return [len(t), t.mean(), t.std(), float(stats.skew(t)),
            float(stats.kurtosis(t)), float(t.min()), float(t.max()),
            *q, float(q[5] - q[3]),          # IQR
            float(q[6] - q[2]),              # decile span q90-q10
            mad, float(np.log(t[t > 0]).mean())]


def f_cdf20(t):
    """FIXED-AREA representation: absolute time by which each 5% of the
    event's photons has arrived (inverse CDF at 5% steps, 19 values)."""
    if len(t) < 5:
        return [0.0] * 19
    return list(np.quantile(t, np.arange(0.05, 1.0, 0.05)))


def f_hist1ns(t):
    """FIXED-WIDTH dual: photon fraction per 1-ns interval, 0-40 ns."""
    n = max(1, len(t))
    h, _ = np.histogram(t, bins=40, range=(0, 40))
    return list(h / n)


# --- reservoir (echo-state) encoding: fixed random RNN driven by the 1-ns
# waveform; final + mean state are the features. No training -> no overfit.
_R = 32
_res_rng = np.random.default_rng(7)
_W_IN = _res_rng.normal(0, 1.0, (_R, 1))
_W = _res_rng.normal(0, 1.0, (_R, _R))
_W *= 0.9 / max(1e-9, np.max(np.abs(np.linalg.eigvals(_W))))   # spectral radius
_LEAK = 0.5


def f_reservoir(t):
    n = max(1, len(t))
    h, _ = np.histogram(t, bins=40, range=(0, 40))
    x = (h / n).astype(np.float64)
    s = np.zeros(_R)
    states = []
    for step in x:
        s = (1 - _LEAK) * s + _LEAK * np.tanh(_W_IN[:, 0] * step + _W @ s)
        states.append(s.copy())
    states = np.array(states)
    return list(states[-1]) + list(states.mean(0))


BLOCKS = {"count": f_count, "abs": f_abs, "gemini7": f_gemini7,
          "early": f_early, "quant": f_quant, "kde16": f_kde16,
          "kdeshape": f_kdeshape, "logwin": f_logwin,
          "cdf20": f_cdf20, "hist1ns": f_hist1ns, "reservoir": f_reservoir,
          "abs13": f_abs13, "abs21": f_abs21}

COMBOS = [
    ("abs+gemini7", ["abs", "gemini7"]),
    ("abs+early", ["abs", "early"]),
    ("abs+kdeshape", ["abs", "kdeshape"]),
    ("abs+kde16", ["abs", "kde16"]),
    ("abs+logwin", ["abs", "logwin"]),
    ("gemini7+kdeshape", ["gemini7", "kdeshape"]),
    ("early+kdeshape", ["early", "kdeshape"]),
    ("abs+gemini7+early", ["abs", "gemini7", "early"]),
    ("abs+early+kdeshape", ["abs", "early", "kdeshape"]),
    ("cdf20+hist1ns", ["cdf20", "hist1ns"]),          # fixed-area + fixed-width duals
    ("abs+early+cdf20", ["abs", "early", "cdf20"]),
    ("early+cdf20", ["early", "cdf20"]),
    ("abs+early+cdf20+hist1ns", ["abs", "early", "cdf20", "hist1ns"]),
    ("abs+reservoir", ["abs", "reservoir"]),
    ("abs+early+reservoir", ["abs", "early", "reservoir"]),
    ("abs13+early", ["abs13", "early"]),
    ("abs21+early", ["abs21", "early"]),
    ("abs21+early+cdf20", ["abs21", "early", "cdf20"]),
    ("ALL", list(BLOCKS)),
]
ENSEMBLE_BLOCKS = ["abs", "gemini7", "kde16", "early", "logwin"]


def zmse(pred, truth, std):
    return float((((pred - truth) / std) ** 2).mean())


def cv_eval(X, T):
    kf = KFold(5, shuffle=True, random_state=0)
    out = []
    for k in range(2):
        fz = []
        for tr, va in kf.split(X):
            std = T[tr, k].std() or 1e-9
            m = HistGradientBoostingRegressor(random_state=0, max_iter=300)
            m.fit(X[tr], T[tr, k])
            fz.append(zmse(m.predict(X[va]), T[va, k], std))
        out.append(float(np.mean(fz)))
    return out


def cv_ensemble(Xs, T):
    kf = KFold(5, shuffle=True, random_state=0)
    out = []
    for k in range(2):
        fz = []
        for tr, va in kf.split(Xs[0]):
            std = T[tr, k].std() or 1e-9
            preds = []
            for X in Xs:
                m = HistGradientBoostingRegressor(random_state=0, max_iter=300)
                m.fit(X[tr], T[tr, k])
                preds.append(m.predict(X[va]))
            fz.append(zmse(np.mean(preds, axis=0), T[va, k], std))
        out.append(float(np.mean(fz)))
    return out


def main() -> int:
    set_seed(0)
    d = np.load(RAW, allow_pickle=True)
    T = d["truth"].astype(np.float64)[:, :2]
    eid, pt = d["event_id"], d["photon_t"]
    N = T.shape[0]
    ev = [pt[np.where(eid == i)[0]] for i in range(N)]
    thin = np.random.default_rng(2)
    print(f"loaded {N} events; blocks: {list(BLOCKS)}\n")

    for ly in LY_EFF:
        frac = ly / LY_SIM
        thinned = []
        for i in range(N):
            t = ev[i][thin.random(len(ev[i])) < frac]
            thinned.append(t if len(t) >= 5 else ev[i][:5])
        mats = {name: np.array([fn(t) for t in thinned], dtype=np.float64)
                for name, fn in BLOCKS.items()}
        rows = []
        for name in BLOCKS:
            rows.append((name, *cv_eval(mats[name], T)))
        for name, parts in COMBOS:
            X = np.hstack([mats[p] for p in parts])
            rows.append((name, *cv_eval(X, T)))
        rows.append(("ENSEMBLE(5)", *cv_ensemble(
            [mats[b] for b in ENSEMBLE_BLOCKS], T)))
        rows.sort(key=lambda r: r[2])
        nc = np.mean([len(t) for t in thinned])
        print(f"=== effective LY {ly} (~{nc:.0f} caught/event) — sorted by v ===")
        print(f"{'features':22} {'u':>7} {'v':>7}")
        for name, zu, zv in rows:
            print(f"{name:22} {zu:7.3f} {zv:7.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
