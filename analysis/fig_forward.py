"""Figure: forward validation per emission region against Geant4 (and JAX), with pulls and physics effects."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import common as C
import style as Sty

REG = ["whole", "hotspot", "centre", "below_hotspot", "far_corner"]
RLAB = dict(whole="whole", hotspot="hot", centre="ctr", below_hotspot="below", far_corner="far")
REF = "g4_unpol"


def load():
    return json.load(open(C.ST / "regions.json"))["regions"]


def pairs(R, a, b, chans=("p0", "p1")):
    """pulls and ratios of method a vs b over all regions/channels where both exist."""
    out = []
    for gid, regs in R.items():
        for name, row in regs.items():
            if a in row and b in row:
                for ch in chans:
                    if ch in row[a] and ch in row[b]:
                        va, vb = row[a][ch], row[b][ch]
                        sa, sb = row[a].get(ch + "_se", 0.0), row[b].get(ch + "_se", 0.0)
                        out.append(dict(gid=gid, region=name, ch=ch, a=va, b=vb, sa=sa, sb=sb,
                                        pull=(va - vb) / np.hypot(sa, sb) if np.hypot(sa, sb) > 0 else np.nan,
                                        ratio=va / vb, ratio_se=va / vb * np.hypot(sa / va, sb / vb)))
    return out


def chi2_summary(P):
    x = np.array([p["pull"] for p in P if np.isfinite(p["pull"])])
    if not len(x):
        return {}
    c2 = float(np.sum(x ** 2))
    return dict(n=len(x), chi2=c2, pvalue=float(stats.chi2.sf(c2, len(x))), mean=float(x.mean()),
                mean_se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan, sd=float(x.std(ddof=1)) if len(x) > 1 else np.nan)


def plot(R):
    Sty.setup()
    gids = [g for g in sorted(R) if any(REF in R[g][r] for r in R[g])]
    fig = plt.figure(figsize=(7.1, 6.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15])
    meths = [("adjoint", -0.24), ("jax", -0.08), ("g4_pol", 0.08), ("network", 0.24)]
    halves = [gids[:(len(gids) + 1) // 2], gids[(len(gids) + 1) // 2:]]
    for row, gset in enumerate(halves):
        ax = fig.add_subplot(gs[row, :])
        xt, xl = [], []
        x = 0
        for gi, g in enumerate(gset):
            x0 = x
            for name in REG:
                rowd = R[g].get(name, {})
                if REF not in rowd:
                    continue
                ref = rowd[REF]
                for ch, dx2 in (("p0", 0.0), ("p1", 0.5)):
                    for m, dx in meths:
                        if m not in rowd:
                            continue
                        r = rowd[m][ch] / ref[ch]
                        se = r * np.hypot(rowd[m].get(ch + "_se", 0) / rowd[m][ch], ref[ch + "_se"] / ref[ch])
                        key = "g4" if m.startswith("g4") else m
                        ax.errorbar(x + dx2 + dx * 0.45, r, yerr=se, fmt=Sty.MRK[key], color=Sty.COL[key], ms=2.6,
                                    mfc="white" if m == "g4_pol" else Sty.COL[key], mew=0.7, elinewidth=0.6)
                xt.append(x + 0.25); xl.append(RLAB[name])
                x += 1.2
            if gi < len(gset) - 1:
                ax.axvline(x - 0.6, color=Sty.GRID, lw=0.8)
            ax.text((x0 + x - 1.2) / 2 + 0.25, 1.17, g, ha="center", fontsize=6, color=Sty.INK2)
        ax.axhline(1, color=Sty.INK2, lw=0.6)
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=5.5)
        ax.set_xlim(-0.6, max(x - 0.6, 1)); ax.set_ylim(0.82, 1.2)
        ax.set_ylabel("ratio to Geant4\n(unpolarised)")
        ax.grid(axis="x", visible=False)
        if row == 0:
            Sty.panel(ax, "a", "Captured fraction per emission region (each pair: channel 0 left, channel 1 right)")
            for m, lab in (("adjoint", "Adjoint, 200k roots"), ("jax", "JAX forward"), ("g4_pol", "Geant4, polarised"),
                           ("network", "Network (628 geom.)")):
                key = "g4" if m.startswith("g4") else m
                ax.plot([], [], Sty.MRK[key], color=Sty.COL[key], mfc="white" if m == "g4_pol" else Sty.COL[key], label=lab, ms=3.5)
            fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="outside lower center", fontsize=6.5)

    # b: pulls
    ax = fig.add_subplot(gs[2, 0])
    summ = {}
    bins = np.linspace(-4, 4, 17)
    for (a, b, col, lab) in (("adjoint", REF, Sty.COL["adjoint"], "adjoint - G4"), ("jax", REF, Sty.COL["jax"], "JAX - G4"),
                             ("adjoint", "jax", Sty.INK, "adjoint - JAX")):
        P = pairs(R, a, b)
        s = chi2_summary(P)
        summ[f"{a}_vs_{b}"] = s
        if not s:
            continue
        xv = np.clip([p["pull"] for p in P], -3.99, 3.99)
        ax.hist(xv, bins=bins, histtype="step", density=True, color=col, lw=1.1,
                label=f"{lab}: $\\chi^2$/n = {s['chi2']:.0f}/{s['n']}")
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, stats.norm.pdf(xx), color=Sty.INK3, ls="--", lw=0.8)
    ax.set_xlabel("pull"); ax.set_ylabel("density"); ax.set_ylim(0, 0.75)
    ax.legend(fontsize=5.6, loc="upper left")
    Sty.panel(ax, "b", "Pulls")

    # c: physics effects in Geant4
    ax = fig.add_subplot(gs[2, 1])
    for k, (a, b, col, lab) in enumerate((("g4_pol", "g4_unpol", Sty.INK, "polarised / unpolarised"),
                                          ("g4_full", "g4_pol", Sty.INK3, "re-entry from air / none"))):
        P = pairs(R, a, b)
        for j, name in enumerate(REG):
            v = [p for p in P if p["region"] == name]
            if not v:
                continue
            r = np.array([p["ratio"] for p in v]); se = np.array([p["ratio_se"] for p in v])
            wv = 1 / se ** 2
            m, e = np.sum(wv * r) / wv.sum(), 1 / np.sqrt(wv.sum())
            ax.errorbar(j + (k - 0.5) * 0.25, m, yerr=e, fmt="o", color=col, ms=3.5, mfc=col if k == 0 else "white",
                        label=lab if j == 0 else None)
            ax.plot(np.full(len(r), j + (k - 0.5) * 0.25), r, "_", color=col, alpha=0.35, ms=5)
        summ[f"{a}_over_{b}"] = dict(ratio=[float(p["ratio"]) for p in P], se=[float(p["ratio_se"]) for p in P])
    ax.axhline(1, color=Sty.INK2, lw=0.6)
    ax.set_xticks(range(len(REG))); ax.set_xticklabels([RLAB[r] for r in REG], fontsize=6)
    ax.set_ylabel("ratio of captured fractions")
    ax.legend(fontsize=5.8, loc="upper right")
    Sty.panel(ax, "c", "Geant4 physics options")

    # d, e: time moments relative to Geant4
    for col_i, (key, letter, title) in enumerate((("t_mean", "d", "Mean arrival time, ch. 0"), ("t_sd60", "e", "Arrival-time s.d., ch. 0"))):
        ax = fig.add_subplot(gs[2, 2 + col_i])
        for k, m in enumerate(("adjoint", "jax", "network")):
            vals = []
            for g in gids:
                for j, name in enumerate(REG):
                    row = R[g].get(name, {})
                    if REF in row and m in row and np.isfinite(row[m].get(key, np.nan)):
                        ref = row[REF][key]
                        se = np.hypot(row[REF].get(key + "_se", 0), row[m].get(key + "_se", 0) or 0)
                        ax.errorbar(j + (k - 1) * 0.22, row[m][key] / ref - 1, yerr=se / ref, fmt=Sty.MRK[m],
                                    color=Sty.COL[m], ms=2.6, elinewidth=0.5, alpha=0.8)
                        vals.append(row[m][key] / ref - 1)
            summ[f"{m}_{key}_rel_vs_g4"] = dict(mean=float(np.mean(vals)) if vals else None,
                                               rms=float(np.sqrt(np.mean(np.square(vals)))) if vals else None)
        ax.axhline(0, color=Sty.INK2, lw=0.6)
        ax.set_xticks(range(len(REG))); ax.set_xticklabels([RLAB[r] for r in REG], fontsize=6)
        ax.set_ylabel("relative difference to Geant4")
        Sty.pct(ax)
        Sty.panel(ax, letter, title)
    Sty.save(fig, "fig1_forward_validation")
    return summ


if __name__ == "__main__":
    summ = plot(load())
    json.dump(summ, open(C.ST / "forward_summary.json", "w"), indent=1)
    for k, v in summ.items():
        if "ratio" not in v:
            print(k, v)
