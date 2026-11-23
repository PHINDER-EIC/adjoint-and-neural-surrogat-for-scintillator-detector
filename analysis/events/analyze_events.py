"""Muon events in B_train_g0180 (held out from training): Geant4 deposits -> detected arrival times by
(i) Geant4 optical transport (unpol configuration), (ii) adjoint maps (2e4 and 2e5 roots), (iii) the network.

  accuracy : per-event and summed comparisons -> events_accuracy.json, events_hist.npz
  timing   : per-event cost of the adjoint and network pipelines vs number of muons -> events_timing.json
  plot     : figure figM7_events (paper_v1/figures) from the two json files

usage: python analyze_events.py accuracy|timing|plot
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve

H = Path(__import__("os").environ.get("PHADJ_EVENTS", paths.DATA / "derived" / "events"))
sys.path.insert(0, str(H.parent))
import common as C  # noqa: E402

REPO = Path(str(paths.REPO))
GID = "B_train_g0180"
NMU = 1000
K = 16                     # points per Geant4 step for the network (and for voxel binning in the timing runs)
DT = 0.25
NB = 240
EDGES = np.arange(NB + 1) * DT
TR, TD = 0.7, 1.8
TAU_OK = None


def emission_mass():
    F = lambda t: 1 - (TD * np.exp(-t / TD) - TR * np.exp(-t / TR)) / (TD - TR)
    return np.diff(F(EDGES))


S_EM = emission_mass()


def conv_emission(h):
    """h [..., 240] per-bin mass of transport time -> arrival-time mass including the emission delay.
    Delays are taken at bin centres and transport times uniform within bins, so each product lands half in bin m, half in m+1."""
    c = fftconvolve(h, S_EM.reshape((1,) * (h.ndim - 1) + (-1,)), axes=-1)[..., :NB + 1]
    out = 0.5 * c[..., :NB]
    out[..., 1:] += 0.5 * c[..., :NB - 1]
    return np.maximum(out, 0)


def mirror(a):
    return a.reshape(20, 20, 8, *a.shape[1:])[::-1, :, ::-1].reshape(a.shape)


def load_map(path, gid):
    with h5py.File(path) as f:
        fl = f["geometries"][gid]["field"]
        h = fl["h_per_ns"][:].reshape(3200, NB) * DT
        p = fl["p_per_emitted"][:].ravel()
        se = fl["p_se"][:].ravel()
    return dict(h0=h, h1=mirror(h), p0=p, p1=mirror(p), se0=se, se1=mirror(se))


def maps():
    return {"adjoint 2e5": load_map(C.TRUTH_H5, GID), "adjoint 2e4": load_map(C.T / "atlas_684_geometries_2e4roots.h5", GID)}


def network():
    import torch
    sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
    import phinder_stage6B_surrogate as S
    torch.set_num_threads(14)
    model, _ = S.load_model(C.T / "networks/network_seed20261003.pt", torch.device("cpu")); model.eval()
    data = S.load_split_data(C.TRUTH_H5, [GID])
    geom, nodes, *_ = S._batch_from_indices(data, np.arange(1), torch.device("cpu"))

    def run(u, batch=8192):
        out = dict(p0=[], p1=[], m0=[], m1=[])
        with torch.no_grad():
            for a in range(0, len(u), batch):
                uu = torch.as_tensor(u[a:a + batch], dtype=torch.float32)
                n = len(uu)
                o = model(geom.expand(n, -1), nodes.expand(n, -1, -1), uu)
                out["p0"].append(o["p0"].numpy()); out["p1"].append(o["p1"].numpy())
                out["m0"].append(model.timing_mass(o["mix0_logits"]).numpy()); out["m1"].append(model.timing_mass(o["mix1_logits"]).numpy())
        return {k: np.concatenate(v) for k, v in out.items()}
    return run, data


def step_points(pre, post, w):
    f = (np.arange(K) + 0.5) / K
    pts = pre[:, None, :] + f[None, :, None] * (post - pre)[:, None, :]
    return pts.reshape(-1, 3), np.repeat(w / K, K)


def vox_index(pos, cell):
    sp = cell / np.array([20, 20, 8])
    v = np.minimum(np.maximum((pos / sp).astype(int), 0), [19, 19, 7])
    return (v[:, 0] * 20 + v[:, 1]) * 8 + v[:, 2]


# ------------------------------------------------------------------------------------------------ accuracy
def accuracy():
    D = np.load(H / "edep_B0180.npz")
    P = np.load(H / "photons_1000.npz")
    cell = D["cell_mm"]
    keep = D["event"] < NMU
    pre, post, sev = D["pre"][keep], D["post"][keep], D["event"][keep]
    n_step = P["n_step"]
    assert len(n_step) == len(sev)
    # Geant4: photons emitted per (event, voxel) and detected photons
    A = np.zeros(NMU * 3200)
    det = {0: [], 1: []}
    n_ph = 0; wall = 0.0; cpu = 0.0
    chunks = sorted((H / "chunks").glob("c*.npz"))
    for c in chunks:
        X = np.load(c); O = np.load(H / "opt" / c.name)
        ev = X["event"]; n_ph += len(ev); wall += float(O["wall_s"]); cpu += float(O["cpu_s"])
        A += np.bincount(ev.astype(np.int64) * 3200 + vox_index(X["pos"].astype(np.float64), cell), minlength=NMU * 3200)
        for k in (0, 1):
            m = O["ch"] == k
            det[k].append(np.stack([ev[O["idx"][m]].astype(np.float64), O["t"][m]], 1))
    A = A.reshape(NMU, 3200)
    det = {k: np.concatenate(v) for k, v in det.items()}
    print(f"{len(chunks)} chunks, {n_ph} photons, detected {len(det[0])} / {len(det[1])}; Geant4 {cpu / 3600:.2f} CPU h", flush=True)
    res = dict(geometry=GID, n_muons=NMU, photons=int(n_ph), photons_per_muon=n_ph / NMU, points_per_step=K,
               g4=dict(cpu_s=cpu, wall_sum_s=wall, photons_per_cpu_s=n_ph / cpu, chunks=len(chunks)), methods={})
    G4n = {k: np.bincount(det[k][:, 0].astype(int), minlength=NMU).astype(float) for k in (0, 1)}
    G4t = {k: np.bincount(det[k][:, 0].astype(int), weights=det[k][:, 1], minlength=NMU) for k in (0, 1)}
    G4h = {k: np.histogram(det[k][:, 1], EDGES)[0].astype(float) for k in (0, 1)}
    hist = {f"g4_{k}": G4h[k] for k in (0, 1)}
    hist.update({f"g4_n{k}": G4n[k] for k in (0, 1)})
    res["g4"].update(detected=[int(G4n[0].sum()), int(G4n[1].sum())],
                     beyond_60ns=[int((det[k][:, 1] >= 60).sum()) for k in (0, 1)])
    # predictions: expected detected photons per event and channel, and arrival-time mass
    preds = {}
    for name, M in maps().items():
        lam = {k: A @ M[f"p{k}"] for k in (0, 1)}
        lt = {k: conv_emission(A @ M[f"h{k}"]) for k in (0, 1)}
        preds[name] = (lam, lt)
    run, _ = network()
    pts, w = step_points(pre, post, n_step.astype(float))
    pe = np.repeat(sev, K)
    t0 = time.perf_counter()
    o = run(pts / cell)
    print(f"network: {len(pts)} points in {time.perf_counter() - t0:.2f} s", flush=True)
    lam = {k: np.bincount(pe, weights=w * o[f"p{k}"], minlength=NMU) for k in (0, 1)}
    lt = {}
    for k in (0, 1):
        m = np.zeros((NMU, NB))
        np.add.at(m, pe, (w * o[f"p{k}"])[:, None] * o[f"m{k}"])
        lt[k] = conv_emission(m)
    preds["network"] = (lam, lt)
    tc = (EDGES[:-1] + EDGES[1:]) / 2
    for name, (lam, lt) in preds.items():
        r = {}
        for k in (0, 1):
            n, L = G4n[k], lam[k]
            tot = float(L.sum()); g = float(n.sum())
            pull = (n - L) / np.sqrt(np.maximum(L, 1e-9))
            # per-event mean arrival time (0-60 ns) and its statistical error from the Geant4 sample
            mt_pred = (lt[k] * tc).sum(1) / lt[k].sum(1)
            m_in = np.bincount(det[k][det[k][:, 1] < 60, 0].astype(int), minlength=NMU).astype(float)
            s_in = np.bincount(det[k][det[k][:, 1] < 60, 0].astype(int), weights=det[k][det[k][:, 1] < 60, 1], minlength=NMU)
            s2_in = np.bincount(det[k][det[k][:, 1] < 60, 0].astype(int), weights=det[k][det[k][:, 1] < 60, 1] ** 2, minlength=NMU)
            ok = m_in > 5
            mt_g4 = s_in[ok] / m_in[ok]
            sd_g4 = np.sqrt(np.maximum(s2_in[ok] / m_in[ok] - mt_g4 ** 2, 0))
            dmt = mt_g4 - mt_pred[ok]
            hp = lt[k].sum(0)
            r[f"ch{k}"] = dict(total_pred=tot, total_g4=g, ratio=g / tot, ratio_se=np.sqrt(g) / tot,
                               pull_mean=float(pull.mean()), pull_sd=float(pull.std()),
                               mean_detected_per_event=float(L.mean()),
                               mt_diff_mean_ns=float(dmt.mean()), mt_diff_rms_ns=float(np.sqrt(np.mean(dmt ** 2))),
                               mt_stat_rms_ns=float(np.sqrt(np.mean(sd_g4 ** 2 / m_in[ok]))),
                               hist_chi2=float(np.sum(((G4h[k] - hp) ** 2 / np.maximum(hp, 1))[hp > 5])),
                               hist_bins=int((hp > 5).sum()))
            hist[f"{name}_{k}"] = hp
            hist[f"{name}_lam{k}"] = L
            hist[f"{name}_mt{k}"] = mt_pred
        # channel asymmetry of the mean arrival time per event (position information)
        res["methods"][name] = r
        print(name, json.dumps({c: {a: round(b, 4) for a, b in v.items()} for c, v in r.items()}), flush=True)
    for k in (0, 1):
        ok = np.bincount(det[k][det[k][:, 1] < 60, 0].astype(int), minlength=NMU)
        hist[f"g4_mt{k}"] = np.bincount(det[k][det[k][:, 1] < 60, 0].astype(int), weights=det[k][det[k][:, 1] < 60, 1],
                                        minlength=NMU) / np.maximum(ok, 1)
        hist[f"g4_m{k}"] = ok
    np.savez_compressed(H / "events_hist.npz", **hist)
    json.dump(res, open(H / "events_accuracy.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ timing
def timeit(fn, reps, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def timing():
    D = np.load(H / "edep_B0180.npz")
    cell = D["cell_mm"]
    import make_photons as MP
    mu_all = MP.Y * MP.visible(D)
    M = maps()["adjoint 2e4"]
    Hs = {k: conv_emission(M[f"h{k}"]) for k in (0, 1)}       # precomputed once per geometry with the map
    run, data = network()
    rng = np.random.default_rng(3)
    out = dict(n_muons=[], steps=[], photons=[], voxels=[], adjoint_s=[], network_s=[], points_per_step=K)
    for nmu in (1, 10, 100, 1000, 10000, 20000):
        sel = D["event"] < nmu
        pre, post, mu = D["pre"][sel], D["post"][sel], mu_all[sel]

        def adj():
            pts, w = step_points(pre, post, mu)
            a = np.bincount(vox_index(pts, cell), weights=w, minlength=3200)
            return [rng.poisson(a @ Hs[k]) for k in (0, 1)]          # detected-photon histograms, both channels

        def net():
            pts, w = step_points(pre, post, mu)
            o = run(pts / cell)
            lam = [conv_emission(((w * o[f"p{k}"])[:, None] * o[f"m{k}"]).sum(0)) for k in (0, 1)]
            return [rng.poisson(l) for l in lam]
        reps = 50 if nmu <= 100 else (10 if nmu <= 1000 else 3)
        ta = timeit(adj, reps)
        tn = timeit(net, max(1, reps // 5)) if nmu <= 1000 else float("nan")
        pts, _ = step_points(pre, post, mu)
        out["n_muons"].append(nmu); out["steps"].append(int(sel.sum())); out["photons"].append(float(mu.sum()))
        out["voxels"].append(int(len(np.unique(vox_index(pts, cell))))); out["adjoint_s"].append(ta); out["network_s"].append(tn)
        print(f"muons {nmu:6d} steps {sel.sum():6d} photons {mu.sum():.3g} voxels {out['voxels'][-1]:4d} "
              f"adjoint {1e3 * ta:.3f} ms network {1e3 * tn:.2f} ms", flush=True)
    # network map once per geometry (3200 voxel centres), then the adjoint pipeline applies
    uvs = data.uvs
    out["network_map_s"] = timeit(lambda: run(uvs), 3)
    out["map_conv_s"] = timeit(lambda: [conv_emission(M[f"h{k}"]) for k in (0, 1)], 3)
    out["g4_edep_s_per_muon"] = float(D["wall_s"]) / int(D["n_muons"])
    json.dump(out, open(H / "events_timing.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ figure
def g4_rates():
    """Geant4 photons/s: this run (unpolarised, per core under the 12-process load), idle single-core unpol/full/bare,
    and the 14-core parallel scaling measured earlier (bench.json, polarised mode)."""
    A = json.load(open(H / "events_accuracy.json"))
    B = json.load(open(H.parent / "bench.json"))
    idle = {m: float(np.load(H / "opt" / f"ovh_{m}.npz")["n"]) / float(np.load(H / "opt" / f"ovh_{m}.npz")["wall_s"])
            for m in ("unpol", "full", "bare")}
    scale14 = B["g4"]["parallel14_pol"]["photons_per_s"] / B["g4"]["single_pol"]["photons_per_s"]
    return dict(per_core_loaded=A["g4"]["photons_per_cpu_s"], idle=idle, scale14=scale14,
                unpol14=idle["unpol"] * scale14, bare14=idle["bare"] * scale14, pol14=B["g4"]["parallel14_pol"]["photons_per_s"])


def plot():
    import matplotlib.pyplot as plt
    import style as Sty
    Sty.setup()
    A = json.load(open(H / "events_accuracy.json")); T = json.load(open(H / "events_timing.json"))
    Hh = dict(np.load(H / "events_hist.npz"))
    R = g4_rates()
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw=dict(width_ratios=[1.05, 0.9, 1.25]))
    # (a) arrival times summed over all events, both channels
    for k, ls in ((0, "-"), (1, "--")):
        ax.stairs(np.maximum(Hh[f"g4_{k}"], 0.1) / NMU / DT, EDGES, color=Sty.COL["g4"], lw=1.0, ls=ls,
                  label="Geant4" if k == 0 else None)
        ax.stairs(np.maximum(Hh[f"adjoint 2e4_{k}"], 1e-3) / NMU / DT, EDGES, color=Sty.COL["adjoint"], lw=0.9, ls=ls,
                  label="adjoint map ($2\\times10^4$ roots)" if k == 0 else None)
        ax.stairs(np.maximum(Hh[f"network_{k}"], 1e-3) / NMU / DT, EDGES, color=Sty.COL["network"], lw=0.9, ls=ls,
                  label="network" if k == 0 else None)
    ax.set_yscale("log"); ax.set_xlim(0, 30); ax.set_ylim(0.05, 300)
    ax.set_xlabel("arrival time at readout (ns)"); ax.set_ylabel("detected photons per muon per ns")
    ax.plot([], [], color=Sty.INK2, lw=0.9, ls="-", label="channel 0"); ax.plot([], [], color=Sty.INK2, lw=0.9, ls="--", label="channel 1")
    ax.legend(loc="upper right", fontsize=5.4)
    Sty.panel(ax, "a", f"{NMU} muons, summed")
    # (b) detected photons per event
    for name, col, mk in (("adjoint 2e4", Sty.COL["adjoint"], "o"), ("network", Sty.COL["network"], "s")):
        for k in (0, 1):
            bx.plot(Hh[f"{name}_lam{k}"], Hh[f"g4_n{k}"], mk, ms=1.4, mew=0, alpha=0.5, color=col, rasterized=True)
        m = A["methods"][name]
        rat = np.mean([m[c]["ratio"] for c in ("ch0", "ch1")]); psd = np.mean([m[c]["pull_sd"] for c in ("ch0", "ch1")])
        bx.plot([], [], mk, ms=3, color=col, label=f"{'adjoint' if 'adj' in name else 'network'}: Geant4/pred. {rat:.3f}, pull s.d. {psd:.2f}")
    lim = [0, max(Hh["g4_n0"].max(), Hh["g4_n1"].max()) * 1.05]
    bx.plot(lim, lim, color=Sty.INK3, lw=0.7, ls=":")
    bx.set_xlim(lim); bx.set_ylim(lim)
    bx.set_xlabel("predicted detected photons per event"); bx.set_ylabel("Geant4 detected photons per event")
    bx.legend(loc="lower right", fontsize=5.0, handletextpad=0.2)
    Sty.panel(bx, "b", "Each event, both channels")
    # (c) cost per event versus photons
    NG = np.geomspace(1e3, 1e10, 60)
    nmax = A["photons"]
    g = R["pol14"]; gc = R["bare14"]
    meas = NG <= nmax
    cx.plot(NG[meas], NG[meas] / g, color=Sty.COL["g4"], lw=1.3, label="Geant4 with Python detection, 14 cores (measured)")
    cx.plot(NG[~meas], NG[~meas] / g, color=Sty.COL["g4"], lw=1.3, ls="--")
    cx.plot(NG, NG / gc, color=Sty.COL["g4"], lw=0.9, ls=":", label="Geant4 C++ tracking only (estimate)")
    t_map = 3.6
    ta1 = T["adjoint_s"][T["n_muons"].index(1)]; tn1 = T["network_s"][T["n_muons"].index(1)]
    cx.plot(NG, np.full_like(NG, t_map + ta1), color=Sty.COL["adjoint"], lw=1.3, label="adjoint, new geometry (map 3.6 s)")
    cx.plot(NG, np.full_like(NG, ta1), color=Sty.COL["adjoint"], lw=1.0, ls="--", label="adjoint, map reused")
    cx.plot(NG, np.full_like(NG, tn1), color=Sty.COL["network"], lw=1.3, label="network, no precomputation")
    ph = np.array(T["photons"]); ok = np.isfinite(T["network_s"])
    cx.plot(ph, T["adjoint_s"], "o", ms=2.6, color=Sty.COL["adjoint"], mfc="white", mew=0.8)
    cx.plot(ph[ok], np.array(T["network_s"])[ok], "s", ms=2.6, color=Sty.COL["network"], mfc="white", mew=0.8)
    cx.plot([], [], "o", ms=2.6, color=Sty.INK2, mfc="white", mew=0.8, label="1 to $2\\times10^4$ muons per event")
    cx.axvline(nmax, color=Sty.INK3, lw=0.6, ls=":", zorder=0)
    cx.text(nmax * 1.25, 12, "Geant4 run\n$\\rightarrow$ extrapolated", fontsize=5.0, color=Sty.INK3, va="bottom")
    cx.set_xscale("log"); cx.set_yscale("log"); cx.set_xlim(1e3, 1e10); cx.set_ylim(1e-4, 1e11)
    cx.set_yticks([1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    rx = cx.twinx(); rx.set_yscale("log"); rx.set_ylim(cx.get_ylim())
    rx.set_yticks([1e-3, 1, 60, 3600, 86400, 3.15e7]); rx.set_yticklabels(["1 ms", "1 s", "1 min", "1 h", "1 day", "1 yr"], fontsize=5.6)
    rx.minorticks_off(); rx.grid(False); rx.spines["right"].set_visible(True)
    cx.set_xlabel(r"photons emitted in the event, $N_\gamma$"); cx.set_ylabel("wall time per event (s)")
    cx.legend(fontsize=4.9, loc="upper left", framealpha=0.95, frameon=True, edgecolor="none", borderaxespad=0.2)
    Sty.panel(cx, "c", "Cost per event, one 14-core CPU")
    Sty.save(fig, "figM7_events")
    import shutil
    for ext in ("pdf", "png"):
        shutil.copy(Sty.FIG / f"figM7_events.{ext}", REPO / "paper_v1/figures" / f"figM7_events.{ext}")
    info = dict(rates=R, t_map=t_map, adjoint_event_1mu_s=ta1, network_event_1mu_s=tn1,
                speedup={f"{n:.0e}": dict(g4_vs_adjoint_new=n / g / (t_map + ta1), g4_vs_adjoint_reused=n / g / ta1,
                                          g4_vs_network=n / g / tn1, g4cpp_vs_adjoint_new=n / gc / (t_map + ta1),
                                          g4cpp_vs_adjoint_reused=n / gc / ta1, g4cpp_vs_network=n / gc / tn1,
                                          g4_s=n / g, g4cpp_s=n / gc)
                         for n in (2.65e4, 1e6, 1e8, 1e9, 1e10)})
    json.dump(info, open(H / "events_speed.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    dict(accuracy=accuracy, timing=timing, plot=plot)[sys.argv[1]]()
