"""How much does guide 1 change channel 0?  Adjoint for channel 0 with the normal two-guide scene, and with guide 1
(its volumes and readout port) removed.  Same roots and same random numbers (common random numbers), so the ratio is
precise.  Writes fd/guide1_effect.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(str(paths.REPO))
T = paths.DATA
sys.path[:0] = [str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"),
                str(ROOT / "Scintillator-Simulator-main/src")]
import numba_adjoint as na  # noqa: E402
import phinder_stage6_adjoint_prototype as proto  # noqa: E402

sys.path.insert(0, str(T / "derived"))
import fd_study as F  # noqa: E402

ROOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
GEOMS = F.GEOMS
SEEDS = F.SEEDS[0]


def tracer(cfg, drop_guide1):
    scene, paths, core_w, core_h = proto._build_from_cfg(cfg)
    if drop_guide1:
        vols = [v for v in scene.volumes if not v.name.startswith("guide1_")]
        ports = [p for p in scene.readout_ports if p.channel == 0]
        scene = proto.Scene(volumes=vols, outside_material=proto.AIR,
                            scintillator_material=proto.EJ204, readout_ports=ports)
    pts, tans = paths[0]
    Z = float(cfg["cell_cm"][2]); th = float(cfg["guide_thickness_cm"])
    tf = np.asarray(tans[-1], float); tf /= np.linalg.norm(tf)
    return na.NumbaAdjointField(scene, terminal_volume="guide0_terminal",
                                endpoint=np.array([pts[-1, 0], pts[-1, 1], Z + 0.5 * th]),
                                outward=np.array([tf[0], tf[1], 0.0]), lateral=np.array([-tf[1], tf[0], 0.0]),
                                vertical=np.array([0.0, 0.0, 1.0]), core_w=core_w, core_h=core_h,
                                cell=cfg["cell_cm"], grid_shape=cfg["grid_shape"], cfg=cfg)


def main():
    bank = {g["id"]: g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"]}
    out = {}
    for gid in GEOMS:
        raw = F.raw_from_u(np.asarray(bank[gid]["spacefill_vector"], float), gid)
        spec, p0, p1, cfg = F.A._geometry_cfg(raw, F.ARGS, *SEEDS)
        res = {}
        for tag, drop in (("both", False), ("guide0_only", True)):
            t0 = time.perf_counter()
            m, _ = tracer(cfg, drop).run(ROOTS, *SEEDS)
            n = m["n"]
            res[tag] = dict(L=m["sumL"] / n, Lt=m["sumLt"] / n, Lvol=m["sumLvol"] / n, Ltvol=m["sumLtvol"] / n,
                            term={k: float(v) / n for k, v in m["terminal_weight"].items()},
                            sec=time.perf_counter() - t0, n=n)
        b, s = res["both"], res["guide0_only"]
        ratio = s["L"] / np.where(b["L"] > 0, b["L"], np.nan)          # p0 without guide 1 / with guide 1
        w = b["L"] / b["L"].sum()
        layer = [float(s["L"][:, :, z].sum() / b["L"][:, :, z].sum()) for z in range(ratio.shape[2])]
        tb = b["Ltvol"] / b["Lvol"]; ts = s["Ltvol"] / s["Lvol"]
        row = dict(cell=spec.cell_cm if isinstance(spec.cell_cm, list) else list(map(float, spec.cell_cm)),
                   whole_cell_ratio=float(s["Lvol"] / b["Lvol"]), layer_ratio=layer,
                   voxel_ratio_q=[float(np.nanquantile(ratio, q)) for q in (0.01, 0.5, 0.99)],
                   voxel_ratio_max=float(np.nanmax(ratio)),
                   wrms_ln_ratio=float(np.sqrt(np.nansum(w * np.log(ratio) ** 2))),
                   mean_t_both=float(tb), mean_t_g0only=float(ts),
                   term_both=b["term"], term_g0only=s["term"], seconds=[b["sec"], s["sec"]], roots=int(b["n"]))
        out[gid] = row
        print(f"{gid:15s} cell {row['cell']}  p0(no g1)/p0 = {row['whole_cell_ratio']:.4f}  layers(bottom..top) "
              + " ".join(f"{x:.3f}" for x in layer)
              + f"  vox 1/50/99% {row['voxel_ratio_q'][0]:.3f}/{row['voxel_ratio_q'][1]:.3f}/{row['voxel_ratio_q'][2]:.3f}"
              f"  <t> {tb:.3f}->{ts:.3f} ns  port1 weight {b['term'].get('READOUT_PORT_1', 0):.4f}  ({b['sec']:.0f}+{s['sec']:.0f}s)",
              flush=True)
        json.dump(out, open(T / "derived/fd/guide1_effect.json", "w"), indent=1)


if __name__ == "__main__":
    main()
