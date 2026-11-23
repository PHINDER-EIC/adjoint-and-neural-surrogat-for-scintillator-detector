"""Export a test geometry's scene (the one JAX and the adjoint tracer use) as closed triangle meshes for Geant4.

Per guide k: core mesh = union of core segments + terminal (internal shared faces removed);
clad shell = clad union outer surface minus its two end caps, + core side faces reversed, + the two end rings.
Core and clad shell are siblings, so the core end faces touch air directly, as in the repo scene.
usage: python g4_export.py <gid> [...]   -> study/g4/geom/<gid>.npz
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(str(paths.REPO))
T = paths.DATA
sys.path[:0] = [str(ROOT / "phinder_bridge/scripts"), str(ROOT / "Scintillator-Simulator-main/src")]
import phinder_stage6_adjoint_geometry_atlas as atlasmod  # noqa: E402
import phinder_stage6_adjoint_prototype as proto  # noqa: E402

OUT = Path(__import__("os").environ.get("PHADJ_WORK", paths.RELEASE / "work")) / "g4/geom"
OUT.mkdir(parents=True, exist_ok=True)
ARGS = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005,
                             max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0,
                             start_epsilon_cm=1e-7)


def outward_quads(v):
    V = np.asarray(v.vertices, float)
    c = V.mean(0)
    out = []
    for f in v.faces:
        q = V[list(f)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if np.dot(n, q.mean(0) - c) < 0:
            q = q[::-1]
        out.append(q)
    return out


def key(q):
    return tuple(sorted(map(tuple, np.round(q, 9))))


def boundary(quads):
    cnt = Counter(key(q) for q in quads)
    return [q for q in quads if cnt[key(q)] == 1]


def normal(q):
    n = np.cross(q[1] - q[0], q[2] - q[0])
    return n / np.linalg.norm(n)


def tris(quads):
    t = []
    for q in quads:
        t += [q[[0, 1, 2]], q[[0, 2, 3]]]
    return np.array(t)


def regions(gid, cell):
    """The five emission regions of jax_study.py (same voxel ranges)."""
    import h5py
    with h5py.File(T / "reference_test40_2e5roots.h5") as f:
        pA = f["geometries"][gid]["field/p_per_emitted"][:]
        gp = f["geometries"][gid]["guide0_points_cm"][:]
    sp = cell / np.array([20, 20, 8])
    ix, iy = np.unravel_index(np.argmax(pA[:, :, 7]), (20, 20))
    ix, iy = int(min(max(ix, 0), 18)), int(min(max(iy, 0), 18))
    best, cx, cy = -1.0, 0, 0
    for (a, b) in ((0, 0), (0, 18), (18, 0), (18, 18)):
        c = np.array([(a + 1) * sp[0], (b + 1) * sp[1]])
        d = np.min(np.linalg.norm(gp[:, :2] - c, axis=1))
        if d > best:
            best, cx, cy = d, a, b
    return {"whole": [[0, 0, 0], [20, 20, 8]], "hotspot": [[ix, iy, 7], [ix + 2, iy + 2, 8]],
            "centre": [[9, 9, 3], [11, 11, 5]], "below_hotspot": [[ix, iy, 0], [ix + 2, iy + 2, 1]],
            "far_corner": [[cx, cy, 6], [cx + 2, cy + 2, 8]]}


def export(gid):
    raw = next(g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"] if g["id"] == gid)
    spec, _, _, cfg = atlasmod._geometry_cfg(raw, ARGS, 1, 2)
    scene, _, core_w, core_h = proto._build_from_cfg(cfg)
    vols = {v.name: v for v in scene.volumes}
    mats = {v.material.name: (float(v.material.refractive_index), float(v.material.absorption_mu)) for v in scene.volumes}
    mats["air"] = (float(scene.outside_material.refractive_index), float(scene.outside_material.absorption_mu))
    res = dict(cell_cm=np.asarray(cfg["cell_cm"], float), core_w=core_w, core_h=core_h,
               materials_json=json.dumps(mats))
    for k in (0, 1):
        core = [q for n, v in vols.items() if n.startswith(f"guide{k}_core") or n == f"guide{k}_terminal" for q in outward_quads(v)]
        clad = [q for n, v in vols.items() if n.startswith(f"guide{k}_clad") for q in outward_quads(v)]
        cb, kb = boundary(core), boundary(clad)
        port = next(p for p in scene.readout_ports if p.channel == k)
        P, N = np.asarray(port.point, float), np.asarray(port.outward_normal, float)
        # end caps of the core: boundary faces whose normal is (anti)parallel to the guide axis at the ends
        start_dir = None
        caps_core = []
        for q in cb:
            n = normal(q)
            on_port = abs(np.dot(q - P, N)).max() < 1e-7 and np.dot(n, N) > 0.999
            if on_port:
                caps_core.append(("port", q))
        # the start cap: the core boundary face of the first core segment that is not a side face
        first = outward_quads(vols[f"guide{k}_core_000"])
        second_keys = {key(q) for q in outward_quads(vols[f"guide{k}_core_001"])}
        side_like = []
        for q in first:
            if key(q) in second_keys:
                start_dir = -normal(q)          # shared face normal points along the guide
        for q in first:
            n = normal(q)
            if np.dot(n, start_dir) > 0.999:
                caps_core.append(("start", q))
        assert len(caps_core) == 2, (gid, k, len(caps_core))
        cap_keys = {key(q) for _, q in caps_core}
        core_side = [q for q in cb if key(q) not in cap_keys]
        # clad end caps: clad boundary faces coplanar with the core caps
        clad_side, rings = [], []
        clad_caps = []
        for q in kb:
            n = normal(q)
            hit = None
            for name, cq in caps_core:
                cn = normal(cq)
                if np.dot(n, cn) > 0.999 and abs(np.dot(q - cq[0], cn)).max() < 1e-7:
                    hit = (name, cq)
            if hit is None:
                clad_side.append(q)
            else:
                clad_caps.append((hit, q))
        assert len(clad_caps) == 2
        for (name, cq), q in clad_caps:
            # match inner vertices to outer ones (nearest), keep the outer winding
            Kq = np.array([cq[np.argmin(np.linalg.norm(cq - c, axis=1))] for c in q])
            for i in range(4):
                j = (i + 1) % 4
                r = np.array([q[i], q[j], Kq[j], Kq[i]])
                if np.dot(normal(r), normal(q)) < 0:
                    r = r[::-1]
                rings.append(r)
        shell = clad_side + [q[::-1] for q in core_side] + rings
        res[f"core{k}"] = tris(core_side + [q for _, q in caps_core])
        res[f"shell{k}"] = tris(shell)
        res[f"port{k}_point"], res[f"port{k}_normal"] = P, N
        # closedness check: every edge used exactly twice
        for nm in (f"core{k}", f"shell{k}"):
            e = Counter()
            for t in res[nm]:
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    e[tuple(sorted((tuple(np.round(t[a], 9)), tuple(np.round(t[b], 9)))))] += 1
            bad = sum(1 for c in e.values() if c != 2)
            res[f"{nm}_open_edges"] = bad
    res["regions_json"] = json.dumps(regions(gid, np.asarray(cfg["cell_cm"], float)))
    np.savez(OUT / f"{gid}.npz", **res)
    print(gid, {k: v.shape for k, v in res.items() if hasattr(v, "shape") and v.ndim > 1},
          {k: v for k, v in res.items() if k.endswith("open_edges")}, flush=True)


if __name__ == "__main__":
    for g in sys.argv[1:]:
        export(g)
