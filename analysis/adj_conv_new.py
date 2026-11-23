"""Adjoint convergence runs for the 40 new designs (bank_blind40, test_* ids), as adj_conv.py does for the original 40.

Same N values and replicates (max_steps 5000 only; the 750-step check is not repeated). Seeds are offset from
adj_conv.py and from the 2e5-root reference (launch 606271 / physics 808291). One npz per geometry in study/adj_new/.
usage: python adj_conv_new.py [gid ...]
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time

import numpy as np

import adj_conv as AC

OUT = AC.T / "derived/adj_new"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    bank = [g for g in json.load(open(AC.T / "banks/bank.json"))["geometries"] if g["id"].startswith("test_")]
    want = set(sys.argv[1:])
    print(f"threads={AC.numba.get_num_threads()} geometries={len(bank)}", flush=True)
    t_all = time.perf_counter()
    for gi, raw in enumerate(bank):
        gid = raw["id"]
        if (want and gid not in want) or (OUT / f"{gid}.npz").exists():
            continue
        tg = time.perf_counter()
        runs = [(n, r) for n in AC.NS for r in range(1 if n >= 50000 else AC.REPS)]
        rec = {k: [] for k in ("N", "rep", "steps", "pvol", "pvol_se", "tvol", "wall", "unresolved")}
        maps = {k: [] for k in ("p", "p_se", "mt", "tsd", "mt_h")}
        for n, r in runs:
            ls, ps = 7_500_000 + 1000 * r + gi, 9_500_000 + 1000 * r + gi + n
            o = AC.one(raw, n, ls, ps, 5000)
            for k in rec:
                rec[k].append({"N": n, "rep": r, "steps": 5000}.get(k, o.get(k)))
            for k in maps:
                maps[k].append(o[k])
        np.savez_compressed(OUT / f"{gid}.npz", **{k: np.array(v) for k, v in rec.items()},
                            **{k: np.stack(v) for k, v in maps.items()})
        w = np.array(rec["wall"]); N = np.array(rec["N"])
        print(f"{gid} done in {time.perf_counter()-tg:.0f}s (tracing {w.sum():.0f}s; 20k {w[N==20000].mean():.2f}s); "
              f"elapsed {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
