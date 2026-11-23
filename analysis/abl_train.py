"""Ablation: the repo surrogate with its ordered-guide BiGRU encoder replaced by an MLP on the flattened node features.

Same inputs (6 guide nodes x [x, y, x-u, y-v, order]), same trunk, heads, losses and training protocol as the
baseline (n = 628, 240 epochs, samples/epoch 148,032, seed 1). The variant class is swapped in at run time; the repo
script is not modified. Writes study/abl/<tag>.pt and <tag>.pred.npz (test-set predictions as in train_study.py).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(str(paths.REPO))
T = paths.DATA
sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

Base = S.Stage6BSurrogate


class NoGRU(Base):
    def __init__(self, *a, hidden=40, trunk=128, **k):
        super().__init__(*a, hidden=hidden, trunk=trunk, **k)
        del self.path_gru
        self.path_mlp = nn.Sequential(nn.Linear(6 * 5, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, 2 * hidden), nn.SiLU())

    def canonical(self, geom6, nodes_norm, uvs):
        B, L, _ = nodes_norm.shape
        uv = uvs[:, :2].unsqueeze(1).expand(-1, L, -1)
        rel = nodes_norm - uv
        order = torch.linspace(0.0, 1.0, L, device=nodes_norm.device, dtype=nodes_norm.dtype).view(1, L, 1).expand(B, -1, -1)
        seq = torch.cat([nodes_norm, rel, order], dim=-1)
        path = self.path_mlp(seq.reshape(B, -1))
        gn = (geom6 - self.geom_mean) / self.geom_std
        z = self.trunk(torch.cat([path, gn, uvs], dim=-1))
        return self.score(z).squeeze(-1), self.mix(z)


S.Stage6BSurrogate = NoGRU

if __name__ == "__main__":
    OUT = T / "derived/abl"; OUT.mkdir(parents=True, exist_ok=True)
    tag = "nogru_n628_e240_s1"
    atlas = T / "surrogate_lc/combined_abl.h5"
    if not atlas.exists():
        import shutil
        shutil.copy2(T / "atlas_684_geometries_2e4roots.h5", atlas)
    sys.path.insert(0, str(T / "derived"))
    os.environ.setdefault("TRAIN_WORKER", "9")
    import train_study as TS          # loads truth data and the label() helper
    TS.label(atlas, 628)
    t0 = time.perf_counter()
    sys.argv = ["x", "train", "--atlas", str(atlas), "--output", str(OUT / f"{tag}.pt"), "--device", "cpu", "--epochs", "240",
                "--samples-per-epoch", str(int(round(65536 * 628 / 278))), "--seed", "1", "--print-every", "20"]
    S.main()
    tmin = (time.perf_counter() - t0) / 60
    m, _ = S.load_model(OUT / f"{tag}.latest.pt", torch.device("cpu"))
    nparam = sum(p.numel() for p in m.parameters())
    TS.OUT = OUT
    TS.save_preds(tag, OUT / f"{tag}.pt", OUT / f"{tag}.latest.pt", dict(n=628, epochs=240, seed=1, variant="no ordered-guide GRU",
                                                                         params=int(nparam), train_minutes=tmin))
    print("ALLDONE", tag, nparam, flush=True)
