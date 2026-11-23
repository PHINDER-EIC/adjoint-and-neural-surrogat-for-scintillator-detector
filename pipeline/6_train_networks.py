"""Step 6: train the three networks on the 628 training geometries (240 epochs, 148,032 samples per epoch, CPU).

The paper uses the final weights (<name>.latest.pt, epoch 240).  About 20 min per network on 14 cores.  Training on
CPU is not bit-reproducible (multithreaded reductions): a rerun follows the paper's loss curve to about 0.1%
(0.3609 vs 0.3612 after epoch 1, equal to 4 decimals by epoch 32), not exactly.
usage: python 6_train_networks.py [--atlas FILE] [--seeds 20261003 1 2]
"""
import argparse
import sys

from _common import EPOCHS, SAMPLES_PER_EPOCH, SEEDS, SURROGATE, WORK, paths, run

ap = argparse.ArgumentParser()
ap.add_argument("--atlas", default=str(WORK / "atlas_684_geometries_2e4roots.h5"),
                help="training atlas; the released one is data/atlas_684_geometries_2e4roots.h5")
ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
a = ap.parse_args()
out = WORK / "networks"
out.mkdir(parents=True, exist_ok=True)
for seed in a.seeds:
    model = out / f"network_seed{seed}.pt"
    cmd = [sys.executable, "-u", SURROGATE, "train", "--atlas", a.atlas, "--output", model, "--device", "cpu",
           "--epochs", EPOCHS, "--samples-per-epoch", SAMPLES_PER_EPOCH, "--seed", seed, "--print-every", 20]
    if model.with_suffix(".latest.pt").exists():
        cmd.append("--resume")
    run(cmd)
