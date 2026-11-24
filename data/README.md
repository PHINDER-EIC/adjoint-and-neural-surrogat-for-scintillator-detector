# Data

The data are on Zenodo ([doi:10.5281/zenodo.24459410](https://doi.org/10.5281/zenodo.24459410)). Download the three HDF5 files and `data.zip` into this folder and unzip
`data.zip` here; it holds `banks/`, `networks/`, `derived/`, `surrogate_lc/`, `xgeom/` and `MANIFEST.sha256`. Then check
the files against the manifest (from this folder):

```
sha256sum -c MANIFEST.sha256
```

| Path | Contents |
|---|---|
| `atlas_684_geometries_2e4roots.h5` | training atlas: adjoint maps of 684 geometries, 2e4 roots each. Attribute `split` marks the paper's 628 train, 16 val and 40 test geometries. Ids are prefixed A_ (default bank), B_ (bank250) and C_ (bank350) |
| `reference_test40_2e5roots.h5` | reference maps of the 40 test geometries, 2e5 roots each |
| `reference_blind40_2e5roots.h5` | reference maps of the 40 additional test designs, 2e5 roots each (plus `train_g0000`, the default design) |
| `banks/` | the geometry banks: design parameters of every geometry (also in git) |
| `networks/network_seed{20261003,1,2}.pt` | the three networks used in the paper (final weights, epoch 240) |
| `surrogate_lc/` | networks of the learning-curve runs (other training-set sizes and seeds) and their scores |
| `xgeom/` | ten straight-guide designs and the X-guide design: bank, 2e5- and 2e4-root maps |
| `derived/` | intermediate results read by the figure, table and analysis scripts |

The main files in `derived/`:
- `adj/`, `adj_new/`, `adj_new_h/`: adjoint maps at 10³–5×10⁴ roots with replicates (convergence study).
- `g4/`:
  - `geom/`: exported scenes;
  - `runs/`: Geant4 chunks;
  - `fast/`, `point/`: timing runs.
- `sur/`, `gp/`, `abl/`: network, Gaussian-process and ablation predictions on the test geometries.
- `fd/`: derivative scans, `fd_atlas.h5` and `fd_scan_net.h5`.
- `events/`: event-study histograms and timings.
- `fig08/`: data of Fig. 8, from a separate training run.
  - `figure-data.h5`: one geometry, its two 2×10⁵-root adjoint maps, three network predictions and their mean, and the derived maps.
  - The same numbers as CSV files (`voxel-data.csv`, `layer-metrics.csv`, guide outlines), plus `metadata.json`.
- `jax/`: forward-engine runs.
- The JSON files at the top level of `derived/`: the summaries behind the tables, figures and text.

## Licence

The data are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The code that made them is MIT-licensed (see `LICENSE` in the repository).
