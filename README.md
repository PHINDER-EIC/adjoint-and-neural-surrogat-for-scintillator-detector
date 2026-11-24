# Geometry-conditioned differentiable light-guide surrogate with adjoint Monte Carlo sampling — code and data

Code and data for the paper "Geometry-conditioned differentiable light-guide surrogate with adjoint Monte Carlo sampling for
scintillator detector co-design". The repository covers the whole chain:
- the geometry banks;
- the adjoint atlases and reference maps;
- the networks;
- the Geant4 validation;
- the analyses behind the numbers;
- one script per data figure.

**Code and data are published separately.** Code is on GitHub (this repository) and data on Zenodo ([doi:10.5281/zenodo.24459410](https://doi.org/10.5281/zenodo.24459410)). The data
go in `data/`; see [`data/README.md`](data/README.md).

## Layout

```
pipeline/          how the data were made, in order (results go to work/, never to data/)
  1_make_banks.py          the four geometry banks (13-D Sobol designs)
  2_build_atlases.py       adjoint maps of the 684 training-bank geometries, 2e4 roots each
  3_combine_atlases.py     the training atlas, with the paper's train / val / test split
  4_make_test_bank.py      the bank of the 40 test geometries
  5_build_references.py    reference maps, 2e5 roots: the 40 test geometries and 40 additional test designs
  6_train_networks.py      the three networks (seeds 20261003, 1, 2)
  7_geant4/                Geant4 11.4.1 validation: scene export, optical simulation, run pool
figures/figNN_<name>/  one folder per data figure: the script and the PDF used in the paper
tables/table3_geant4_agreement/   the statistics behind Table III, and the table as typeset
analysis/          the study scripts behind the other numbers (convergence, Geant4, scaling, derivatives, speed, events)
code/              shared modules (paths, metrics and loaders, plot style, emission regions)
  simulator/       the simulator and network code, in the layout of the phinder-adjoint-optics repository:
                   Scintillator-Simulator-main/src/Photon_Simulator (scene, materials, forward engines incl. JAX),
                   Scintillator-Simulator-main/extras/numba_adjoint (reverse tracer, atlas driver),
                   Scintillator-Simulator-main/extras/numba_engine (forward engine, numba),
                   Scintillator-Simulator-main/extras/jax_compaction, phinder_bridge/scripts (atlas builder, network)
data/              from Zenodo (geometry banks are also in git)
```

## Installation

Python 3.11:

```
pip install -r requirements.txt
```

Geant4 (pipeline step 7 and `analysis/events`) runs in a separate environment:

```
pip install -r requirements-geant4.txt
```

It also needs the `G4ENSDFSTATE3.0` dataset, with `G4ENSDFSTATEDATA` pointing at it. Set `G4_PYTHON` to that
environment's Python when starting `7_geant4/g4_pool.py` from the main environment.

Environment variables (all optional):
- `PHADJ_DATA`: data folder (default `data/`).
- `PHADJ_WORK`: pipeline output (default `work/`).
- `PHADJ_REPO`: simulator code (default `code/simulator`).
- `FIG_DIR`: where figures are written (default: next to the script).

## Reproducing the figures

```
cd figures/fig06_geant4_comparison
python fig06_geant4_comparison.py
```

| Paper | Folder | PDF | Data read |
|---|---|---|---|
| Fig. 3 | `fig03_forward_and_reverse_sampling` | `fig_reciprocal_schematic.pdf` | none: a 2D trace of one root, computed in the script |
| Fig. 4 | `fig04_detection_maps_vs_roots` | `fig_backtrace_maps.pdf` | reference maps, `banks/bank_test40.json`; reruns the reverse tracer (~1.5 min) |
| Fig. 6 | `fig06_geant4_comparison` | `figM1_validation_geant4_ch0.pdf` | reference maps, networks, Geant4 region results (`derived/regions.json`, `regions_hist_single.npz`, `ch0_table3.json`, `g4/geom`) |
| Fig. 7 | `fig07_adjoint_convergence` | `figM2_adjoint_convergence_80.pdf` | both reference files, `derived/adj`, `adj_new`, `bench.json`, `ref_noise.json` |
| Fig. 8 | `fig08_ensemble_vs_reference` | `Fig-8.pdf` | `derived/fig08/figure-data.h5` (see below) |
| Fig. 9 | `fig09_error_vs_training_geometries` | `figM4_surrogate_scaling.pdf` | `derived/scaling_combined.json`, `adj_convergence.json` |
| Fig. 10 | `fig10_dt10_top_bottom_layers` | `fig10_early_time_quantiles.pdf` | reference maps, training atlas, network seed 20261003 |
| Figs. 11, 12 | `fig11_fig12_dt10_middle_layer_and_vertical_section` | `fig10b_t10_middle_layer.pdf`, `fig10c_t10_cross_section.pdf` | `derived/t10_maps.npz` (from the Fig. 10 script) |
| Fig. 13 | `fig13_geometry_derivatives` | `figM5_derivatives_scan.pdf` | `derived/fd/best_derivs.json` |
| Fig. 14 | `fig14_cost_on_matched_tasks` | `figM8_cost_matched_v2.pdf` | timing and accuracy summaries in `derived/` |
| Fig. 15 | `fig15_events_end_to_end` | `figM7_events.pdf` | `derived/events/events_hist.npz` |
| Table III | `tables/table3_geant4_agreement` | `tab_geant4.tex` | as Fig. 6 |

Run from a clean checkout with the Zenodo data, every figure script reproduces the PDF used in the paper pixel for
pixel, and the Table III scripts reproduce their statistics files byte for byte.

Fig. 8 comes from a separate training run with a new split and three networks (seeds 62801–62803).
`derived/fig08/figure-data.h5` holds everything the figure shows for its geometry (`sobol_independent__021`):
- the two 2×10⁵-root adjoint maps A and B;
- the three network predictions and their mean;
- the derived maps.

Those networks and that run's geometry bank are not part of this release.

Not included, because they are schematics, were drawn by hand, or were made outside this code: Figs. 1, 2, 5 and 16, and Tables I and II.

## Rebuilding the data

Run the pipeline steps in order from `pipeline/`. Each writes to `work/`, so the results can be compared with `data/`.

| Step | Time (14-core CPU) | Check made for this release |
|---|---|---|
| 1 banks | ~1 min | all four banks identical to `data/banks` |
| 2 atlases | ~70 min | maps bit-identical (one geometry per bank) |
| 3 combine | ~1 min | split labels identical (`--check`) |
| 4 test bank | seconds | identical to `data/banks/bank_test40.json` |
| 5 references | ~2 h | bit-identical (one geometry) |
| 6 networks | ~20 min per network | follows the paper's loss curve to ~0.1%; multithreaded CPU training is not bit-reproducible |
| 7 Geant4 | ~1 min per 10⁵ photons | scene export and a 10⁵-photon run bit-identical to the stored ones |

Seeds:
- Banks: Sobol seeds 629100, 731200, 842300 and 953400.
- Tracer: launch and physics seeds fixed per split.
- Split permutation: 20261028.
- Networks: 20261003, 1 and 2.
- Geant4: SHA-256 of the run tag.

## The HDF5 files

The training atlas and the two reference files have the same layout. Each geometry is a group `geometries/<id>`:

- `field/p_per_emitted` (20, 20, 8): detection probability of readout 0 per emitted photon in each voxel; `field/p_se` is its standard error.
- `field/h_per_ns` (20, 20, 8, 240): arrival-time density at readout 0, 0–60 ns in 0.25-ns bins (edges in `/time_edges_ns`).
- `field/mean_transport_time_ns` (20, 20, 8): mean arrival time.
- `x_edges_cm`, `y_edges_cm`, `z_edges_cm`: voxel edges.
- `cell_cm`, `design_vector` (13 design parameters), `guide0_points_cm`, `guide1_points_cm`: the geometry.
- Attributes:
  - `split` (train / val / test), `design_json`, `width_cm`, `radius_cm`;
  - `guide_thickness_cm` (0.10) and `cladding_thickness_cm` (0.001);
  - root counts and seeds;
  - whole-cell means (`volume_mean_p`, `volume_mean_transport_time_ns`).

Readout 1 is the mirror image of readout 0 (x → D_x − x, z → D_z − z) and is not stored. The root datasets `u_edges`,
`v_edges` and `s_edges` hold the guide-frame coordinates used by the network.

## Licence

Code: MIT licence (see `LICENSE`). Data on Zenodo ([doi:10.5281/zenodo.24459410](https://doi.org/10.5281/zenodo.24459410)): Creative Commons Attribution 4.0 International (CC BY 4.0).
