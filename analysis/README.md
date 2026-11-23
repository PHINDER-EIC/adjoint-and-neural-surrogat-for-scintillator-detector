# Analysis scripts

The study scripts behind the numbers in the paper, with their paths pointed at this repository.
They read `data/` (`data/derived` for intermediate results) and write their outputs back to `data/derived`, so a rerun replaces the released file with its recomputed version.
Run them from this folder: `python <script>.py`.

These were rerun from this folder and reproduce their stored outputs exactly: `ref_noise.py`, `fd_best.py`, `fig_scaling.py`, `fig_adjoint.py`.
The others are the versions that produced the stored results, but they were not rerun here, because many take hours (training runs, Geant4, derivative scans).

`events/` holds the end-to-end event study (Fig. 15): `g4_edep.py` (Geant4 energy deposits), `make_photons.py`, `run_opt.py` / `g4_ev_opt.py` (Geant4 optical transport, in the Geant4 environment), and `analyze_events.py` (adjoint and network predictions, `events_hist.npz`). It works in `data/derived/events` (override with `PHADJ_EVENTS`). The raw event chunks were not kept.

## Adjoint convergence and noise

| Script | What it does | Writes (in data/derived) |
|---|---|---|
| `adj_conv.py` | Adjoint (numba) convergence study on the 40 test geometries. |  |
| `adj_conv_new.py` | Adjoint convergence runs for the 40 new designs (bank_blind40, test_* ids), as adj_conv.py does for the original 40. |  |
| `adj_h_new.py` | 2e4-root adjoint maps WITH arrival-time histograms for the 40 new designs (the convergence runs in adj_new/ keep only moments). Same settings as the atlas (max_steps 5000 as in adj_conv). Seeds independent of the referen… |  |
| `ref_noise.py` | Measure the noise of the 2e5-root reference maps without using the reference. | `ref_noise.json` |
| `region_noise.py` | Measured standard errors of the adjoint region values used in the Geant4 comparison (regions.json). | `region_noise.json`, `regions.json` |
| `bench.py` | Clean speed benchmarks (run on an otherwise idle machine). Writes study/bench.json. |  |
| `fig_adjoint.py` | Figure: adjoint (numba reverse tracer) convergence with root count, vs the 200k-root maps. | `adj_convergence.json` |
| `launch_check.py` | Launch phase-space check of the reverse tracer: cosine-weighted directions and uniform port coordinates. |  |
| `split_study.py` | Sensitivity of the reverse tracer to the Fresnel-split threshold (split when both children weigh >= threshold, else one analog child). threshold > 0.5 never splits (pure analog). 20k roots, 2 independent seed pairs, 4 te… |  |

## Geant4 comparison

| Script | What it does | Writes (in data/derived) |
|---|---|---|
| `fix_sd_se.py` | Standard error of the arrival-time spread (0-60 ns) for the Monte Carlo sources in regions.json. | `regions.json` |
| `make_g4_agreement.py` | Agreement of each method with Geant4 (unpolarised, matched configuration) over all emission regions and channels. Writes study/g4_agreement.json (used by paper_figs.m1 and the Geant4 table). Earlier versions of this file… |  |
| `g4_fast.py` | Geant4 optical transport without per-step Python: the cost of Geant4 itself, as in a compiled application. |  |
| `g4_fast_bench.py` | Measured Geant4 cost without per-step Python (g4_fast.py), on all 14 cores, and actual forward maps. | `g4_fast.json` |
| `g4_fast_time.py` | Voxel error of the mean arrival time (t-bar) of the Geant4 forward maps in g4_fast.json (B_train_g0180, channel 0), with the paper's metric (p-weighted rms relative difference, reference noise subtracted), and its predic… | `g4_fast.json` |
| `g4_point_bench.py` | Geant4 (g4_fast.py, no Python in the photon loop) for the Fig. 13 task on this CPU: N = 1e3 ... 1e6 photons from one voxel at the cell centre of B_train_g0180 (voxel 10,10,4), one process and 14 processes (N/14 each). tr… |  |
| `g4_dbg.py` | Geant4 (11.4, geant4_pybind) optical forward simulation of a Stage-6B test geometry. |  |
| `fig_forward.py` | Figure: forward validation per emission region against Geant4 (and JAX), with pulls and physics effects. | `forward_summary.json` |
| `fig_g4map.py` | Figure: voxel-by-voxel comparison of the adjoint map with a Geant4 (unpolarised) forward map. | `g4map_summary.json` |
| `fig_time.py` | Figure: channel-0 arrival-time distributions per emission region: Geant4, JAX, adjoint map, network. |  |
| `guide1_effect.py` | How much does guide 1 change channel 0? Adjoint for channel 0 with the normal two-guide scene, and with guide 1 (its volumes and readout port) removed. Same roots and same random numbers (common random numbers), so the r… |  |
| `ratio80.py` | Whole-cell sensitivity-to-error ratios on the 80 test geometries (Sec. VI D): spread of the whole-cell mean arrival time t-bar (p-weighted mean of voxel mean times, uniform emission) and of ln p-bar across the geometries… | `ratio80.json` |
| `diag_review2_lib.py` | (a) Test set vs new designs: per-geometry error contributions, bootstrap intervals, permutation test. (b) Does the network capture how the mean arrival time depends on guide length (the main effect of the guide path on t… |  |
| `dist_metrics.py` | Kolmogorov-Smirnov distance (max |F-G|) and Wasserstein-1 distance (integral |F-G| dt, in ns) of the channel-0 arrival-time distribution, normalized over 0-60 ns on the 0.25-ns grid (CDF-based, so no rebinning needed). (… | `dist_metrics.json` |
| `dist_more.py` | More tests of the channel-0 arrival-time distribution f_0(t) (follow-up of dist_metrics.py / js_regions.py). | `dist_more.json` |
| `dist_chi2_time.py` | Where in time is the network's chi2 excess against Geant4? Same bins and merging as dist_more.py; each merged group is assigned to t < 2 ns or t >= 2 ns by its first bin. Prints the share of chi2 and of chi2 - ndf below … |  |
| `voxel_js.py` | Voxel-level Jensen-Shannon divergence of the channel-0 arrival-time shape f_0(t), network and 2e4-root adjoint maps vs the 2e5-root reference, on all 80 test geometries (Sec. VI C, Fig. 8). | `voxel_js.json` |
| `js_regions.py` | Jensen-Shannon divergence of the channel-0 arrival-time distribution f_0(t) in the Table III regions, each method vs Geant4 (unpolarized). Shapes only: each histogram is normalized over 0-60 ns. Bins as in Fig. 6: 0.25 n… | `js_regions.json` |

## Network: training, scaling, held-out designs

| Script | What it does | Writes (in data/derived) |
|---|---|---|
| `run_lc.py` | Learning curve of the repo's Stage-6B neural surrogate vs the Gaussian-process stand-in. |  |
| `train_study.py` | Surrogate training study: data scaling (with seeds) and epoch scaling, all scored on the 200k-root test maps. |  |
| `scaling_combined.py` | Scaling with training size on the test set, the 40 new designs, and both pooled (80 geometries). | `scaling_combined.json` |
| `fig_scaling.py` | Figure: surrogate accuracy vs training-set size and training length (40 test geometries, 200k reference). | `scaling.json` |
| `gp_study.py` | GP stand-in scaling: PCA-48 + ARD Matern-5/2 GP per component on the 13-D design vector, for log p0, mean time and time s.d. (0-60 ns) maps. Training sets = the same first-n pools as the network (make_combined.py label l… |  |
| `abl_train.py` | Ablation: the repo surrogate with its ordered-guide BiGRU encoder replaced by an MLP on the flattened node features. |  |
| `blind_eval.py` | One-shot blind evaluation on 40 fresh geometries (new Sobol seed 953400, never seen during any training, model selection or analysis). Frozen models: 3-network ensemble and single networks (628 training geometries, 240 e… | `blind.json` |
| `blind_metrics.py` | Recompute the network metrics of blind.json (40 new designs) with the current common.pmetrics / tmetrics. Same frozen models and reference as blind_eval.py; the GP entry (not used in paper v1) is kept from the old file b… | `blind.json`, `blind_preds.npz` |
| `straight10.py` | Network on ten straight-guide designs (xgeom/straight10, no bends; absent from training) plus X_test_g0000: voxel errors of p, t-bar, s_t (C.pmetrics / C.tmetrics: p-weighted, reference noise subtracted, as for the 80 te… | `straight10.json` |
| `straight10_make_bank.py` | Ten straight-guide designs (no bends), as xgeom/make_bank.py for X_test_g0000 but spread over the design family: cell D_x, D_y in [4, 6] cm, D_z in [0.7, 1.3] cm, width w in [0.4, 0.8] cm, R/w in [1.5, 2.5] (Sobol, scipy… |  |
| `xgeom_make_bank.py` | Geometry 00: two straight guides that cross as an X (no bends). 5 x 5 x 1 cm cell, width and radius of ref_5x5x1. The four 'turn' points lie on the straight line, so the rounding code adds no fillet (collinear points are… |  |
| `stats_rel.py` | Statistics quoted in Sec. VI.C, recomputed with the current residual of p (common.resid; relative difference by default): new designs vs test (bootstrap, permutation), offset and scatter, residual correlations, track and… |  |
| `t10_study.py` | Early-arrival quantiles: t10 per voxel and channel, and the channel asymmetry dt10 = t10(ch1) - t10(ch0). | `t10.json`, `t10_maps.npz` |
| `t10_80.py` | Channel asymmetry dt10 = t10(ch1) - t10(ch0) on all 80 test geometries (original 40 + 40 new designs), network (each of the three seeds) and independent 2e4-root adjoint maps, against the 2e5-root references; same defini… | `t10_80.json`, `t10_80_arrays.npz` |
| `fig_maps.py` | Figure: detection-probability maps and residual maps of each method for one test geometry. |  |
| `fig_residuals.py` | Figure: residual statistics of every method against the 200k-root reference maps (40 test geometries). | `residual_stats.json` |

## Geometry derivatives

| Script | What it does | Writes (in data/derived) |
|---|---|---|
| `fd_study.py` | Derivative fidelity: d ln(whole-volume p0) / d u for design parameters, adjoint finite differences vs surrogates. |  |
| `fd_scan.py` | Line scans through the finite-difference cases: is the timing-derivative disagreement from the FD step, the reference noise, or the network? |  |
| `fd_scan_analyze.py` | Analyse fd/scan_runs.json + fd/time_runs.json (adjoint at u0 + s e_k, 8 seeds each) against fd/scan_net.json. Per seed: least-squares line and quadratic through all available s; derivative = mean over seeds, error = sd/s… |  |
| `fd_stencils.py` | All 24 derivatives from the line scans: central differences at every symmetric step h, five-point stencils [f(-2h) - 8 f(-h) + 8 f(h) - f(2h)] / 12h, and the least-squares line through the symmetric points, per seed (8 s… |  |
| `fd_best.py` | Best derivative per case from the line scans: symmetric least-squares line when >= 4 symmetric points (|s| <= 0.16); otherwise a quadratic least-squares fit through all valid points with |s| <= 0.16 if that is more preci… |  |
| `fd_time.py` | Finite-difference derivatives of whole-cell timing quantities with replicated seed pairs. |  |
| `fd_time_check.py` | Check: truncated-histogram mean (fd_time.py) vs untruncated volume mean transport time (fd_study.py) on the same runs. Rows with the largest old/new dt discrepancy; 8 seed pairs, 5e4 roots. Writes fd/time_check.json. |  |
| `fd_net_time.py` | Network derivatives of whole-cell p, mean arrival time, arrival-time s.d. and t10 at the perturbed designs of fd_study.py. Whole-cell distribution = voxel mean of p0 * timing mass (uniform emission), same definitions as … |  |
| `fd_compare_time.py` | Compare replicated adjoint FD (fd/time_derivs.json) with the old 2-pair FD (fd/adjoint.json) and the network (fd/network_time.json). Writes fd/time_compare.json. |  |
| `fd_surrogates.py` | Surrogate derivatives for the designs of fd_study.py, and the derivative-fidelity figure. |  |

## Speed and cost

| Script | What it does | Writes (in data/derived) |
|---|---|---|
| `fig_speed.py` | Figure: speed vs accuracy of every method for full 20x20x8 detection-probability maps (one geometry, both channels). | `speed_summary.json` |
| `jax_study.py` | Forward JAX (engine B) runs for the speed/accuracy study. |  |
| `numba_fwd_bench.py` | Speed of the numba forward engine (extras/numba_engine) on this 14-core CPU, for Fig. 11 and Fig. 13. | `numba_fwd.json` |
| `numba_plot_data.py` | Plot-ready data for adding the numba forward engine to Fig. 11 and Fig. 13 (no figure is changed). Writes paper_v1/data/fig11_numba.csv, fig11_numba_lines.csv, fig13_this_machine.csv and numba_summary.json. |  |
| `event_cost.py` | Per-event cost of turning a response map (or the surrogate) into the detected arrival-time distribution of an event. | `event_cost.json` |

## Tables

| Script | What it does | Writes (in data/derived) |
|---|---|---|
| `paper_tables_v2.py` | Tables for paper v1 (simplified): no GP, single network as 'the surrogate', narrow columns. Writes paper_v1/tables/*.tex. table* only where a single column is too narrow. |  |

