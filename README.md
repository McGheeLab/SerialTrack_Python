# SerialTrack Python

A modern Python reimplementation of [SerialTrack](https://github.com/FranckLab/SerialTrack), a particle tracking velocimetry (PTV) framework for measuring full-field displacements and strains from 2D and 3D image sequences. Originally developed in MATLAB by the [Franck Lab](https://www.franck.engin.brown.edu/), this Python version delivers the same scientific rigor with JIT-accelerated performance, modular architecture, and an optional PySide6 GUI.

---

## Overview

SerialTrack uses a serial approach to particle tracking: rather than solving a global assignment problem in one shot, it iteratively refines particle correspondences through an **ADMM (Alternating Direction Method of Multipliers)** loop that alternates between local topology-based matching and global displacement regularization. This makes it especially robust for large deformations, high particle densities, and 3D volumetric data.

### Key Features

- **2D and 3D support** — works with planar image sequences or volumetric stacks
- **Numba JIT acceleration** — core matching kernels run at near-native speed
- **Three tracking modes** — incremental, cumulative, and double-frame
- **Multiple detection methods** — LoG + polynomial sub-pixel (TracTrac) or blob centroid + radial symmetry (TPT)
- **POD-GPR motion prediction** — intelligent initial guesses for faster convergence on long sequences
- **Trajectory stitching** — automatic merging of broken trajectory segments with gap filling
- **Multi-format export** — HDF5, MATLAB `.mat`, NumPy `.npz`, and per-frame CSV
- **GUI-ready architecture** — protocol-based design with optional PySide6 integration

---

## Installation

### Dependencies

```bash
pip install numpy scipy scikit-image numba tifffile scikit-learn h5py
```

For the optional GUI:

```bash
pip install PySide6 matplotlib
```

### From Source

```bash
git clone https://github.com/your-org/serialtrack-python.git
cd serialtrack-python
pip install -e .
```

---

## Quick Start

```python
from serialtrack.config import DetectionConfig, TrackingConfig
from serialtrack.io import ImageLoader
from serialtrack.tracking import SerialTracker

# Load a 2D image sequence
images = ImageLoader.load_2d_sequence("path/to/images/", pattern="*.tif*")

# Configure detection and tracking
det_cfg = DetectionConfig(threshold=0.4, bead_radius=3.0, method=2)
trk_cfg = TrackingConfig(f_o_s=60.0, n_neighbors_max=25, max_iter=20)

# Run tracking
tracker = SerialTracker(det_cfg, trk_cfg)
session = tracker.track_images(images)

# Extract results
trajectories = session.build_stitched_trajectories()  # (N_traj, n_frames, D)
print(f"Tracked {len(session.frame_results)} frame pairs")
print(f"Mean tracking ratio: {session.tracking_ratios.mean():.1%}")
```

---

## Architecture

```
serialtrack/
├── config.py           # Configuration dataclasses (DetectionConfig, TrackingConfig)
├── io.py               # Image loading (TIFF, MAT, NPY sequences)
├── detection.py        # Particle detection & sub-pixel localization
├── matching.py         # Topology-based & nearest-neighbor particle matching
├── outliers.py         # Westerweel universal outlier detection
├── regularization.py   # Global-step solvers (MLS, grid regularization, ADMM-AL)
├── fields.py           # Displacement & strain field computation
├── prediction.py       # POD-GPR initial guess prediction
├── tracking.py         # Main ADMM tracking engine & session management
├── trajectories.py     # Trajectory segment stitching & merging
├── results.py          # Export, persistence & GUI bridge
└── run.py              # PySide6 GUI entry point
```

---

## Pipeline Workflow

The SerialTrack pipeline processes an image sequence through six stages. Each stage is handled by a dedicated module, and the main tracking engine (`tracking.py`) orchestrates them into a coherent ADMM loop.

### Stage 1 — Image Loading (`io.py`)

The `ImageLoader` class supports multiple input formats and normalizes everything into NumPy arrays using SerialTrack's coordinate convention (`(x, y)` for 2D, `(x, y, z)` for 3D — transposed from the typical row-column image layout).

**Supported formats:**
- 2D TIFF/PNG sequences from a folder
- 3D TIFF slice stacks (one file per z-slice, assembled into a volume)
- MATLAB `.mat` volumetric files (v5 via scipy, v7.3 via h5py)
- Raw NumPy `.npy` arrays

RGB images are automatically converted to grayscale. MATLAB volumes are permuted from `(row, col, slice)` to `(x, y, z)` to match SerialTrack's internal convention.

### Stage 2 — Particle Detection (`detection.py`)

The `ParticleDetector` class locates particles at sub-pixel accuracy using one of two methods:

**TracTrac method** (`DetectionMethod.TRACTRAC`):
1. Threshold the image and filter connected components by size (`min_size` to `max_size`)
2. Apply a Laplacian of Gaussian (LoG) filter at the specified `bead_radius` scale
3. Find local maxima via a dilation-based peak detector
4. Refine to sub-pixel accuracy using 3-point parabolic fitting along each axis (Numba-accelerated)

**TPT method** (`DetectionMethod.TPT`):
1. Threshold and compute connected-component centroids
2. Extract image patches around each centroid
3. Apply radial symmetry sub-voxel localization (Liu et al. 2013) — a Numba-parallelized kernel that builds a weighted normal system from intensity gradient votes and solves via Cramer's rule

Both methods support optional Richardson-Lucy PSF deconvolution as a preprocessing step, and can detect either bright-on-dark (`color="white"`) or dark-on-bright (`color="black"`) particles.

### Stage 3 — Particle Matching (`matching.py`)

Particle matching establishes correspondences between detected particles across frames. SerialTrack's distinguishing feature is **rotation-invariant topology matching**, which identifies particles by the geometric signature of their local neighborhood rather than by absolute position.

**Topology matching** (`TopologyMatcher`, used when `n_neighbors > 2`):
1. Build a KD-tree for each frame's particle coordinates
2. For each particle, find K nearest self-neighbors and construct a **rotation-invariant local coordinate frame**: `ex` points to the nearest neighbor, `ez = cross(r₁, r₂)` oriented by the third neighbor, `ey = cross(ez, ex)`
3. Transform neighbor offsets into this frame and compute spherical coordinates `(r, φ, θ)`
4. Reorder neighbors by azimuthal angle starting from the nearest, producing a feature vector of sorted distances, angular gaps, and polar angles
5. For each particle in frame A, evaluate all candidate particles in frame B (within a ball of radius `√D × f_o_s`) and select the candidate whose feature vector minimizes the weighted Euclidean distance
6. Enforce one-to-one matching by keeping only the best match per B particle

All feature construction and brute-force matching kernels are Numba JIT-compiled with `parallel=True` for throughput.

**Nearest-neighbor fallback** (`NearestNeighborMatcher`, used when `n_neighbors ≤ 2`): simple KD-tree closest-point matching with an `f_o_s` distance cutoff.

After matching, the `compute_displacement` function builds a `track_a2b` index array mapping each A particle to its B partner (or -1 if untracked), and optionally applies Westerweel outlier removal.

### Stage 4 — Outlier Removal & Adaptive Search (`outliers.py`)

**Westerweel universal outlier detection** implements the normalized median residual test from Westerweel & Scarano (2005). For each tracked particle, the algorithm computes the median displacement of its K nearest tracked neighbors, then flags particles whose displacement deviates from the local median by more than `outlier_threshold` times the median absolute deviation (with a 0.075 px fluctuation floor).

**Missing-particle culling** activates in late ADMM iterations (when `n_neighbors < 4`) and identifies particles in both frames that lack a plausible nearby partner after the global warp, removing them from subsequent matching to avoid spurious links.

**Adaptive field-of-search** (`update_f_o_s`) shrinks the search window between ADMM iterations based on displacement quantiles (median + 0.5 × IQR), with a floor of max(2 px, 10% of current f_o_s) to prevent collapse.

### Stage 5 — Global Regularization (`regularization.py`)

The global step interpolates sparse matched displacements onto all particles (or a regular grid) to produce a smooth displacement field. Three solvers are available:

**Moving Least Squares** (`GlobalSolver.MLS`): fits a local linear model at each query point using distance-weighted neighbors from a KD-tree. Fast and mesh-free, suitable for the first ADMM iteration.

**Grid Regularization** (`GlobalSolver.REGULARIZATION`): scatters particle displacements onto a regular grid using RBF interpolation, then applies a thin-plate spline smoothing penalty via sparse matrix solve. Produces smooth fields appropriate for computing strain gradients.

**ADMM with Augmented Lagrangian** (`GlobalSolver.ADMM`): iterates between the scattered interpolation and a derivative-penalized regularization step with an L-curve strategy for automatic penalty parameter selection. The derivative operator is built via Kronecker products of 1D finite-difference matrices, and the regularization subproblem is solved with `scipy.sparse.linalg.spsolve`.

The `DisplacementRegularizer` class automatically selects MLS for the first iteration (when the displacement field is rough) and switches to the configured solver for subsequent iterations.

### Stage 6 — ADMM Tracking Loop (`tracking.py`)

The `_ADMMFrameTracker` class orchestrates the iterative refinement for each frame pair, following the exact structure of the MATLAB `f_track_serial_match3D.m`:

```
For each frame pair (A → B):
  1. Initialize: coords_b_curr = coords_b + init_disp (if available)
  2. For iter = 1 to max_iter:
     a. Compute n_neighbors (exponential decay from max to min)
     b. LOCAL STEP: match coords_a ↔ coords_b_curr using topology matching
     c. GLOBAL STEP: regularize matched displacements → temp_disp
     d. CONVERGENCE CHECK (before warping — matching MATLAB behavior)
        - If ‖temp_disp‖ < threshold or match_ratio ≈ 1.0 for 5+ iters → stop
     e. WARP: disp_b2a += temp_disp; coords_b_curr = coords_b + disp_b2a
     f. UPDATE: shrink f_o_s based on displacement quantiles
     g. CULL: remove missing particles when n_neighbors < 4
```

The neighbor count starts high (broad search) and decays exponentially toward `n_neighbors_min`, progressively tightening the matching criteria as the displacement field converges.

### Tracking Modes

The `SerialTracker` class supports three tracking modes that determine how frame pairs are formed:

**Incremental** (`TrackingMode.INCREMENTAL`): tracks frame-to-frame (1→2, 2→3, 3→4, ...). Best for small inter-frame displacements. Trajectory segments are built by chaining consecutive links and then stitched across gaps.

**Cumulative** (`TrackingMode.CUMULATIVE`): tracks everything relative to the first frame (1→2, 1→3, 1→4, ...). Best when particle appearance changes slowly and displacements from the reference are moderate. Produces one trajectory per reference particle directly.

**Double-frame** (`TrackingMode.DOUBLE_FRAME`): treats each consecutive pair independently with no history. Useful for PIV-style analysis where temporal coherence isn't assumed.

### Motion Prediction (`prediction.py`)

For incremental and cumulative modes, the `InitialGuessPredictor` provides warm-start displacements that dramatically accelerate ADMM convergence on later frames:

- **Frame 3** (1 history frame): simple 2× extrapolation of the previous displacement
- **Frames 4–6** (2 history frames): linear extrapolation from the two most recent results
- **Frame 7+** (5+ history frames): **POD-GPR** — applies PCA to a snapshot matrix of recent displacement fields to extract dominant spatial modes, then fits a Gaussian Process Regressor to each mode's temporal coefficient and predicts the next time step

All strategies interpolate previous displacement fields to current particle positions using Delaunay-based `LinearNDInterpolator`.

### Post-Processing

**Displacement & strain fields** (`fields.py`): after each frame pair converges, displacements are scattered onto a regular grid and differentiated via `np.gradient` to produce the full deformation gradient tensor. Strain is computed in both the deformed and reference configurations. An alternative MLS-based strain gauge (`compute_strain_mls`) provides direct scattered-point strain estimates without gridding.

**Trajectory stitching** (`trajectories.py`): in incremental mode, frame-to-frame links often produce short segments broken by detection gaps. The merging algorithm extrapolates each segment forward and backward using PCHIP interpolation (or nearest-neighbor for Brownian motion), searches for other segments whose endpoints are nearby, and merges matches across multiple passes. Configurable via `TrajectoryConfig` (distance threshold, minimum segment length, maximum gap, number of passes).

---

## Configuration Reference

### DetectionConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `TRACTRAC` (2) | Detection strategy: `TPT` (1) or `TRACTRAC` (2) |
| `threshold` | 0.4 | Normalized intensity threshold [0–1] |
| `bead_radius` | 3.0 | Expected particle radius in pixels (0 = centroid only) |
| `min_size` | 2 | Minimum blob area/volume in pixels |
| `max_size` | 1000 | Maximum blob area/volume in pixels |
| `color` | `"white"` | Foreground: `"white"` (bright particles) or `"black"` (dark) |
| `psf` | `None` | PSF array for Richardson-Lucy deconvolution |
| `deconv_iters` | 6 | Number of deconvolution iterations |
| `win_size` | `(5,5,5)` | Radial symmetry window (TPT, 3D only) |

### TrackingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `f_o_s` | 60.0 | Initial field-of-search radius [px] |
| `n_neighbors_max` | 25 | Starting neighbor count for topology features |
| `n_neighbors_min` | 1 | Final neighbor count (convergence target) |
| `loc_solver` | `TOPOLOGY` (1) | Local solver: topology (1) or histogram→topology (2) |
| `solver` | `REGULARIZATION` (2) | Global solver: MLS (1), regularization (2), or ADMM (3) |
| `smoothness` | 0.1 | Regularization smoothing parameter |
| `outlier_threshold` | 5.0 | Westerweel normalized residual cutoff |
| `max_iter` | 20 | Maximum ADMM iterations per frame pair |
| `iter_stop_threshold` | 1e-2 | Convergence threshold on displacement update norm |
| `mode` | `INCREMENTAL` (1) | Tracking mode: incremental (1), cumulative (2), double-frame (3) |
| `use_prev_results` | `False` | Enable POD-GPR motion prediction |
| `xstep`, `ystep`, `zstep` | 1.0 | Physical pixel size per dimension |
| `tstep` | 1.0 | Physical time step per frame |

### TrajectoryConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dist_threshold` | 1.0 | Max distance to connect split segments [px] |
| `extrap_method` | `"pchip"` | Extrapolation: `"pchip"` (smooth) or `"nearest"` (Brownian) |
| `min_segment_length` | 10 | Min frames for a segment to be extrapolated |
| `max_gap_length` | 0 | Max frame gap between merged segments (0 = adjacent only) |
| `merge_passes` | 4 | Number of merge iterations |

---

## Export & Persistence

The `results.py` module provides full session serialization and multiple export formats:

```python
from serialtrack.results import SessionExporter, ExportFormat

exporter = SessionExporter()

# Full session round-trip (preserves configs, all frame results, trajectories)
exporter.save(session, "results.h5", fmt=ExportFormat.HDF5)
exporter.save(session, "results.mat", fmt=ExportFormat.MAT)
exporter.save(session, "results.npz", fmt=ExportFormat.NPZ)

# Per-frame particle CSV (positions, displacements, tracking indices)
exporter.save(session, "output_dir/", fmt=ExportFormat.CSV)
```

The results system also includes a thread-safe `ResultAccumulator` for live GUI updates, a `SessionSummary` with per-frame statistics, and a `SignalBridge` protocol for PySide6 signal integration without hard-coupling the core engine to Qt.

---

## GUI

Launch the PySide6 interface:

```bash
python run.py
```

The GUI provides interactive pages for image loading, detection parameter tuning, mask editing, tracking execution with live progress, post-processing visualization, and strain analysis — all backed by the same core modules described above.

---

## Relationship to MATLAB SerialTrack

This Python implementation is a direct port of the algorithms from the MATLAB [SerialTrack](https://github.com/FranckLab/SerialTrack) codebase. The table below maps MATLAB source files to their Python equivalents:

| MATLAB Source | Python Module | Description |
|---------------|---------------|-------------|
| `BeadPara`, `MPTPara` structs | `config.py` | Configuration parameters |
| `funReadImage3.m` | `io.py` | Image loading |
| `funPTT.m`, `funPTT_tracTrac.m` | `detection.py` | Particle detection |
| `f_track_serial_match3D.m` | `matching.py` | Topology matching |
| `removeOutlierTPT.m` | `outliers.py` | Outlier removal |
| `funScatter2Grid3D.m`, `regularizeNd.m` | `regularization.py` | Regularization |
| `funCompDefGrad3.m`, `funDerivativeOp3.m` | `fields.py` | Strain computation |
| `funInitGuess3.m`, `funPOR_GPR.m` | `prediction.py` | Motion prediction |
| `run_Serial_MPT_3D_hardpar_*.m` | `tracking.py` | Main tracking loop |
| Trajectory merge sections | `trajectories.py` | Trajectory stitching |

---

## Citation

If you use SerialTrack in your research, please cite the original work:

> Yang, J. and Bhattacharya, K., "Combining Image Compression with Digital Image Correlation," *Experimental Mechanics*, 2019.

> Bar-Kochba, E., Toyjanova, J., Andrews, E., Kim, K., and Franck, C., "A Fast Iterative Digital Volume Correlation Algorithm for Large Deformations," *Experimental Mechanics*, 2015.

---

## License

See [LICENSE](LICENSE) for details.