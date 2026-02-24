#!/usr/bin/env python3
"""
SerialTrack Python — Synthetic Test Example
=============================================

Generates synthetic particle data with **known** deformations and runs the
full SerialTrack ADMM tracking pipeline, then validates tracked displacements
and strains against the analytical ground truth.

Deformation types:  simple shear, uniaxial tension, hydrostatic expansion,
                    combined shear + tension
Particle types:     hard (rigid beads), soft (shape-deforming beads)
Dimensions:         2-D and 3-D

Setup
-----
Your module files (config.py, detection.py, tracking.py, etc.) use relative
imports, so they must live inside a package folder.  This script will
automatically create that folder for you if needed:

  your_project/
  ├── config.py
  ├── detection.py
  ├── tracking.py
  ├── ...                       ← loose module files
  ├── example_synthetic_tests.py   ← this script
  └── serialtrack/              ← auto-created package folder
      ├── __init__.py
      ├── config.py  → (copied)
      └── ...

Just place this file next to your module .py files and run it.

Usage
-----
  python example_synthetic_tests.py                  # run everything
  python example_synthetic_tests.py --quick           # 2-D hard only
  python example_synthetic_tests.py --coords-only     # skip image gen
  python example_synthetic_tests.py --verbose         # ADMM iteration log
  python example_synthetic_tests.py --save-images     # write TIFFs

Requirements
------------
  pip install numpy scipy scikit-image numba scikit-learn
  (tifffile needed only for --save-images)
"""

from __future__ import annotations

import sys, os, shutil, time, argparse, logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════════════
#  0.  AUTO-SETUP: create serialtrack/ package from loose module files
# ═══════════════════════════════════════════════════════════════════════

_MODULE_FILES = [
    "__init__.py", "config.py", "detection.py", "fields.py", "io.py",
    "matching.py", "outliers.py", "prediction.py", "regularization.py",
    "results.py", "tracking.py", "trajectories.py",
]

def _auto_setup_package():
    """If `import serialtrack` fails, build the package folder from loose files."""
    try:
        import serialtrack
        return  # already importable — nothing to do
    except ImportError:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.join(script_dir, "serialtrack")

    # Check if loose module files exist next to this script
    found = [f for f in _MODULE_FILES if os.path.isfile(os.path.join(script_dir, f))]
    if not found:
        print("ERROR: Cannot find module files (config.py, tracking.py, etc.)")
        print(f"       Looked in: {script_dir}")
        print("       Place this script next to your SerialTrack .py files,")
        print("       or install serialtrack as a package.")
        sys.exit(1)

    print(f"Setting up serialtrack/ package from {len(found)} module files...")
    os.makedirs(pkg_dir, exist_ok=True)

    for f in found:
        src = os.path.join(script_dir, f)
        dst = os.path.join(pkg_dir, f)
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copy2(src, dst)

    # Make sure __init__.py exists even if not in the loose files
    init_path = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init_path):
        with open(init_path, "w") as fh:
            fh.write("# auto-generated\n")

    # Add the script directory to sys.path so `import serialtrack` works
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print(f"  -> created {pkg_dir}/")


_auto_setup_package()

import serialtrack as st  # now guaranteed to work


# ═══════════════════════════════════════════════════════════════════════
#  1.  PARTICLE SEEDING
# ═══════════════════════════════════════════════════════════════════════

def poisson_disc_sample(
    domain: np.ndarray,
    min_dist: float,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dart-throwing Poisson-disc-like sampler.

    Returns up to *n_target* points inside [margin, domain-margin]^D
    separated by at least *min_dist*.
    """
    ndim = len(domain)
    margin = max(8.0, min_dist)
    pts = []
    for _ in range(n_target * 80):
        pt = rng.uniform(margin, domain - margin, size=ndim)
        if len(pts) == 0:
            pts.append(pt)
        else:
            arr = np.asarray(pts)
            if np.min(np.linalg.norm(arr - pt, axis=1)) > min_dist:
                pts.append(pt)
        if len(pts) >= n_target:
            break
    return np.asarray(pts, dtype=np.float64)


def seed_beads(
    coords: np.ndarray,
    sigma: np.ndarray,
    image_size: Tuple[int, ...],
    F_local: Optional[np.ndarray] = None,
    noise: float = 0.03,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Render Gaussian beads into a synthetic image (replaces seedBeadsN.m).

    Parameters
    ----------
    coords     : (N, D) bead centres
    sigma      : (D,)   Gaussian std-dev per axis
    image_size : image shape
    F_local    : (N, D, D) deformation gradient per bead.
                 None -> hard (isotropic) beads.
                 Provided -> soft beads (Gaussian stretched by F).
    noise      : additive Gaussian noise level
    rng        : random generator
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ndim = len(image_size)
    img = np.zeros(image_size, dtype=np.float64)
    hw = (np.ceil(sigma) * 5).astype(int)

    for i, c in enumerate(coords):
        ci = np.round(c).astype(int)

        # Skip beads entirely outside the image
        if np.any(ci < -hw) or np.any(ci >= np.array(image_size) + hw):
            continue

        slices, ranges = [], []
        skip = False
        for d in range(ndim):
            lo = max(ci[d] - hw[d], 0)
            hi = min(ci[d] + hw[d] + 1, image_size[d])
            if lo >= hi:
                skip = True
                break
            slices.append(slice(lo, hi))
            ranges.append(np.arange(lo, hi, dtype=np.float64))
        if skip:
            continue
        grids = np.meshgrid(*ranges, indexing="ij")
        dx = np.stack([g - c[d] for d, g in enumerate(grids)], axis=-1)

        # Soft beads: map deformed-config coords back through F^{-1}
        if F_local is not None:
            try:
                Finv = np.linalg.inv(F_local[i])
            except np.linalg.LinAlgError:
                Finv = np.eye(ndim)
            dx = np.einsum("ij,...j->...i", Finv, dx)

        exponent = sum(-(dx[..., d] / (2 * sigma[d])) ** 2 for d in range(ndim))
        img[tuple(slices)] += np.exp(exponent)

    img = np.clip(img, 0, 1)
    img += noise * rng.standard_normal(image_size)
    return np.clip(img, 0, 1)


# ═══════════════════════════════════════════════════════════════════════
#  2.  DEFORMATION MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeformationCase:
    """One row of the synthetic test matrix.

    The deformation is applied homogeneously about the domain centre:

        u(x) = (F - I) * (x - x_centre)

    where F = F_hydro * F_tension * F_shear:

        F_shear   : identity + gamma in the (0,1) slot
        F_tension : stretch lambda_x along axis 0
        F_hydro   : uniform scaling lambda_h on all axes

    You can combine any of the three by setting the corresponding
    parameters to non-trivial values.
    """
    name: str
    ndim: int
    particle_type: str            # "hard" | "soft"
    n_particles: int = 300
    domain_size: float = 512.0    # px per side
    n_steps: int = 3              # load increments
    bead_sigma: float = 3.0       # Gaussian radius [px]

    # deformation parameters (applied about domain centre)
    shear_gamma: float = 0.0     # F_01 = gamma  (simple shear in xy)
    stretch_lambda: float = 1.0  # F_00 = lambda (uniaxial tension in x)
    hydro_lambda: float = 1.0    # F = lambda*I  (hydrostatic expansion)

    def F_at(self, frac: float) -> np.ndarray:
        """Deformation gradient at load fraction *frac* in [0, 1]."""
        D = self.ndim
        F = np.eye(D)
        F[0, 1] = frac * self.shear_gamma
        F[0, 0] *= 1.0 + frac * (self.stretch_lambda - 1.0)
        h = 1.0 + frac * (self.hydro_lambda - 1.0)
        F *= h
        return F

    def displace(
        self, x0: np.ndarray, frac: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply deformation to reference coords *x0*.
        Returns (x_def, u, F).
        """
        D = self.ndim
        F = self.F_at(frac)
        centre = np.full(D, self.domain_size / 2.0)
        u = (x0 - centre) @ (F - np.eye(D)).T
        return x0 + u, u, F

    def tag(self) -> str:
        parts = []
        if self.shear_gamma:
            parts.append(f"gamma={self.shear_gamma}")
        if self.stretch_lambda != 1:
            parts.append(f"lam_x={self.stretch_lambda}")
        if self.hydro_lambda != 1:
            parts.append(f"lam_h={self.hydro_lambda}")
        return ", ".join(parts) or "identity"


# ═══════════════════════════════════════════════════════════════════════
#  3.  TEST MATRIX
# ═══════════════════════════════════════════════════════════════════════

def build_test_matrix(quick: bool = False) -> List[DeformationCase]:
    """Construct the full matrix of test cases.

    Covers every combination of:
        dimension       : 2-D, 3-D
        particle type   : hard, soft
        deformation     : shear, tension, hydrostatic, combined

    With --quick only 2-D hard-particle cases are returned.
    """
    cases: List[DeformationCase] = []

    ptypes_2d = ["hard"] if quick else ["hard", "soft"]
    ptypes_3d = [] if quick else ["hard", "soft"]

    # ---- 2-D cases (512x512, sigma=3, ~300 well-separated particles) ----
    for pt in ptypes_2d:
        cases += [
            DeformationCase(
                name=f"2D-{pt}-shear",
                ndim=2, particle_type=pt,
                n_particles=300, domain_size=512.0, n_steps=3,
                bead_sigma=3.0, shear_gamma=0.15,
            ),
            DeformationCase(
                name=f"2D-{pt}-tension",
                ndim=2, particle_type=pt,
                n_particles=300, domain_size=512.0, n_steps=3,
                bead_sigma=3.0, stretch_lambda=1.20,
            ),
            DeformationCase(
                name=f"2D-{pt}-hydrostatic",
                ndim=2, particle_type=pt,
                n_particles=300, domain_size=512.0, n_steps=3,
                bead_sigma=3.0, hydro_lambda=1.10,
            ),
            DeformationCase(
                name=f"2D-{pt}-combined",
                ndim=2, particle_type=pt,
                n_particles=300, domain_size=512.0, n_steps=3,
                bead_sigma=3.0, shear_gamma=0.10, stretch_lambda=1.15,
            ),
        ]

    # ---- 3-D cases (100^3, sigma=1.5, ~400 particles) ----
    for pt in ptypes_3d:
        cases += [
            DeformationCase(
                name=f"3D-{pt}-shear",
                ndim=3, particle_type=pt,
                n_particles=400, domain_size=100.0, n_steps=2,
                bead_sigma=1.5, shear_gamma=0.12,
            ),
            DeformationCase(
                name=f"3D-{pt}-tension",
                ndim=3, particle_type=pt,
                n_particles=400, domain_size=100.0, n_steps=2,
                bead_sigma=1.5, stretch_lambda=1.15,
            ),
            DeformationCase(
                name=f"3D-{pt}-hydrostatic",
                ndim=3, particle_type=pt,
                n_particles=400, domain_size=100.0, n_steps=2,
                bead_sigma=1.5, hydro_lambda=1.08,
            ),
            DeformationCase(
                name=f"3D-{pt}-combined",
                ndim=3, particle_type=pt,
                n_particles=400, domain_size=100.0, n_steps=2,
                bead_sigma=1.5, shear_gamma=0.08, stretch_lambda=1.12,
            ),
        ]

    return cases


# ═══════════════════════════════════════════════════════════════════════
#  4.  DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def generate_images(
    case: DeformationCase,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Generate a synthetic image sequence for one test case.

    Returns
    -------
    images     : list of (n_steps + 1) images  [ref, step1, step2, ...]
    coords_ref : (N, D) ground-truth reference positions
    true_disps : list of (N, D) ground-truth displacements per step
    true_Fs    : list of (D, D) deformation gradients per step
    """
    D = case.ndim
    domain = np.full(D, case.domain_size)
    sigma = np.full(D, case.bead_sigma)
    imsize = tuple(int(case.domain_size) for _ in range(D))

    # Minimum inter-particle distance: 6*sigma ensures non-overlapping beads
    # for reliable detection.  (MATLAB uses ~5*sigma for 3D volumes.)
    min_dist = max(6.0 * case.bead_sigma, 8.0)
    x0 = poisson_disc_sample(domain, min_dist, case.n_particles, rng)

    # Reference image - always isotropic beads
    images = [seed_beads(x0, sigma, imsize, rng=rng)]
    true_u: List[np.ndarray] = []
    true_F: List[np.ndarray] = []

    for step in range(1, case.n_steps + 1):
        frac = step / case.n_steps
        xd, u, F = case.displace(x0, frac)

        # Soft particles: each bead's shape is deformed by F
        F_local = np.tile(F, (len(xd), 1, 1)) if case.particle_type == "soft" else None

        images.append(seed_beads(xd, sigma, imsize, F_local=F_local, rng=rng))
        true_u.append(u)
        true_F.append(F)

    return images, x0, true_u, true_F


def generate_coords(
    case: DeformationCase,
    rng: np.random.Generator,
    loc_noise: float = 0.05,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Generate coordinate-only sequence (no images - much faster).

    Simulates "perfect detection" with small localisation noise.
    """
    D = case.ndim
    domain = np.full(D, case.domain_size)
    x0 = poisson_disc_sample(domain, max(6.0 * case.bead_sigma, 8.0), case.n_particles, rng)

    frames = [x0 + rng.normal(0, loc_noise, x0.shape)]
    true_u: List[np.ndarray] = []
    true_F: List[np.ndarray] = []

    for step in range(1, case.n_steps + 1):
        frac = step / case.n_steps
        xd, u, F = case.displace(x0, frac)
        frames.append(xd + rng.normal(0, loc_noise, xd.shape))
        true_u.append(u)
        true_F.append(F)

    return frames, x0, true_u, true_F


# ═══════════════════════════════════════════════════════════════════════
#  5.  BUILD SerialTrack CONFIGS
# ═══════════════════════════════════════════════════════════════════════

def make_configs(case: DeformationCase):
    """Create detection + tracking configs tuned for the synthetic data."""
    det = st.DetectionConfig(
        method=st.DetectionMethod.TRACTRAC,
        threshold=0.15,
        bead_radius=case.bead_sigma,
        min_size=max(2, int(case.bead_sigma)),
        max_size=2000,
    )

    fos = case.domain_size * 0.35

    trk = st.TrackingConfig(
        mode=st.TrackingMode.CUMULATIVE,
        solver=st.GlobalSolver.ADMM,
        f_o_s=fos,
        n_neighbors_max=25,
        n_neighbors_min=1,
        smoothness=1e-2,
        outlier_threshold=5.0,
        max_iter=20,
        iter_stop_threshold=1e-2,
        strain_n_neighbors=15,
        strain_f_o_s=fos,
        dist_missing=5.0,
    )

    # For 3-D coordinate-only tracking we must tell the config it is 3-D
    # (normally inferred from image shape in track_images).
    if case.ndim == 3:
        s = int(case.domain_size)
        trk.roi_x = (0, s)
        trk.roi_y = (0, s)
        trk.roi_z = (0, s)

    return det, trk


# ═══════════════════════════════════════════════════════════════════════
#  6.  VALIDATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    """Stores validation metrics for one test case."""
    name: str
    n_ref: int
    n_detected: int
    n_tracked: int
    ratio: float
    rms_err: float      # displacement RMS error [px]
    max_err: float      # displacement max error [px]
    F_err: float        # ||F_estimated - F_true||  (Frobenius)
    wall: float         # wall-clock time [s]
    ok: bool            # pass / fail

    def line(self) -> str:
        tag = "PASS" if self.ok else "FAIL"
        return (f"  [{tag}] {self.name:<28s}  "
                f"det={self.n_detected:<4d} "
                f"ratio={self.ratio:.3f}  "
                f"RMS={self.rms_err:.4f}px  "
                f"max={self.max_err:.4f}px  "
                f"|dF|={self.F_err:.5f}  "
                f"t={self.wall:.1f}s")


def _match_detected_to_gt(
    detected: np.ndarray,
    gt: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Match detected particles to ground-truth via nearest-neighbour.

    Returns gt_idx: array of length len(detected) where
        gt_idx[i] = index into gt of the closest ground-truth particle,
                    or -1 if no ground-truth particle is within max_dist.
    """
    if len(detected) == 0 or len(gt) == 0:
        return np.full(len(detected), -1, dtype=np.int64)

    tree = cKDTree(gt)
    dists, idxs = tree.query(detected, k=1)
    gt_idx = np.where(dists < max_dist, idxs, -1).astype(np.int64)
    return gt_idx


def validate(
    session: st.TrackingSession,
    x0: np.ndarray,
    true_u_list: List[np.ndarray],
    true_F_list: List[np.ndarray],
    case: DeformationCase,
    wall: float,
    is_image_based: bool = False,
) -> Result:
    """Compare tracked output against analytical ground truth.

    When is_image_based=True, the detected coordinates don't share
    indices with ground truth, so we first match session.coords_ref
    to x0 via nearest-neighbour, then propagate through the tracking
    correspondences.
    """
    D = case.ndim
    n_detected = len(session.coords_ref)

    # ---- Build mapping: detected-ref index -> ground-truth index ----
    if is_image_based:
        max_match_dist = 3.0 * case.bead_sigma
        gt_for_det = _match_detected_to_gt(session.coords_ref, x0, max_match_dist)
    else:
        # Coords-only: indices align directly (same array passed in)
        gt_for_det = np.arange(len(session.coords_ref), dtype=np.int64)

    errs: List[float] = []
    total_tracked = 0

    for res, true_u in zip(session.frame_results, true_u_list):
        good = res.track_a2b >= 0
        ia = np.where(good)[0]     # indices into session.coords_ref
        ib = res.track_a2b[ia]     # matched indices in deformed frame
        if len(ia) == 0:
            continue

        # Which of these detected-ref particles matched a ground-truth?
        gt_ia = gt_for_det[ia]       # ground-truth index for each tracked particle
        has_gt = gt_ia >= 0
        ia_valid = ia[has_gt]
        ib_valid = ib[has_gt]
        gt_valid = gt_ia[has_gt]

        total_tracked += len(ia_valid)

        if len(ia_valid) == 0:
            continue

        # Tracked displacement = deformed_detected - reference_detected
        tracked_u = res.coords_b[ib_valid] - session.coords_ref[ia_valid]
        # Ground-truth displacement for those same particles
        gt_u = true_u[gt_valid]

        err = np.linalg.norm(tracked_u - gt_u, axis=1)
        errs.extend(err.tolist())

    errs_arr = np.asarray(errs) if errs else np.array([0.0])
    rms = float(np.sqrt(np.mean(errs_arr ** 2)))
    mx  = float(np.max(errs_arr))

    # ---- Strain error via MLS on last frame ----
    F_err = 0.0
    last   = session.frame_results[-1]
    F_true = true_F_list[-1]
    good = last.track_a2b >= 0
    ia = np.where(good)[0]
    ib = last.track_a2b[ia]
    gt_ia = gt_for_det[ia]
    has_gt = gt_ia >= 0
    ia_v = ia[has_gt]
    ib_v = ib[has_gt]
    gt_v = gt_ia[has_gt]

    if len(ia_v) >= D + 2:
        disp_tr = last.coords_b[ib_v] - session.coords_ref[ia_v]
        ref_coords = session.coords_ref[ia_v]
        try:
            _, F_mls, valid = st.compute_strain_mls(
                disp_tr, ref_coords,
                f_o_s=case.domain_size * 0.5,
                n_neighbors=min(20, len(ia_v) - 1),
            )
            if np.any(valid):
                F_est = np.eye(D) + np.nanmean(F_mls[valid], axis=0)
                F_err = float(np.linalg.norm(F_est - F_true))
        except Exception:
            pass

    ratio = float(session.tracking_ratios.mean()) if len(session.tracking_ratios) else 0.0
    ok = (ratio > 0.70) and (rms < 2.0)

    return Result(
        name=case.name, n_ref=len(x0), n_detected=n_detected,
        n_tracked=total_tracked,
        ratio=ratio, rms_err=rms, max_err=mx, F_err=F_err, wall=wall, ok=ok,
    )


# ═══════════════════════════════════════════════════════════════════════
#  7.  RUNNERS
# ═══════════════════════════════════════════════════════════════════════

def run_coords(case: DeformationCase, rng: np.random.Generator) -> Result:
    """Coordinate-only test (fast - no image rendering / detection)."""
    t0 = time.perf_counter()
    frames, x0, true_u, true_F = generate_coords(case, rng)

    det, trk = make_configs(case)
    tracker = st.SerialTracker(det, trk)
    session = tracker.track_coordinates(frames)

    return validate(session, x0, true_u, true_F, case,
                    time.perf_counter() - t0, is_image_based=False)


def run_images(case: DeformationCase, rng: np.random.Generator,
               save: bool = False) -> Result:
    """Full image-based test (detection -> tracking -> validation)."""
    t0 = time.perf_counter()
    images, x0, true_u, true_F = generate_images(case, rng)

    if save:
        _save_tiffs(images, case)

    det, trk = make_configs(case)
    tracker = st.SerialTracker(det, trk)
    session = tracker.track_images(images)

    return validate(session, x0, true_u, true_F, case,
                    time.perf_counter() - t0, is_image_based=True)


def _save_tiffs(images: List[np.ndarray], case: DeformationCase):
    """Optionally write synthetic images to disk as TIFF stacks."""
    try:
        import tifffile
    except ImportError:
        print("  (tifffile not installed -- skipping TIFF save)")
        return
    outdir = os.path.join("synthetic_data", case.name)
    os.makedirs(outdir, exist_ok=True)
    for i, img in enumerate(images):
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        tifffile.imwrite(os.path.join(outdir, f"frame_{i:04d}.tif"), arr)
    print(f"  -> saved {len(images)} frames to {outdir}/")


# ═══════════════════════════════════════════════════════════════════════
#  8.  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SerialTrack Python -- Synthetic Validation Suite"
    )
    ap.add_argument("--quick",       action="store_true",
                    help="Run only 2-D hard-particle cases (fastest)")
    ap.add_argument("--coords-only", action="store_true",
                    help="Skip image generation; test tracking engine only")
    ap.add_argument("--save-images", action="store_true",
                    help="Write synthetic images to ./synthetic_data/ as TIFFs")
    ap.add_argument("--verbose",     action="store_true",
                    help="Print ADMM iteration-level logs")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s | %(message)s",
    )

    cases = build_test_matrix(quick=args.quick)
    rng = np.random.default_rng(args.seed)
    mode = "coordinates" if args.coords_only else "images"

    hdr = "SerialTrack Python -- Synthetic Validation"
    print(f"\n{'=' * 72}\n  {hdr}\n{'=' * 72}")
    print(f"  mode={mode}  cases={len(cases)}  seed={args.seed}")
    print(f"{'=' * 72}\n")

    results: List[Result] = []

    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {c.name}  "
              f"({c.ndim}D, {c.particle_type}, N~{c.n_particles}, "
              f"domain={int(c.domain_size)}, sigma={c.bead_sigma})")
        print(f"        deformation: {c.tag()}")

        F = c.F_at(1.0)
        np.set_printoptions(precision=4, suppress=True)
        Fstr = np.array2string(F, prefix=" " * 16)
        print(f"        F(final) =\n                {Fstr}")

        try:
            if args.coords_only:
                r = run_coords(c, rng)
            else:
                r = run_images(c, rng, save=args.save_images)
            results.append(r)
            print(r.line())
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append(
                Result(c.name, c.n_particles, 0, 0, 0.0,
                       999.0, 999.0, 999.0, 0.0, False)
            )
            print(f"  [ERROR] {exc}")
        print()

    # ---- summary table ----
    print(f"{'=' * 72}\n  SUMMARY\n{'=' * 72}")
    for r in results:
        print(r.line())

    ok  = sum(r.ok for r in results)
    tot = len(results)
    ttot = sum(r.wall for r in results)
    print(f"\n  {ok}/{tot} passed   total time: {ttot:.1f}s")

    if ok < tot:
        print("\n  Failed:")
        for r in results:
            if not r.ok:
                print(f"    - {r.name}: ratio={r.ratio:.3f}, RMS={r.rms_err:.4f}")

    print(f"{'=' * 72}\n")
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())