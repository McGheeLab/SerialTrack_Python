#!/usr/bin/env python3
"""
Hydrogel Bead Phantom Generator + SerialTrack Analysis + Error Reporting
=========================================================================

Builds a synthetic 3-D volume representing a hydrogel with embedded
fluorescent beads, applies a user-specified deformation history, saves
the image sequence as TIFF stacks, runs SerialTrack particle tracking,
and generates error-analysis plots comparing tracked vs ground truth.

Everything is controlled by a single JSON configuration file.

Usage
-----
    python hydrogel_phantom.py

A file-picker dialogue will open — browse to your JSON config file.
You can also pass the path directly:

    python hydrogel_phantom.py  config.json

The JSON file specifies:
    volume      — domain size, voxel pitch, bead density & radius, noise
    deformation — modes + magnitudes, loading time, capture rate
    workflow    — output path, which steps to run, seed, verbosity

See the example configs shipped alongside this script, or call:
    python hydrogel_phantom.py --write-examples
to generate them in the current directory.

Requirements
------------
    pip install numpy scipy scikit-image numba scikit-learn tifffile matplotlib
"""

from __future__ import annotations

import sys, os, shutil, json, time, argparse, logging, copy
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("hydrogel_phantom")


# ═══════════════════════════════════════════════════════════════════════
#  0.  AUTO-SETUP — create serialtrack/ package from loose module files
# ═══════════════════════════════════════════════════════════════════════

_MODULE_FILES = [
    "__init__.py", "config.py", "detection.py", "fields.py", "io.py",
    "matching.py", "outliers.py", "prediction.py", "regularization.py",
    "results.py", "tracking.py", "trajectories.py",
]


def _auto_setup_package():
    try:
        import serialtrack          # noqa: F401
        return
    except ImportError:
        pass
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.join(script_dir, "serialtrack")
    found = [f for f in _MODULE_FILES
             if os.path.isfile(os.path.join(script_dir, f))]
    if not found:
        print("ERROR: Cannot find SerialTrack module files "
              "(config.py, tracking.py, ...)")
        print(f"       Looked in: {script_dir}")
        print("       Place this script next to your .py module files.")
        sys.exit(1)
    print(f"Setting up serialtrack/ package from {len(found)} module files ...")
    os.makedirs(pkg_dir, exist_ok=True)
    for f in found:
        src = os.path.join(script_dir, f)
        dst = os.path.join(pkg_dir, f)
        if (not os.path.exists(dst)
                or os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copy2(src, dst)
    init_path = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init_path):
        with open(init_path, "w") as fh:
            fh.write("# auto-generated\n")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    print(f"  -> created {pkg_dir}/")


_auto_setup_package()
import serialtrack as st            # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  1.  CONFIGURATION DATA-CLASSES
# ═══════════════════════════════════════════════════════════════════════

DEFORMATION_MODES = [
    "shear_xy", "shear_xz", "shear_yz",
    "tension_x", "tension_y", "tension_z",
    "hydrostatic",
]


@dataclass
class VolumeConfig:
    """Physical and imaging parameters for the hydrogel phantom."""
    size_x: int = 128
    size_y: int = 128
    size_z: int = 64
    voxel_dx: float = 0.5
    voxel_dy: float = 0.5
    voxel_dz: float = 1.0
    bead_density: float = 0.003
    bead_radius: float = 2.0
    bead_peak_intensity: float = 1.0
    background_noise: float = 0.02
    poisson_noise: bool = False
    psf_sigma: Optional[Tuple[float, float, float]] = None
    bit_depth: int = 16

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.size_x, self.size_y, self.size_z)

    @property
    def n_voxels(self) -> int:
        return self.size_x * self.size_y * self.size_z

    @property
    def n_beads_target(self) -> int:
        return int(self.bead_density * self.n_voxels)

    @property
    def voxel_pitch(self) -> np.ndarray:
        return np.array([self.voxel_dx, self.voxel_dy, self.voxel_dz])


@dataclass
class DeformationConfig:
    """Deformation prescription.

    Shear modes: magnitude is gamma (dimensionless shear strain).
    Tension / hydrostatic modes: magnitude is the stretch ratio lambda.

    The deformation ramps linearly from identity (t=0) to full load
    (t = total_time).  Frames are captured every *capture_interval*.
    """
    modes: Dict[str, float] = field(default_factory=dict)
    total_time: float = 10.0
    capture_interval: float = 2.0

    @property
    def frame_times(self) -> np.ndarray:
        n = int(round(self.total_time / self.capture_interval)) + 1
        return np.linspace(0.0, self.total_time, n)

    @property
    def n_frames(self) -> int:
        return len(self.frame_times)

    @property
    def load_fractions(self) -> np.ndarray:
        t = self.frame_times
        return t / self.total_time if self.total_time > 0 else np.array([0.0])


@dataclass
class WorkflowConfig:
    """Which steps to run and where to write output."""
    output: str = "./phantom_output"
    generate: bool = True
    analyse: bool = True
    plot: bool = True
    seed: int = 42
    verbose: bool = False


# ── JSON <-> dataclass helpers ────────────────────────────────────────

def _vol_from_dict(d: dict) -> VolumeConfig:
    sz = d.get("volume_size", [128, 128, 64])
    vs = d.get("voxel_size", [0.5, 0.5, 1.0])
    psf = d.get("psf_sigma", None)
    return VolumeConfig(
        size_x=sz[0], size_y=sz[1], size_z=sz[2],
        voxel_dx=vs[0], voxel_dy=vs[1], voxel_dz=vs[2],
        bead_density=d.get("bead_density", 0.003),
        bead_radius=d.get("bead_radius", 2.0),
        bead_peak_intensity=d.get("bead_peak_intensity", 1.0),
        background_noise=d.get("background_noise", 0.02),
        poisson_noise=d.get("poisson_noise", False),
        psf_sigma=tuple(psf) if psf else None,
        bit_depth=d.get("bit_depth", 16),
    )


def _def_from_dict(d: dict) -> DeformationConfig:
    modes = d.get("modes", {"shear_xy": 0.15})
    for k in modes:
        if k not in DEFORMATION_MODES:
            raise ValueError(
                f"Unknown deformation mode '{k}'.  "
                f"Valid modes: {DEFORMATION_MODES}"
            )
    return DeformationConfig(
        modes=modes,
        total_time=d.get("total_time", 10.0),
        capture_interval=d.get("capture_interval", 2.0),
    )


def _wf_from_dict(d: dict) -> WorkflowConfig:
    return WorkflowConfig(
        output=d.get("output", "./phantom_output"),
        generate=d.get("generate", True),
        analyse=d.get("analyse", True),
        plot=d.get("plot", True),
        seed=d.get("seed", 42),
        verbose=d.get("verbose", False),
    )


def load_config(path: str):
    """Load a JSON config and return (VolumeConfig, DeformConfig, WorkflowConfig)."""
    with open(path) as f:
        raw = json.load(f)
    return (
        _vol_from_dict(raw.get("volume", {})),
        _def_from_dict(raw.get("deformation", {})),
        _wf_from_dict(raw.get("workflow", {})),
    )


# ═══════════════════════════════════════════════════════════════════════
#  2.  EXAMPLE JSON CONFIGS
# ═══════════════════════════════════════════════════════════════════════

EXAMPLE_CONFIGS = {
    # ── simple shear ──────────────────────────────────────────────
    "config_shear.json": {
        "_comment": "Simple shear in the XY plane, 6 frames over 10 s",
        "volume": {
            "volume_size": [128, 128, 128],
            "voxel_size": [0.5, 0.5, 1.0],
            "bead_density": 0.002,
            "bead_radius": 1.5,
            "background_noise": 0.02,
            "poisson_noise": False,
            "psf_sigma": None,
            "bit_depth": 16
        },
        "deformation": {
            "modes": {
                "shear_xy": 0.15
            },
            "total_time": 10.0,
            "capture_interval": 2.0
        },
        "workflow": {
            "output": "./run_shear",
            "generate": True,
            "analyse": True,
            "plot": True,
            "seed": 42,
            "verbose": False
        }
    },
    # ── combined shear + tension ──────────────────────────────────
    "config_combined.json": {
        "_comment": "Combined shear + uniaxial tension + hydrostatic",
        "volume": {
            "volume_size": [100, 100, 80],
            "voxel_size": [0.5, 0.5, 1.0],
            "bead_density": 0.003,
            "bead_radius": 1.5,
            "background_noise": 0.02,
            "poisson_noise": False,
            "psf_sigma": None,
            "bit_depth": 16
        },
        "deformation": {
            "modes": {
                "shear_xy": 0.10,
                "tension_z": 1.15,
                "hydrostatic": 1.05
            },
            "total_time": 8.0,
            "capture_interval": 2.0
        },
        "workflow": {
            "output": "./run_combined",
            "generate": True,
            "analyse": True,
            "plot": True,
            "seed": 42,
            "verbose": False
        }
    },
    # ── hydrostatic with realistic noise ──────────────────────────
    "config_hydrostatic_noisy.json": {
        "_comment": "Hydrostatic expansion with PSF blur and Poisson noise",
        "volume": {
            "volume_size": [128, 128, 64],
            "voxel_size": [0.5, 0.5, 1.0],
            "bead_density": 0.002,
            "bead_radius": 2.5,
            "background_noise": 0.03,
            "poisson_noise": True,
            "psf_sigma": [0.5, 0.5, 1.0],
            "bit_depth": 16
        },
        "deformation": {
            "modes": {
                "hydrostatic": 1.10
            },
            "total_time": 6.0,
            "capture_interval": 3.0
        },
        "workflow": {
            "output": "./run_hydro_noisy",
            "generate": True,
            "analyse": True,
            "plot": True,
            "seed": 123,
            "verbose": False
        }
    },
    # ── analyse-only (re-run analysis on existing data) ───────────
    "config_analyse_only.json": {
        "_comment": [
            "Skip generation, just analyse + plot existing data.",
            "Point 'output' at a folder that already contains",
            "frame_NNNN/ folders and ground_truth.npz."
        ],
        "volume": {
            "volume_size": [128, 128, 128],
            "voxel_size": [0.5, 0.5, 1.0],
            "bead_density": 0.002,
            "bead_radius": 1.5,
            "background_noise": 0.02
        },
        "deformation": {
            "modes": {"shear_xy": 0.15},
            "total_time": 10.0,
            "capture_interval": 2.0
        },
        "workflow": {
            "output": "./run_shear",
            "generate": False,
            "analyse": True,
            "plot": True,
            "seed": 42,
            "verbose": False
        }
    },
}


def write_example_configs(directory: str = "."):
    """Dump all example JSON configs to *directory*."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, cfg in EXAMPLE_CONFIGS.items():
        p = d / name
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  wrote {p}")


# ═══════════════════════════════════════════════════════════════════════
#  3.  DEFORMATION GRADIENT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_deformation_gradient(
    modes: Dict[str, float],
    frac: float,
) -> np.ndarray:
    """Construct 3x3 F from mode superposition at load fraction *frac*.

    Composition order: shears (additive) -> stretches (row-wise
    multiplicative) -> hydrostatic (uniform scale).
    """
    F = np.eye(3)
    if "shear_xy" in modes:
        F[0, 1] += frac * modes["shear_xy"]
    if "shear_xz" in modes:
        F[0, 2] += frac * modes["shear_xz"]
    if "shear_yz" in modes:
        F[1, 2] += frac * modes["shear_yz"]
    if "tension_x" in modes:
        F[0, :] *= 1.0 + frac * (modes["tension_x"] - 1.0)
    if "tension_y" in modes:
        F[1, :] *= 1.0 + frac * (modes["tension_y"] - 1.0)
    if "tension_z" in modes:
        F[2, :] *= 1.0 + frac * (modes["tension_z"] - 1.0)
    if "hydrostatic" in modes:
        F *= 1.0 + frac * (modes["hydrostatic"] - 1.0)
    return F


def apply_deformation(
    x0: np.ndarray,
    F: np.ndarray,
    centre: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map reference coordinates through F about centre.
    Returns (x_deformed, displacement).
    """
    u = (x0 - centre) @ (F - np.eye(3)).T
    return x0 + u, u


# ═══════════════════════════════════════════════════════════════════════
#  4.  VOLUME GENERATION
# ═══════════════════════════════════════════════════════════════════════

def poisson_disc_3d(
    domain: np.ndarray,
    min_dist: float,
    n_target: int,
    rng: np.random.Generator,
    margin: float = 8.0,
) -> np.ndarray:
    """Dart-throwing quasi-uniform sampler in 3-D, KDTree-accelerated."""
    pts: List[np.ndarray] = []
    lo = np.full(3, margin)
    hi = domain - margin
    max_attempts = max(n_target * 200, 100_000)
    tree = None
    rebuild_every = 50  # rebuild KDTree periodically for O(N log N) checks

    for attempt in range(max_attempts):
        pt = rng.uniform(lo, hi)
        if len(pts) == 0:
            pts.append(pt)
        else:
            if tree is None or len(pts) % rebuild_every == 0:
                tree = cKDTree(np.asarray(pts))
            d, _ = tree.query(pt, k=1)
            if d > min_dist:
                pts.append(pt)
                tree = None  # invalidate, will rebuild next check
        if len(pts) >= n_target:
            break

    return np.asarray(pts, dtype=np.float64)


def render_volume(
    coords: np.ndarray,
    sigma: np.ndarray,
    vol_shape: Tuple[int, int, int],
    F_per_bead: Optional[np.ndarray] = None,
    peak: float = 1.0,
    noise_std: float = 0.02,
    poisson: bool = False,
    psf_sigma: Optional[Tuple[float, ...]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Render Gaussian beads into a 3-D volume."""
    if rng is None:
        rng = np.random.default_rng(0)
    vol = np.zeros(vol_shape, dtype=np.float64)
    hw = (np.ceil(sigma) * 4).astype(int)

    for i, c in enumerate(coords):
        ci = np.round(c).astype(int)
        if np.any(ci < -hw) or np.any(ci >= np.array(vol_shape) + hw):
            continue
        slices, ranges = [], []
        skip = False
        for d in range(3):
            lo = max(ci[d] - hw[d], 0)
            hi = min(ci[d] + hw[d] + 1, vol_shape[d])
            if lo >= hi:
                skip = True
                break
            slices.append(slice(lo, hi))
            ranges.append(np.arange(lo, hi, dtype=np.float64))
        if skip:
            continue

        grids = np.meshgrid(*ranges, indexing="ij")
        dx = np.stack([g - c[d] for d, g in enumerate(grids)], axis=-1)

        if F_per_bead is not None:
            try:
                Finv = np.linalg.inv(F_per_bead[i])
            except np.linalg.LinAlgError:
                Finv = np.eye(3)
            dx = np.einsum("ij,...j->...i", Finv, dx)

        exponent = (-(dx[..., 0] / (2 * sigma[0])) ** 2
                    - (dx[..., 1] / (2 * sigma[1])) ** 2
                    - (dx[..., 2] / (2 * sigma[2])) ** 2)
        vol[tuple(slices)] += peak * np.exp(exponent)

    if psf_sigma is not None:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=psf_sigma)

    vmax = vol.max()
    if vmax > 0:
        vol /= vmax

    if poisson and vol.max() > 0:
        photon_scale = 1000.0
        vol = rng.poisson(vol * photon_scale).astype(np.float64) / photon_scale

    vol += noise_std * rng.standard_normal(vol_shape)
    return np.clip(vol, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
#  5.  TIFF I/O (compatible with serialtrack.io.ImageLoader)
# ═══════════════════════════════════════════════════════════════════════

def save_volume_tiff(
    vol: np.ndarray,
    outdir: Path,
    frame_idx: int,
    bit_depth: int = 16,
) -> Path:
    """Save a 3-D volume as per-z-slice TIFFs.

    Layout:  outdir / frame_NNNN / slice_NNNN.tif
    Convention follows ImageLoader.load_3d_tiff_stack().
    """
    import tifffile
    frame_dir = outdir / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    maxval = (2 ** bit_depth) - 1
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    for z in range(vol.shape[2]):
        slc = (vol[:, :, z].T * maxval).astype(dtype)
        tifffile.imwrite(str(frame_dir / f"slice_{z:04d}.tif"), slc)
    return frame_dir


def load_volume_tiff(frame_dir: Path) -> np.ndarray:
    return st.ImageLoader.load_3d_tiff_stack(str(frame_dir))


# ═══════════════════════════════════════════════════════════════════════
#  6.  GROUND-TRUTH CONTAINER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GroundTruth:
    vol_config: dict
    deform_config: dict
    coords_ref: np.ndarray
    frame_times: np.ndarray
    coords_per_frame: List[np.ndarray]
    displacements: List[np.ndarray]
    F_per_frame: List[np.ndarray]
    centre: np.ndarray

    def save(self, path: Path):
        np.savez_compressed(
            str(path),
            coords_ref=self.coords_ref,
            frame_times=self.frame_times,
            centre=self.centre,
            **{f"coords_{i}": c for i, c in enumerate(self.coords_per_frame)},
            **{f"disp_{i}": u for i, u in enumerate(self.displacements)},
            **{f"F_{i}": F for i, F in enumerate(self.F_per_frame)},
        )
        meta = {"volume": self.vol_config,
                "deformation": self.deform_config,
                "n_beads": len(self.coords_ref),
                "n_frames": len(self.frame_times)}
        with open(str(path).replace(".npz", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def load(path: Path) -> "GroundTruth":
        data = np.load(str(path))
        coords_ref = data["coords_ref"]
        frame_times = data["frame_times"]
        centre = data["centre"]
        n_frames = len(frame_times)
        coords = [data[f"coords_{i}"] for i in range(n_frames)]
        disps = [data[f"disp_{i}"] for i in range(n_frames - 1)]
        Fs = [data[f"F_{i}"] for i in range(n_frames - 1)]
        meta_path = str(path).replace(".npz", "_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return GroundTruth(
            vol_config=meta.get("volume", {}),
            deform_config=meta.get("deformation", {}),
            coords_ref=coords_ref, frame_times=frame_times,
            coords_per_frame=coords, displacements=disps,
            F_per_frame=Fs, centre=centre,
        )


# ═══════════════════════════════════════════════════════════════════════
#  7.  PHANTOM GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def generate_phantom(
    vol_cfg: VolumeConfig,
    def_cfg: DeformationConfig,
    outdir: Path,
    seed: int = 42,
) -> GroundTruth:
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    domain = np.array(vol_cfg.shape, dtype=np.float64)
    sigma = np.full(3, vol_cfg.bead_radius)
    centre = domain / 2.0
    margin = max(8.0, 2.0 * vol_cfg.bead_radius)
    n_target = vol_cfg.n_beads_target

    # Choose inter-bead spacing: start conservative (6*sigma), but
    # relax toward 3.5*sigma if the target count can't be reached.
    eff_domain = domain - 2 * margin
    for mult in [6.0, 5.0, 4.0, 3.5]:
        min_dist = max(mult * vol_cfg.bead_radius, 6.0)
        max_beads = int(np.prod(eff_domain / min_dist))
        if max_beads >= n_target:
            break

    if max_beads < n_target:
        print(f"  WARNING: Volume too small for {n_target} beads at "
              f"sigma={vol_cfg.bead_radius:.1f}.")
        print(f"           Max ~{max_beads} beads with min_dist="
              f"{min_dist:.1f} voxels.")
        print(f"           Increase volume_size, reduce bead_density, "
              f"or reduce bead_radius.")

    print(f"  Volume:  {vol_cfg.shape}  "
          f"({vol_cfg.n_voxels / 1e6:.2f} M voxels)")
    print(f"  Voxel pitch: {vol_cfg.voxel_pitch} um")
    print(f"  Bead target: {n_target}  "
          f"(density = {vol_cfg.bead_density:.4f} beads/voxel)")
    print(f"  Bead sigma:  {vol_cfg.bead_radius:.1f} voxels  "
          f"(min separation = {min_dist:.1f} voxels)")

    x0 = poisson_disc_3d(domain, min_dist, n_target, rng)
    print(f"  Seeded:  {len(x0)} beads")

    frame_times = def_cfg.frame_times
    fracs = def_cfg.load_fractions
    n_frames = len(frame_times)

    print(f"  Frames:  {n_frames}  "
          f"(t = 0 .. {def_cfg.total_time} s, "
          f"dt = {def_cfg.capture_interval} s)")
    print(f"  Modes:   {def_cfg.modes}")

    F_full = build_deformation_gradient(def_cfg.modes, 1.0)
    np.set_printoptions(precision=4, suppress=True)
    print(f"  F(full) =\n    {np.array2string(F_full, prefix='    ')}")

    coords_all = [x0.copy()]
    disps_all: List[np.ndarray] = []
    Fs_all: List[np.ndarray] = []

    print(f"\n  Rendering frame 0/{n_frames - 1} (reference) ...",
          end=" ", flush=True)
    vol_ref = render_volume(
        x0, sigma, vol_cfg.shape,
        peak=vol_cfg.bead_peak_intensity,
        noise_std=vol_cfg.background_noise,
        poisson=vol_cfg.poisson_noise,
        psf_sigma=vol_cfg.psf_sigma, rng=rng,
    )
    save_volume_tiff(vol_ref, outdir, 0, vol_cfg.bit_depth)
    print("done")

    for fi in range(1, n_frames):
        frac = fracs[fi]
        F = build_deformation_gradient(def_cfg.modes, frac)
        x_def, u = apply_deformation(x0, F, centre)
        coords_all.append(x_def.copy())
        disps_all.append(u.copy())
        Fs_all.append(F.copy())

        F_per_bead = np.tile(F, (len(x_def), 1, 1))
        print(f"  Rendering frame {fi}/{n_frames - 1}  "
              f"(t = {frame_times[fi]:.1f} s,  frac = {frac:.3f}) ...",
              end=" ", flush=True)
        vol_def = render_volume(
            x_def, sigma, vol_cfg.shape,
            F_per_bead=F_per_bead,
            peak=vol_cfg.bead_peak_intensity,
            noise_std=vol_cfg.background_noise,
            poisson=vol_cfg.poisson_noise,
            psf_sigma=vol_cfg.psf_sigma, rng=rng,
        )
        save_volume_tiff(vol_def, outdir, fi, vol_cfg.bit_depth)
        print("done")

    gt = GroundTruth(
        vol_config=asdict(vol_cfg),
        deform_config={"modes": def_cfg.modes,
                       "total_time": def_cfg.total_time,
                       "capture_interval": def_cfg.capture_interval},
        coords_ref=x0, frame_times=frame_times,
        coords_per_frame=coords_all, displacements=disps_all,
        F_per_frame=Fs_all, centre=centre,
    )
    gt.save(outdir / "ground_truth.npz")
    print(f"\n  Ground truth -> {outdir / 'ground_truth.npz'}")
    return gt


# ═══════════════════════════════════════════════════════════════════════
#  8.  SerialTrack ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(
    outdir: Path,
    vol_cfg: VolumeConfig,
    def_cfg: DeformationConfig,
    verbose: bool = False,
) -> st.TrackingSession:
    frame_dirs = sorted(outdir.glob("frame_*"))
    if len(frame_dirs) < 2:
        raise FileNotFoundError(
            f"Need >= 2 frame_* folders in {outdir}")

    print(f"\n  Loading {len(frame_dirs)} volumes from {outdir} ...")
    volumes: List[np.ndarray] = []
    for fd in frame_dirs:
        v = load_volume_tiff(fd)
        volumes.append(v)
        print(f"    {fd.name}: shape = {v.shape}")

    det = st.DetectionConfig(
        method=st.DetectionMethod.TRACTRAC,
        threshold=0.15,
        bead_radius=vol_cfg.bead_radius,
        min_size=max(2, int(vol_cfg.bead_radius)),
        max_size=5000,
    )

    fos = min(vol_cfg.size_x, vol_cfg.size_y, vol_cfg.size_z) * 0.35
    trk = st.TrackingConfig(
        mode=st.TrackingMode.CUMULATIVE,
        solver=st.GlobalSolver.ADMM,
        f_o_s=fos,
        n_neighbors_max=25, n_neighbors_min=1,
        smoothness=1e-2,
        outlier_threshold=5.0,
        max_iter=20, iter_stop_threshold=1e-2,
        strain_n_neighbors=15, strain_f_o_s=fos,
        dist_missing=5.0,
        xstep=vol_cfg.voxel_dx,
        ystep=vol_cfg.voxel_dy,
        zstep=vol_cfg.voxel_dz,
    )

    tracker = st.SerialTracker(det, trk)

    def progress(frame, total, result):
        print(f"    Frame {frame}/{total}:  "
              f"ratio = {result.match_ratio:.3f},  "
              f"iters = {result.n_iterations},  "
              f"time = {result.wall_time:.1f} s")

    print(f"\n  Running SerialTrack  "
          f"(f_o_s = {fos:.0f}, ADMM, cumulative) ...")
    t0 = time.perf_counter()
    session = tracker.track_images(volumes, progress_cb=progress)
    elapsed = time.perf_counter() - t0

    print(f"\n  Tracking complete: {elapsed:.1f} s")
    print(f"  Detected  {len(session.coords_ref)} particles in reference")
    print(f"  Ratios    {session.tracking_ratios}")
    return session


# ═══════════════════════════════════════════════════════════════════════
#  9.  ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FrameError:
    frame_idx: int
    time: float
    load_frac: float
    F_true: np.ndarray
    n_gt_beads: int
    n_detected: int
    n_matched_gt: int
    n_tracked: int
    tracking_ratio: float
    disp_errors: np.ndarray
    rms_disp_error: float
    max_disp_error: float
    mean_disp_error: float
    F_estimated: Optional[np.ndarray]
    F_error_frobenius: float
    component_errors: Optional[np.ndarray]


def compute_errors(
    session: st.TrackingSession,
    gt: GroundTruth,
    vol_cfg: VolumeConfig,
) -> List[FrameError]:
    max_match_dist = 3.0 * vol_cfg.bead_radius
    tree_gt = cKDTree(gt.coords_ref)
    dists, gt_idx = tree_gt.query(session.coords_ref, k=1)
    gt_for_det = np.where(dists < max_match_dist, gt_idx, -1).astype(np.int64)
    n_matched_ref = int(np.sum(gt_for_det >= 0))

    print(f"\n  Detection -> GT matching:  "
          f"{n_matched_ref} / {len(session.coords_ref)} detected "
          f"matched to GT  (< {max_match_dist:.1f} px)")

    errors: List[FrameError] = []
    for fi, (res, true_u, F_true) in enumerate(
        zip(session.frame_results, gt.displacements, gt.F_per_frame)
    ):
        t = gt.frame_times[fi + 1]
        frac = t / gt.frame_times[-1] if gt.frame_times[-1] > 0 else 0

        good = res.track_a2b >= 0
        ia = np.where(good)[0]
        ib = res.track_a2b[ia]
        gt_ia = gt_for_det[ia]
        has_gt = gt_ia >= 0
        ia_v, ib_v, gt_v = ia[has_gt], ib[has_gt], gt_ia[has_gt]

        if len(ia_v) > 0:
            tracked_u = res.coords_b[ib_v] - session.coords_ref[ia_v]
            err_norms = np.linalg.norm(tracked_u - true_u[gt_v], axis=1)
        else:
            err_norms = np.array([0.0])

        rms = float(np.sqrt(np.mean(err_norms ** 2)))
        mx  = float(np.max(err_norms))
        mn  = float(np.mean(err_norms))

        F_est, F_err_frob, comp_err = None, 0.0, None
        if len(ia_v) >= 5:
            disp_tr = res.coords_b[ib_v] - session.coords_ref[ia_v]
            try:
                _, F_mls, valid = st.compute_strain_mls(
                    disp_tr, session.coords_ref[ia_v],
                    f_o_s=min(vol_cfg.size_x, vol_cfg.size_y,
                              vol_cfg.size_z) * 0.5,
                    n_neighbors=min(20, len(ia_v) - 1),
                )
                if np.any(valid):
                    F_est = np.eye(3) + np.nanmean(F_mls[valid], axis=0)
                    F_err_frob = float(np.linalg.norm(F_est - F_true))
                    comp_err = F_est - F_true
            except Exception as e:
                log.warning("MLS strain failed frame %d: %s", fi + 1, e)

        errors.append(FrameError(
            frame_idx=fi + 1, time=t, load_frac=frac, F_true=F_true,
            n_gt_beads=len(gt.coords_ref), n_detected=len(session.coords_ref),
            n_matched_gt=n_matched_ref, n_tracked=len(ia_v),
            tracking_ratio=res.match_ratio, disp_errors=err_norms,
            rms_disp_error=rms, max_disp_error=mx, mean_disp_error=mn,
            F_estimated=F_est, F_error_frobenius=F_err_frob,
            component_errors=comp_err,
        ))
    return errors


def print_error_summary(errors: List[FrameError]):
    print(f"\n  {'=' * 78}")
    print(f"  {'Frame':>5s}  {'Time':>6s}  {'Frac':>5s}  "
          f"{'Tracked':>8s}  {'Ratio':>6s}  "
          f"{'RMS':>8s}  {'Max':>8s}  {'|dF|':>8s}")
    print(f"  {'-' * 78}")
    for e in errors:
        print(f"  {e.frame_idx:5d}  {e.time:6.2f}  {e.load_frac:5.3f}  "
              f"{e.n_tracked:8d}  {e.tracking_ratio:6.3f}  "
              f"{e.rms_disp_error:8.4f}  {e.max_disp_error:8.4f}  "
              f"{e.F_error_frobenius:8.5f}")
    print(f"  {'=' * 78}")


# ═══════════════════════════════════════════════════════════════════════
#  10.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def generate_plots(
    errors: List[FrameError],
    gt: GroundTruth,
    session: st.TrackingSession,
    outdir: Path,
    vol_cfg: VolumeConfig,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plotdir = outdir / "plots"
    plotdir.mkdir(exist_ok=True)

    times = np.array([e.time for e in errors])
    fracs = np.array([e.load_frac for e in errors])
    rms   = [e.rms_disp_error for e in errors]
    mx    = [e.max_disp_error for e in errors]
    mn    = [e.mean_disp_error for e in errors]
    ratios = [e.tracking_ratio for e in errors]
    F_err_frob = [e.F_error_frobenius for e in errors]

    # ── 1  Displacement error vs time ─────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(times, rms, "o-", color="#2196F3", lw=2, label="RMS")
    ax.plot(times, mn, "s--", color="#4CAF50", lw=1.5, label="Mean")
    ax.plot(times, mx, "^--", color="#f44336", lw=1.5, alpha=.7, label="Max")
    ax.set_ylabel("Displacement error [voxels]")
    ax.set_title("Displacement Tracking Error vs. Time")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, ratios, "o-", color="#9C27B0", lw=2)
    ax.set_ylabel("Tracking ratio"); ax.set_xlabel("Time [s]")
    ax.set_ylim([0, 1.05])
    ax.axhline(0.8, color="gray", ls=":", alpha=0.5, label="80 %")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plotdir / "01_displacement_error_vs_time.png", dpi=150)
    plt.close(fig)

    # ── 2  Per-frame error histograms ─────────────────────────────
    nf = len(errors)
    ncols = min(nf, 4); nrows = (nf + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(np.asarray(axes))
    for idx, e in enumerate(errors):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if len(e.disp_errors) > 1:
            ax.hist(e.disp_errors, bins=40, color="#2196F3",
                    alpha=0.7, edgecolor="white")
        ax.axvline(e.rms_disp_error, color="#f44336", ls="--",
                   label=f"RMS={e.rms_disp_error:.3f}")
        ax.set_title(f"Frame {e.frame_idx} (t={e.time:.1f}s)", fontsize=10)
        ax.set_xlabel("Error [voxels]"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for idx in range(nf, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    plt.suptitle("Per-Particle Displacement Error Distributions", fontsize=13)
    plt.tight_layout()
    fig.savefig(plotdir / "02_displacement_error_histograms.png", dpi=150)
    plt.close(fig)

    # ── 3  Strain error ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(times, F_err_frob, "o-", color="#FF9800", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||F_est - F_true|| (Frobenius)")
    ax.set_title("Deformation Gradient Error vs. Time")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    last = errors[-1]
    if last.component_errors is not None:
        ce = last.component_errors.ravel()
        labels = ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]
        cols = ["#f44336" if abs(v) > 0.01 else "#4CAF50" for v in ce]
        ax.bar(labels, ce, color=cols, edgecolor="white", alpha=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("F_est - F_true")
        ax.set_title(f"Component-wise F Error (Frame {last.frame_idx})")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Strain not computed\n(too few particles)",
                ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(plotdir / "03_strain_error.png", dpi=150)
    plt.close(fig)

    # ── 4  Spatial error map (projections) ────────────────────────
    last_res = session.frame_results[-1]
    tree_gt = cKDTree(gt.coords_ref)
    d_, idx_ = tree_gt.query(session.coords_ref, k=1)
    gt_for_det = np.where(d_ < 3.0 * vol_cfg.bead_radius, idx_, -1)
    good = last_res.track_a2b >= 0
    ia = np.where(good)[0]; ib = last_res.track_a2b[ia]
    gt_ia = gt_for_det[ia]; has_gt = gt_ia >= 0

    if np.sum(has_gt) > 10:
        ia_v = ia[has_gt]; ib_v = ib[has_gt]; gt_v = gt_ia[has_gt]
        pos = session.coords_ref[ia_v]
        tracked_u = last_res.coords_b[ib_v] - session.coords_ref[ia_v]
        gt_u = gt.displacements[-1][gt_v]
        err_vec = tracked_u - gt_u
        err_mag = np.linalg.norm(err_vec, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (d1, d2, lab) in zip(
            axes, [(0,1,"XY"),(0,2,"XZ"),(1,2,"YZ")]
        ):
            vmax = max(1.0, np.percentile(err_mag, 95))
            sc = ax.scatter(pos[:,d1], pos[:,d2], c=err_mag, s=8,
                            cmap="hot", vmin=0, vmax=vmax)
            ax.set_xlabel(f"{'XYZ'[d1]} [vox]")
            ax.set_ylabel(f"{'XYZ'[d2]} [vox]")
            ax.set_title(f"{lab} — Error Magnitude"); ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax, label="Error [vox]")
        plt.suptitle(f"Spatial Error Distribution "
                     f"(Frame {last.frame_idx})", fontsize=13)
        plt.tight_layout()
        fig.savefig(plotdir / "04_spatial_error_map.png", dpi=150)
        plt.close(fig)

        # ── 5  Quiver comparison ──────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        step = max(1, len(ia_v) // 200); s = slice(None, None, step)
        ax = axes[0]
        ax.quiver(pos[s,0], pos[s,1], gt_u[s,0], gt_u[s,1],
                  color="#2196F3", alpha=.6, label="Ground truth")
        ax.quiver(pos[s,0], pos[s,1], tracked_u[s,0], tracked_u[s,1],
                  color="#f44336", alpha=.4, label="Tracked")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("XY Displacement Field"); ax.legend()
        ax.set_aspect("equal")

        ax = axes[1]
        ax.quiver(pos[s,0], pos[s,1], err_vec[s,0], err_vec[s,1],
                  err_mag[s], cmap="hot", alpha=.7)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("Displacement Error Vectors"); ax.set_aspect("equal")
        plt.suptitle(f"Tracked vs True (Frame {last.frame_idx})", fontsize=13)
        plt.tight_layout()
        fig.savefig(plotdir / "05_quiver_comparison.png", dpi=150)
        plt.close(fig)

    # ── 6  Summary dashboard ──────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(fracs, rms, "o-", color="#2196F3", lw=2)
    ax.set_xlabel("Load fraction"); ax.set_ylabel("RMS err [vox]")
    ax.set_title("RMS vs Load"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(fracs, ratios, "o-", color="#9C27B0", lw=2)
    ax.set_xlabel("Load fraction"); ax.set_ylabel("Tracking ratio")
    ax.set_ylim([0, 1.05])
    ax.set_title("Ratio vs Load"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(fracs, F_err_frob, "o-", color="#FF9800", lw=2)
    ax.set_xlabel("Load fraction"); ax.set_ylabel("||dF||")
    ax.set_title("Strain Err vs Load"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    if len(last.disp_errors) > 1:
        ax.hist(last.disp_errors, bins=50, color="#2196F3",
                alpha=0.7, edgecolor="white", density=True)
    ax.axvline(last.rms_disp_error, color="#f44336", ls="--", lw=2,
               label=f"RMS = {last.rms_disp_error:.4f}")
    ax.axvline(last.mean_disp_error, color="#4CAF50", ls="--", lw=2,
               label=f"Mean = {last.mean_disp_error:.4f}")
    ax.set_xlabel("Error [vox]"); ax.set_ylabel("Density")
    ax.set_title(f"Error at Full Load (N = {last.n_tracked})")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    modes_str = ", ".join(f"{k}={v}"
        for k, v in gt.deform_config.get("modes", {}).items())
    summary = (
        f"Volume: {gt.vol_config.get('size_x','?')} x "
        f"{gt.vol_config.get('size_y','?')} x "
        f"{gt.vol_config.get('size_z','?')}   "
        f"Beads: {last.n_gt_beads}   "
        f"Detected: {last.n_detected}   "
        f"Tracked (last): {last.n_tracked}\n"
        f"Deformation: {modes_str}   "
        f"Frames: {len(errors)+1}   "
        f"Total time: {gt.frame_times[-1]:.1f} s\n"
        f"Final RMS: {last.rms_disp_error:.4f} vox   "
        f"|dF|: {last.F_error_frobenius:.5f}   "
        f"Ratio: {last.tracking_ratio:.3f}"
    )
    ax.text(0.5, 0.5, summary, ha="center", va="center",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8),
            transform=ax.transAxes)
    plt.suptitle("SerialTrack Synthetic Validation — Dashboard",
                 fontsize=14, y=0.98)
    fig.savefig(plotdir / "06_summary_dashboard.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Plots saved to {plotdir}/")
    for i in range(1, 7):
        name = sorted(plotdir.glob(f"{i:02d}_*.png"))
        if name:
            print(f"    {name[0].name}")


# ═══════════════════════════════════════════════════════════════════════
#  11.  MAIN — JSON-driven entry point
# ═══════════════════════════════════════════════════════════════════════

def _pick_config_file() -> Optional[str]:
    """Open a file-picker dialogue and return the selected JSON path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("ERROR: tkinter is not available.  Pass the config path as "
              "an argument instead:\n    python hydrogel_phantom.py config.json")
        return None

    root = tk.Tk()
    root.withdraw()          # hide the empty root window
    root.attributes("-topmost", True)   # dialogue on top
    path = filedialog.askopenfilename(
        title="Select Hydrogel Phantom Config",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


def main() -> int:
    # --write-examples flag still works from the command line
    if len(sys.argv) == 2 and sys.argv[1] == "--write-examples":
        print("Writing example config files:")
        write_example_configs(".")
        return 0

    # If a path was passed on the command line, use it; otherwise open a
    # file-picker dialogue so the user can browse to the JSON config.
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        config_path = sys.argv[1]
    else:
        print("Opening file dialog — select a .json config file ...")
        config_path = _pick_config_file()
        if config_path is None:
            print("No file selected.  Exiting.")
            return 1

    if not os.path.isfile(config_path):
        print(f"ERROR: config file not found: {config_path}")
        return 1

    vol_cfg, def_cfg, wf_cfg = load_config(config_path)

    logging.basicConfig(
        level=logging.INFO if wf_cfg.verbose else logging.WARNING,
        format="%(name)s | %(message)s",
    )

    outdir = Path(wf_cfg.output)

    # Print loaded config
    banner = "Hydrogel Bead Phantom Generator"
    print(f"\n{'=' * 72}\n  {banner}\n{'=' * 72}")
    print(f"  Config: {os.path.abspath(config_path)}")
    print(f"  Output: {outdir.resolve()}")
    print(f"  Steps:  generate={wf_cfg.generate}  "
          f"analyse={wf_cfg.analyse}  plot={wf_cfg.plot}")
    print(f"  Seed:   {wf_cfg.seed}")
    print(f"{'=' * 72}\n")

    # ── STEP 1: Generate ──────────────────────────────────────────
    if wf_cfg.generate:
        print("STEP 1: Generating phantom data\n")
        gt = generate_phantom(vol_cfg, def_cfg, outdir, seed=wf_cfg.seed)
    else:
        print("STEP 1: Skipped (generate = false)\n")
        gt_path = outdir / "ground_truth.npz"
        if not gt_path.exists():
            print(f"ERROR: {gt_path} not found. "
                  f"Set \"generate\": true first.")
            return 1
        gt = GroundTruth.load(gt_path)
        print(f"  Loaded ground truth: {len(gt.coords_ref)} beads, "
              f"{len(gt.frame_times)} frames")

    # ── STEP 2: Analyse ───────────────────────────────────────────
    if wf_cfg.analyse:
        print(f"\n{'=' * 72}")
        print("STEP 2: SerialTrack Analysis")
        print(f"{'=' * 72}\n")
        session = run_analysis(outdir, vol_cfg, def_cfg,
                               verbose=wf_cfg.verbose)

        # ── STEP 3: Error Analysis ────────────────────────────────
        print(f"\n{'=' * 72}")
        print("STEP 3: Error Analysis")
        print(f"{'=' * 72}")
        errors = compute_errors(session, gt, vol_cfg)
        print_error_summary(errors)

        # ── STEP 4: Plots ─────────────────────────────────────────
        if wf_cfg.plot:
            print(f"\n{'=' * 72}")
            print("STEP 4: Generating Plots")
            print(f"{'=' * 72}")
            generate_plots(errors, gt, session, outdir, vol_cfg)

    # ── Copy config into output for reproducibility ───────────────
    shutil.copy2(config_path, str(outdir / "config.json"))

    print(f"\n{'=' * 72}")
    print(f"  Done.  Output in: {outdir.resolve()}")
    print(f"{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())