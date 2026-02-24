"""
SerialTrack Python — Results export, persistence & GUI bridge
==============================================================
    serialtrack/results.py

Provides:
  - Save / load entire TrackingSession to HDF5, .mat, .npz
  - Per-frame CSV export for particle data
  - Summary statistics and text reports
  - GUI-ready signal bridge (PySide6-compatible callback protocol)
  - Thread-safe result accumulator for live GUI updates

Dependencies:
    pip install numpy scipy h5py
    Optional: pip install PySide6  (only for SignalBridge at runtime)
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Tuple, Union,
    runtime_checkable,
)

import numpy as np

log = logging.getLogger("serialtrack.results")


# ═══════════════════════════════════════════════════════════════
#  Export format enum
# ═══════════════════════════════════════════════════════════════

class ExportFormat(IntEnum):
    """Supported file formats for session persistence."""
    HDF5 = 1
    MAT = 2
    NPZ = 3
    CSV = 4   # per-frame particle CSV (no session round-trip)


# ═══════════════════════════════════════════════════════════════
#  Summary statistics
# ═══════════════════════════════════════════════════════════════

@dataclass
class FrameSummary:
    """Lightweight per-frame statistics for GUI display."""
    frame_idx: int
    n_particles_detected: int
    n_particles_tracked: int
    match_ratio: float
    n_admm_iterations: int
    wall_time_s: float
    mean_disp_magnitude: float
    max_disp_magnitude: float
    rms_disp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame_idx,
            "detected": self.n_particles_detected,
            "tracked": self.n_particles_tracked,
            "ratio": round(self.match_ratio, 4),
            "iters": self.n_admm_iterations,
            "time_s": round(self.wall_time_s, 3),
            "mean_disp": round(self.mean_disp_magnitude, 4),
            "max_disp": round(self.max_disp_magnitude, 4),
            "rms_disp": round(self.rms_disp, 4),
        }


@dataclass
class SessionSummary:
    """Aggregate statistics for the full tracking session."""
    n_frames: int
    n_ref_particles: int
    ndim: int
    tracking_mode: str
    solver: str
    total_wall_time_s: float
    mean_tracking_ratio: float
    min_tracking_ratio: float
    max_tracking_ratio: float
    frame_summaries: List[FrameSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_ref_particles": self.n_ref_particles,
            "ndim": self.ndim,
            "tracking_mode": self.tracking_mode,
            "solver": self.solver,
            "total_wall_time_s": round(self.total_wall_time_s, 2),
            "mean_tracking_ratio": round(self.mean_tracking_ratio, 4),
            "min_tracking_ratio": round(self.min_tracking_ratio, 4),
            "max_tracking_ratio": round(self.max_tracking_ratio, 4),
            "frames": [f.to_dict() for f in self.frame_summaries],
        }

    def to_text_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = [
            "=" * 60,
            "  SerialTrack — Tracking Session Report",
            "=" * 60,
            f"  Frames:              {self.n_frames}",
            f"  Reference particles: {self.n_ref_particles}",
            f"  Dimensions:          {self.ndim}D",
            f"  Tracking mode:       {self.tracking_mode}",
            f"  Solver:              {self.solver}",
            f"  Total time:          {self.total_wall_time_s:.2f} s",
            "",
            f"  Tracking ratio:  mean={self.mean_tracking_ratio:.4f}"
            f"  min={self.min_tracking_ratio:.4f}"
            f"  max={self.max_tracking_ratio:.4f}",
            "",
            "  Per-frame breakdown:",
            "  " + "-" * 56,
            f"  {'Frame':>5}  {'Det':>6}  {'Trk':>6}  {'Ratio':>7}"
            f"  {'Iters':>5}  {'Time':>6}  {'RMS':>8}",
            "  " + "-" * 56,
        ]
        for fs in self.frame_summaries:
            lines.append(
                f"  {fs.frame_idx:5d}  {fs.n_particles_detected:6d}"
                f"  {fs.n_particles_tracked:6d}  {fs.match_ratio:7.4f}"
                f"  {fs.n_admm_iterations:5d}  {fs.wall_time_s:6.2f}"
                f"  {fs.rms_disp:8.4f}"
            )
        lines.append("  " + "-" * 56)
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Build summary from a TrackingSession
# ═══════════════════════════════════════════════════════════════

def summarise_session(session) -> SessionSummary:
    """Compute summary statistics from a TrackingSession.

    Parameters
    ----------
    session : TrackingSession
        The completed tracking session (from tracking.py).

    Returns
    -------
    SessionSummary
    """
    from .config import TrackingMode, GlobalSolver

    cfg = session.tracking_config
    frame_sums = []

    for res in session.frame_results:
        n_tracked = int(np.sum(res.track_a2b >= 0))
        disp_mag = np.linalg.norm(res.disp_b2a, axis=1) if len(res.disp_b2a) > 0 else np.array([0.0])
        tracked_mask = res.track_b2a >= 0
        tracked_disp = disp_mag[tracked_mask] if np.any(tracked_mask) else disp_mag

        frame_sums.append(FrameSummary(
            frame_idx=res.frame_idx,
            n_particles_detected=len(res.coords_b),
            n_particles_tracked=n_tracked,
            match_ratio=res.match_ratio,
            n_admm_iterations=res.n_iterations,
            wall_time_s=res.wall_time,
            mean_disp_magnitude=float(np.mean(tracked_disp)) if len(tracked_disp) > 0 else 0.0,
            max_disp_magnitude=float(np.max(tracked_disp)) if len(tracked_disp) > 0 else 0.0,
            rms_disp=float(np.sqrt(np.mean(tracked_disp ** 2))) if len(tracked_disp) > 0 else 0.0,
        ))

    ratios = session.tracking_ratios
    return SessionSummary(
        n_frames=session.n_frames,
        n_ref_particles=len(session.coords_ref),
        ndim=session.coords_ref.shape[1],
        tracking_mode=TrackingMode(cfg.mode).name,
        solver=GlobalSolver(cfg.solver).name,
        total_wall_time_s=sum(r.wall_time for r in session.frame_results),
        mean_tracking_ratio=float(np.mean(ratios)) if len(ratios) > 0 else 0.0,
        min_tracking_ratio=float(np.min(ratios)) if len(ratios) > 0 else 0.0,
        max_tracking_ratio=float(np.max(ratios)) if len(ratios) > 0 else 0.0,
        frame_summaries=frame_sums,
    )


# ═══════════════════════════════════════════════════════════════
#  Save / Load — HDF5
# ═══════════════════════════════════════════════════════════════

def save_hdf5(session, path: Union[str, Path]) -> None:
    """Save a TrackingSession to HDF5 format.

    Structure::

        /config/         — JSON-encoded detection & tracking config
        /coords_ref      — (N, D) reference particle positions
        /frames/0001/    — per-frame group
            coords_b     — (Nb, D)
            disp_b2a     — (Nb, D)
            track_a2b    — (Na,)
            track_b2a    — (Nb,)
            match_ratio  — scalar
            n_iterations — scalar
            wall_time    — scalar
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as f:
        # ── Metadata ──
        f.attrs["serialtrack_version"] = "2.0.0"
        f.attrs["n_frames"] = session.n_frames
        f.attrs["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # ── Config (JSON serialised) ──
        cfg_grp = f.create_group("config")
        cfg_grp.attrs["detection"] = _config_to_json(session.detection_config)
        cfg_grp.attrs["tracking"] = _config_to_json(session.tracking_config)

        # ── Reference coords ──
        f.create_dataset("coords_ref", data=session.coords_ref,
                         compression="gzip", compression_opts=4)

        # ── Per-frame results ──
        frames_grp = f.create_group("frames")
        for res in session.frame_results:
            key = f"{res.frame_idx:04d}"
            g = frames_grp.create_group(key)
            g.create_dataset("coords_b", data=res.coords_b,
                             compression="gzip", compression_opts=4)
            g.create_dataset("disp_b2a", data=res.disp_b2a,
                             compression="gzip", compression_opts=4)
            g.create_dataset("track_a2b", data=res.track_a2b,
                             compression="gzip", compression_opts=4)
            g.create_dataset("track_b2a", data=res.track_b2a,
                             compression="gzip", compression_opts=4)
            g.attrs["match_ratio"] = res.match_ratio
            g.attrs["n_iterations"] = res.n_iterations
            g.attrs["wall_time"] = res.wall_time

            # Optional gridded fields
            if res.disp_field is not None:
                _save_disp_field(g, "disp_field", res.disp_field)
            if res.strain_field is not None:
                _save_strain_field(g, "strain_field", res.strain_field)
            if res.disp_field_ref is not None:
                _save_disp_field(g, "disp_field_ref", res.disp_field_ref)
            if res.strain_field_ref is not None:
                _save_strain_field(g, "strain_field_ref", res.strain_field_ref)

    log.info("Session saved to %s", path)


def load_hdf5(path: Union[str, Path]):
    """Load a TrackingSession from HDF5.

    Returns a TrackingSession instance (lazy import to avoid circular deps).
    """
    import h5py
    from .config import DetectionConfig, TrackingConfig
    from .tracking import TrackingSession, FrameResult
    from .fields import DisplacementField, StrainField

    path = Path(path)
    with h5py.File(str(path), "r") as f:
        det_cfg = _json_to_config(f["config"].attrs["detection"], DetectionConfig)
        trk_cfg = _json_to_config(f["config"].attrs["tracking"], TrackingConfig)
        coords_ref = f["coords_ref"][:]

        session = TrackingSession(
            detection_config=det_cfg,
            tracking_config=trk_cfg,
            coords_ref=coords_ref,
        )

        for key in sorted(f["frames"].keys()):
            g = f["frames"][key]
            res = FrameResult(
                frame_idx=int(key),
                coords_b=g["coords_b"][:],
                disp_b2a=g["disp_b2a"][:],
                track_a2b=g["track_a2b"][:],
                track_b2a=g["track_b2a"][:],
                match_ratio=float(g.attrs["match_ratio"]),
                n_iterations=int(g.attrs["n_iterations"]),
                wall_time=float(g.attrs["wall_time"]),
            )
            if "disp_field" in g:
                res.disp_field = _load_disp_field(g, "disp_field")
            if "strain_field" in g:
                res.strain_field = _load_strain_field(g, "strain_field")
            if "disp_field_ref" in g:
                res.disp_field_ref = _load_disp_field(g, "disp_field_ref")
            if "strain_field_ref" in g:
                res.strain_field_ref = _load_strain_field(g, "strain_field_ref")

            session.frame_results.append(res)

    log.info("Session loaded from %s (%d frames)", path, len(session.frame_results))
    return session


# ── HDF5 helpers for gridded fields ──

def _save_disp_field(parent_group, name, dfield) -> None:
    g = parent_group.create_group(name)
    for i, grid in enumerate(dfield.grids):
        g.create_dataset(f"grid_{i}", data=grid, compression="gzip")
    g.create_dataset("components", data=dfield.components, compression="gzip")
    g.create_dataset("pixel_steps", data=dfield.pixel_steps)
    g.attrs["time_step"] = dfield.time_step


def _load_disp_field(parent_group, name):
    from .fields import DisplacementField
    g = parent_group[name]
    grids = []
    i = 0
    while f"grid_{i}" in g:
        grids.append(g[f"grid_{i}"][:])
        i += 1
    return DisplacementField(
        grids=tuple(grids),
        components=g["components"][:],
        pixel_steps=g["pixel_steps"][:],
        time_step=float(g.attrs.get("time_step", 1.0)),
    )


def _save_strain_field(parent_group, name, sfield) -> None:
    g = parent_group.create_group(name)
    g.create_dataset("F_tensor", data=sfield.F_tensor, compression="gzip")
    g.create_dataset("eps_tensor", data=sfield.eps_tensor, compression="gzip")
    for i, grid in enumerate(sfield.grids):
        g.create_dataset(f"grid_{i}", data=grid, compression="gzip")
    g.create_dataset("pixel_steps", data=sfield.pixel_steps)


def _load_strain_field(parent_group, name):
    from .fields import StrainField
    g = parent_group[name]
    grids = []
    i = 0
    while f"grid_{i}" in g:
        grids.append(g[f"grid_{i}"][:])
        i += 1
    return StrainField(
        F_tensor=g["F_tensor"][:],
        eps_tensor=g["eps_tensor"][:],
        grids=tuple(grids),
        pixel_steps=g["pixel_steps"][:],
    )


# ═══════════════════════════════════════════════════════════════
#  Save / Load — MATLAB .mat (scipy)
# ═══════════════════════════════════════════════════════════════

def save_mat(session, path: Union[str, Path]) -> None:
    """Save core tracking results in MATLAB-compatible .mat format.

    Saves arrays matching the MATLAB SerialTrack output variables:
      parCoord_prev, uvw_B2A_prev, track_A2B_prev
    """
    from scipy.io import savemat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(session.frame_results)
    # Build cell-array-like structures (object arrays)
    par_coords = np.empty(n + 1, dtype=object)
    par_coords[0] = session.coords_ref
    uvw = np.empty(n, dtype=object)
    tracks = np.empty(n, dtype=object)

    for i, res in enumerate(session.frame_results):
        par_coords[i + 1] = res.coords_b
        uvw[i] = res.disp_b2a
        tracks[i] = res.track_a2b

    mdict = {
        "parCoord_prev": par_coords,
        "uvw_B2A_prev": uvw,
        "track_A2B_prev": tracks,
    }
    savemat(str(path), mdict, do_compression=True)
    log.info("MAT file saved to %s", path)


def load_mat(path: Union[str, Path]):
    """Load particle coordinates and tracks from a .mat file."""
    from scipy.io import loadmat
    data = loadmat(str(path), squeeze_me=True)
    return {
        "parCoord_prev": data.get("parCoord_prev"),
        "uvw_B2A_prev": data.get("uvw_B2A_prev"),
        "track_A2B_prev": data.get("track_A2B_prev"),
    }


# ═══════════════════════════════════════════════════════════════
#  Save — NumPy .npz (lightweight, fast)
# ═══════════════════════════════════════════════════════════════

def save_npz(session, path: Union[str, Path]) -> None:
    """Save session to compressed .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {"coords_ref": session.coords_ref}
    meta = {
        "n_frames": session.n_frames,
        "tracking_mode": int(session.tracking_config.mode),
        "solver": int(session.tracking_config.solver),
    }

    for i, res in enumerate(session.frame_results):
        prefix = f"f{i:04d}_"
        arrays[prefix + "coords_b"] = res.coords_b
        arrays[prefix + "disp_b2a"] = res.disp_b2a
        arrays[prefix + "track_a2b"] = res.track_a2b
        arrays[prefix + "track_b2a"] = res.track_b2a
        meta[prefix + "match_ratio"] = res.match_ratio
        meta[prefix + "n_iterations"] = res.n_iterations
        meta[prefix + "wall_time"] = res.wall_time

    # Store metadata as a JSON string in a byte array
    arrays["_meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )

    np.savez_compressed(str(path), **arrays)
    log.info("NPZ file saved to %s", path)


# ═══════════════════════════════════════════════════════════════
#  Export — Per-frame CSV (particle-level data)
# ═══════════════════════════════════════════════════════════════

def export_frame_csv(
    frame_result,
    path: Union[str, Path],
    pixel_steps: Optional[np.ndarray] = None,
    time_step: float = 1.0,
) -> None:
    """Export a single FrameResult to CSV.

    Columns: x, y, [z,] ux, uy, [uz,] disp_mag, tracked
    Physical units are applied if pixel_steps is provided.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res = frame_result
    ndim = res.coords_b.shape[1]
    ps = pixel_steps if pixel_steps is not None else np.ones(ndim)

    n = len(res.coords_b)
    tracked = res.track_b2a >= 0

    coords_phys = res.coords_b * ps
    disp_phys = res.disp_b2a * ps / time_step  # velocity
    disp_mag = np.linalg.norm(disp_phys, axis=1)

    coord_names = ["x", "y", "z"][:ndim]
    disp_names = [f"v{c}" for c in coord_names]
    header = ",".join(coord_names + disp_names + ["disp_mag", "tracked"])

    data = np.column_stack([
        coords_phys,
        disp_phys,
        disp_mag[:, np.newaxis],
        tracked[:, np.newaxis].astype(float),
    ])

    np.savetxt(str(path), data, delimiter=",", header=header, comments="",
               fmt="%.6f")
    log.info("Frame CSV saved to %s (%d particles)", path, n)


def export_all_frames_csv(
    session,
    output_dir: Union[str, Path],
    prefix: str = "frame",
) -> List[Path]:
    """Export all frames as individual CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = session.tracking_config
    paths = []

    for res in session.frame_results:
        p = output_dir / f"{prefix}_{res.frame_idx:04d}.csv"
        export_frame_csv(res, p, cfg.steps, cfg.tstep)
        paths.append(p)

    return paths


def export_trajectories_csv(
    session,
    path: Union[str, Path],
    use_stitching: bool = False,
) -> None:
    """Export trajectory matrix to a single CSV.

    Columns: particle_id, frame, x, y, [z]
    One row per (particle, frame) where the particle was tracked.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_stitching:
        traj = session.build_stitched_trajectories()
    else:
        traj = session.get_trajectories()

    # traj shape: (N_particles, n_frames, D)
    N, nf, ndim = traj.shape
    rows = []
    for pid in range(N):
        for fi in range(nf):
            if not np.any(np.isnan(traj[pid, fi])):
                row = [pid, fi] + traj[pid, fi].tolist()
                rows.append(row)

    coord_names = ["x", "y", "z"][:ndim]
    header = ",".join(["particle_id", "frame"] + coord_names)
    data = np.array(rows)
    np.savetxt(str(path), data, delimiter=",", header=header,
               comments="", fmt=["%d", "%d"] + ["%.6f"] * ndim)
    log.info("Trajectories CSV saved to %s (%d rows)", path, len(rows))


# ═══════════════════════════════════════════════════════════════
#  Unified save dispatcher
# ═══════════════════════════════════════════════════════════════

def save_session(
    session,
    path: Union[str, Path],
    fmt: Optional[ExportFormat] = None,
) -> None:
    """Save a TrackingSession to disk.

    Format is auto-detected from extension if ``fmt`` is None:
        .h5 / .hdf5 → HDF5
        .mat         → MATLAB
        .npz         → NumPy compressed
    """
    path = Path(path)
    if fmt is None:
        ext = path.suffix.lower()
        fmt_map = {
            ".h5": ExportFormat.HDF5, ".hdf5": ExportFormat.HDF5,
            ".mat": ExportFormat.MAT,
            ".npz": ExportFormat.NPZ,
        }
        fmt = fmt_map.get(ext)
        if fmt is None:
            raise ValueError(
                f"Cannot infer format from extension '{ext}'. "
                f"Supported: .h5, .hdf5, .mat, .npz"
            )

    if fmt == ExportFormat.HDF5:
        save_hdf5(session, path)
    elif fmt == ExportFormat.MAT:
        save_mat(session, path)
    elif fmt == ExportFormat.NPZ:
        save_npz(session, path)
    else:
        raise ValueError(f"Unsupported format for session save: {fmt}")


# ═══════════════════════════════════════════════════════════════
#  GUI Signal Bridge — PySide6-compatible callback protocol
# ═══════════════════════════════════════════════════════════════
#
#  Design philosophy:
#    The core tracking engine knows NOTHING about PySide6.
#    It only calls plain Python callables via progress_cb.
#    This module provides an adapter that translates those
#    callbacks into Qt signals when PySide6 is available.
#
#  Usage in PySide6 GUI:
#    bridge = SignalBridge()
#    bridge.frame_completed.connect(my_slot)
#    bridge.progress_updated.connect(progress_bar.setValue)
#    session = tracker.track_images(images, progress_cb=bridge.on_progress)
#
# ═══════════════════════════════════════════════════════════════

@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress callbacks accepted by the tracking engine.

    The tracker calls: ``progress_cb(frame_idx, total_frames, frame_result)``
    Any callable matching this signature works — no Qt dependency needed.
    """
    def __call__(
        self, frame_idx: int, total_frames: int, frame_result: Any
    ) -> None: ...


class CallbackAccumulator:
    """Thread-safe result accumulator using plain callbacks.

    Use this when you don't have PySide6 but still want to collect
    intermediate results during tracking (e.g. for a web UI or logging).

    Attributes
    ----------
    summaries : list of FrameSummary
        Accumulated per-frame summaries, available during tracking.
    is_cancelled : bool
        Set to True to request cancellation (checked between frames).
    """

    def __init__(
        self,
        on_frame: Optional[Callable[[FrameSummary], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.on_frame = on_frame
        self.on_progress = on_progress
        self.on_log = on_log
        self.summaries: List[FrameSummary] = []
        self.is_cancelled = False
        self._start_time = time.perf_counter()

    def __call__(self, frame_idx: int, total_frames: int, frame_result) -> None:
        """Called by the tracking engine after each frame."""
        if self.is_cancelled:
            raise KeyboardInterrupt("Tracking cancelled by user")

        # Build summary
        disp_mag = np.linalg.norm(frame_result.disp_b2a, axis=1)
        tracked = frame_result.track_b2a >= 0
        t_disp = disp_mag[tracked] if np.any(tracked) else disp_mag

        summary = FrameSummary(
            frame_idx=frame_result.frame_idx,
            n_particles_detected=len(frame_result.coords_b),
            n_particles_tracked=int(np.sum(frame_result.track_a2b >= 0)),
            match_ratio=frame_result.match_ratio,
            n_admm_iterations=frame_result.n_iterations,
            wall_time_s=frame_result.wall_time,
            mean_disp_magnitude=float(np.mean(t_disp)) if len(t_disp) > 0 else 0.0,
            max_disp_magnitude=float(np.max(t_disp)) if len(t_disp) > 0 else 0.0,
            rms_disp=float(np.sqrt(np.mean(t_disp ** 2))) if len(t_disp) > 0 else 0.0,
        )
        self.summaries.append(summary)

        # Fire callbacks
        if self.on_frame is not None:
            self.on_frame(summary)
        if self.on_progress is not None:
            self.on_progress(frame_idx, total_frames)
        if self.on_log is not None:
            self.on_log(
                f"Frame {frame_idx}/{total_frames}: "
                f"ratio={summary.match_ratio:.3f}, "
                f"time={summary.wall_time_s:.2f}s"
            )

    def cancel(self) -> None:
        """Request cancellation of tracking."""
        self.is_cancelled = True

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time


def make_qt_bridge():
    """Create a PySide6 QObject signal bridge (lazy import).

    Returns None if PySide6 is not installed.
    Call this from your GUI code::

        bridge = make_qt_bridge()
        if bridge:
            bridge.frame_completed.connect(self.on_frame_done)
            session = tracker.track_images(images, progress_cb=bridge)

    The bridge emits these signals:
        frame_completed(FrameSummary)
        progress_updated(int, int)           — (current, total)
        tracking_finished()
        log_message(str)
    """
    try:
        from PySide6.QtCore import QObject, Signal

        class _QtSignalBridge(QObject):
            """Qt signal bridge — lives in the GUI thread.

            The tracking engine runs in a worker thread and calls
            ``bridge(frame_idx, total, result)`` which is a plain
            Python call.  We then emit Qt signals that are
            auto-queued to the GUI thread.
            """
            frame_completed = Signal(object)     # FrameSummary
            progress_updated = Signal(int, int)  # current, total
            tracking_finished = Signal()
            log_message = Signal(str)

            def __init__(self, parent=None):
                super().__init__(parent)
                self._accumulator = CallbackAccumulator(
                    on_frame=lambda s: self.frame_completed.emit(s),
                    on_progress=lambda c, t: self.progress_updated.emit(c, t),
                    on_log=lambda m: self.log_message.emit(m),
                )

            def __call__(self, frame_idx, total_frames, frame_result):
                self._accumulator(frame_idx, total_frames, frame_result)

            def cancel(self):
                self._accumulator.cancel()

            @property
            def summaries(self):
                return self._accumulator.summaries

        return _QtSignalBridge()

    except ImportError:
        log.debug("PySide6 not available — Qt signal bridge disabled")
        return None


# ═══════════════════════════════════════════════════════════════
#  Config serialization helpers
# ═══════════════════════════════════════════════════════════════

def _config_to_json(cfg) -> str:
    """Serialize a dataclass config to JSON string.

    Handles numpy arrays and enums gracefully.
    """
    d = {}
    for fld in cfg.__dataclass_fields__:
        val = getattr(cfg, fld)
        if isinstance(val, np.ndarray):
            d[fld] = val.tolist()
        elif isinstance(val, IntEnum):
            d[fld] = int(val)
        elif hasattr(val, "__dataclass_fields__"):
            d[fld] = json.loads(_config_to_json(val))
        elif val is None:
            d[fld] = None
        else:
            try:
                json.dumps(val)
                d[fld] = val
            except (TypeError, ValueError):
                d[fld] = str(val)
    return json.dumps(d)


def _json_to_config(json_str: str, config_class):
    """Deserialize JSON string back to a config dataclass.

    Best-effort: unknown fields are ignored, missing fields use defaults.
    """
    d = json.loads(json_str)
    valid = {}
    for fld_name, fld_obj in config_class.__dataclass_fields__.items():
        if fld_name in d and d[fld_name] is not None:
            val = d[fld_name]
            fld_type = fld_obj.type
            # Reconstruct numpy arrays from lists
            if isinstance(val, list) and "ndarray" in str(fld_type):
                valid[fld_name] = np.array(val)
            # Reconstruct tuples
            elif isinstance(val, list) and "Tuple" in str(fld_type):
                valid[fld_name] = tuple(val)
            else:
                valid[fld_name] = val
    return config_class(**valid)