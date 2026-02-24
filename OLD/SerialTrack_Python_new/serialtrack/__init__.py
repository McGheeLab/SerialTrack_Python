"""
SerialTrack — ScalE and Rotation Invariant Augmented Lagrangian Particle Tracking
==================================================================================

A Python reimplementation of the MATLAB SerialTrack algorithm by
Jin Yang et al. (Franck Lab, UW-Madison).

Reference
---------
Yang et al., "SerialTrack: ScalE and Rotation Invariant Augmented
Lagrangian Particle Tracking", SoftwareX 19, 101204 (2022).

Quick start
-----------
>>> import serialtrack as st
>>>
>>> # Configure
>>> det = st.DetectionConfig(threshold=0.4, bead_radius=3)
>>> trk = st.TrackingConfig(
...     mode=st.TrackingMode.CUMULATIVE,
...     solver=st.GlobalSolver.ADMM,
...     f_o_s=60,
... )
>>>
>>> # Load images
>>> images = st.ImageLoader.load_2d_sequence("./imgFolder")
>>>
>>> # Track
>>> session = st.track(images, det, trk)
>>> print(session.tracking_ratios)
>>> trajectories = session.get_trajectories()
"""

__version__ = "2.0.0a1"

# ── Config enums & dataclasses (always available) ──
from .config import (
    DetectionMethod,
    DetectionConfig,
    GlobalSolver,
    LocalSolver,
    TrackingMode,
    TrackingConfig,
    TrajectoryConfig,
)

# ── Core modules ──
from .detection import ParticleDetector
from .matching import TopologyMatcher, NearestNeighborMatcher, compute_displacement
from .outliers import remove_outliers, find_not_missing, update_f_o_s
from .regularization import DisplacementRegularizer, scatter_to_grid, scatter_to_grid_multi
from .fields import DisplacementField, StrainField, compute_strain_mls, compute_gridded_strain
from .prediction import InitialGuessPredictor
from .tracking import SerialTracker, TrackingSession, FrameResult, track
from .trajectories import (
    TrajectorySegment,
    build_segments_incremental,
    build_segments_cumulative,
    merge_segments,
    segments_to_matrix,
)
from .io import ImageLoader

__all__ = [
    # Config
    "DetectionMethod", "DetectionConfig",
    "GlobalSolver", "LocalSolver", "TrackingMode",
    "TrackingConfig", "TrajectoryConfig",
    # Core
    "ParticleDetector",
    "TopologyMatcher", "NearestNeighborMatcher", "compute_displacement",
    "remove_outliers", "find_not_missing", "update_f_o_s",
    "DisplacementRegularizer", "scatter_to_grid", "scatter_to_grid_multi",
    "DisplacementField", "StrainField", "compute_strain_mls", "compute_gridded_strain",
    "InitialGuessPredictor",
    "SerialTracker", "TrackingSession", "FrameResult", "track",
    "TrajectorySegment", "build_segments_incremental", "build_segments_cumulative",
    "merge_segments", "segments_to_matrix",
    "ImageLoader",
]