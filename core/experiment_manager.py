"""
Experiment Manager — session history, save/load, per-experiment data persistence.

Each experiment run creates an ExperimentRecord containing:
  - Unique ID and timestamp
  - Detection, tracking, postprocessing configs
  - Image paths / references
  - Results (TrackingSession) once complete
  - Postprocessing results (displacement, strain, stress fields)
  - In-memory data cache for fast experiment switching

Records are serializable to JSON manifest + optional NPZ data files.
"""
from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import numpy as np

from PySide6.QtCore import QObject, Signal


class ExperimentStatus(Enum):
    CONFIGURED = "configured"
    DETECTING = "detecting"
    TRACKING = "tracking"
    POSTPROCESSING = "postprocessing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class PostProcessRecord:
    """One post-processing run on an experiment."""
    record_id: str = ""
    timestamp: str = ""
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    # Stored paths to output files
    output_dir: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = uuid.uuid4().hex[:8]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ExperimentRecord:
    """A complete experiment record with in-memory data cache."""
    exp_id: str = ""
    name: str = ""
    timestamp: str = ""
    status: str = "configured"
    description: str = ""

    # Configuration snapshots (serializable dicts)
    image_config: Dict[str, Any] = field(default_factory=dict)
    detection_config: Dict[str, Any] = field(default_factory=dict)
    tracking_config: Dict[str, Any] = field(default_factory=dict)
    mask_config: Dict[str, Any] = field(default_factory=dict)
    enhancement_config: Dict[str, Any] = field(default_factory=dict)

    # Paths
    image_paths: List[str] = field(default_factory=list)
    output_dir: str = ""

    # Results metadata
    n_frames: int = 0
    n_particles_ref: int = 0
    mean_tracking_ratio: float = 0.0

    # Post-processing runs
    postprocess_runs: List[PostProcessRecord] = field(default_factory=list)

    # Stress analysis runs
    stress_runs: List[PostProcessRecord] = field(default_factory=list)

    # ── In-memory data cache (not serialized to JSON) ──
    # These hold actual results so switching experiments restores state
    _tracking_session: Any = field(default=None, repr=False)
    _postprocess_results: Dict[str, Any] = field(default_factory=dict, repr=False)
    _stress_results: Dict[str, Any] = field(default_factory=dict, repr=False)
    _image_volumes: Optional[List] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.exp_id:
            self.exp_id = uuid.uuid4().hex[:8]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.name:
            self.name = f"Experiment_{self.exp_id}"

    # ── Data cache access ──

    def store_tracking_session(self, session):
        """Cache the tracking session for this experiment."""
        self._tracking_session = session
        if session and hasattr(session, 'coords_ref'):
            self.n_particles_ref = len(session.coords_ref)

    def get_tracking_session(self):
        return self._tracking_session

    def store_postprocess_results(self, results: Dict[str, Any]):
        self._postprocess_results = results

    def get_postprocess_results(self) -> Dict[str, Any]:
        return self._postprocess_results

    def store_stress_results(self, results: Dict[str, Any]):
        self._stress_results = results

    def get_stress_results(self) -> Dict[str, Any]:
        return self._stress_results

    def store_image_volumes(self, volumes: List):
        self._image_volumes = volumes

    def get_image_volumes(self) -> Optional[List]:
        return self._image_volumes

    # ── Serialization (excludes large data caches) ──

    def to_dict(self) -> dict:
        d = {
            "exp_id": self.exp_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "status": self.status,
            "description": self.description,
            "image_config": self.image_config,
            "detection_config": self.detection_config,
            "tracking_config": self.tracking_config,
            "mask_config": self.mask_config,
            "enhancement_config": self.enhancement_config,
            "image_paths": self.image_paths,
            "output_dir": self.output_dir,
            "n_frames": self.n_frames,
            "n_particles_ref": self.n_particles_ref,
            "mean_tracking_ratio": self.mean_tracking_ratio,
            "postprocess_runs": [
                {"record_id": r.record_id, "timestamp": r.timestamp,
                 "description": r.description, "config": r.config,
                 "output_dir": r.output_dir}
                for r in self.postprocess_runs
            ],
            "stress_runs": [
                {"record_id": r.record_id, "timestamp": r.timestamp,
                 "description": r.description, "config": r.config,
                 "output_dir": r.output_dir}
                for r in self.stress_runs
            ],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentRecord:
        pp = [PostProcessRecord(**r) for r in d.pop("postprocess_runs", [])]
        sr = [PostProcessRecord(**r) for r in d.pop("stress_runs", [])]
        # Filter out non-dataclass fields
        valid_fields = {f for f in cls.__dataclass_fields__ if not f.startswith('_')}
        rec = cls(**{k: v for k, v in d.items() if k in valid_fields})
        rec.postprocess_runs = pp
        rec.stress_runs = sr
        return rec


class ExperimentManager(QObject):
    """Manages a timeline of experiment records for the current session.

    When the active experiment changes, pages should save their current
    results to the old experiment and load from the new one.

    Signals:
        experiment_added(str)       — exp_id
        experiment_updated(str)     — exp_id
        experiment_removed(str)     — exp_id
        active_changed(str)         — exp_id
    """
    experiment_added = Signal(str)
    experiment_updated = Signal(str)
    experiment_removed = Signal(str)
    active_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records: Dict[str, ExperimentRecord] = {}
        self._order: List[str] = []
        self._active_id: Optional[str] = None
        self._session_dir: Optional[Path] = None

    # ── Properties ──────────────────────────────────────────
    @property
    def active(self) -> Optional[ExperimentRecord]:
        if self._active_id and self._active_id in self._records:
            return self._records[self._active_id]
        return None

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    @property
    def records(self) -> List[ExperimentRecord]:
        return [self._records[eid] for eid in self._order
                if eid in self._records]

    @property
    def count(self) -> int:
        return len(self._records)

    # ── CRUD ────────────────────────────────────────────────
    def create_experiment(self, name: str = "",
                          description: str = "") -> ExperimentRecord:
        rec = ExperimentRecord(name=name, description=description)
        self._records[rec.exp_id] = rec
        self._order.append(rec.exp_id)
        self.experiment_added.emit(rec.exp_id)
        if self._active_id is None:
            self.set_active(rec.exp_id)
        return rec

    def get(self, exp_id: str) -> Optional[ExperimentRecord]:
        return self._records.get(exp_id)

    def update(self, exp_id: str, **kwargs):
        rec = self._records.get(exp_id)
        if rec is None:
            return
        for k, v in kwargs.items():
            if hasattr(rec, k) and not k.startswith('_'):
                setattr(rec, k, v)
        self.experiment_updated.emit(exp_id)

    def remove(self, exp_id: str):
        if exp_id in self._records:
            del self._records[exp_id]
            self._order.remove(exp_id)
            self.experiment_removed.emit(exp_id)
            if self._active_id == exp_id:
                self._active_id = self._order[-1] if self._order else None
                if self._active_id:
                    self.active_changed.emit(self._active_id)

    def set_active(self, exp_id: str):
        if exp_id in self._records:
            # Before switching, notify pages to save state to current experiment
            # (handled by pages via on_experiment_changed signal)
            self._active_id = exp_id
            self.active_changed.emit(exp_id)

    def duplicate(self, exp_id: str) -> Optional[ExperimentRecord]:
        src = self._records.get(exp_id)
        if src is None:
            return None
        d = src.to_dict()
        d.pop("exp_id")
        d.pop("timestamp")
        d["name"] = f"{src.name} (copy)"
        d["status"] = "configured"
        d["postprocess_runs"] = []
        d["stress_runs"] = []
        new = ExperimentRecord.from_dict(d)
        self._records[new.exp_id] = new
        self._order.append(new.exp_id)
        self.experiment_added.emit(new.exp_id)
        return new

    # ── Session save/load ───────────────────────────────────
    def save_session(self, filepath: str):
        """Save all experiments to a JSON manifest file."""
        p = Path(filepath)
        data = {
            "version": "2.0",
            "experiments": [self._records[eid].to_dict()
                           for eid in self._order
                           if eid in self._records],
            "active_id": self._active_id,
        }
        p.write_text(json.dumps(data, indent=2, default=str))

    def load_session(self, filepath: str):
        """Load experiments from a JSON manifest file."""
        p = Path(filepath)
        data = json.loads(p.read_text())

        self._records.clear()
        self._order.clear()
        self._active_id = None

        for ed in data.get("experiments", []):
            rec = ExperimentRecord.from_dict(ed)
            self._records[rec.exp_id] = rec
            self._order.append(rec.exp_id)
            self.experiment_added.emit(rec.exp_id)

        active = data.get("active_id")
        if active and active in self._records:
            self.set_active(active)
        elif self._order:
            self.set_active(self._order[0])

    def set_session_dir(self, directory: str):
        self._session_dir = Path(directory)
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_dir(self, exp_id: str) -> Path:
        base = self._session_dir or Path("./serialtrack_output")
        d = base / exp_id
        d.mkdir(parents=True, exist_ok=True)
        return d