"""
Experiment Manager — session history, save/load, per-experiment data persistence.

Switching lifecycle (orchestrated by MainWindow):
  1. Pages save state → old ExperimentRecord cache
  2. Manager switches active ID, emits active_changed(old_id, new_id)
  3. Pages load state ← new ExperimentRecord cache & refresh UI
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
    record_id: str = ""
    timestamp: str = ""
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = uuid.uuid4().hex[:8]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ExperimentRecord:
    """Complete experiment record with in-memory data cache."""
    exp_id: str = ""
    name: str = ""
    timestamp: str = ""
    status: str = "configured"
    description: str = ""

    image_config: Dict[str, Any] = field(default_factory=dict)
    detection_config: Dict[str, Any] = field(default_factory=dict)
    tracking_config: Dict[str, Any] = field(default_factory=dict)
    mask_config: Dict[str, Any] = field(default_factory=dict)
    enhancement_config: Dict[str, Any] = field(default_factory=dict)

    image_paths: List[str] = field(default_factory=list)
    output_dir: str = ""

    n_frames: int = 0
    n_particles_ref: int = 0
    mean_tracking_ratio: float = 0.0

    postprocess_runs: List[PostProcessRecord] = field(default_factory=list)
    stress_runs: List[PostProcessRecord] = field(default_factory=list)

    # In-memory data cache (NOT serialized to JSON)
    _tracking_session: Any = field(default=None, repr=False)
    _postprocess_results: Dict[str, Any] = field(default_factory=dict, repr=False)
    _stress_results: Dict[str, Any] = field(default_factory=dict, repr=False)
    _image_volumes: Optional[List] = field(default=None, repr=False)
    _frame_data: Optional[List] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.exp_id:
            self.exp_id = uuid.uuid4().hex[:8]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.name:
            self.name = f"Experiment_{self.exp_id}"

    def store_tracking_session(self, session):
        self._tracking_session = session
        if session and hasattr(session, 'coords_ref'):
            self.n_particles_ref = len(session.coords_ref)

    def get_tracking_session(self):
        return self._tracking_session

    def store_postprocess_results(self, results):
        self._postprocess_results = results or {}

    def get_postprocess_results(self):
        return self._postprocess_results

    def store_stress_results(self, results):
        self._stress_results = results or {}

    def get_stress_results(self):
        return self._stress_results

    def store_image_volumes(self, volumes):
        self._image_volumes = volumes

    def get_image_volumes(self):
        return self._image_volumes

    def store_frame_data(self, frame_data):
        self._frame_data = frame_data

    def get_frame_data(self):
        return self._frame_data

    def has_data(self):
        return (self._tracking_session is not None
                or bool(self._postprocess_results)
                or bool(self._stress_results)
                or self._image_volumes is not None)

    def to_dict(self):
        return {
            "exp_id": self.exp_id, "name": self.name,
            "timestamp": self.timestamp, "status": self.status,
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
                for r in self.postprocess_runs],
            "stress_runs": [
                {"record_id": r.record_id, "timestamp": r.timestamp,
                 "description": r.description, "config": r.config,
                 "output_dir": r.output_dir}
                for r in self.stress_runs],
        }

    @classmethod
    def from_dict(cls, d):
        pp = [PostProcessRecord(**r) for r in d.pop("postprocess_runs", [])]
        sr = [PostProcessRecord(**r) for r in d.pop("stress_runs", [])]
        valid = {f for f in cls.__dataclass_fields__ if not f.startswith('_')}
        rec = cls(**{k: v for k, v in d.items() if k in valid})
        rec.postprocess_runs = pp
        rec.stress_runs = sr
        return rec


class ExperimentManager(QObject):
    """Manages experiment records with cache and disk persistence.

    Signals:
        experiment_added(str)
        experiment_updated(str)
        experiment_removed(str)
        active_changed(str, str)  — (old_exp_id, new_exp_id)
    """
    experiment_added = Signal(str)
    experiment_updated = Signal(str)
    experiment_removed = Signal(str)
    active_changed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records: Dict[str, ExperimentRecord] = {}
        self._order: List[str] = []
        self._active_id: Optional[str] = None
        self._session_dir: Optional[Path] = None

    @property
    def active(self):
        if self._active_id and self._active_id in self._records:
            return self._records[self._active_id]
        return None

    @property
    def active_id(self):
        return self._active_id

    @property
    def records(self):
        return [self._records[eid] for eid in self._order if eid in self._records]

    @property
    def count(self):
        return len(self._records)

    def create_experiment(self, name="", description=""):
        rec = ExperimentRecord(name=name, description=description)
        self._records[rec.exp_id] = rec
        self._order.append(rec.exp_id)
        self.experiment_added.emit(rec.exp_id)
        if self._active_id is None:
            self._active_id = rec.exp_id
            self.active_changed.emit("", rec.exp_id)
        return rec

    def get(self, exp_id):
        return self._records.get(exp_id)

    def update(self, exp_id, **kwargs):
        rec = self._records.get(exp_id)
        if rec is None:
            return
        for k, v in kwargs.items():
            if hasattr(rec, k) and not k.startswith('_'):
                setattr(rec, k, v)
        self.experiment_updated.emit(exp_id)

    def remove(self, exp_id):
        if exp_id in self._records:
            del self._records[exp_id]
            self._order.remove(exp_id)
            self.experiment_removed.emit(exp_id)
            if self._active_id == exp_id:
                old = exp_id
                self._active_id = self._order[-1] if self._order else None
                if self._active_id:
                    self.active_changed.emit(old, self._active_id)

    def set_active(self, exp_id):
        if exp_id not in self._records:
            return
        old_id = self._active_id or ""
        if old_id == exp_id:
            return
        self._active_id = exp_id
        self.active_changed.emit(old_id, exp_id)

    def duplicate(self, exp_id):
        src = self._records.get(exp_id)
        if src is None:
            return None
        d = src.to_dict()
        d.pop("exp_id"); d.pop("timestamp")
        d["name"] = f"{src.name} (copy)"
        d["status"] = "configured"
        d["postprocess_runs"] = []; d["stress_runs"] = []
        new = ExperimentRecord.from_dict(d)
        self._records[new.exp_id] = new
        self._order.append(new.exp_id)
        self.experiment_added.emit(new.exp_id)
        return new

    # ── Session save/load ───────────────────────────────────

    def save_session(self, filepath):
        p = Path(filepath)
        data_dir = p.parent / (p.stem + "_data")
        data_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "version": "2.0",
            "experiments": [self._records[eid].to_dict()
                           for eid in self._order if eid in self._records],
            "active_id": self._active_id,
            "data_dir": str(data_dir.name),
        }
        p.write_text(json.dumps(manifest, indent=2, default=str))

        for eid in self._order:
            rec = self._records.get(eid)
            if rec and rec.has_data():
                self._save_experiment_npz(rec, data_dir / f"{eid}.npz")

    def _save_experiment_npz(self, rec, path):
        arrays = {}
        pp = rec.get_postprocess_results()
        if pp and pp.get("frames"):
            frames = pp["frames"]
            arrays["pp_n_frames"] = np.array([len(frames)])
            if pp.get("config"):
                arrays["pp_config_json"] = np.void(
                    json.dumps(pp["config"], default=str).encode())
            for i, frame in enumerate(frames):
                if frame is None:
                    arrays[f"pp_{i}_none"] = np.array([1])
                    continue
                for key in ("disp_components", "eps_tensor", "F_tensor",
                            "grids", "velocity"):
                    if key not in frame:
                        continue
                    val = frame[key]
                    if isinstance(val, list) and len(val) > 0 and isinstance(
                            val[0], (list, np.ndarray)):
                        for gi, g in enumerate(val):
                            arrays[f"pp_{i}_{key}_{gi}"] = np.asarray(g)
                        arrays[f"pp_{i}_{key}_len"] = np.array([len(val)])
                    else:
                        arrays[f"pp_{i}_{key}"] = np.asarray(val)

        st = rec.get_stress_results()
        if st and st.get("frames"):
            frames = st["frames"]
            arrays["st_n_frames"] = np.array([len(frames)])
            if st.get("material"):
                arrays["st_material_json"] = np.void(
                    json.dumps(st["material"], default=str).encode())
            if st.get("model"):
                arrays["st_model"] = np.void(st["model"].encode())
            for i, frame in enumerate(frames):
                if frame is None:
                    arrays[f"st_{i}_none"] = np.array([1])
                    continue
                for key in ("sigma_tensor", "von_mises"):
                    if key in frame:
                        arrays[f"st_{i}_{key}"] = np.asarray(frame[key])

        fd = rec.get_frame_data()
        if fd:
            arrays["frame_data_json"] = np.void(
                json.dumps(fd, default=str).encode())

        if arrays:
            np.savez_compressed(str(path), **arrays)

    def load_session(self, filepath):
        p = Path(filepath)
        data = json.loads(p.read_text())
        self._records.clear()
        self._order.clear()
        self._active_id = None
        data_dir = p.parent / data.get("data_dir", p.stem + "_data")

        for ed in data.get("experiments", []):
            rec = ExperimentRecord.from_dict(ed)
            self._records[rec.exp_id] = rec
            self._order.append(rec.exp_id)
            npz_path = data_dir / f"{rec.exp_id}.npz"
            if npz_path.exists():
                self._load_experiment_npz(rec, npz_path)
            self.experiment_added.emit(rec.exp_id)

        active = data.get("active_id")
        if active and active in self._records:
            self._active_id = active
            self.active_changed.emit("", active)
        elif self._order:
            self._active_id = self._order[0]
            self.active_changed.emit("", self._order[0])

    def _load_experiment_npz(self, rec, path):
        try:
            npz = np.load(str(path), allow_pickle=True)
        except Exception:
            return

        if "pp_n_frames" in npz:
            n = int(npz["pp_n_frames"][0])
            pp_config = {}
            if "pp_config_json" in npz:
                pp_config = json.loads(bytes(npz["pp_config_json"]))
            frames = []
            for i in range(n):
                if f"pp_{i}_none" in npz:
                    frames.append(None)
                    continue
                frame = {}
                for key in ("disp_components", "eps_tensor", "F_tensor",
                            "grids", "velocity"):
                    lk = f"pp_{i}_{key}_len"
                    if lk in npz:
                        cnt = int(npz[lk][0])
                        frame[key] = [npz[f"pp_{i}_{key}_{gi}"].tolist()
                                      for gi in range(cnt)]
                    elif f"pp_{i}_{key}" in npz:
                        frame[key] = npz[f"pp_{i}_{key}"].tolist()
                frames.append(frame)
            rec.store_postprocess_results({
                "frames": frames, "config": pp_config, "n_frames": n})

        if "st_n_frames" in npz:
            n = int(npz["st_n_frames"][0])
            st_result = {"frames": []}
            if "st_material_json" in npz:
                st_result["material"] = json.loads(bytes(npz["st_material_json"]))
            if "st_model" in npz:
                st_result["model"] = bytes(npz["st_model"]).decode()
            for i in range(n):
                if f"st_{i}_none" in npz:
                    st_result["frames"].append(None)
                    continue
                frame = {}
                for key in ("sigma_tensor", "von_mises"):
                    if f"st_{i}_{key}" in npz:
                        frame[key] = npz[f"st_{i}_{key}"].tolist()
                st_result["frames"].append(frame)
            rec.store_stress_results(st_result)

        if "frame_data_json" in npz:
            rec.store_frame_data(json.loads(bytes(npz["frame_data_json"])))
        npz.close()

    def set_session_dir(self, directory):
        self._session_dir = Path(directory)
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_dir(self, exp_id):
        base = self._session_dir or Path("./serialtrack_output")
        d = base / exp_id
        d.mkdir(parents=True, exist_ok=True)
        return d