"""
ND2 Microscopy Processor & Mask Generator v3
=============================================

Multi-tab pipeline with:
  - Step-by-step processing with undo/revert/re-run
  - Numba/multiprocessing acceleration
  - Per-channel color assignment
  - Separate tabs: Import → Preprocess → Segment → Mask
  - Flexible preview (per-channel, composite, overlay)
  - Display histogram controls (brightness, contrast, gamma)
  - Size-constrained watershed & template matching
  - Global status indicator

Requirements:
    pip install PySide6 nd2 numpy scipy scikit-image psutil matplotlib tifffile numba

Usage:
    python nd2_processor.py
"""

import sys, os, json, traceback, functools, time as _time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from copy import deepcopy

import numpy as np

# ── Numba ───────────────────────────────────────────────────────────────────
try:
    from numba import njit, prange, config as numba_config
    # Try TBB first, fall back to other available backends
    for _layer in ("tbb", "omp", "workqueue"):
        try:
            numba_config.THREADING_LAYER = _layer
            break
        except Exception:
            continue
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(fn): return fn
        if args and callable(args[0]): return args[0]
        return _wrap
    prange = range

# ── Qt ──────────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QFileDialog, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QProgressBar, QMessageBox,
    QSlider, QScrollArea, QRadioButton, QTextEdit, QColorDialog,
    QFormLayout, QSplitter, QSizePolicy, QListWidget, QListWidgetItem,
    QFrame
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QBrush, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# ── Optional science imports ────────────────────────────────────────────────
try:
    import nd2; HAS_ND2 = True
except ImportError:
    HAS_ND2 = False

try:
    import psutil; HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from skimage import exposure, filters, morphology, measure, segmentation, feature
    from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_triangle
    from skimage.morphology import (
        ball, disk, remove_small_holes, remove_small_objects,
        binary_closing, binary_opening, binary_dilation, binary_erosion,
    )
    from skimage.feature import blob_log, blob_dog, peak_local_max, match_template
    from skimage.segmentation import watershed
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from scipy import ndimage as ndi
    from scipy.ndimage import distance_transform_edt, label as scipy_label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tifffile; HAS_TIFF = True
except ImportError:
    HAS_TIFF = False


# ═════════════════════════════════════════════════════════════════════════════
# NUMBA KERNELS
# ═════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def _median_3d_numba(volume, size):
    nz, ny, nx = volume.shape
    out = np.empty_like(volume)
    r = size // 2
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                z0, z1 = max(0, z-r), min(nz, z+r+1)
                y0, y1 = max(0, y-r), min(ny, y+r+1)
                x0, x1 = max(0, x-r), min(nx, x+r+1)
                out[z, y, x] = np.median(volume[z0:z1, y0:y1, x0:x1].ravel())
    return out


@njit(parallel=True, cache=True)
def _lbp_3d_numba(volume, radius):
    nz, ny, nx = volume.shape
    result = np.zeros((nz, ny, nx), dtype=np.uint8)
    offsets_z = np.array([-radius, radius, 0, 0, 0, 0])
    offsets_y = np.array([0, 0, -radius, radius, 0, 0])
    offsets_x = np.array([0, 0, 0, 0, -radius, radius])
    for z in prange(radius, nz - radius):
        for y in range(radius, ny - radius):
            for x in range(radius, nx - radius):
                val = np.uint8(0)
                center = volume[z, y, x]
                for bit in range(6):
                    if volume[z+offsets_z[bit], y+offsets_y[bit], x+offsets_x[bit]] <= center:
                        val |= np.uint8(1 << bit)
                result[z, y, x] = val
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKERS
# ═════════════════════════════════════════════════════════════════════════════

def _clahe_single_slice(args):
    sl, kernel_size, clip_limit = args
    sl_f = sl.astype(np.float32)
    if sl_f.max() > sl_f.min():
        sl_f = (sl_f - sl_f.min()) / (sl_f.max() - sl_f.min())
    return exposure.equalize_adapthist(
        sl_f, kernel_size=min(kernel_size, min(sl.shape)), clip_limit=clip_limit
    ).astype(np.float32)


def _blob_single_slice(args):
    sl, min_sigma, max_sigma, thresh = args
    sl_f = sl.astype(np.float64)
    if sl_f.max() > sl_f.min():
        sl_f = (sl_f - sl_f.min()) / (sl_f.max() - sl_f.min())
    return blob_log(sl_f, min_sigma=min_sigma, max_sigma=max_sigma,
                    threshold=thresh, num_sigma=5)


N_WORKERS = max(1, mp.cpu_count() - 1)


# ═════════════════════════════════════════════════════════════════════════════
# ACCELERATED OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════

class Accel:

    @staticmethod
    def clahe_3d(vol, kernel_size=64, clip_limit=0.01, _progress=None):
        nz = vol.shape[0]
        args = [(vol[z], kernel_size, clip_limit) for z in range(nz)]
        out = np.empty_like(vol, dtype=np.float32)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for i, result in enumerate(pool.map(_clahe_single_slice, args)):
                out[i] = result
                if _progress and i % max(1, nz//20) == 0:
                    _progress(int(100*i/nz))
        if _progress: _progress(100)
        return out

    @staticmethod
    def median_3d(vol, size=3, _progress=None):
        if HAS_NUMBA and vol.ndim == 3:
            try:
                r = _median_3d_numba(vol.astype(np.float32), size)
                if _progress: _progress(100)
                return r
            except Exception as e:
                print(f"Numba median failed ({e}), falling back to scipy")
        if HAS_SCIPY:
            r = ndi.median_filter(vol.astype(np.float32), size=size)
            if _progress: _progress(100)
            return r.astype(np.float32)
        raise ImportError("Need numba or scipy for median filter")

    @staticmethod
    def lbp_3d(vol, radius=1, _progress=None):
        if HAS_NUMBA and vol.ndim == 3:
            try:
                r = _lbp_3d_numba(vol.astype(np.float32), radius)
                if _progress: _progress(100)
                return r
            except Exception as e:
                print(f"Numba LBP failed ({e}), falling back to numpy")
        # Numpy vectorised fallback
        pad = np.pad(vol.astype(np.float32), radius, mode="reflect")
        nz, ny, nx = vol.shape; r_ = radius
        result = np.zeros_like(vol, dtype=np.uint8)
        c = pad[r_:r_+nz, r_:r_+ny, r_:r_+nx]
        for bit, (dz, dy, dx) in enumerate([(-r_,0,0),(r_,0,0),(0,-r_,0),(0,r_,0),(0,0,-r_),(0,0,r_)]):
            n = pad[r_+dz:r_+dz+nz, r_+dy:r_+dy+ny, r_+dx:r_+dx+nx]
            result |= ((c >= n).astype(np.uint8) << bit)
        if _progress: _progress(100)
        return result

    @staticmethod
    def color_deconvolution(multichannel, mixing_matrix=None, _progress=None):
        nc = multichannel.shape[0]
        if mixing_matrix is None:
            mixing_matrix = np.eye(nc)*0.9 + 0.1*np.ones((nc, nc))
            np.fill_diagonal(mixing_matrix, 1.0)
        try:
            unmix = np.linalg.inv(mixing_matrix)
        except np.linalg.LinAlgError:
            unmix = np.linalg.pinv(mixing_matrix)
        flat = multichannel.reshape(nc, -1).astype(np.float64)
        result = np.clip(unmix @ flat, 0, None).reshape(multichannel.shape)
        if _progress: _progress(100)
        return result.astype(np.float32)

    @staticmethod
    def threshold(vol, method="otsu", _progress=None):
        v = vol.astype(np.float64)
        if v.max() > v.min():
            v = (v - v.min()) / (v.max() - v.min())
        funcs = {"otsu": threshold_otsu, "li": threshold_li,
                 "yen": threshold_yen, "triangle": threshold_triangle}
        t = funcs.get(method, threshold_otsu)(v)
        if _progress: _progress(100)
        return (v > t).astype(np.uint8)

    @staticmethod
    def close_holes(binary, min_um=10, max_um=200, voxel=(1,1,1), _progress=None):
        vv = np.prod(voxel)
        max_vox = max(1, int((4/3*np.pi*(max_um/2)**3)/vv))
        min_vox = max(1, int((4/3*np.pi*(min_um/2)**3)/vv))
        filled = remove_small_holes(binary.astype(bool), area_threshold=max_vox)
        inv = ~filled
        cleaned = remove_small_objects(inv, min_size=min_vox)
        if _progress: _progress(100)
        return (~cleaned).astype(np.uint8)

    @staticmethod
    def morph_close_3d(binary, radius=3, _progress=None):
        r = binary_closing(binary.astype(bool), ball(radius)).astype(np.uint8)
        if _progress: _progress(100)
        return r

    @staticmethod
    def morph_open_3d(binary, radius=3, _progress=None):
        r = binary_opening(binary.astype(bool), ball(radius)).astype(np.uint8)
        if _progress: _progress(100)
        return r

    @staticmethod
    def blob_detect_3d(vol, min_sigma=5, max_sigma=30, thresh=0.1, _progress=None):
        nz = vol.shape[0]
        args = [(vol[z], min_sigma, max_sigma, thresh) for z in range(nz)]
        label_vol = np.zeros_like(vol, dtype=np.int32)
        cur = 1
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for z, blobs in enumerate(pool.map(_blob_single_slice, args)):
                for y, x, sigma in blobs:
                    ri = int(sigma * np.sqrt(2))
                    yc, xc = int(y), int(x)
                    y0, y1 = max(0, yc-ri), min(vol.shape[1], yc+ri+1)
                    x0, x1 = max(0, xc-ri), min(vol.shape[2], xc+ri+1)
                    yy, xx = np.ogrid[y0:y1, x0:x1]
                    mask = ((yy-yc)**2 + (xx-xc)**2) <= ri**2
                    label_vol[z, y0:y1, x0:x1][mask] = cur
                    cur += 1
                if _progress and z % max(1, nz//20) == 0:
                    _progress(int(100*z/nz))
        if _progress: _progress(100)
        return label_vol

    @staticmethod
    def watershed_size_constrained(binary, vol_for_markers, voxel_um=(1,1,1),
                                    min_radius_um=20, max_radius_um=80, _progress=None):
        if not HAS_SCIPY or not HAS_SKIMAGE:
            raise ImportError("scipy + scikit-image required")
        if _progress: _progress(5)
        dist = distance_transform_edt(binary.astype(bool), sampling=voxel_um)
        if _progress: _progress(20)
        min_dist_vox = max(2, int(min_radius_um / max(voxel_um[1], voxel_um[2])))
        coords = peak_local_max(dist, min_distance=min_dist_vox,
                                threshold_abs=min_radius_um * 0.3,
                                labels=binary.astype(np.intp))
        if _progress: _progress(35)
        valid = []
        for c in coords:
            r_est = dist[tuple(c)]
            if min_radius_um * 0.5 <= r_est <= max_radius_um * 1.5:
                valid.append(c)
        valid = np.array(valid) if valid else np.empty((0, 3), dtype=int)
        if _progress: _progress(45)
        markers = np.zeros_like(binary, dtype=np.int32)
        for i, c in enumerate(valid):
            markers[tuple(c)] = i + 1
        markers = ndi.label(ndi.binary_dilation(markers > 0, iterations=2))[0]
        if _progress: _progress(55)
        labels = watershed(-dist, markers, mask=binary.astype(bool))
        if _progress: _progress(80)
        voxel_vol = np.prod(voxel_um)
        min_vol = (4/3)*np.pi*(min_radius_um**3) * 0.3
        max_vol = (4/3)*np.pi*(max_radius_um**3) * 3.0
        min_vox = int(min_vol / voxel_vol)
        max_vox = int(max_vol / voxel_vol)
        props = measure.regionprops(labels)
        for p in props:
            if p.area < min_vox or p.area > max_vox:
                labels[labels == p.label] = 0
        labels, _ = ndi.label(labels > 0)
        if _progress: _progress(100)
        return labels

    @staticmethod
    def template_match_ellipsoid(vol, major_um, minor_um, pixel_um,
                                  shape_type="ellipsoid", _progress=None):
        major_px = major_um / pixel_um
        minor_px = minor_um / pixel_um
        sz = int(max(major_px, minor_px) * 1.5) | 1
        cy, cx = sz//2, sz//2
        yy, xx = np.ogrid[:sz, :sz]
        if shape_type == "sphere":
            r = (major_px + minor_px) / 4
            tmpl = (((yy-cy)**2 + (xx-cx)**2) <= r**2).astype(np.float32)
        elif shape_type == "cuboid":
            tmpl = np.zeros((sz, sz), dtype=np.float32)
            hw, hh = int(major_px/2), int(minor_px/2)
            tmpl[max(0,cy-hh):cy+hh+1, max(0,cx-hw):cx+hw+1] = 1.0
        else:
            a, b = major_px/2, minor_px/2
            if a > 0 and b > 0:
                tmpl = (((yy-cy)/b)**2 + ((xx-cx)/a)**2 <= 1).astype(np.float32)
            else:
                tmpl = np.ones((3, 3), dtype=np.float32)
        nz = vol.shape[0]
        score = np.zeros_like(vol, dtype=np.float32)
        scales = [0.8, 1.0, 1.2]
        for z in range(nz):
            sl = vol[z].astype(np.float32)
            if sl.max() > sl.min():
                sl = (sl - sl.min()) / (sl.max() - sl.min())
            best = np.zeros_like(sl)
            for sc in scales:
                from scipy.ndimage import zoom as scipy_zoom
                t = scipy_zoom(tmpl, sc, order=1)
                if t.shape[0] > sl.shape[0] or t.shape[1] > sl.shape[1]:
                    continue
                r = match_template(sl, t, pad_input=True)
                best = np.maximum(best, r)
            score[z] = best
            if _progress and z % max(1, nz//20) == 0:
                _progress(int(100*z/nz))
        if _progress: _progress(100)
        return score


# ═════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_CHANNEL_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 255, 255),
]


@dataclass
class ND2Metadata:
    filepath: str = ""
    n_timepoints: int = 1
    n_zslices: int = 1
    n_channels: int = 1
    n_multipoints: int = 1
    n_large_images: int = 1
    height: int = 0
    width: int = 0
    dtype: np.dtype = np.dtype("uint16")
    pixel_size_um: float = 1.0
    z_step_um: float = 1.0
    channel_names: List[str] = field(default_factory=list)
    voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def bytes_per_pixel(self): return self.dtype.itemsize

    @property
    def single_frame_bytes(self): return self.height * self.width * self.bytes_per_pixel

    def estimate_bytes(self, t_r, z_r, c_r, m_r, l_r):
        n = lambda r: r[1] - r[0] + 1
        return n(t_r)*n(z_r)*n(c_r)*n(m_r)*n(l_r)*self.single_frame_bytes


class LoadMode(Enum):
    HYPERSTACK = "hyperstack"
    EXPORT = "export"


@dataclass
class MaskClass:
    name: str
    label_value: int
    color: Tuple[int, int, int]
    channel_index: Optional[int] = None


# ═════════════════════════════════════════════════════════════════════════════
# STEP HISTORY — the core of the undo/rerun system
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStep:
    """One completed processing step."""
    name: str               # human-readable label
    params: Dict[str, Any]  # parameters used
    # Per-channel results: key = "ch0","ch1",... value = (Z,Y,X) array
    outputs: Dict[str, np.ndarray]
    timestamp: float = 0.0

    def summary(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items() if k != "_progress")
        return f"{self.name}  ({p})"


class PipelineHistory:
    """
    Manages an ordered list of processing steps.
    Each step's output becomes the next step's input.
    Supports revert-to-step-N and re-run from any point.
    """

    def __init__(self):
        self._steps: List[PipelineStep] = []
        self._base: Dict[str, np.ndarray] = {}  # the raw starting volumes

    def set_base(self, volumes: Dict[str, np.ndarray]):
        """Set the starting point (e.g. raw channel data)."""
        self._base = {k: v.copy() for k, v in volumes.items()}
        self._steps.clear()

    @property
    def steps(self) -> List[PipelineStep]:
        return list(self._steps)

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    def current_volumes(self) -> Dict[str, np.ndarray]:
        """Get the volumes at the current head of the pipeline."""
        if self._steps:
            return self._steps[-1].outputs
        return self._base

    def volumes_at(self, step_index: int) -> Dict[str, np.ndarray]:
        """Get volumes at a specific step (−1 = base)."""
        if step_index < 0 or not self._steps:
            return self._base
        idx = min(step_index, len(self._steps) - 1)
        return self._steps[idx].outputs

    def push(self, step: PipelineStep):
        """Add a new step."""
        step.timestamp = _time.time()
        self._steps.append(step)

    def revert_to(self, step_index: int):
        """
        Discard all steps after step_index.
        step_index = −1 means revert to base (discard all).
        """
        if step_index < 0:
            self._steps.clear()
        else:
            self._steps = self._steps[:step_index + 1]

    def reset(self):
        """Full reset, keeping the base data."""
        self._steps.clear()

    def has_base(self) -> bool:
        return bool(self._base)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED DATA STORE
# ═════════════════════════════════════════════════════════════════════════════

class DataStore:
    def __init__(self):
        self.raw: Optional[np.ndarray] = None
        self.metadata: Optional[ND2Metadata] = None
        self.channel_colors: List[Tuple[int,int,int]] = list(DEFAULT_CHANNEL_COLORS)
        self.preprocess_history = PipelineHistory()
        self.segment_history = PipelineHistory()
        self.combined_mask: Optional[np.ndarray] = None
        self.mask_classes: List[MaskClass] = [
            MaskClass("Void", 0, (0, 0, 0)),
            MaskClass("Functional", 1, (255, 50, 50)),
            MaskClass("Inert", 2, (50, 50, 255)),
            MaskClass("Cells", 3, (50, 255, 50)),
        ]

    @property
    def n_channels(self): return self.raw.shape[2] if self.raw is not None else 0
    @property
    def n_z(self): return self.raw.shape[1] if self.raw is not None else 0
    @property
    def n_t(self): return self.raw.shape[0] if self.raw is not None else 0
    @property
    def shape_yx(self):
        if self.raw is not None: return self.raw.shape[3], self.raw.shape[4]
        return 0, 0

    def reset_all(self):
        self.preprocess_history.reset()
        self.segment_history.reset()
        self.combined_mask = None


# ═════════════════════════════════════════════════════════════════════════════
# ND2 READER
# ═════════════════════════════════════════════════════════════════════════════

class ND2Reader:
    @staticmethod
    def read_metadata(filepath):
        if not HAS_ND2: raise ImportError("pip install nd2")
        meta = ND2Metadata(filepath=filepath)
        with nd2.ND2File(filepath) as f:
            meta.dtype = f.dtype; s = f.sizes
            meta.n_timepoints = s.get("T",1); meta.n_zslices = s.get("Z",1)
            meta.n_channels = s.get("C",1)
            meta.n_multipoints = s.get("P", s.get("M",1))
            meta.n_large_images = s.get("L", s.get("S",1))
            meta.height = s.get("Y",0); meta.width = s.get("X",0)
            try:
                v = f.voxel_size()
                meta.pixel_size_um = getattr(v,"x",1.0)
                meta.z_step_um = getattr(v,"z",1.0)
                meta.voxel_size_um = (meta.z_step_um, meta.pixel_size_um, meta.pixel_size_um)
            except Exception: pass
            try: meta.channel_names = [ch.channel.name for ch in f.metadata.channels]
            except Exception: meta.channel_names = [f"Ch{i}" for i in range(meta.n_channels)]
        return meta

    @staticmethod
    def load_hyperstack(filepath, t_r, z_r, c_r, m_r, l_r, callback=None):
        with nd2.ND2File(filepath) as f:
            full = f.to_dask(); sizes = f.sizes; dim_order = list(sizes.keys())
            sl = {"T":slice(t_r[0],t_r[1]+1),"Z":slice(z_r[0],z_r[1]+1),
                  "C":slice(c_r[0],c_r[1]+1)}
            for k in ("P","M"):
                if k in sizes: sl[k]=slice(m_r[0],m_r[1]+1)
            for k in ("L","S"):
                if k in sizes: sl[k]=slice(l_r[0],l_r[1]+1)
            idx = tuple(sl.get(d, slice(None)) for d in dim_order)
            if callback: callback(40)
            data = np.asarray(full[idx])
            if callback: callback(90)
            target = ["T","Z","C","Y","X"]
            shape5 = []
            for d in target:
                if d in sl:
                    s_ = sl[d]; shape5.append(s_.stop - s_.start)
                elif d=="Y": shape5.append(sizes.get("Y", data.shape[-2]))
                elif d=="X": shape5.append(sizes.get("X", data.shape[-1]))
                else: shape5.append(1)
            try: data = data.reshape(shape5)
            except Exception:
                while data.ndim < 5: data = data[np.newaxis]
            if callback: callback(100)
            return data

    @staticmethod
    def export_frames(filepath, output_dir, t_r, z_r, c_r, m_r, l_r, callback=None):
        if not HAS_TIFF: raise ImportError("pip install tifffile")
        os.makedirs(output_dir, exist_ok=True)
        with nd2.ND2File(filepath) as f:
            data = np.asarray(f.to_dask()); sizes = f.sizes; dim_order = list(sizes.keys())
            total = max(1,(t_r[1]-t_r[0]+1)*(z_r[1]-z_r[0]+1)*(c_r[1]-c_r[0]+1)
                        *(m_r[1]-m_r[0]+1)*(l_r[1]-l_r[0]+1))
            count = 0
            for t in range(t_r[0],t_r[1]+1):
                for m in range(m_r[0],m_r[1]+1):
                    for li in range(l_r[0],l_r[1]+1):
                        for z in range(z_r[0],z_r[1]+1):
                            for c in range(c_r[0],c_r[1]+1):
                                idx_map = {"T":t,"Z":z,"C":c,"Y":slice(None),"X":slice(None)}
                                for k in ("P","M"):
                                    if k in sizes: idx_map[k]=m
                                for k in ("L","S"):
                                    if k in sizes: idx_map[k]=li
                                idx = tuple(idx_map.get(d,slice(None)) for d in dim_order)
                                try:
                                    frame = np.asarray(data[idx])
                                    fn = f"T{t:04d}_Z{z:04d}_C{c:02d}_M{m:03d}_L{li:03d}.tif"
                                    tifffile.imwrite(os.path.join(output_dir, fn), frame)
                                except Exception: pass
                                count += 1
                                if callback: callback(int(100*count/total))


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ═════════════════════════════════════════════════════════════════════════════

class GenericWorker(QThread):
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, func, *a, **kw):
        super().__init__()
        self._f, self._a, self._kw = func, a, kw

    def run(self):
        try:
            r = self._f(*self._a, **self._kw, _progress=self.progress.emit)
            self.finished.emit(r)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL STATUS BAR WIDGET
# ═════════════════════════════════════════════════════════════════════════════

class StatusWidget(QWidget):
    """
    A persistent status widget showing:
    - Spinning indicator when busy
    - Current operation name
    - Progress bar
    """

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)

        self.indicator = QLabel("●")
        self.indicator.setFixedWidth(18)
        self.indicator.setAlignment(Qt.AlignCenter)
        self._set_idle()
        layout.addWidget(self.indicator)

        self.label = QLabel("Ready")
        self.label.setMinimumWidth(250)
        layout.addWidget(self.label, 1)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setFixedWidth(200)
        self.pbar.setTextVisible(True)
        layout.addWidget(self.pbar)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_state = 0

    def set_busy(self, text: str):
        self.label.setText(text)
        self.indicator.setStyleSheet("color: #ffaa00; font-size: 16px; font-weight: bold;")
        self.indicator.setText("◉")
        self.pbar.setValue(0)
        self._pulse_state = 0
        self._pulse_timer.start(400)

    def set_progress(self, value: int):
        self.pbar.setValue(value)

    def set_done(self, text: str = "Ready"):
        self._pulse_timer.stop()
        self.label.setText(text)
        self.pbar.setValue(100)
        self._set_idle()

    def set_error(self, text: str):
        self._pulse_timer.stop()
        self.label.setText(f"ERROR: {text[:80]}")
        self.indicator.setStyleSheet("color: #ff3333; font-size: 16px; font-weight: bold;")
        self.indicator.setText("✗")

    def _set_idle(self):
        self.indicator.setStyleSheet("color: #44cc44; font-size: 16px; font-weight: bold;")
        self.indicator.setText("●")

    def _pulse(self):
        symbols = ["◐", "◓", "◑", "◒"]
        self.indicator.setText(symbols[self._pulse_state % 4])
        self._pulse_state += 1


# ═════════════════════════════════════════════════════════════════════════════
# PREVIEW WIDGET
# ═════════════════════════════════════════════════════════════════════════════

class PreviewWidget(QWidget):

    def __init__(self, store: DataStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._volumes: Dict[str, np.ndarray] = {}  # direct reference to current volumes
        self._overlay: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        self.spin_t = QSpinBox(); self.spin_t.setPrefix("T:")
        self.spin_z = QSpinBox(); self.spin_z.setPrefix("Z:")
        self.spin_c = QSpinBox(); self.spin_c.setPrefix("C:")
        for sp in (self.spin_t, self.spin_z, self.spin_c):
            sp.valueChanged.connect(self.refresh)
            ctrl.addWidget(sp)

        self.combo_view = QComboBox()
        self.combo_view.addItems([
            "Single Channel (colored)", "All Channels Separate",
            "Composite (merged)", "Raw + Mask Overlay",
        ])
        self.combo_view.currentIndexChanged.connect(self.refresh)
        ctrl.addWidget(QLabel("View:")); ctrl.addWidget(self.combo_view)
        layout.addLayout(ctrl)

        disp = QHBoxLayout()
        disp.addWidget(QLabel("Bright:"))
        self.sl_bright = QSlider(Qt.Horizontal); self.sl_bright.setRange(-100,100); self.sl_bright.setValue(0)
        self.sl_bright.valueChanged.connect(self.refresh); disp.addWidget(self.sl_bright)
        disp.addWidget(QLabel("Contrast:"))
        self.sl_contrast = QSlider(Qt.Horizontal); self.sl_contrast.setRange(-100,100); self.sl_contrast.setValue(0)
        self.sl_contrast.valueChanged.connect(self.refresh); disp.addWidget(self.sl_contrast)
        disp.addWidget(QLabel("γ:"))
        self.sl_gamma = QDoubleSpinBox(); self.sl_gamma.setRange(0.1,5.0); self.sl_gamma.setValue(1.0)
        self.sl_gamma.setSingleStep(0.1); self.sl_gamma.valueChanged.connect(self.refresh)
        disp.addWidget(self.sl_gamma)
        disp.addWidget(QLabel("α:"))
        self.sl_alpha = QSlider(Qt.Horizontal); self.sl_alpha.setRange(0,100); self.sl_alpha.setValue(50)
        self.sl_alpha.valueChanged.connect(self.refresh); disp.addWidget(self.sl_alpha)
        layout.addLayout(disp)

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def set_volumes(self, vols: Dict[str, np.ndarray]):
        self._volumes = vols

    def set_overlay(self, vol: Optional[np.ndarray]):
        self._overlay = vol

    def update_ranges(self):
        s = self.store
        self.spin_t.setMaximum(max(0, s.n_t - 1))
        self.spin_z.setMaximum(max(0, s.n_z - 1))
        self.spin_c.setMaximum(max(0, s.n_channels - 1))

    def _adjust(self, img):
        f = img.astype(np.float64)
        if f.max() > f.min(): f = (f - f.min()) / (f.max() - f.min())
        f = np.power(np.clip(f, 1e-10, None), 1.0 / self.sl_gamma.value())
        c = self.sl_contrast.value() / 100.0
        f = np.clip((f - 0.5) * (1 + c) + 0.5, 0, 1)
        b = self.sl_bright.value() / 200.0
        return np.clip(f + b, 0, 1)

    def _colorize(self, gray, color):
        rgb = np.zeros((*gray.shape, 3), dtype=np.float64)
        for i in range(3): rgb[..., i] = gray * (color[i] / 255.0)
        return rgb

    def _get_slice(self, t, z, c):
        key = f"ch{c}"
        if key in self._volumes:
            vol = self._volumes[key]
            return vol[min(z, vol.shape[0]-1)].astype(np.float32)
        if self.store.raw is not None:
            return self.store.raw[min(t, self.store.n_t-1),
                                  min(z, self.store.n_z-1),
                                  min(c, self.store.n_channels-1)].astype(np.float32)
        return np.zeros((64, 64), dtype=np.float32)

    @Slot()
    def refresh(self):
        self.fig.clear()
        s = self.store
        if s.raw is None and not self._volumes:
            self.canvas.draw(); return

        t = self.spin_t.value(); z = self.spin_z.value(); c = self.spin_c.value()
        mode = self.combo_view.currentIndex()
        nc = s.n_channels or max(1, len(self._volumes))

        if mode == 0:
            ax = self.fig.add_subplot(111)
            raw = self._get_slice(t, z, c)
            adj = self._adjust(raw)
            clr = s.channel_colors[c % len(s.channel_colors)]
            ax.imshow(self._colorize(adj, clr), origin="lower")
            nm = s.metadata.channel_names[c] if s.metadata and c < len(s.metadata.channel_names) else f"Ch{c}"
            ax.set_title(f"{nm}  T{t} Z{z}", fontsize=10); ax.axis("off")

        elif mode == 1:
            axes = self.fig.subplots(1, max(1, nc), squeeze=False)[0]
            for ci in range(nc):
                raw = self._get_slice(t, z, ci)
                adj = self._adjust(raw)
                clr = s.channel_colors[ci % len(s.channel_colors)]
                axes[ci].imshow(self._colorize(adj, clr), origin="lower")
                nm = s.metadata.channel_names[ci] if s.metadata and ci < len(s.metadata.channel_names) else f"Ch{ci}"
                axes[ci].set_title(nm, fontsize=9); axes[ci].axis("off")

        elif mode == 2:
            ax = self.fig.add_subplot(111)
            yx = s.shape_yx if s.shape_yx[0] > 0 else (64, 64)
            composite = np.zeros((*yx, 3), dtype=np.float64)
            for ci in range(nc):
                raw = self._get_slice(t, z, ci)
                adj = self._adjust(raw)
                clr = s.channel_colors[ci % len(s.channel_colors)]
                composite += self._colorize(adj, clr)
            ax.imshow(np.clip(composite, 0, 1), origin="lower")
            ax.set_title(f"Composite T{t} Z{z}", fontsize=10); ax.axis("off")

        elif mode == 3:
            axes = self.fig.subplots(1, 2, squeeze=False)[0]
            yx = s.shape_yx if s.shape_yx[0] > 0 else (64, 64)
            composite = np.zeros((*yx, 3), dtype=np.float64)
            for ci in range(nc):
                raw = self._get_slice(t, z, ci)
                adj = self._adjust(raw)
                clr = s.channel_colors[ci % len(s.channel_colors)]
                composite += self._colorize(adj, clr)
            composite = np.clip(composite, 0, 1)
            axes[0].imshow(composite, origin="lower")
            axes[0].set_title("Composite", fontsize=9); axes[0].axis("off")

            axes[1].imshow(composite, origin="lower")
            alpha = self.sl_alpha.value() / 100.0
            ov = self._overlay if self._overlay is not None else s.combined_mask
            if ov is not None and z < ov.shape[0]:
                ms = ov[z]
                rgba = np.zeros((*ms.shape, 4), dtype=np.float64)
                for mc in s.mask_classes:
                    if mc.label_value == 0: continue
                    region = ms == mc.label_value
                    rgba[region, 0] = mc.color[0]/255; rgba[region, 1] = mc.color[1]/255
                    rgba[region, 2] = mc.color[2]/255; rgba[region, 3] = alpha
                axes[1].imshow(rgba, origin="lower")
                for mc in s.mask_classes:
                    if mc.label_value > 0:
                        axes[1].plot([], [], "s", color=np.array(mc.color)/255, label=mc.name, ms=8)
                axes[1].legend(loc="upper right", fontsize=7, framealpha=0.7)
            axes[1].set_title("Mask Overlay", fontsize=9); axes[1].axis("off")

        self.fig.tight_layout()
        self.canvas.draw()


# ═════════════════════════════════════════════════════════════════════════════
# STEP HISTORY LIST WIDGET
# ═════════════════════════════════════════════════════════════════════════════

class StepListWidget(QWidget):
    """
    Displays pipeline steps as a list with:
    - Click to preview that step's output
    - "Revert Here" to discard later steps
    - "Reset All" to start over
    """

    step_selected = Signal(int)    # user clicked a step to preview
    revert_requested = Signal(int) # revert to step N (−1 = base)
    reset_requested = Signal()

    def __init__(self, label: str = "Pipeline Steps"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{label}</b>"))
        self.btn_reset = QPushButton("⟲ Reset All")
        self.btn_reset.setFixedHeight(26)
        self.btn_reset.clicked.connect(lambda: self.reset_requested.emit())
        header.addWidget(self.btn_reset)
        layout.addLayout(header)

        self.list_w = QListWidget()
        self.list_w.currentRowChanged.connect(self._on_select)
        layout.addWidget(self.list_w)

        self.btn_revert = QPushButton("⮌ Revert to Selected Step")
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(self._on_revert)
        layout.addWidget(self.btn_revert)

        self.lbl_info = QLabel("No steps yet. Run an operation to begin.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #888;")
        layout.addWidget(self.lbl_info)

    def refresh(self, history: PipelineHistory):
        self.list_w.clear()
        # Base item
        base_item = QListWidgetItem("⓪  Base (raw data)")
        base_item.setData(Qt.UserRole, -1)
        base_item.setForeground(QBrush(QColor(120, 200, 120)))
        self.list_w.addItem(base_item)

        for i, step in enumerate(history.steps):
            icon = "●" if i == history.n_steps - 1 else "○"
            item = QListWidgetItem(f"{icon}  Step {i+1}: {step.summary()}")
            item.setData(Qt.UserRole, i)
            if i == history.n_steps - 1:
                item.setForeground(QBrush(QColor(100, 200, 255)))
            self.list_w.addItem(item)

        # Select last
        self.list_w.setCurrentRow(self.list_w.count() - 1)
        n = history.n_steps
        self.lbl_info.setText(f"{n} step{'s' if n != 1 else ''} completed. "
                              f"Select a step to preview, or revert to re-run from that point.")

    def _on_select(self, row):
        if row < 0: return
        item = self.list_w.item(row)
        step_idx = item.data(Qt.UserRole)
        self.btn_revert.setEnabled(True)
        self.step_selected.emit(step_idx)

    def _on_revert(self):
        row = self.list_w.currentRow()
        if row < 0: return
        item = self.list_w.item(row)
        step_idx = item.data(Qt.UserRole)
        self.revert_requested.emit(step_idx)


# ═════════════════════════════════════════════════════════════════════════════
# EXPORT MIXIN
# ═════════════════════════════════════════════════════════════════════════════

class ExportMixin:
    def _export_volume_dialog(self, volumes, store, parent_widget):
        if not HAS_TIFF:
            QMessageBox.warning(parent_widget, "Missing", "pip install tifffile"); return
        if not volumes:
            QMessageBox.warning(parent_widget, "No Data", "Nothing to export."); return
        msg = QMessageBox(parent_widget)
        msg.setWindowTitle("Export Format"); msg.setText("Choose export format:")
        btn_ind = msg.addButton("Individual TIFFs", QMessageBox.ActionRole)
        btn_rgb = msg.addButton("RGB TIFF (composite)", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Cancel); msg.exec()
        clicked = msg.clickedButton()
        out_dir = QFileDialog.getExistingDirectory(parent_widget, "Export Directory")
        if not out_dir: return
        if clicked == btn_ind:
            for key, vol in volumes.items():
                tifffile.imwrite(os.path.join(out_dir, f"{key}.tif"), vol.astype(np.float32))
            QMessageBox.information(parent_widget, "Done", f"Exported {len(volumes)} volumes.")
        elif clicked == btn_rgb:
            ref = list(volumes.values())[0]; nz, ny, nx = ref.shape
            rgb = np.zeros((nz, ny, nx, 3), dtype=np.float32)
            for ci, (_, vol) in enumerate(volumes.items()):
                clr = np.array(store.channel_colors[ci % len(store.channel_colors)]) / 255.0
                v = vol.astype(np.float32)
                if v.max() > v.min(): v = (v-v.min())/(v.max()-v.min())
                for ch in range(3): rgb[...,ch] += v * clr[ch]
            tifffile.imwrite(os.path.join(out_dir, "composite_rgb.tif"),
                             np.clip(rgb*255,0,255).astype(np.uint8), photometric="rgb")
            QMessageBox.information(parent_widget, "Done", "RGB stack exported.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: FILE IMPORT
# ═════════════════════════════════════════════════════════════════════════════

class ImportTab(QWidget):
    data_loaded = Signal()

    def __init__(self, store: DataStore, status: StatusWidget):
        super().__init__()
        self.store = store
        self.status = status
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        fg = QGroupBox("ND2 File")
        fl = QHBoxLayout(fg)
        self.file_lbl = QLabel("No file selected"); self.file_lbl.setMinimumWidth(400)
        btn = QPushButton("Browse…"); btn.clicked.connect(self._browse)
        fl.addWidget(self.file_lbl, 1); fl.addWidget(btn)
        layout.addWidget(fg)

        self.meta_txt = QTextEdit(); self.meta_txt.setReadOnly(True); self.meta_txt.setMaximumHeight(90)
        layout.addWidget(self.meta_txt)

        dg = QGroupBox("Dimension Ranges"); dl = QGridLayout(dg)
        self.dim_spins: Dict[str, Tuple[QSpinBox, QSpinBox]] = {}
        for row, (key, label) in enumerate([("T","Timelapse"),("Z","Z-Series"),
                ("C","Channels"),("M","Multi-Points"),("L","Large Images")]):
            dl.addWidget(QLabel(label), row, 0)
            s0, s1 = QSpinBox(), QSpinBox()
            for s in (s0, s1): s.setMinimum(0); s.setEnabled(False); s.valueChanged.connect(self._update_mem)
            dl.addWidget(QLabel("From:"), row, 1); dl.addWidget(s0, row, 2)
            dl.addWidget(QLabel("To:"), row, 3); dl.addWidget(s1, row, 4)
            self.dim_spins[key] = (s0, s1)
        layout.addWidget(dg)

        mg = QGroupBox("Memory"); ml = QHBoxLayout(mg)
        self.mem_lbl = QLabel("—"); self.mem_fit = QLabel("")
        ml.addWidget(self.mem_lbl, 1); ml.addWidget(self.mem_fit)
        layout.addWidget(mg)

        mog = QGroupBox("Load Mode"); mol = QHBoxLayout(mog)
        self.r_hyper = QRadioButton("Hyperstack (RAM)"); self.r_hyper.setChecked(True)
        self.r_export = QRadioButton("Direct Export (TIFFs)")
        mol.addWidget(self.r_hyper); mol.addWidget(self.r_export)
        layout.addWidget(mog)

        ccg = QGroupBox("Channel Colors"); self.color_layout = QHBoxLayout(ccg)
        self.color_btns: List[QPushButton] = []
        layout.addWidget(ccg)

        al = QHBoxLayout()
        self.btn_go = QPushButton("Load / Export"); self.btn_go.setEnabled(False)
        self.btn_go.clicked.connect(self._start)
        self.btn_reset = QPushButton("⟲ Reset All")
        self.btn_reset.clicked.connect(self._reset)
        al.addWidget(self.btn_go); al.addWidget(self.btn_reset)
        layout.addLayout(al)
        layout.addStretch()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open ND2", "", "ND2 (*.nd2);;All (*)")
        if not p: return
        self.file_lbl.setText(p)
        try:
            self.store.metadata = ND2Reader.read_metadata(p)
            self._populate(); self.btn_go.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _populate(self):
        m = self.store.metadata
        tots = {"T":m.n_timepoints,"Z":m.n_zslices,"C":m.n_channels,
                "M":m.n_multipoints,"L":m.n_large_images}
        for k,(s0,s1) in self.dim_spins.items():
            n = tots.get(k,1)
            s0.setMaximum(max(0,n-1)); s1.setMaximum(max(0,n-1))
            s0.setValue(0); s1.setValue(max(0,n-1)); s0.setEnabled(True); s1.setEnabled(True)
        ch = ", ".join(m.channel_names) if m.channel_names else "N/A"
        self.meta_txt.setPlainText(
            f"File: {Path(m.filepath).name}\n"
            f"T={m.n_timepoints} Z={m.n_zslices} C={m.n_channels} "
            f"M={m.n_multipoints} L={m.n_large_images}\n"
            f"{m.width}×{m.height} {m.dtype} | px={m.pixel_size_um:.4f}µm Δz={m.z_step_um:.4f}µm\n"
            f"Channels: {ch}")
        for b in self.color_btns:
            self.color_layout.removeWidget(b); b.deleteLater()
        self.color_btns.clear()
        for ci in range(m.n_channels):
            clr = self.store.channel_colors[ci % len(self.store.channel_colors)]
            name = m.channel_names[ci] if ci < len(m.channel_names) else f"Ch{ci}"
            b = QPushButton(name); b.setFixedHeight(28)
            b.setStyleSheet(f"background-color:rgb({clr[0]},{clr[1]},{clr[2]});"
                            f"color:{'black' if sum(clr)>380 else 'white'};font-weight:bold;")
            b.clicked.connect(functools.partial(self._pick_color, ci))
            self.color_layout.addWidget(b); self.color_btns.append(b)
        self._update_mem()

    def _pick_color(self, ci):
        cur = self.store.channel_colors[ci % len(self.store.channel_colors)]
        c = QColorDialog.getColor(QColor(*cur), self, f"Color for Ch{ci}")
        if c.isValid():
            while len(self.store.channel_colors) <= ci:
                self.store.channel_colors.append((255,255,255))
            self.store.channel_colors[ci] = (c.red(), c.green(), c.blue())
            self._populate()

    def _ranges(self):
        return {k:(s0.value(),s1.value()) for k,(s0,s1) in self.dim_spins.items()}

    def _update_mem(self):
        m = self.store.metadata
        if not m: return
        r = self._ranges(); est = m.estimate_bytes(r["T"],r["Z"],r["C"],r["M"],r["L"])
        oh = int(est*1.5)
        self.mem_lbl.setText(f"Data: {est/1e9:.2f} GB | +50%: {oh/1e9:.2f} GB")
        if HAS_PSUTIL:
            avail = psutil.virtual_memory().available; ok = oh < avail
            self.mem_fit.setText(f"{'✓ Fits' if ok else '✗ NO FIT'} ({avail/1e9:.1f} GB free)")
            self.mem_fit.setStyleSheet("color:green;font-weight:bold" if ok else "color:red;font-weight:bold")
            if not ok: self.r_export.setChecked(True)

    def _start(self):
        m = self.store.metadata; r = self._ranges()
        mode = LoadMode.HYPERSTACK if self.r_hyper.isChecked() else LoadMode.EXPORT
        if mode == LoadMode.EXPORT:
            d = QFileDialog.getExistingDirectory(self, "Export Dir")
            if not d: return
            def _run_export(_progress=None):
                ND2Reader.export_frames(m.filepath, d, r["T"],r["Z"],r["C"],r["M"],r["L"], callback=_progress)
                return None
            self._launch(_run_export, "Exporting frames…")
        else:
            def _run_load(_progress=None):
                return ND2Reader.load_hyperstack(m.filepath, r["T"],r["Z"],r["C"],r["M"],r["L"], callback=_progress)
            self._launch(_run_load, "Loading hyperstack…")

    def _launch(self, func, msg):
        self.btn_go.setEnabled(False)
        self.status.set_busy(msg)
        self._worker = GenericWorker(func)
        self._worker.progress.connect(self.status.set_progress)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _done(self, data):
        self.btn_go.setEnabled(True)
        if data is not None:
            self.store.raw = data
            # Seed preprocessing history base
            base = {}
            for ci in range(data.shape[2]):
                base[f"ch{ci}"] = data[0, :, ci, :, :].astype(np.float32)
            self.store.preprocess_history.set_base(base)
            self.store.segment_history = PipelineHistory()
            self.store.combined_mask = None
            self.status.set_done(f"Loaded: {data.shape} {data.dtype}")
        else:
            self.status.set_done("Export complete.")
        self.data_loaded.emit()

    def _on_error(self, e):
        self.btn_go.setEnabled(True)
        self.status.set_error(e.split('\n')[0])
        QMessageBox.critical(self, "Error", e)

    def _reset(self):
        self.store.raw = None
        self.store.reset_all()
        self.status.set_done("Reset — all data cleared.")
        self.data_loaded.emit()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: PREPROCESSING (step-by-step)
# ═════════════════════════════════════════════════════════════════════════════

class PreprocessTab(QWidget, ExportMixin):
    processing_done = Signal()

    def __init__(self, store: DataStore, status: StatusWidget):
        super().__init__()
        self.store = store
        self.status = status
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        layout = QVBoxLayout(self); layout.addWidget(splitter)

        # ── Left: controls + step list ──────────────────────────────────
        ctrl_w = QWidget(); ctrl = QVBoxLayout(ctrl_w)
        scr = QScrollArea(); scr.setWidgetResizable(True); scr.setWidget(ctrl_w)
        scr.setMaximumWidth(420); splitter.addWidget(scr)

        # Step history
        self.step_list = StepListWidget("Preprocessing Steps")
        self.step_list.step_selected.connect(self._on_step_selected)
        self.step_list.revert_requested.connect(self._on_revert)
        self.step_list.reset_requested.connect(self._on_reset)
        ctrl.addWidget(self.step_list)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); ctrl.addWidget(sep)
        ctrl.addWidget(QLabel("<b>Run Next Step:</b>"))

        # CLAHE
        g1 = QGroupBox("3D CLAHE")
        l1 = QFormLayout(g1)
        self.sp_clahe_k = QSpinBox(); self.sp_clahe_k.setRange(8,256); self.sp_clahe_k.setValue(64)
        self.sp_clahe_c = QDoubleSpinBox(); self.sp_clahe_c.setRange(0.001,0.1); self.sp_clahe_c.setValue(0.01)
        self.sp_clahe_c.setSingleStep(0.005)
        l1.addRow("Kernel:", self.sp_clahe_k); l1.addRow("Clip:", self.sp_clahe_c)
        self.btn_clahe = QPushButton("▶ Run CLAHE"); self.btn_clahe.clicked.connect(self._run_clahe)
        l1.addRow(self.btn_clahe)
        ctrl.addWidget(g1)

        # Deconvolution
        g2 = QGroupBox("Color Deconvolution")
        l2 = QVBoxLayout(g2)
        self.btn_deconv = QPushButton("▶ Run Deconvolution"); self.btn_deconv.clicked.connect(self._run_deconv)
        l2.addWidget(self.btn_deconv)
        ctrl.addWidget(g2)

        # Median
        g3 = QGroupBox(f"3D Median Filter {'(numba)' if HAS_NUMBA else '(scipy)'}")
        l3 = QFormLayout(g3)
        self.sp_med = QSpinBox(); self.sp_med.setRange(1,15); self.sp_med.setValue(3)
        l3.addRow("Kernel:", self.sp_med)
        self.btn_med = QPushButton("▶ Run Median"); self.btn_med.clicked.connect(self._run_median)
        l3.addRow(self.btn_med)
        ctrl.addWidget(g3)

        # LBP
        g4 = QGroupBox(f"3D LBP {'(numba)' if HAS_NUMBA else '(numpy)'}")
        l4 = QFormLayout(g4)
        self.sp_lbp_r = QSpinBox(); self.sp_lbp_r.setRange(1,5); self.sp_lbp_r.setValue(1)
        l4.addRow("Radius:", self.sp_lbp_r)
        self.btn_lbp = QPushButton("▶ Run LBP"); self.btn_lbp.clicked.connect(self._run_lbp)
        l4.addRow(self.btn_lbp)
        ctrl.addWidget(g4)

        # Export
        self.btn_export = QPushButton("💾 Export Current Result")
        self.btn_export.clicked.connect(self._export)
        ctrl.addWidget(self.btn_export)
        ctrl.addStretch()

        # ── Right: preview ──────────────────────────────────────────────
        self.preview = PreviewWidget(self.store)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)

    def on_data_loaded(self):
        h = self.store.preprocess_history
        self.step_list.refresh(h)
        self.preview.update_ranges()
        self.preview.set_volumes(h.current_volumes())
        self.preview.refresh()

    def _set_buttons_enabled(self, enabled: bool):
        for b in (self.btn_clahe, self.btn_deconv, self.btn_med, self.btn_lbp, self.btn_export):
            b.setEnabled(enabled)

    def _run_step(self, name, params, func):
        """Generic: run a single step on current head volumes."""
        h = self.store.preprocess_history
        if not h.has_base():
            QMessageBox.warning(self, "No Data", "Load data in the Import tab first."); return
        cur = h.current_volumes()
        self._set_buttons_enabled(False)
        self.status.set_busy(f"Running: {name}…")

        def _work(_progress=None):
            results = {}
            nc = len([k for k in cur if k.startswith("ch") and not k.startswith("ch_")])
            for i, (key, vol) in enumerate(cur.items()):
                results[key] = func(vol, _progress=lambda p: _progress(
                    int((i * 100 + p) / max(1, len(cur)))) if _progress else None)
                if _progress: _progress(int(100 * (i+1) / max(1, len(cur))))
            return results, name, params

        self._worker = GenericWorker(_work)
        self._worker.progress.connect(self.status.set_progress)
        self._worker.finished.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_step_done(self, result):
        results, name, params = result
        step = PipelineStep(name=name, params=params, outputs=results)
        self.store.preprocess_history.push(step)
        self._set_buttons_enabled(True)
        self.step_list.refresh(self.store.preprocess_history)
        self.preview.set_volumes(results)
        self.preview.refresh()
        self.status.set_done(f"✓ {name} complete ({self.store.preprocess_history.n_steps} steps)")
        self.processing_done.emit()

    def _on_error(self, e):
        self._set_buttons_enabled(True)
        self.status.set_error(e.split('\n')[0])
        QMessageBox.critical(self, "Error", e)

    # ── Individual step runners ─────────────────────────────────────────
    def _run_clahe(self):
        k = self.sp_clahe_k.value(); c = self.sp_clahe_c.value()
        self._run_step("CLAHE", {"kernel": k, "clip": c},
                       lambda vol, _progress=None: Accel.clahe_3d(vol, k, c, _progress))

    def _run_deconv(self):
        h = self.store.preprocess_history
        if not h.has_base():
            QMessageBox.warning(self, "No Data", "Load data first."); return
        cur = h.current_volumes()
        nc = len([k for k in cur if k.startswith("ch")])
        if nc < 2:
            QMessageBox.warning(self, "Need ≥2 channels", "Deconvolution needs multiple channels."); return

        self._set_buttons_enabled(False)
        self.status.set_busy("Running: Color Deconvolution…")

        def _work(_progress=None):
            stack = np.stack([cur[f"ch{i}"] for i in range(nc)], axis=0)
            unmixed = Accel.color_deconvolution(stack, _progress=_progress)
            results = dict(cur)  # copy
            for i in range(nc):
                results[f"ch{i}"] = unmixed[i]
            return results, "Color Deconvolution", {"channels": nc}

        self._worker = GenericWorker(_work)
        self._worker.progress.connect(self.status.set_progress)
        self._worker.finished.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _run_median(self):
        sz = self.sp_med.value()
        self._run_step("Median Filter", {"size": sz},
                       lambda vol, _progress=None: Accel.median_3d(vol, sz, _progress))

    def _run_lbp(self):
        r = self.sp_lbp_r.value()
        self._run_step("3D LBP", {"radius": r},
                       lambda vol, _progress=None: Accel.lbp_3d(vol, r, _progress))

    def _on_step_selected(self, step_idx):
        h = self.store.preprocess_history
        vols = h.volumes_at(step_idx)
        self.preview.set_volumes(vols)
        self.preview.refresh()

    def _on_revert(self, step_idx):
        h = self.store.preprocess_history
        h.revert_to(step_idx)
        self.step_list.refresh(h)
        self.preview.set_volumes(h.current_volumes())
        self.preview.refresh()
        self.status.set_done(f"Reverted to step {step_idx + 1}" if step_idx >= 0 else "Reverted to base")

    def _on_reset(self):
        self.store.preprocess_history.reset()
        self.step_list.refresh(self.store.preprocess_history)
        self.preview.set_volumes(self.store.preprocess_history.current_volumes())
        self.preview.refresh()
        self.status.set_done("Preprocessing reset to base.")

    def _export(self):
        vols = self.store.preprocess_history.current_volumes()
        self._export_volume_dialog(vols, self.store, self)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: SEGMENTATION (step-by-step)
# ═════════════════════════════════════════════════════════════════════════════

class SegmentTab(QWidget, ExportMixin):
    segmentation_done = Signal()

    def __init__(self, store: DataStore, status: StatusWidget):
        super().__init__()
        self.store = store
        self.status = status
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        layout = QVBoxLayout(self); layout.addWidget(splitter)

        ctrl_w = QWidget(); ctrl = QVBoxLayout(ctrl_w)
        scr = QScrollArea(); scr.setWidgetResizable(True); scr.setWidget(ctrl_w)
        scr.setMaximumWidth(440); splitter.addWidget(scr)

        # Step history
        self.step_list = StepListWidget("Segmentation Steps")
        self.step_list.step_selected.connect(self._on_step_selected)
        self.step_list.revert_requested.connect(self._on_revert)
        self.step_list.reset_requested.connect(self._on_reset)
        ctrl.addWidget(self.step_list)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); ctrl.addWidget(sep)
        ctrl.addWidget(QLabel("<b>Run Next Step:</b>"))

        # Threshold
        g1 = QGroupBox("Threshold")
        l1 = QFormLayout(g1)
        self.combo_thresh = QComboBox(); self.combo_thresh.addItems(["otsu","li","yen","triangle"])
        l1.addRow("Method:", self.combo_thresh)
        self.btn_thresh = QPushButton("▶ Run Threshold"); self.btn_thresh.clicked.connect(self._run_thresh)
        l1.addRow(self.btn_thresh)
        ctrl.addWidget(g1)

        # Hole closing
        g2 = QGroupBox("Close Holes")
        l2 = QFormLayout(g2)
        self.sp_hmin = QDoubleSpinBox(); self.sp_hmin.setRange(0.1,500); self.sp_hmin.setValue(10); self.sp_hmin.setSuffix(" µm")
        self.sp_hmax = QDoubleSpinBox(); self.sp_hmax.setRange(1,5000); self.sp_hmax.setValue(200); self.sp_hmax.setSuffix(" µm")
        l2.addRow("Min:", self.sp_hmin); l2.addRow("Max:", self.sp_hmax)
        self.btn_holes = QPushButton("▶ Run Close Holes"); self.btn_holes.clicked.connect(self._run_holes)
        l2.addRow(self.btn_holes)
        ctrl.addWidget(g2)

        # Morphological
        g3 = QGroupBox("Morphological Close / Open")
        l3 = QFormLayout(g3)
        self.sp_morph_r = QSpinBox(); self.sp_morph_r.setRange(1,20); self.sp_morph_r.setValue(3)
        self.combo_morph = QComboBox(); self.combo_morph.addItems(["Close", "Open"])
        l3.addRow("Radius:", self.sp_morph_r); l3.addRow("Op:", self.combo_morph)
        self.btn_morph = QPushButton("▶ Run Morphology"); self.btn_morph.clicked.connect(self._run_morph)
        l3.addRow(self.btn_morph)
        ctrl.addWidget(g3)

        # Blob detection
        g4 = QGroupBox("Blob Detection (LoG)")
        l4 = QFormLayout(g4)
        self.sp_bmin = QDoubleSpinBox(); self.sp_bmin.setRange(1,100); self.sp_bmin.setValue(5)
        self.sp_bmax = QDoubleSpinBox(); self.sp_bmax.setRange(5,200); self.sp_bmax.setValue(30)
        l4.addRow("σ_min:", self.sp_bmin); l4.addRow("σ_max:", self.sp_bmax)
        self.btn_blob = QPushButton("▶ Run Blob Detection"); self.btn_blob.clicked.connect(self._run_blob)
        l4.addRow(self.btn_blob)
        ctrl.addWidget(g4)

        # Watershed
        g5 = QGroupBox("Size-Constrained Watershed")
        l5 = QFormLayout(g5)
        self.sp_ws_rmin = QDoubleSpinBox(); self.sp_ws_rmin.setRange(1,500); self.sp_ws_rmin.setValue(25); self.sp_ws_rmin.setSuffix(" µm")
        self.sp_ws_rmax = QDoubleSpinBox(); self.sp_ws_rmax.setRange(5,1000); self.sp_ws_rmax.setValue(60); self.sp_ws_rmax.setSuffix(" µm")
        l5.addRow("Min R:", self.sp_ws_rmin); l5.addRow("Max R:", self.sp_ws_rmax)
        self.btn_ws = QPushButton("▶ Run Watershed"); self.btn_ws.clicked.connect(self._run_ws)
        l5.addRow(self.btn_ws)
        ctrl.addWidget(g5)

        # Template matching
        g6 = QGroupBox("Template Matching")
        l6 = QFormLayout(g6)
        self.sp_tmaj = QDoubleSpinBox(); self.sp_tmaj.setRange(1,1000); self.sp_tmaj.setValue(100); self.sp_tmaj.setSuffix(" µm")
        self.sp_tmin = QDoubleSpinBox(); self.sp_tmin.setRange(1,1000); self.sp_tmin.setValue(75); self.sp_tmin.setSuffix(" µm")
        self.combo_shape = QComboBox(); self.combo_shape.addItems(["ellipsoid","sphere","cuboid"])
        l6.addRow("Major:", self.sp_tmaj); l6.addRow("Minor:", self.sp_tmin)
        l6.addRow("Shape:", self.combo_shape)
        self.btn_tmpl = QPushButton("▶ Run Template Match"); self.btn_tmpl.clicked.connect(self._run_tmpl)
        l6.addRow(self.btn_tmpl)
        ctrl.addWidget(g6)

        # Export
        self.btn_export = QPushButton("💾 Export Current Result")
        self.btn_export.clicked.connect(self._export)
        ctrl.addWidget(self.btn_export)
        ctrl.addStretch()

        # Preview
        self.preview = PreviewWidget(self.store)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)

    def on_data_loaded(self):
        # Seed segmentation base from preprocessing output
        pre_vols = self.store.preprocess_history.current_volumes()
        if pre_vols:
            self.store.segment_history.set_base(pre_vols)
        h = self.store.segment_history
        self.step_list.refresh(h)
        self.preview.update_ranges()
        self.preview.set_volumes(h.current_volumes())
        self.preview.refresh()

    def _all_buttons(self):
        return [self.btn_thresh, self.btn_holes, self.btn_morph,
                self.btn_blob, self.btn_ws, self.btn_tmpl, self.btn_export]

    def _set_buttons_enabled(self, en):
        for b in self._all_buttons(): b.setEnabled(en)

    def _run_step(self, name, params, func):
        h = self.store.segment_history
        if not h.has_base():
            QMessageBox.warning(self, "No Data", "Preprocess data first."); return
        cur = h.current_volumes()
        self._set_buttons_enabled(False)
        self.status.set_busy(f"Running: {name}…")

        def _work(_progress=None):
            results = {}
            total = max(1, len(cur))
            for i, (key, vol) in enumerate(cur.items()):
                results[key] = func(vol, _progress=lambda p: _progress(
                    int((i*100+p)/total)) if _progress else None)
                if _progress: _progress(int(100*(i+1)/total))
            return results, name, params

        self._worker = GenericWorker(_work)
        self._worker.progress.connect(self.status.set_progress)
        self._worker.finished.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _run_step_custom(self, name, params, func):
        """For steps that need custom multi-channel handling (watershed, etc.)."""
        h = self.store.segment_history
        if not h.has_base():
            QMessageBox.warning(self, "No Data", "Preprocess data first."); return
        cur = h.current_volumes()
        self._set_buttons_enabled(False)
        self.status.set_busy(f"Running: {name}…")

        def _work(_progress=None):
            return func(cur, _progress), name, params

        self._worker = GenericWorker(_work)
        self._worker.progress.connect(self.status.set_progress)
        self._worker.finished.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_step_done(self, result):
        results, name, params = result
        step = PipelineStep(name=name, params=params, outputs=results)
        self.store.segment_history.push(step)
        self._set_buttons_enabled(True)
        self.step_list.refresh(self.store.segment_history)
        self.preview.set_volumes(results)
        self.preview.refresh()
        self.status.set_done(f"✓ {name} complete ({self.store.segment_history.n_steps} steps)")
        self.segmentation_done.emit()

    def _on_error(self, e):
        self._set_buttons_enabled(True)
        self.status.set_error(e.split('\n')[0])
        QMessageBox.critical(self, "Error", e)

    # ── Step runners ────────────────────────────────────────────────────
    def _run_thresh(self):
        m = self.combo_thresh.currentText()
        self._run_step("Threshold", {"method": m},
                       lambda vol, _progress=None: Accel.threshold(vol, m, _progress).astype(np.float32))

    def _run_holes(self):
        mn, mx = self.sp_hmin.value(), self.sp_hmax.value()
        vox = self.store.metadata.voxel_size_um if self.store.metadata else (1,1,1)
        self._run_step("Close Holes", {"min_µm": mn, "max_µm": mx},
                       lambda vol, _progress=None: Accel.close_holes(
                           vol, mn, mx, vox, _progress).astype(np.float32))

    def _run_morph(self):
        r = self.sp_morph_r.value(); op = self.combo_morph.currentText()
        fn = Accel.morph_close_3d if op == "Close" else Accel.morph_open_3d
        self._run_step(f"Morpho {op}", {"radius": r},
                       lambda vol, _progress=None: fn(vol, r, _progress).astype(np.float32))

    def _run_blob(self):
        bmin, bmax = self.sp_bmin.value(), self.sp_bmax.value()
        def _func(cur, _progress=None):
            results = dict(cur)
            for key, vol in cur.items():
                if key.startswith("ch"):
                    blobs = Accel.blob_detect_3d(vol, bmin, bmax, _progress=_progress)
                    results[f"blob_{key}"] = blobs.astype(np.float32)
            return results
        self._run_step_custom("Blob Detection", {"σ_min": bmin, "σ_max": bmax}, _func)

    def _run_ws(self):
        rmin, rmax = self.sp_ws_rmin.value(), self.sp_ws_rmax.value()
        vox = self.store.metadata.voxel_size_um if self.store.metadata else (1,1,1)
        def _func(cur, _progress=None):
            results = dict(cur)
            for key, vol in cur.items():
                if key.startswith("ch"):
                    binary = (vol > 0.5).astype(np.uint8) if vol.max() <= 1 else Accel.threshold(vol, "otsu")
                    lbl = Accel.watershed_size_constrained(binary, vol, vox, rmin, rmax, _progress)
                    results[f"ws_{key}"] = lbl.astype(np.float32)
            return results
        self._run_step_custom("Watershed", {"min_R": rmin, "max_R": rmax}, _func)

    def _run_tmpl(self):
        maj, mn = self.sp_tmaj.value(), self.sp_tmin.value()
        px = self.store.metadata.pixel_size_um if self.store.metadata else 1.0
        shape = self.combo_shape.currentText()
        def _func(cur, _progress=None):
            results = dict(cur)
            for key, vol in cur.items():
                if key.startswith("ch"):
                    score = Accel.template_match_ellipsoid(vol, maj, mn, px, shape, _progress)
                    results[f"tmpl_{key}"] = score
            return results
        self._run_step_custom("Template Match", {"major": maj, "minor": mn, "shape": shape}, _func)

    def _on_step_selected(self, step_idx):
        h = self.store.segment_history
        self.preview.set_volumes(h.volumes_at(step_idx))
        self.preview.refresh()

    def _on_revert(self, step_idx):
        h = self.store.segment_history
        h.revert_to(step_idx)
        self.step_list.refresh(h)
        self.preview.set_volumes(h.current_volumes())
        self.preview.refresh()
        self.status.set_done(f"Reverted to step {step_idx+1}" if step_idx >= 0 else "Reverted to base")

    def _on_reset(self):
        self.store.segment_history.reset()
        self.step_list.refresh(self.store.segment_history)
        self.preview.set_volumes(self.store.segment_history.current_volumes())
        self.preview.refresh()
        self.status.set_done("Segmentation reset to base.")

    def _export(self):
        vols = self.store.segment_history.current_volumes()
        self._export_volume_dialog(
            {k: v.astype(np.float32) for k, v in vols.items()}, self.store, self)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: MASK GENERATION
# ═════════════════════════════════════════════════════════════════════════════

class MaskTab(QWidget, ExportMixin):
    def __init__(self, store: DataStore, status: StatusWidget):
        super().__init__()
        self.store = store
        self.status = status
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        layout = QVBoxLayout(self); layout.addWidget(splitter)

        ctrl_w = QWidget(); ctrl = QVBoxLayout(ctrl_w)
        scr = QScrollArea(); scr.setWidgetResizable(True); scr.setWidget(ctrl_w)
        scr.setMaximumWidth(420); splitter.addWidget(scr)

        g1 = QGroupBox("Class ↔ Source Mapping")
        l1 = QGridLayout(g1)
        l1.addWidget(QLabel("Class"), 0, 0); l1.addWidget(QLabel("Source"), 0, 1)
        l1.addWidget(QLabel("Color"), 0, 2)
        self.class_rows: List[Tuple[QLabel, QComboBox, QPushButton]] = []
        for i, mc in enumerate(self.store.mask_classes):
            if mc.label_value == 0: continue
            lbl = QLabel(mc.name)
            combo = QComboBox(); combo.addItem("(none)")
            cbtn = QPushButton("■"); cbtn.setFixedSize(28, 28)
            cbtn.setStyleSheet(f"background:rgb({mc.color[0]},{mc.color[1]},{mc.color[2]})")
            cbtn.clicked.connect(functools.partial(self._pick_class_color, i))
            l1.addWidget(lbl, i, 0); l1.addWidget(combo, i, 1); l1.addWidget(cbtn, i, 2)
            self.class_rows.append((lbl, combo, cbtn))
        ctrl.addWidget(g1)

        g2 = QGroupBox("Overlap Priority")
        l2 = QVBoxLayout(g2)
        l2.addWidget(QLabel("Later classes overwrite earlier ones in overlap regions.\n"
                            "Order: Functional → Inert → Cells"))
        ctrl.addWidget(g2)

        self.btn_build = QPushButton("▶ Build Combined Mask")
        self.btn_build.clicked.connect(self._build)
        ctrl.addWidget(self.btn_build)

        self.btn_reset = QPushButton("⟲ Reset Mask")
        self.btn_reset.clicked.connect(self._reset_mask)
        ctrl.addWidget(self.btn_reset)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); ctrl.addWidget(sep)
        ctrl.addWidget(QLabel("<b>Export:</b>"))
        self.btn_elbl = QPushButton("💾 Label TIFF"); self.btn_elbl.clicked.connect(self._export_label)
        self.btn_ergb = QPushButton("💾 RGB TIFF"); self.btn_ergb.clicked.connect(self._export_rgb)
        self.btn_eind = QPushButton("💾 Per-slice TIFFs"); self.btn_eind.clicked.connect(self._export_slices)
        ctrl.addWidget(self.btn_elbl); ctrl.addWidget(self.btn_ergb); ctrl.addWidget(self.btn_eind)
        ctrl.addStretch()

        self.preview = PreviewWidget(self.store)
        self.preview.combo_view.setCurrentIndex(3)
        splitter.addWidget(self.preview); splitter.setStretchFactor(1, 1)

    def on_data_loaded(self):
        self._update_combos()
        self.preview.update_ranges()
        self.preview.set_volumes(self.store.preprocess_history.current_volumes())
        self.preview.refresh()

    def _update_combos(self):
        sources = ["(none)"]
        seg = self.store.segment_history.current_volumes()
        for k in sorted(seg.keys()): sources.append(f"seg:{k}")
        pre = self.store.preprocess_history.current_volumes()
        for k in sorted(pre.keys()):
            if not k.startswith("lbp") and not k.startswith("deconv"):
                sources.append(f"pre:{k}")
        for _, combo, _ in self.class_rows:
            cur = combo.currentText(); combo.clear()
            for src in sources: combo.addItem(src)
            idx = combo.findText(cur)
            if idx >= 0: combo.setCurrentIndex(idx)

    def _pick_class_color(self, class_idx):
        mc = self.store.mask_classes[class_idx]
        c = QColorDialog.getColor(QColor(*mc.color), self, f"Color for {mc.name}")
        if c.isValid():
            mc.color = (c.red(), c.green(), c.blue())
            self.class_rows[class_idx - 1][2].setStyleSheet(
                f"background:rgb({c.red()},{c.green()},{c.blue()})")

    def _resolve(self, src):
        if src == "(none)": return None
        parts = src.split(":", 1)
        if len(parts) != 2: return None
        kind, key = parts
        if kind == "seg":
            vols = self.store.segment_history.current_volumes()
            if key in vols:
                v = vols[key]
                return (v > 0.5).astype(np.uint8) if v.max() <= 1 else Accel.threshold(v, "otsu")
        if kind == "pre":
            vols = self.store.preprocess_history.current_volumes()
            if key in vols:
                return Accel.threshold(vols[key], "otsu")
        return None

    def _build(self):
        binaries = {}; priority = []
        for (lbl, combo, _), mc in zip(
            self.class_rows, [c for c in self.store.mask_classes if c.label_value > 0]):
            vol = self._resolve(combo.currentText())
            if vol is not None:
                binaries[mc.name] = vol; priority.append(mc.name)
        if not binaries:
            QMessageBox.warning(self, "Nothing", "Assign at least one source."); return
        ref = list(binaries.values())[0]
        mask = np.zeros(ref.shape, dtype=np.uint8)
        for name in priority:
            mc = next(c for c in self.store.mask_classes if c.name == name)
            mask[binaries[name] > 0] = mc.label_value
        self.store.combined_mask = mask
        self.preview.set_overlay(mask)
        self.preview.combo_view.setCurrentIndex(3)
        self.preview.refresh()
        self.status.set_done(f"Mask built: classes {np.unique(mask).tolist()}")

    def _reset_mask(self):
        self.store.combined_mask = None
        self.preview.set_overlay(None)
        self.preview.refresh()
        self.status.set_done("Mask cleared.")

    def _export_label(self):
        if self.store.combined_mask is None:
            QMessageBox.warning(self, "No Mask", "Build first."); return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "mask_labels.tif", "TIFF (*.tif)")
        if p:
            tifffile.imwrite(p, self.store.combined_mask.astype(np.uint8))
            self.status.set_done(f"Label mask → {p}")

    def _export_rgb(self):
        if self.store.combined_mask is None:
            QMessageBox.warning(self, "No Mask", "Build first."); return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "mask_rgb.tif", "TIFF (*.tif)")
        if not p: return
        m = self.store.combined_mask
        rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
        for mc in self.store.mask_classes:
            region = m == mc.label_value
            rgb[region, 0] = mc.color[0]; rgb[region, 1] = mc.color[1]; rgb[region, 2] = mc.color[2]
        tifffile.imwrite(p, rgb, photometric="rgb")
        self.status.set_done(f"RGB mask → {p}")

    def _export_slices(self):
        if self.store.combined_mask is None:
            QMessageBox.warning(self, "No Mask", "Build first."); return
        d = QFileDialog.getExistingDirectory(self, "Output Dir")
        if not d: return
        m = self.store.combined_mask
        for z in range(m.shape[0]):
            sl = m[z]; rgb = np.zeros((*sl.shape, 3), dtype=np.uint8)
            for mc in self.store.mask_classes:
                region = sl == mc.label_value
                rgb[region, 0] = mc.color[0]; rgb[region, 1] = mc.color[1]; rgb[region, 2] = mc.color[2]
            tifffile.imwrite(os.path.join(d, f"mask_Z{z:04d}.tif"), rgb, photometric="rgb")
        self.status.set_done(f"Exported {m.shape[0]} slices → {d}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ND2 Microscopy Processor & Mask Generator v3")
        self.setMinimumSize(1350, 880)

        # Global status widget in status bar
        self.status_widget = StatusWidget()
        self.statusBar().addPermanentWidget(self.status_widget, 1)

        self.store = DataStore()
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)

        self.tab_import     = ImportTab(self.store, self.status_widget)
        self.tab_preprocess = PreprocessTab(self.store, self.status_widget)
        self.tab_segment    = SegmentTab(self.store, self.status_widget)
        self.tab_mask       = MaskTab(self.store, self.status_widget)

        self.tabs.addTab(self.tab_import,     "📂 Import")
        self.tabs.addTab(self.tab_preprocess, "⚙ Preprocess")
        self.tabs.addTab(self.tab_segment,    "🔬 Segment")
        self.tabs.addTab(self.tab_mask,       "🎨 Mask")

        self.tab_import.data_loaded.connect(self._on_data)
        self.tab_preprocess.processing_done.connect(self._on_preprocess)
        self.tab_segment.segmentation_done.connect(self._on_segment)

    def _on_data(self):
        self.tab_preprocess.on_data_loaded()
        self.tab_segment.on_data_loaded()
        self.tab_mask.on_data_loaded()

    def _on_preprocess(self):
        self.tab_segment.on_data_loaded()
        self.tab_mask.on_data_loaded()
        self.tab_mask._update_combos()

    def _on_segment(self):
        self.tab_mask.on_data_loaded()
        self.tab_mask._update_combos()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(48, 48, 48))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(32, 32, 32))
    pal.setColor(QPalette.AlternateBase, QColor(48, 48, 48))
    pal.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    pal.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
    pal.setColor(QPalette.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.Button, QColor(55, 55, 55))
    pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.BrightText, QColor(255, 60, 60))
    pal.setColor(QPalette.Highlight, QColor(42, 130, 218))
    pal.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)

    numba_info = "✗"
    if HAS_NUMBA:
        try:
            from numba import config as _nc
            numba_info = f"✓ (threading: {_nc.THREADING_LAYER})"
        except Exception:
            numba_info = "✓ (threading: unknown)"
    print(f"Numba: {numba_info} | Workers: {N_WORKERS} | "
          f"nd2: {'✓' if HAS_ND2 else '✗'} | skimage: {'✓' if HAS_SKIMAGE else '✗'}")
    print("  Tip: For best numba performance, install TBB: pip install tbb")

    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()