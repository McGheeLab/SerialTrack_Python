"""
Images Page — Load images, preview slices, apply enhancement pipeline.

Supports: ND2, TIFF stacks (single-file multi-frame), TIFF series,
MAT files, NPY, DICOM (.dcm), MRC/MRC2, HDF5.
Enhancement plugins are loaded from the plugin registry.
"""
from __future__ import annotations
from typing import Optional, List
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QFileDialog, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QFrame, QLineEdit,
    QFormLayout, QCheckBox, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QThread

from widgets.common import ImageViewer, ParamEditor, MplCanvas
from core.plugin_registry import PluginBase, EnhancementPlugin
from core.settings import Settings


# Supported extensions for file dialog and folder scanning
SUPPORTED_EXTS = {
    ".tif", ".tiff",  # TIFF single or stack
    ".nd2",           # Nikon ND2
    ".mat",           # MATLAB
    ".npy", ".npz",   # NumPy
    ".dcm",           # DICOM
    ".mrc", ".mrc2",  # MRC electron microscopy
    ".h5", ".hdf5",   # HDF5
    ".raw",           # Raw binary (needs metadata)
}

FILTER_STRING = (
    "All Supported (*.tif *.tiff *.nd2 *.mat *.npy *.npz *.dcm *.mrc *.h5 *.hdf5);;"
    "TIFF (*.tif *.tiff);;"
    "ND2 (*.nd2);;"
    "MATLAB (*.mat);;"
    "NumPy (*.npy *.npz);;"
    "DICOM (*.dcm);;"
    "MRC (*.mrc *.mrc2);;"
    "HDF5 (*.h5 *.hdf5);;"
    "All Files (*)"
)


class LoadWorker(QThread):
    """Background thread for loading images."""
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int, int)
    log = Signal(str)

    def __init__(self, paths, load_mode, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.load_mode = load_mode

    def run(self):
        try:
            volumes = []
            total = len(self.paths)
            for i, p in enumerate(self.paths):
                p = Path(p)
                ext = p.suffix.lower()
                self.log.emit(f"Loading {p.name} ({i+1}/{total})...")

                if ext in (".tif", ".tiff"):
                    volumes.extend(self._load_tiff(p))

                elif ext == ".mat":
                    import scipy.io as sio
                    data = sio.loadmat(str(p), simplify_cells=True)
                    keys = [k for k in data if not k.startswith("_")]
                    vol = np.asarray(data[keys[0]])
                    if vol.ndim == 3:
                        vol = vol.transpose(1, 0, 2)
                    volumes.append(vol.astype(np.float64))

                elif ext == ".npy":
                    volumes.append(np.load(str(p)).astype(np.float64))

                elif ext == ".npz":
                    npz = np.load(str(p))
                    keys = list(npz.keys())
                    if keys:
                        vol = npz[keys[0]].astype(np.float64)
                        volumes.append(vol)

                elif ext == ".nd2":
                    volumes.extend(self._load_nd2(p))

                elif ext == ".dcm":
                    volumes.extend(self._load_dicom(p))

                elif ext in (".mrc", ".mrc2"):
                    volumes.extend(self._load_mrc(p))

                elif ext in (".h5", ".hdf5"):
                    volumes.extend(self._load_hdf5(p))

                else:
                    self.log.emit(f"  Skipping unsupported format: {ext}")

                self.progress.emit(i + 1, total)

            if not volumes:
                self.error.emit("No volumes could be loaded from the selected files.")
                return

            self.log.emit(f"Loaded {len(volumes)} volumes successfully.")
            self.finished.emit(volumes)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _load_tiff(self, p: Path) -> List[np.ndarray]:
        """Load TIFF: handles single images, stacks, and multi-series."""
        import tifffile
        vols = []
        img = tifffile.imread(str(p)).astype(np.float64)

        if img.ndim == 2:
            # Single 2D image
            vols.append(img.T)  # (x, y)

        elif img.ndim == 3:
            # Could be (T, Y, X) time series or (Z, Y, X) volume
            # Heuristic: if first dim is small (<10) treat as time series of 2D
            # Otherwise treat as single 3D volume
            if img.shape[0] <= 10 and img.shape[0] < img.shape[1]:
                # Time series of 2D images
                for t in range(img.shape[0]):
                    vols.append(img[t].T)
                self.log.emit(f"  Interpreted as {img.shape[0]}-frame 2D series")
            else:
                # Single 3D volume: (Z, Y, X) → (X, Y, Z)
                vols.append(img.transpose(2, 1, 0))
                self.log.emit(f"  Interpreted as 3D volume {img.shape}")

        elif img.ndim == 4:
            # (T, Z, Y, X) time series of volumes
            for t in range(img.shape[0]):
                vols.append(img[t].transpose(2, 1, 0))
            self.log.emit(f"  Loaded {img.shape[0]}-frame 3D series")

        elif img.ndim == 5:
            # (T, C, Z, Y, X) — take first channel
            self.log.emit(f"  5D data detected, using first channel")
            for t in range(img.shape[0]):
                vols.append(img[t, 0].transpose(2, 1, 0))

        return vols

    def _load_nd2(self, p: Path) -> List[np.ndarray]:
        """Load Nikon ND2 files."""
        vols = []
        try:
            import nd2
            with nd2.ND2File(str(p)) as f:
                data = f.asarray().astype(np.float64)
                self.log.emit(f"  ND2 raw shape: {data.shape}, ndim={data.ndim}")

                if data.ndim == 2:
                    vols.append(data.T)
                elif data.ndim == 3:
                    # (Z, Y, X) single volume
                    vols.append(data.transpose(2, 1, 0))
                elif data.ndim == 4:
                    # (T, Z, Y, X) or (C, Z, Y, X)
                    # If first dim is small and matches channel count, assume channels
                    if data.shape[0] <= 4:
                        # Assume (C, Z, Y, X) — use first channel
                        vols.append(data[0].transpose(2, 1, 0))
                        self.log.emit(f"  Interpreted as {data.shape[0]} channels, using ch0")
                    else:
                        for t in range(data.shape[0]):
                            vols.append(data[t].transpose(2, 1, 0))
                        self.log.emit(f"  Loaded {data.shape[0]} time points")
                elif data.ndim == 5:
                    # (T, C, Z, Y, X)
                    self.log.emit(f"  5D ND2: {data.shape}, using ch0")
                    for t in range(data.shape[0]):
                        vols.append(data[t, 0].transpose(2, 1, 0))
                else:
                    vols.append(data)

        except ImportError:
            self.error.emit(
                "nd2 package not installed. Install with: pip install nd2")
        except Exception as e:
            self.log.emit(f"  ND2 load error: {e}")
            # Fallback: try tifffile
            self.log.emit("  Attempting tifffile fallback...")
            try:
                import tifffile
                img = tifffile.imread(str(p)).astype(np.float64)
                if img.ndim == 3:
                    vols.append(img.transpose(2, 1, 0))
                elif img.ndim == 2:
                    vols.append(img.T)
            except Exception:
                pass

        return vols

    def _load_dicom(self, p: Path) -> List[np.ndarray]:
        """Load DICOM files."""
        vols = []
        try:
            import pydicom
            ds = pydicom.dcmread(str(p))
            arr = ds.pixel_array.astype(np.float64)
            if arr.ndim == 2:
                vols.append(arr.T)
            elif arr.ndim == 3:
                vols.append(arr.transpose(2, 1, 0))
            self.log.emit(f"  DICOM loaded: {arr.shape}")
        except ImportError:
            self.log.emit("  pydicom not installed — skipping DICOM")
        except Exception as e:
            self.log.emit(f"  DICOM error: {e}")
        return vols

    def _load_mrc(self, p: Path) -> List[np.ndarray]:
        """Load MRC/MRC2 electron microscopy files."""
        vols = []
        try:
            import mrcfile
            with mrcfile.open(str(p), mode='r') as mrc:
                arr = mrc.data.astype(np.float64)
                if arr.ndim == 2:
                    vols.append(arr.T)
                elif arr.ndim == 3:
                    vols.append(arr.transpose(2, 1, 0))
                self.log.emit(f"  MRC loaded: {arr.shape}")
        except ImportError:
            self.log.emit("  mrcfile not installed — skipping MRC")
        except Exception as e:
            self.log.emit(f"  MRC error: {e}")
        return vols

    def _load_hdf5(self, p: Path) -> List[np.ndarray]:
        """Load HDF5 files (first dataset found)."""
        vols = []
        try:
            import h5py
            with h5py.File(str(p), 'r') as f:
                # Find first dataset
                def find_datasets(group, path=""):
                    datasets = []
                    for key in group:
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            datasets.append(f"{path}/{key}")
                        elif isinstance(item, h5py.Group):
                            datasets.extend(find_datasets(item, f"{path}/{key}"))
                    return datasets

                ds_paths = find_datasets(f)
                if ds_paths:
                    arr = np.array(f[ds_paths[0]]).astype(np.float64)
                    if arr.ndim == 2:
                        vols.append(arr.T)
                    elif arr.ndim == 3:
                        vols.append(arr.transpose(2, 1, 0))
                    elif arr.ndim == 4:
                        for t in range(arr.shape[0]):
                            vols.append(arr[t].transpose(2, 1, 0))
                    self.log.emit(f"  HDF5 dataset '{ds_paths[0]}': shape {arr.shape}")
        except ImportError:
            self.log.emit("  h5py not installed — skipping HDF5")
        except Exception as e:
            self.log.emit(f"  HDF5 error: {e}")
        return vols


class EnhanceWorker(QThread):
    """Background thread for running enhancement pipeline."""
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int)
    step_info = Signal(str)

    def __init__(self, volumes, pipeline, parent=None):
        super().__init__(parent)
        self.volumes = volumes
        self.pipeline = pipeline  # List of (plugin_class, params)

    def run(self):
        try:
            results = []
            for vi, vol in enumerate(self.volumes):
                current = vol.copy()
                for si, (plugin_cls, params) in enumerate(self.pipeline):
                    plugin = plugin_cls()
                    self.step_info.emit(
                        f"Volume {vi+1}/{len(self.volumes)}: {plugin.name}"
                    )
                    current = plugin.execute(
                        current, params,
                        progress_cb=lambda p: self.progress.emit(p)
                    )
                results.append(current)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ImagesPage(QWidget):
    """Image loading & enhancement page."""
    images_loaded = Signal(list)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._raw_volumes: List[np.ndarray] = []
        self._enhanced_volumes: List[np.ndarray] = []
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel: controls ──────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(380)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(8)

        # Load section
        load_grp = QGroupBox("Load Images")
        load_lay = QVBoxLayout(load_grp)

        self.btn_load_files = QPushButton("Load Files...")
        self.btn_load_files.setObjectName("primaryBtn")
        self.btn_load_files.clicked.connect(self._load_files)
        load_lay.addWidget(self.btn_load_files)

        self.btn_load_folder = QPushButton("Load Folder...")
        self.btn_load_folder.clicked.connect(self._load_folder)
        load_lay.addWidget(self.btn_load_folder)

        # Supported formats info
        formats_lbl = QLabel(
            "Supported: .nd2 · .tif/.tiff · .mat · .npy · .dcm · .mrc · .h5")
        formats_lbl.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        formats_lbl.setWordWrap(True)
        load_lay.addWidget(formats_lbl)

        self.load_info = QLabel("No images loaded")
        self.load_info.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        load_lay.addWidget(self.load_info)

        ctrl_layout.addWidget(load_grp)

        # Enhancement pipeline
        enh_grp = QGroupBox("Enhancement Pipeline")
        enh_lay = QVBoxLayout(enh_grp)

        # Available methods
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        plugins = PluginBase.get_plugins("enhancement")
        for p in plugins:
            self.method_combo.addItem(p.name)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self.method_combo)

        self.btn_add_step = QPushButton("+ Add")
        self.btn_add_step.clicked.connect(self._add_pipeline_step)
        method_row.addWidget(self.btn_add_step)
        enh_lay.addLayout(method_row)

        # Method parameters
        self.method_params = ParamEditor()
        enh_lay.addWidget(self.method_params)

        # Pipeline list
        enh_lay.addWidget(QLabel("Pipeline Steps:"))
        self.pipeline_list = QListWidget()
        self.pipeline_list.setMaximumHeight(120)
        enh_lay.addWidget(self.pipeline_list)

        pipe_btns = QHBoxLayout()
        self.btn_remove_step = QPushButton("Remove")
        self.btn_remove_step.clicked.connect(self._remove_pipeline_step)
        pipe_btns.addWidget(self.btn_remove_step)
        self.btn_clear_pipeline = QPushButton("Clear All")
        self.btn_clear_pipeline.clicked.connect(self._clear_pipeline)
        pipe_btns.addWidget(self.btn_clear_pipeline)
        enh_lay.addLayout(pipe_btns)

        # Run
        self.btn_run_enhance = QPushButton("▶  Run Enhancement")
        self.btn_run_enhance.setObjectName("successBtn")
        self.btn_run_enhance.clicked.connect(self._run_enhancement)
        enh_lay.addWidget(self.btn_run_enhance)

        self.btn_revert = QPushButton("↩  Revert to Original")
        self.btn_revert.clicked.connect(self._revert_enhancement)
        enh_lay.addWidget(self.btn_revert)

        self.enhance_progress = QProgressBar()
        self.enhance_progress.setVisible(False)
        enh_lay.addWidget(self.enhance_progress)

        self.enhance_status = QLabel("")
        self.enhance_status.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        enh_lay.addWidget(self.enhance_status)

        ctrl_layout.addWidget(enh_grp)
        ctrl_layout.addStretch()

        ctrl_scroll.setWidget(ctrl_widget)
        splitter.addWidget(ctrl_scroll)

        # ── Right panel: viewer ───────────────────────────────
        self.viewer = ImageViewer(show_toolbar=True)
        splitter.addWidget(self.viewer)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        # Init method params
        self._pipeline = []  # List of (plugin_class, params_dict)
        self._on_method_changed()

    # ── Load ──────────────────────────────────────────────────

    def _load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Image Files", "", FILTER_STRING)
        if paths:
            self._start_load(paths)

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Load Image Folder")
        if folder:
            p = Path(folder)
            files = sorted([
                f for f in p.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTS
            ])
            if files:
                self._start_load([str(f) for f in files])
            else:
                QMessageBox.warning(self, "No Files",
                                    "No supported files found in the selected folder.")

    def _start_load(self, paths):
        self.load_info.setText(f"Loading {len(paths)} files...")
        self.btn_load_files.setEnabled(False)
        self.btn_load_folder.setEnabled(False)

        self._worker = LoadWorker(paths, "auto")
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._on_load_error)
        self._worker.progress.connect(
            lambda i, t: self.load_info.setText(f"Loading {i}/{t}...")
        )
        self._worker.log.connect(
            lambda msg: self.enhance_status.setText(msg)
        )
        self._worker.start()

    def _on_loaded(self, volumes):
        self._raw_volumes = volumes
        self._enhanced_volumes = volumes.copy()
        self.btn_load_files.setEnabled(True)
        self.btn_load_folder.setEnabled(True)

        info_parts = [f"{len(volumes)} volumes"]
        if volumes:
            s = volumes[0].shape
            info_parts.append(f"shape={s}")
            dtype_str = str(volumes[0].dtype)
            info_parts.append(f"dtype={dtype_str}")
        self.load_info.setText(" | ".join(info_parts))

        self.viewer.set_data(volumes)

        # Store paths in experiment
        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                exp.n_frames = len(volumes)
                exp.image_config = {
                    "n_volumes": len(volumes),
                    "shape": list(volumes[0].shape) if volumes else [],
                    "ndim": volumes[0].ndim if volumes else 0,
                }
                self.main_window.exp_manager.update(
                    exp.exp_id, n_frames=len(volumes)
                )
            self.main_window.set_status(f"Loaded {len(volumes)} images", "ready")

    def _on_load_error(self, msg):
        self.btn_load_files.setEnabled(True)
        self.btn_load_folder.setEnabled(True)
        self.load_info.setText("Load failed")
        QMessageBox.warning(self, "Load Error", msg)

    # ── Enhancement pipeline ──────────────────────────────────

    def _on_method_changed(self):
        name = self.method_combo.currentText()
        plugin_cls = PluginBase.get_plugin("enhancement", name)
        if plugin_cls:
            inst = plugin_cls()
            self.method_params.set_params(inst.get_params())

    def _add_pipeline_step(self):
        name = self.method_combo.currentText()
        plugin_cls = PluginBase.get_plugin("enhancement", name)
        if plugin_cls:
            params = self.method_params.get_values()
            self._pipeline.append((plugin_cls, params))
            item = QListWidgetItem(f"{len(self._pipeline)}. {name}")
            item.setToolTip(str(params))
            self.pipeline_list.addItem(item)

    def _remove_pipeline_step(self):
        row = self.pipeline_list.currentRow()
        if row >= 0:
            self.pipeline_list.takeItem(row)
            self._pipeline.pop(row)
            # Renumber
            for i in range(self.pipeline_list.count()):
                item = self.pipeline_list.item(i)
                name = self._pipeline[i][0].name
                item.setText(f"{i+1}. {name}")

    def _clear_pipeline(self):
        self._pipeline.clear()
        self.pipeline_list.clear()

    def _run_enhancement(self):
        if not self._raw_volumes:
            QMessageBox.warning(self, "No Data", "Load images first.")
            return
        if not self._pipeline:
            QMessageBox.information(self, "No Steps", "Add enhancement steps first.")
            return

        self.enhance_progress.setVisible(True)
        self.enhance_progress.setValue(0)
        self.btn_run_enhance.setEnabled(False)

        self._worker = EnhanceWorker(self._raw_volumes, self._pipeline)
        self._worker.finished.connect(self._on_enhanced)
        self._worker.error.connect(self._on_enhance_error)
        self._worker.progress.connect(self.enhance_progress.setValue)
        self._worker.step_info.connect(self.enhance_status.setText)
        self._worker.start()

    def _on_enhanced(self, results):
        self._enhanced_volumes = results
        self.viewer.set_data(results)
        self.enhance_progress.setVisible(False)
        self.btn_run_enhance.setEnabled(True)
        self.enhance_status.setText("Enhancement complete ✓")

        if self.main_window:
            self.main_window.set_status("Enhancement complete", "ready")

    def _on_enhance_error(self, msg):
        self.enhance_progress.setVisible(False)
        self.btn_run_enhance.setEnabled(True)
        self.enhance_status.setText(f"Error: {msg}")
        QMessageBox.warning(self, "Enhancement Error", msg)

    def _revert_enhancement(self):
        if self._raw_volumes:
            self._enhanced_volumes = self._raw_volumes.copy()
            self.viewer.set_data(self._raw_volumes)
            self.enhance_status.setText("Reverted to original")

    # ── Public API ────────────────────────────────────────────

    def get_volumes(self) -> List[np.ndarray]:
        """Return current (possibly enhanced) volumes."""
        return self._enhanced_volumes if self._enhanced_volumes else self._raw_volumes

    def on_experiment_changed(self, exp_id: str):
        pass  # Could reload images for new experiment

    def on_activated(self):
        pass