"""
Images Page — Load images, preview slices, apply enhancement pipeline.

Supports: ND2, TIFF stacks, MAT files, NPY.
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

class LoadWorker(QThread):
    """Background thread for loading images."""
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int, int)
    
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
                if p.suffix.lower() in (".tif", ".tiff"):
                    import tifffile
                    img = tifffile.imread(str(p)).astype(np.float64)
                    if img.ndim == 2:
                        img = img.T  # (x, y)
                    elif img.ndim == 3:
                        img = img.transpose(1, 0, 2)  # (x, y, z)
                    volumes.append(img)
                elif p.suffix.lower() == ".mat":
                    import scipy.io as sio
                    data = sio.loadmat(str(p), simplify_cells=True)
                    keys = [k for k in data if not k.startswith("_")]
                    vol = np.asarray(data[keys[0]])
                    if vol.ndim == 3:
                        vol = vol.transpose(1, 0, 2)
                    volumes.append(vol.astype(np.float64))
                elif p.suffix.lower() == ".npy":
                    volumes.append(np.load(str(p)).astype(np.float64))
                elif p.suffix.lower() == ".nd2":
                    try:
                        import nd2
                        with nd2.ND2File(str(p)) as f:
                            data = f.asarray().astype(np.float64)
                            if data.ndim == 4:  # (T, Z, Y, X)
                                for t in range(data.shape[0]):
                                    volumes.append(data[t].transpose(2, 1, 0))
                            elif data.ndim == 3:  # (Z, Y, X) or (T, Y, X)
                                volumes.append(data.transpose(2, 1, 0))
                            else:
                                volumes.append(data.T)
                    except ImportError:
                        self.error.emit("nd2 package not installed. Install with: pip install nd2")
                        return
                else:
                    self.error.emit(f"Unsupported format: {p.suffix}")
                    return
                self.progress.emit(i + 1, total)

            self.finished.emit(volumes)
        except Exception as e:
            self.error.emit(str(e))


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
            self, "Load Image Files", "",
            "Image Files (*.tif *.tiff *.mat *.npy *.nd2);;All Files (*)"
        )
        if paths:
            self._start_load(paths)

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Load Image Folder")
        if folder:
            p = Path(folder)
            files = sorted(
                list(p.glob("*.tif")) + list(p.glob("*.tiff")) +
                list(p.glob("*.mat")) + list(p.glob("*.npy"))
            )
            if files:
                self._start_load([str(f) for f in files])
            else:
                QMessageBox.warning(self, "No Files", "No supported files found.")

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
        pass  # Now handled by save/load_from_experiment

    def on_activated(self):
        pass

    def save_to_experiment(self, exp):
        """Persist loaded volumes and paths into experiment cache."""
        vols = self.get_volumes()
        if vols:
            exp.store_image_volumes(vols)
        # Save image paths
        paths = []
        for i in range(self.file_list.count()):
            paths.append(self.file_list.item(i).text())
        exp.image_paths = paths

    def load_from_experiment(self, exp):
        """Restore volumes and file list from experiment cache."""
        vols = exp.get_image_volumes()
        if vols is not None and len(vols) > 0:
            self._raw_volumes = vols
            self._enhanced_volumes = vols.copy() if hasattr(vols, 'copy') else list(vols)
            # Update file list
            self.file_list.clear()
            for p in exp.image_paths:
                self.file_list.addItem(p)
            # Update preview
            if hasattr(self, 'preview_canvas') and vols:
                self._show_preview(0)
            self.status_label.set_status("ready", f"{len(vols)} volumes")
        else:
            self._raw_volumes = []
            self._enhanced_volumes = []
            self.file_list.clear()
            if hasattr(self, 'preview_canvas'):
                self.preview_canvas.clear()
                self.preview_canvas.draw()
            self.status_label.set_status("idle", "No images")