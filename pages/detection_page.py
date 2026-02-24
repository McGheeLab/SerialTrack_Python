"""
Detection Preview Page — Detect particles and visualize with circle overlays.

Users can adjust detection parameters and rerun until satisfied.
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QScrollArea, QProgressBar,
    QFrame, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QThread

from widgets.common import MplCanvas, ParamEditor
from core.settings import Settings
from core.plugin_registry import ParamSpec


class DetectWorker(QThread):
    """Run particle detection in background."""
    finished = Signal(object)  # coords array
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, image, config_dict, parent=None):
        super().__init__(parent)
        self.image = image
        self.config = config_dict

    def run(self):
        try:
            from serialtrack.config import DetectionConfig, DetectionMethod
            from serialtrack.detection import ParticleDetector

            method_map = {"TracTrac (LoG)": 2, "TPT (Radial Symmetry)": 1}
            cfg = DetectionConfig(
                method=DetectionMethod(method_map.get(
                    self.config.get("method", "TracTrac (LoG)"), 2)),
                threshold=self.config.get("threshold", 0.4),
                bead_radius=self.config.get("bead_radius", 3.0),
                min_size=int(self.config.get("min_size", 2)),
                max_size=int(self.config.get("max_size", 1000)),
                color=self.config.get("color", "white"),
            )

            self.progress.emit("Running detection...")
            detector = ParticleDetector(cfg)
            coords = detector.detect(self.image)
            coords = ParticleDetector.clip_to_bounds(coords, self.image.shape)
            self.finished.emit(coords)
        except Exception as e:
            self.error.emit(str(e))


class DetectionPage(QWidget):
    """Particle detection preview with adjustable parameters."""
    detection_complete = Signal(object)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._coords: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── Controls ──────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(380)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(8)

        # Detection parameters
        det_grp = QGroupBox("Detection Parameters")
        det_lay = QVBoxLayout(det_grp)

        specs = [
            ParamSpec("method", "Method", "choice",
                      "TracTrac (LoG)", choices=["TracTrac (LoG)", "TPT (Radial Symmetry)"],
                      tooltip="TracTrac: LoG blob detection + polynomial sub-pixel.\n"
                              "TPT: Centroid + radial symmetry sub-voxel (3D only)."),
            ParamSpec("threshold", "Intensity Threshold", "float", 0.4, 0.01, 1.0, 0.01,
                      tooltip="Minimum normalized intensity to consider as particle.\n"
                              "Lower = more detections (may include noise)."),
            ParamSpec("bead_radius", "Bead Radius (px)", "float", 3.0, 0.0, 50.0, 0.5,
                      tooltip="Expected bead radius in pixels.\n"
                              "Used as sigma for LoG filter. 0 = centroid only."),
            ParamSpec("min_size", "Min Blob Size (px²)", "int", 2, 1, 10000, 1,
                      tooltip="Minimum connected-component area/volume to keep."),
            ParamSpec("max_size", "Max Blob Size (px²)", "int", 1000, 10, 100000, 10,
                      tooltip="Maximum connected-component area/volume to keep."),
            ParamSpec("color", "Foreground", "choice", "white",
                      choices=["white", "black"],
                      tooltip="Are beads bright on dark ('white') or dark on bright ('black')?"),
        ]

        self.det_params = ParamEditor()
        self.det_params.set_params(specs)
        det_lay.addWidget(self.det_params)

        ctrl_layout.addWidget(det_grp)

        # Auto-estimate section
        auto_grp = QGroupBox("Auto-Estimate Parameters")
        auto_lay = QVBoxLayout(auto_grp)

        auto_specs = [
            ParamSpec("bead_diameter_um", "Bead Diameter (µm)", "float", 1.0, 0.1, 100.0, 0.1,
                      tooltip="Physical bead diameter for auto-estimation"),
            ParamSpec("pixel_size_um", "Pixel Size (µm/px)", "float", 0.5, 0.01, 10.0, 0.01,
                      tooltip="Physical pixel size for converting µm to px"),
            ParamSpec("bead_density", "Approx Bead Density", "choice", "medium",
                      choices=["sparse", "medium", "dense"],
                      tooltip="Rough bead density affects threshold estimation"),
        ]
        self.auto_params = ParamEditor()
        self.auto_params.set_params(auto_specs)
        auto_lay.addWidget(self.auto_params)

        self.btn_auto = QPushButton("Auto-Estimate")
        self.btn_auto.setObjectName("primaryBtn")
        self.btn_auto.clicked.connect(self._auto_estimate)
        auto_lay.addWidget(self.btn_auto)

        ctrl_layout.addWidget(auto_grp)

        # Run
        self.btn_detect = QPushButton("▶  Run Detection")
        self.btn_detect.setObjectName("successBtn")
        self.btn_detect.clicked.connect(self._run_detection)
        ctrl_layout.addWidget(self.btn_detect)

        self.detect_progress = QProgressBar()
        self.detect_progress.setVisible(False)
        self.detect_progress.setRange(0, 0)  # Indeterminate
        ctrl_layout.addWidget(self.detect_progress)

        # Results
        self.results_label = QLabel("No detection results yet")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        ctrl_layout.addWidget(self.results_label)

        # Z slice for 3D viewing
        z_grp = QGroupBox("Z Slice (3D)")
        z_lay = QHBoxLayout(z_grp)
        z_lay.addWidget(QLabel("Slice:"))
        from PySide6.QtWidgets import QSlider, QSpinBox
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.valueChanged.connect(self._redraw)
        z_lay.addWidget(self.z_slider)
        self.z_label = QLabel("0")
        z_lay.addWidget(self.z_label)
        ctrl_layout.addWidget(z_grp)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_widget)
        splitter.addWidget(ctrl_scroll)

        # ── Image viewer ──────────────────────────────────────
        self.canvas = MplCanvas(figsize=(6, 5), toolbar=True)
        splitter.addWidget(self.canvas)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    def _auto_estimate(self):
        vals = self.auto_params.get_values()
        bead_um = vals.get("bead_diameter_um", 1.0)
        px_um = vals.get("pixel_size_um", 0.5)
        density = vals.get("bead_density", "medium")

        bead_radius_px = (bead_um / 2.0) / px_um

        threshold_map = {"sparse": 0.5, "medium": 0.35, "dense": 0.2}
        threshold = threshold_map.get(density, 0.35)

        min_size = max(2, int(0.3 * np.pi * bead_radius_px**2))
        max_size = max(min_size * 10, int(5.0 * np.pi * bead_radius_px**2))

        self.det_params.set_values({
            "bead_radius": round(bead_radius_px, 1),
            "threshold": threshold,
            "min_size": min_size,
            "max_size": max_size,
        })
        self.results_label.setText(
            f"Auto: radius={bead_radius_px:.1f}px, thresh={threshold}, "
            f"size=[{min_size}, {max_size}]"
        )

    def _run_detection(self):
        # Get image from Images page
        if self.main_window and "images" in self.main_window.pages:
            vols = self.main_window.pages["images"].get_volumes()
            if not vols:
                QMessageBox.warning(self, "No Images", "Load images first.")
                return
            self._current_image = vols[0]
        else:
            QMessageBox.warning(self, "No Images", "Load images first.")
            return

        config = self.det_params.get_values()
        self.detect_progress.setVisible(True)
        self.btn_detect.setEnabled(False)

        self._worker = DetectWorker(self._current_image, config)
        self._worker.finished.connect(self._on_detected)
        self._worker.error.connect(self._on_detect_error)
        self._worker.progress.connect(self.results_label.setText)
        self._worker.start()

    def _on_detected(self, coords):
        self._coords = coords
        self.detect_progress.setVisible(False)
        self.btn_detect.setEnabled(True)

        if self._current_image is not None and self._current_image.ndim == 3:
            nz = self._current_image.shape[2]
            self.z_slider.setRange(0, nz - 1)
            self.z_slider.setValue(nz // 2)

        n = len(coords) if coords is not None else 0
        ndim = coords.shape[1] if n > 0 else 0
        self.results_label.setText(
            f"Detected {n} particles ({ndim}D)\n"
            f"Click 'Run Detection' to adjust and rerun."
        )
        self.results_label.setStyleSheet(f"color: {Settings.ACCENT_GREEN};")

        self.detection_complete.emit(coords)
        self._redraw()

        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                exp.n_particles_ref = n
                exp.detection_config = self.det_params.get_values()
                self.main_window.exp_manager.update(exp.exp_id, n_particles_ref=n)
            self.main_window.set_status(f"Detected {n} particles", "ready")

    def _on_detect_error(self, msg):
        self.detect_progress.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.results_label.setText(f"Error: {msg}")
        self.results_label.setStyleSheet(f"color: {Settings.ACCENT_RED};")
        QMessageBox.warning(self, "Detection Error", msg)

    def _redraw(self):
        if self._current_image is None:
            return

        self.canvas.clear()
        ax = self.canvas.add_subplot(111)

        img = self._current_image
        z = self.z_slider.value()
        self.z_label.setText(str(z))

        if img.ndim == 3:
            z = min(z, img.shape[2] - 1)
            slice_img = img[:, :, z]
        else:
            slice_img = img

        ax.imshow(slice_img.T, cmap="gray", origin="lower", aspect="equal")

        # Draw detected particles
        if self._coords is not None and len(self._coords) > 0:
            bead_r = self.det_params.get_values().get("bead_radius", 3.0)
            coords = self._coords

            if coords.shape[1] == 3:
                # Filter to particles near this Z slice
                z_tol = max(2, bead_r)
                mask = np.abs(coords[:, 2] - z) < z_tol
                vis_coords = coords[mask]
            else:
                vis_coords = coords

            if len(vis_coords) > 0:
                ax.scatter(vis_coords[:, 0], vis_coords[:, 1],
                          s=max(10, bead_r * 15), facecolors='none',
                          edgecolors='#50fa7b', linewidths=0.8, alpha=0.8)

            ax.set_title(
                f"{len(vis_coords)} particles visible (Z={z})",
                color="#f8f8f2", fontsize=10
            )

        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        self.canvas.draw()

    def get_coords(self) -> Optional[np.ndarray]:
        return self._coords

    def on_experiment_changed(self, exp_id: str):
        pass

    def on_activated(self):
        if self._current_image is None:
            if self.main_window and "images" in self.main_window.pages:
                vols = self.main_window.pages["images"].get_volumes()
                if vols:
                    self._current_image = vols[0]
                    if self._current_image.ndim == 3:
                        nz = self._current_image.shape[2]
                        self.z_slider.setRange(0, nz - 1)
                        self.z_slider.setValue(nz // 2)
                    self._redraw()
