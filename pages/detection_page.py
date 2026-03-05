"""
Detection Preview Page — Detect particles and visualize with circle overlays.

Users can adjust detection parameters and rerun until satisfied.
Supports three detection methods:
  - TracTrac (LoG)          — Laplacian of Gaussian + sub-pixel polynomial
  - TPT (Radial Symmetry)   — Centroid + radial symmetry sub-voxel (3D)
  - StarDist (Deep Learning) — Star-convex polygon instance segmentation
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


# ─────────────────────────────────────────────────────────────
#  Check StarDist availability at import time (non-blocking)
# ─────────────────────────────────────────────────────────────

def _check_stardist() -> bool:
    """Return True if stardist and csbdeep are importable."""
    try:
        import stardist          # noqa: F401
        from csbdeep.utils import normalize  # noqa: F401
        return True
    except ImportError:
        return False


_STARDIST_AVAILABLE = _check_stardist()


# ─────────────────────────────────────────────────────────────
#  Detection worker (background thread)
# ─────────────────────────────────────────────────────────────

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
            from serialtrack.config import (
                DetectionConfig, DetectionMethod, StarDistConfig,
            )
            from serialtrack.detection import ParticleDetector

            method_str = self.config.get("method", "TracTrac (LoG)")
            method_map = {
                "TracTrac (LoG)": DetectionMethod.TRACTRAC,
                "TPT (Radial Symmetry)": DetectionMethod.TPT,
                "StarDist (Deep Learning)": DetectionMethod.STARDIST,
            }
            method = method_map.get(method_str, DetectionMethod.TRACTRAC)

            # Build StarDist config if selected
            sd_cfg = StarDistConfig()
            if method == DetectionMethod.STARDIST:
                sd_model = self.config.get("sd_model", "2D_versatile_fluo")
                # model_basedir: empty string → None (use pretrained)
                sd_basedir = self.config.get("sd_model_basedir", "")
                sd_basedir = sd_basedir.strip() if sd_basedir else ""
                sd_cfg = StarDistConfig(
                    model_name=sd_model,
                    model_basedir=sd_basedir if sd_basedir else None,
                    prob_thresh=self.config.get("sd_prob_thresh"),
                    nms_thresh=self.config.get("sd_nms_thresh"),
                    normalize_input=self.config.get("sd_normalize", True),
                    norm_pmin=self.config.get("sd_norm_pmin", 1.0),
                    norm_pmax=self.config.get("sd_norm_pmax", 99.8),
                    use_gpu=self.config.get("sd_use_gpu", True),
                )
                # Parse prob/nms: None means "use model default"
                if sd_cfg.prob_thresh is not None:
                    try:
                        sd_cfg.prob_thresh = float(sd_cfg.prob_thresh)
                        if sd_cfg.prob_thresh <= 0:
                            sd_cfg.prob_thresh = None
                    except (TypeError, ValueError):
                        sd_cfg.prob_thresh = None
                if sd_cfg.nms_thresh is not None:
                    try:
                        sd_cfg.nms_thresh = float(sd_cfg.nms_thresh)
                        if sd_cfg.nms_thresh <= 0:
                            sd_cfg.nms_thresh = None
                    except (TypeError, ValueError):
                        sd_cfg.nms_thresh = None

            cfg = DetectionConfig(
                method=method,
                threshold=self.config.get("threshold", 0.4),
                bead_radius=self.config.get("bead_radius", 3.0),
                min_size=int(self.config.get("min_size", 2)),
                max_size=int(self.config.get("max_size", 1000)),
                color=self.config.get("color", "white"),
                stardist=sd_cfg,
            )

            self.progress.emit(f"Running {method_str} detection...")
            detector = ParticleDetector(cfg)
            coords = detector.detect(self.image)
            coords = ParticleDetector.clip_to_bounds(coords, self.image.shape)
            self.finished.emit(coords)
        except ImportError as e:
            self.error.emit(
                f"Missing dependency: {e}\n\n"
                "For StarDist, install with:\n"
                "  pip install stardist tensorflow"
            )
        except RuntimeError as e:
            # RuntimeError is raised by _load_pretrained_ssl with
            # detailed download/SSL instructions
            self.error.emit(str(e))
        except Exception as e:
            err_str = str(e).lower()
            if 'ssl' in err_str or 'certificate' in err_str:
                self.error.emit(
                    f"SSL certificate error: {e}\n\n"
                    "Try one of:\n"
                    "  1. Click 'Pre-Download Model' in StarDist params\n"
                    "  2. Fix certificates: pip install --upgrade certifi\n"
                    "  3. macOS: run Install Certificates.command\n"
                    "  4. Manually download model and set 'Model Base Dir'"
                )
            else:
                self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────
#  Detection page widget
# ─────────────────────────────────────────────────────────────

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

        # --- Build method choices depending on StarDist availability ---
        method_choices = ["TracTrac (LoG)", "TPT (Radial Symmetry)"]
        if _STARDIST_AVAILABLE:
            method_choices.append("StarDist (Deep Learning)")

        method_tooltip = (
            "TracTrac: LoG blob detection + polynomial sub-pixel.\n"
            "TPT: Centroid + radial symmetry sub-voxel (3D only)."
        )
        if _STARDIST_AVAILABLE:
            method_tooltip += (
                "\nStarDist: Deep-learning instance segmentation.\n"
                "  Uses pretrained fluorescence microscopy models."
            )
        else:
            method_tooltip += (
                "\n\n⚠ StarDist not available. Install with:\n"
                "  pip install stardist tensorflow"
            )

        specs = [
            ParamSpec("method", "Method", "choice",
                      "TracTrac (LoG)", choices=method_choices,
                      tooltip=method_tooltip),
            ParamSpec("threshold", "Intensity Threshold", "float", 0.4, 0.01, 1.0, 0.01,
                      tooltip="Minimum normalized intensity to consider as particle.\n"
                              "Lower = more detections (may include noise).\n"
                              "(Not used by StarDist — use Prob Threshold instead.)"),
            ParamSpec("bead_radius", "Bead Radius (px)", "float", 3.0, 0.0, 50.0, 0.5,
                      tooltip="Expected bead radius in pixels.\n"
                              "Used as sigma for LoG filter. 0 = centroid only.\n"
                              "(Not used by StarDist.)"),
            ParamSpec("min_size", "Min Blob Size (px²)", "int", 2, 1, 10000, 1,
                      tooltip="Minimum connected-component area/volume to keep.\n"
                              "(Also applied as post-filter for StarDist.)"),
            ParamSpec("max_size", "Max Blob Size (px²)", "int", 1000, 10, 100000, 10,
                      tooltip="Maximum connected-component area/volume to keep.\n"
                              "(Also applied as post-filter for StarDist.)"),
            ParamSpec("color", "Foreground", "choice", "white",
                      choices=["white", "black"],
                      tooltip="Are beads bright on dark ('white') or dark on bright ('black')?"),
        ]

        self.det_params = ParamEditor()
        self.det_params.set_params(specs)
        det_lay.addWidget(self.det_params)

        ctrl_layout.addWidget(det_grp)

        # ── StarDist parameters group ─────────────────────────
        self.sd_grp = QGroupBox("StarDist Parameters")
        sd_lay = QVBoxLayout(self.sd_grp)

        if not _STARDIST_AVAILABLE:
            # Show install instructions instead of parameters
            install_label = QLabel(
                "StarDist is not installed.\n\n"
                "To enable deep-learning bead detection:\n"
                "  pip install stardist tensorflow\n\n"
                "Restart the application after installing."
            )
            install_label.setWordWrap(True)
            install_label.setStyleSheet(f"color: {Settings.FG_SECONDARY}; padding: 8px;")
            sd_lay.addWidget(install_label)
        else:
            sd_model_choices = [
                "2D_versatile_fluo",
                "2D_paper_dsb2018",
                "3D_demo",
            ]
            sd_specs = [
                ParamSpec("sd_model", "Model", "choice",
                          "2D_versatile_fluo", choices=sd_model_choices,
                          tooltip="Pretrained StarDist model.\n"
                                  "2D_versatile_fluo: Best for fluorescence beads (2D).\n"
                                  "2D_paper_dsb2018: Data Science Bowl 2018 model.\n"
                                  "3D_demo: Demo model for 3D volumes.\n\n"
                                  "The correct 2D/3D model is auto-selected\n"
                                  "based on your image dimensionality."),
                ParamSpec("sd_model_basedir", "Model Base Dir", "str", "",
                          tooltip="Leave empty to use pretrained models (auto-downloaded).\n\n"
                                  "Set to a local directory path to use a custom or\n"
                                  "manually-downloaded model. The model folder should be\n"
                                  "inside this directory.\n\n"
                                  "Useful if automatic download fails (SSL issues)."),
                ParamSpec("sd_prob_thresh", "Prob Threshold", "float", 0.5, 0.0, 1.0, 0.05,
                          tooltip="Object probability threshold.\n"
                                  "Lower = more detections (may include noise).\n"
                                  "Set to 0 to use model default.\n"
                                  "Typical range: 0.3–0.7."),
                ParamSpec("sd_nms_thresh", "NMS Threshold", "float", 0.4, 0.0, 1.0, 0.05,
                          tooltip="Non-maximum suppression (overlap) threshold.\n"
                                  "Lower = more aggressive overlap removal.\n"
                                  "Set to 0 to use model default.\n"
                                  "Typical range: 0.3–0.5."),
                ParamSpec("sd_normalize", "Normalize Input", "choice", "Yes",
                          choices=["Yes", "No"],
                          tooltip="Apply percentile normalization before prediction.\n"
                                  "Recommended for raw microscopy images."),
                ParamSpec("sd_norm_pmin", "Norm Percentile Low", "float", 1.0, 0.0, 50.0, 0.5,
                          tooltip="Lower percentile for intensity normalization."),
                ParamSpec("sd_norm_pmax", "Norm Percentile High", "float", 99.8, 50.0, 100.0, 0.1,
                          tooltip="Upper percentile for intensity normalization."),
                ParamSpec("sd_use_gpu", "Use GPU", "choice", "Yes",
                          choices=["Yes", "No"],
                          tooltip="Use GPU acceleration if available.\n"
                                  "Requires CUDA-compatible GPU and tensorflow-gpu."),
            ]
            self.sd_params = ParamEditor()
            self.sd_params.set_params(sd_specs)
            sd_lay.addWidget(self.sd_params)

            # Pre-download button for convenience
            self.btn_sd_download = QPushButton("Pre-Download Model")
            self.btn_sd_download.setToolTip(
                "Download the selected pretrained model to\n"
                "~/.serialtrack/stardist_models/ for offline use.\n"
                "Also works around SSL certificate issues."
            )
            self.btn_sd_download.clicked.connect(self._download_stardist_model)
            sd_lay.addWidget(self.btn_sd_download)

        ctrl_layout.addWidget(self.sd_grp)

        # Initially hide StarDist group (shown when method is selected)
        self.sd_grp.setVisible(False)

        # Connect method change to show/hide StarDist params
        # ParamEditor emits value_changed when any param changes
        if hasattr(self.det_params, 'value_changed'):
            self.det_params.value_changed.connect(self._on_method_changed)

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

    # ── Method change handler ─────────────────────────────────

    def _on_method_changed(self):
        """Show/hide StarDist parameter group based on selected method."""
        vals = self.det_params.get_values()
        method = vals.get("method", "TracTrac (LoG)")
        is_stardist = method == "StarDist (Deep Learning)"
        self.sd_grp.setVisible(is_stardist)

    def _download_stardist_model(self):
        """Pre-download StarDist model to local cache."""
        if not _STARDIST_AVAILABLE:
            return

        sd_vals = self.sd_params.get_values()
        model_name = sd_vals.get("sd_model", "2D_versatile_fluo")

        self.results_label.setText(f"Downloading StarDist model '{model_name}'...")
        self.results_label.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        self.btn_sd_download.setEnabled(False)

        # Run in background thread to avoid blocking GUI
        from PySide6.QtCore import QThread, Signal as QSignal

        class DownloadWorker(QThread):
            done = QSignal(str)
            failed = QSignal(str)

            def __init__(self, name, parent=None):
                super().__init__(parent)
                self.model_name = name

            def run(self):
                try:
                    from serialtrack.detection import ParticleDetector
                    path = ParticleDetector.download_pretrained_model(
                        self.model_name,
                    )
                    self.done.emit(str(path))
                except Exception as e:
                    self.failed.emit(str(e))

        def _on_download_done(path):
            self.btn_sd_download.setEnabled(True)
            self.results_label.setText(
                f"Model downloaded to:\n{path}\n\n"
                f"You can set 'Model Base Dir' to its parent folder\n"
                f"for offline use."
            )
            self.results_label.setStyleSheet(f"color: {Settings.ACCENT_GREEN};")
            # Auto-fill the basedir field
            from pathlib import Path as P
            self.sd_params.set_values({
                "sd_model_basedir": str(P(path).parent),
            })

        def _on_download_failed(msg):
            self.btn_sd_download.setEnabled(True)
            self.results_label.setText(f"Download failed: {msg}")
            self.results_label.setStyleSheet(f"color: {Settings.ACCENT_RED};")

        self._dl_worker = DownloadWorker(model_name)
        self._dl_worker.done.connect(_on_download_done)
        self._dl_worker.failed.connect(_on_download_failed)
        self._dl_worker.start()

    # ── Auto-estimate ─────────────────────────────────────────

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

        # Also update StarDist prob_thresh based on density
        if _STARDIST_AVAILABLE and hasattr(self, 'sd_params'):
            sd_prob_map = {"sparse": 0.6, "medium": 0.5, "dense": 0.35}
            self.sd_params.set_values({
                "sd_prob_thresh": sd_prob_map.get(density, 0.5),
            })

        self.results_label.setText(
            f"Auto: radius={bead_radius_px:.1f}px, thresh={threshold}, "
            f"size=[{min_size}, {max_size}]"
        )

    # ── Run detection ─────────────────────────────────────────

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

        # Merge StarDist params if that method is selected
        method = config.get("method", "TracTrac (LoG)")
        if method == "StarDist (Deep Learning)":
            if not _STARDIST_AVAILABLE:
                QMessageBox.warning(
                    self, "StarDist Not Available",
                    "StarDist is not installed.\n\n"
                    "Install with:\n"
                    "  pip install stardist tensorflow\n\n"
                    "Then restart the application.",
                )
                return
            sd_vals = self.sd_params.get_values()
            # Convert Yes/No choices to booleans
            sd_vals["sd_normalize"] = sd_vals.get("sd_normalize", "Yes") == "Yes"
            sd_vals["sd_use_gpu"] = sd_vals.get("sd_use_gpu", "Yes") == "Yes"
            # Convert 0 thresholds to None (= model default)
            prob = sd_vals.get("sd_prob_thresh", 0.5)
            sd_vals["sd_prob_thresh"] = prob if prob > 0 else None
            nms = sd_vals.get("sd_nms_thresh", 0.4)
            sd_vals["sd_nms_thresh"] = nms if nms > 0 else None
            config.update(sd_vals)

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
        method = self.det_params.get_values().get("method", "")
        self.results_label.setText(
            f"Detected {n} particles ({ndim}D) — {method}\n"
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
        # Refresh StarDist visibility based on current method selection
        self._on_method_changed()

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

    def save_to_experiment(self, exp):
        """Persist detection state into experiment cache."""
        config = self.det_params.get_values()
        # Include StarDist params if available
        if _STARDIST_AVAILABLE and hasattr(self, 'sd_params'):
            config['_stardist_params'] = self.sd_params.get_values()
        exp.detection_config = config
        # Store coords in experiment (transient, not saved to disk)
        exp._detection_coords = self._coords

    def load_from_experiment(self, exp):
        """Restore detection state from experiment record."""
        # Restore detection params
        if exp.detection_config:
            # Split out StarDist params
            config = dict(exp.detection_config)
            sd_saved = config.pop('_stardist_params', None)
            self.det_params.set_values(config)
            if sd_saved and _STARDIST_AVAILABLE and hasattr(self, 'sd_params'):
                self.sd_params.set_values(sd_saved)

        # Restore coords
        self._coords = getattr(exp, '_detection_coords', None)

        # Reload image from images page (images page loads first)
        if self.main_window and "images" in self.main_window.pages:
            vols = self.main_window.pages["images"].get_volumes()
            if vols:
                self._current_image = vols[0]
            else:
                self._current_image = None
        else:
            self._current_image = None

        # Refresh visibility
        self._on_method_changed()

        # Refresh UI
        if self._current_image is not None:
            if self._current_image.ndim == 3:
                nz = self._current_image.shape[2]
                self.z_slider.setRange(0, nz - 1)
                self.z_slider.setValue(nz // 2)
            n = len(self._coords) if self._coords is not None else 0
            self.results_label.setText(f"{n} particles loaded")
            self.results_label.setStyleSheet(
                f"color: {Settings.ACCENT_GREEN if n > 0 else Settings.FG_SECONDARY};")
            self._redraw()
        else:
            # No image data: clear everything
            self._coords = None
            self.canvas.clear()
            self.canvas.draw()
            self.results_label.setText("No image data loaded")
            self.results_label.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
