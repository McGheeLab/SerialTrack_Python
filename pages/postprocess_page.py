"""
Post-Processing Page — Compute displacement, velocity, and strain fields.

Supports modular post-processing methods, data validation,
and saving results with unique histories per experiment.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QScrollArea, QProgressBar,
    QTabWidget, QFrame, QMessageBox, QCheckBox, QListWidget,
    QListWidgetItem, QTextEdit,
)
from PySide6.QtCore import Qt, Signal, QThread

from widgets.common import ParamEditor, MplCanvas, StatusIndicator
from core.settings import Settings
from core.plugin_registry import ParamSpec
from core.experiment_manager import PostProcessRecord


class PostProcessWorker(QThread):
    """Run post-processing in background."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    log_msg = Signal(str)

    def __init__(self, session, config, parent=None):
        super().__init__(parent)
        self.session = session
        self.config = config

    def run(self):
        try:
            from serialtrack.fields import compute_gridded_strain, compute_strain_mls

            results = {"frames": []}
            session = self.session
            cfg = self.config
            n_frames = len(session.frame_results)

            grid_step_val = cfg.get("grid_step", 10.0)
            smoothness = cfg.get("smoothness", 1e-3)
            ndim = session.coords_ref.shape[1]
            grid_step = np.full(ndim, grid_step_val)
            pixel_steps = np.array([
                cfg.get("xstep", 1.0),
                cfg.get("ystep", 1.0),
            ])
            if ndim == 3:
                pixel_steps = np.append(pixel_steps, cfg.get("zstep", 1.0))

            max_disp_thresh = cfg.get("max_disp_threshold", float('inf'))
            max_vel_thresh = cfg.get("max_velocity_threshold", float('inf'))

            for i, res in enumerate(session.frame_results):
                self.log_msg.emit(f"Processing frame {i+1}/{n_frames}...")

                tracked = res.track_b2a >= 0
                if np.sum(tracked) < 4:
                    self.log_msg.emit(f"  Frame {i+1}: too few tracked particles, skipping")
                    results["frames"].append(None)
                    self.progress.emit(int(100 * (i + 1) / n_frames))
                    continue

                coords = res.coords_b[tracked]
                disp = -res.disp_b2a[tracked]  # a→b displacement

                # Validation: remove large displacements
                disp_mag = np.linalg.norm(disp, axis=1)
                valid = disp_mag < max_disp_thresh
                if not np.all(valid):
                    n_removed = np.sum(~valid)
                    self.log_msg.emit(f"  Removed {n_removed} outliers (disp > {max_disp_thresh})")
                    coords = coords[valid]
                    disp = disp[valid]

                try:
                    dfield, sfield = compute_gridded_strain(
                        coords=coords, disp=disp,
                        grid_step=grid_step, smoothness=smoothness,
                        pixel_steps=pixel_steps,
                    )

                    frame_result = {
                        "grids": [g.tolist() for g in dfield.grids],
                        "disp_components": dfield.components.tolist(),
                        "F_tensor": sfield.F_tensor.tolist(),
                        "eps_tensor": sfield.eps_tensor.tolist(),
                        "n_tracked": len(coords),
                    }

                    # Compute velocity if tstep given
                    tstep = cfg.get("tstep", 1.0)
                    vel = dfield.components * np.array(pixel_steps).reshape(-1, *([1]*ndim)) / tstep
                    frame_result["velocity"] = vel.tolist()

                    results["frames"].append(frame_result)
                except Exception as e:
                    self.log_msg.emit(f"  Frame {i+1} failed: {e}")
                    results["frames"].append(None)

                self.progress.emit(int(100 * (i + 1) / n_frames))

            results["config"] = cfg
            results["n_frames"] = n_frames
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class PostProcessPage(QWidget):
    """Post-processing page with modular methods and history."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._worker = None
        self._results: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── Controls ──────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(400)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(8)

        # Grid parameters
        grid_grp = QGroupBox("Grid Interpolation")
        grid_lay = QVBoxLayout(grid_grp)
        self.grid_params = ParamEditor()
        self.grid_params.set_params([
            ParamSpec("grid_step", "Grid Step (px)", "float", 10.0, 1.0, 100.0, 1.0,
                      tooltip="Grid spacing for scatter→grid interpolation.\n"
                              "Smaller = finer resolution but noisier."),
            ParamSpec("smoothness", "Smoothness", "float", 1e-3, 1e-8, 10.0, 1e-4,
                      tooltip="Regularization smoothness for gridding.\n"
                              "Higher = smoother fields."),
        ])
        grid_lay.addWidget(self.grid_params)
        ctrl_layout.addWidget(grid_grp)

        # Physical scales
        scale_grp = QGroupBox("Physical Scales")
        scale_lay = QVBoxLayout(scale_grp)
        self.scale_params = ParamEditor()
        self.scale_params.set_params([
            ParamSpec("xstep", "X Step (µm/px)", "float", 1.0, 0.001, 100.0, 0.01),
            ParamSpec("ystep", "Y Step (µm/px)", "float", 1.0, 0.001, 100.0, 0.01),
            ParamSpec("zstep", "Z Step (µm/px)", "float", 1.0, 0.001, 100.0, 0.01),
            ParamSpec("tstep", "Time Step (s)", "float", 1.0, 0.001, 3600.0, 0.1),
        ])
        scale_lay.addWidget(self.scale_params)
        ctrl_layout.addWidget(scale_grp)

        # Data validation
        val_grp = QGroupBox("Data Validation")
        val_lay = QVBoxLayout(val_grp)
        self.val_params = ParamEditor()
        self.val_params.set_params([
            ParamSpec("max_disp_threshold", "Max Displacement (px)", "float",
                      100.0, 1.0, 10000.0, 10.0,
                      tooltip="Displacements larger than this are removed as outliers."),
            ParamSpec("max_velocity_threshold", "Max Velocity (µm/s)", "float",
                      1000.0, 0.1, 100000.0, 10.0,
                      tooltip="Velocities larger than this are flagged."),
        ])
        val_lay.addWidget(self.val_params)
        ctrl_layout.addWidget(val_grp)

        # Output selection
        out_grp = QGroupBox("Output Fields")
        out_lay = QVBoxLayout(out_grp)
        self.cb_displacement = QCheckBox("Displacement field")
        self.cb_displacement.setChecked(True)
        out_lay.addWidget(self.cb_displacement)
        self.cb_velocity = QCheckBox("Velocity field")
        self.cb_velocity.setChecked(True)
        out_lay.addWidget(self.cb_velocity)
        self.cb_strain = QCheckBox("Strain field (infinitesimal)")
        self.cb_strain.setChecked(True)
        out_lay.addWidget(self.cb_strain)
        self.cb_def_grad = QCheckBox("Deformation gradient F")
        out_lay.addWidget(self.cb_def_grad)
        ctrl_layout.addWidget(out_grp)

        # Run
        self.btn_run = QPushButton("▶  Run Post-Processing")
        self.btn_run.setObjectName("successBtn")
        self.btn_run.clicked.connect(self._run_postprocess)
        ctrl_layout.addWidget(self.btn_run)

        self.pp_progress = QProgressBar()
        self.pp_progress.setVisible(False)
        ctrl_layout.addWidget(self.pp_progress)

        # History
        hist_grp = QGroupBox("Processing History")
        hist_lay = QVBoxLayout(hist_grp)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(120)
        hist_lay.addWidget(self.history_list)
        ctrl_layout.addWidget(hist_grp)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_widget)
        splitter.addWidget(ctrl_scroll)

        # ── Results viewer ────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)

        self.result_canvas = MplCanvas(figsize=(8, 6), toolbar=True)
        right_lay.addWidget(self.result_canvas)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        right_lay.addWidget(self.log_edit)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    def _run_postprocess(self):
        if not self.main_window:
            return

        analysis_page = self.main_window.pages.get("analysis")
        session = analysis_page.get_session() if analysis_page else None

        if session is None:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return

        config = {}
        config.update(self.grid_params.get_values())
        config.update(self.scale_params.get_values())
        config.update(self.val_params.get_values())

        self.pp_progress.setVisible(True)
        self.pp_progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.log_edit.clear()

        self._worker = PostProcessWorker(session, config)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.pp_progress.setValue)
        self._worker.log_msg.connect(lambda m: self.log_edit.append(m))
        self._worker.start()

    def _on_finished(self, results):
        self._results = results
        self.pp_progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.log_edit.append("\n✓ Post-processing complete!")

        # Add to history
        record = PostProcessRecord(
            description=f"Grid={results['config'].get('grid_step', 10)}, "
                       f"Smooth={results['config'].get('smoothness', 1e-3)}",
            config=results["config"],
        )
        item = QListWidgetItem(f"[{record.timestamp}] {record.description}")
        item.setData(Qt.UserRole, record.record_id)
        self.history_list.addItem(item)

        # Save to experiment
        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                exp.postprocess_runs.append(record)
                exp.store_postprocess_results(results)
                self.main_window.exp_manager.update(exp.exp_id)

        # Quick preview plot
        self._preview_results(results)

    def _on_error(self, msg):
        self.pp_progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.log_edit.append(f"\n❌ Error: {msg}")
        QMessageBox.warning(self, "Post-Processing Error", msg)

    def _preview_results(self, results):
        """Show a quick preview of the computed fields."""
        self.result_canvas.clear()

        valid_frames = [f for f in results.get("frames", []) if f is not None]
        if not valid_frames:
            return

        # Show last frame's displacement field
        last = valid_frames[-1]
        disp = np.array(last["disp_components"])
        ndim = disp.shape[0]

        if ndim >= 2:
            n_plots = min(ndim, 3)
            labels = ["u_x", "u_y", "u_z"][:n_plots]

            for i in range(n_plots):
                ax = self.result_canvas.add_subplot(1, n_plots, i + 1)
                comp = disp[i]
                if comp.ndim == 3:
                    comp = comp[:, :, comp.shape[2]//2]
                im = ax.imshow(comp.T, cmap="coolwarm", origin="lower")
                ax.set_title(labels[i], color="#f8f8f2", fontsize=10)
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.05)
                self.result_canvas.figure.colorbar(im, cax=cax)

        self.result_canvas.draw()

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def on_experiment_changed(self, exp_id: str):
        # Load history for this experiment
        self.history_list.clear()
        if self.main_window:
            exp = self.main_window.exp_manager.get(exp_id)
            if exp:
                for r in exp.postprocess_runs:
                    item = QListWidgetItem(f"[{r.timestamp}] {r.description}")
                    item.setData(Qt.UserRole, r.record_id)
                    self.history_list.addItem(item)

    def on_activated(self):
        pass

    # ── Experiment switching ──────────────────────────────────

    def save_to_experiment(self, exp):
        """Persist current post-process results into experiment cache."""
        exp.store_postprocess_results(self._results)

    def load_from_experiment(self, exp):
        """Restore post-process results from experiment cache."""
        self._results = exp.get_postprocess_results() or {}

        # Refresh history list
        self.history_list.clear()
        for r in exp.postprocess_runs:
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(f"[{r.timestamp}] {r.description}")
            item.setData(Qt.UserRole, r.record_id)
            self.history_list.addItem(item)

        # Refresh frame selector and preview
        frames = self._results.get("frames", [])
        if frames:
            self.sb_frame.setMaximum(max(0, len(frames) - 1))
            self._update_preview()
            self.status.set_status("ready", f"{len(frames)} frames loaded")
        else:
            self.sb_frame.setMaximum(0)
            self.result_canvas.clear()
            self.result_canvas.draw()
            self.status.set_status("idle", "No post-process data")