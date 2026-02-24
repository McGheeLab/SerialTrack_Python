"""
Analysis Page — Run tracking with real-time progress, ETA, and live plots.

Shows: tracking ratio over time, number of beads tracked, timing,
displacement statistics — all updating live during the run.
"""
from __future__ import annotations
from typing import Optional, List
import time
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QProgressBar, QTextEdit, QFrame,
    QCheckBox, QMessageBox, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer

from widgets.common import MplCanvas, StatusIndicator
from core.settings import Settings


class TrackingWorker(QThread):
    """Run SerialTrack in background thread."""
    frame_done = Signal(dict)       # per-frame summary
    progress = Signal(int, int)     # current, total
    log_msg = Signal(str)
    finished = Signal(object)       # TrackingSession
    error = Signal(str)

    def __init__(self, volumes, det_config, trk_config, mask=None, parent=None):
        super().__init__(parent)
        self.volumes = volumes
        self.det_config = det_config
        self.trk_config = trk_config
        self.mask = mask
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from serialtrack.config import (
                DetectionConfig, DetectionMethod, TrackingConfig,
                TrackingMode, GlobalSolver, LocalSolver, TrajectoryConfig,
            )
            from serialtrack.tracking import SerialTracker

            # Build detection config
            method_map = {"TracTrac (LoG)": 2, "TPT (Radial Symmetry)": 1}
            det_cfg = DetectionConfig(
                method=DetectionMethod(method_map.get(
                    self.det_config.get("method", "TracTrac (LoG)"), 2)),
                threshold=self.det_config.get("threshold", 0.4),
                bead_radius=self.det_config.get("bead_radius", 3.0),
                min_size=int(self.det_config.get("min_size", 2)),
                max_size=int(self.det_config.get("max_size", 1000)),
                color=self.det_config.get("color", "white"),
            )

            # Build tracking config
            mode_map = {"Incremental": 1, "Cumulative": 2, "Double Frame": 3}
            solver_map = {"MLS": 1, "Regularization": 2, "ADMM": 3}
            loc_map = {"Topology": 1, "Histogram then Topology": 2}

            traj_cfg = TrajectoryConfig(
                dist_threshold=self.trk_config.get("traj_dist_threshold", 1.0),
                extrap_method=self.trk_config.get("traj_extrap_method", "pchip"),
                min_segment_length=int(self.trk_config.get("traj_min_segment", 10)),
                max_gap_length=int(self.trk_config.get("traj_max_gap", 0)),
                merge_passes=int(self.trk_config.get("traj_merge_passes", 4)),
            )

            trk_cfg = TrackingConfig(
                mode=TrackingMode(mode_map.get(
                    self.trk_config.get("mode", "Incremental"), 1)),
                f_o_s=self.trk_config.get("f_o_s", 60.0),
                n_neighbors_max=int(self.trk_config.get("n_neighbors_max", 25)),
                n_neighbors_min=int(self.trk_config.get("n_neighbors_min", 1)),
                loc_solver=LocalSolver(loc_map.get(
                    self.trk_config.get("loc_solver", "Topology"), 1)),
                solver=GlobalSolver(solver_map.get(
                    self.trk_config.get("solver", "Regularization"), 2)),
                smoothness=self.trk_config.get("smoothness", 0.1),
                outlier_threshold=self.trk_config.get("outlier_threshold", 5.0),
                max_iter=int(self.trk_config.get("max_iter", 20)),
                iter_stop_threshold=self.trk_config.get("iter_stop_threshold", 1e-2),
                strain_n_neighbors=int(self.trk_config.get("strain_n_neighbors", 20)),
                strain_f_o_s=self.trk_config.get("strain_f_o_s", 60.0),
                use_prev_results=self.trk_config.get("use_prev_results", False),
                dist_missing=self.trk_config.get("dist_missing", 5.0),
                xstep=self.trk_config.get("xstep", 1.0),
                ystep=self.trk_config.get("ystep", 1.0),
                zstep=self.trk_config.get("zstep", 1.0),
                tstep=self.trk_config.get("tstep", 1.0),
                trajectory=traj_cfg,
                mask=self.mask,
            )

            # Progress callback
            def progress_cb(frame_idx, total_frames, frame_result):
                if self._cancelled:
                    raise KeyboardInterrupt("Cancelled")

                n_tracked = int(np.sum(frame_result.track_a2b >= 0))
                disp_mag = np.linalg.norm(frame_result.disp_b2a, axis=1)
                tracked = frame_result.track_b2a >= 0
                t_disp = disp_mag[tracked] if np.any(tracked) else disp_mag

                summary = {
                    "frame": frame_result.frame_idx,
                    "detected": len(frame_result.coords_b),
                    "tracked": n_tracked,
                    "ratio": frame_result.match_ratio,
                    "iters": frame_result.n_iterations,
                    "time": frame_result.wall_time,
                    "mean_disp": float(np.mean(t_disp)) if len(t_disp) > 0 else 0,
                    "max_disp": float(np.max(t_disp)) if len(t_disp) > 0 else 0,
                    "rms_disp": float(np.sqrt(np.mean(t_disp**2))) if len(t_disp) > 0 else 0,
                }
                self.frame_done.emit(summary)
                self.progress.emit(frame_idx, total_frames)
                self.log_msg.emit(
                    f"Frame {frame_idx}/{total_frames}: "
                    f"ratio={summary['ratio']:.3f}, "
                    f"tracked={summary['tracked']}, "
                    f"time={summary['time']:.2f}s"
                )

            tracker = SerialTracker(det_cfg, trk_cfg)
            self.log_msg.emit("Starting tracking...")
            session = tracker.track_images(self.volumes, progress_cb=progress_cb)
            self.finished.emit(session)

        except KeyboardInterrupt:
            self.log_msg.emit("Tracking cancelled by user.")
        except Exception as e:
            self.error.emit(str(e))


class AnalysisPage(QWidget):
    """Analysis runner with live plotting."""
    analysis_complete = Signal(object)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._worker: Optional[TrackingWorker] = None
        self._frame_data: List[dict] = []
        self._session = None
        self._start_time = 0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Top control bar ───────────────────────────────────
        ctrl_row = QHBoxLayout()

        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setObjectName("successBtn")
        self.btn_run.setMinimumWidth(160)
        self.btn_run.clicked.connect(self._start_analysis)
        ctrl_row.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("⬛  Cancel")
        self.btn_cancel.setObjectName("dangerBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_analysis)
        ctrl_row.addWidget(self.btn_cancel)

        ctrl_row.addSpacing(20)

        self.cb_concurrent_pp = QCheckBox("Concurrent Post-Processing")
        self.cb_concurrent_pp.setToolTip(
            "Run post-processing concurrently as tracking data becomes available"
        )
        ctrl_row.addWidget(self.cb_concurrent_pp)

        ctrl_row.addStretch()

        self.status_label = StatusIndicator("Ready")
        ctrl_row.addWidget(self.status_label)

        layout.addLayout(ctrl_row)

        # Progress
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        progress_row.addWidget(self.progress_bar)
        self.eta_label = QLabel("ETA: --")
        self.eta_label.setMinimumWidth(120)
        progress_row.addWidget(self.eta_label)
        layout.addLayout(progress_row)

        # ── Main content: plots + log ─────────────────────────
        splitter = QSplitter(Qt.Vertical)

        # Plots area
        plots_widget = QWidget()
        plots_layout = QHBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(4)

        # Tracking ratio plot
        self.ratio_canvas = MplCanvas(figsize=(4, 3), toolbar=False)
        plots_layout.addWidget(self.ratio_canvas)

        # Beads tracked plot
        self.beads_canvas = MplCanvas(figsize=(4, 3), toolbar=False)
        plots_layout.addWidget(self.beads_canvas)

        # Displacement / timing plot
        self.disp_canvas = MplCanvas(figsize=(4, 3), toolbar=False)
        plots_layout.addWidget(self.disp_canvas)

        splitter.addWidget(plots_widget)

        # Log output
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(200)
        self.log_edit.setStyleSheet(
            "QTextEdit { background-color: #1a1c24; color: #8be9fd; "
            "font: 9pt 'Consolas'; border: 1px solid #44475a; border-radius: 5px; }"
        )
        splitter.addWidget(self.log_edit)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _start_analysis(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Running", "Analysis is already running.")
            return

        # Get data from other pages
        if not self.main_window:
            return

        images_page = self.main_window.pages.get("images")
        detection_page = self.main_window.pages.get("detection")
        params_page = self.main_window.pages.get("parameters")
        mask_page = self.main_window.pages.get("mask")

        volumes = images_page.get_volumes() if images_page else []
        if not volumes or len(volumes) < 2:
            QMessageBox.warning(self, "No Data", "Load at least 2 images first.")
            return

        det_config = detection_page.det_params.get_values() if detection_page else {}
        trk_config = params_page.get_config() if params_page else {}
        mask = mask_page.get_mask() if mask_page else None

        # Reset state
        self._frame_data.clear()
        self.log_edit.clear()
        self._start_time = time.time()

        # Update UI
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, len(volumes))
        self.progress_bar.setValue(0)
        self.status_label.set_status("running", "Running...")

        # Update experiment status
        exp = self.main_window.exp_manager.active
        if exp:
            self.main_window.exp_manager.update(exp.exp_id, status="tracking")

        # Start worker
        self._worker = TrackingWorker(volumes, det_config, trk_config, mask)
        self._worker.frame_done.connect(self._on_frame_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_msg.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self._on_log("Analysis started...")

    def _cancel_analysis(self):
        if self._worker:
            self._worker.cancel()
            self._on_log("Cancellation requested...")

    def _on_frame_done(self, summary: dict):
        self._frame_data.append(summary)
        self._update_plots()

    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)

        # ETA calculation
        elapsed = time.time() - self._start_time
        frames_done = current - 1  # subtract reference
        if frames_done > 0:
            time_per_frame = elapsed / frames_done
            remaining = (total - current) * time_per_frame
            if remaining > 3600:
                eta_str = f"{remaining/3600:.1f}h"
            elif remaining > 60:
                eta_str = f"{remaining/60:.1f}m"
            else:
                eta_str = f"{remaining:.0f}s"
            self.eta_label.setText(f"ETA: {eta_str}")

    def _on_log(self, msg: str):
        self.log_edit.append(msg)
        # Auto scroll
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, session):
        self._session = session
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        elapsed = time.time() - self._start_time
        n_frames = len(self._frame_data)
        mean_ratio = np.mean([d["ratio"] for d in self._frame_data]) if self._frame_data else 0

        self.status_label.set_status("ready", "Complete")
        self.eta_label.setText(f"Done: {elapsed:.1f}s")
        self._on_log(
            f"\n{'='*50}\n"
            f"Analysis complete!\n"
            f"  Frames: {n_frames}\n"
            f"  Mean tracking ratio: {mean_ratio:.4f}\n"
            f"  Total time: {elapsed:.1f}s\n"
            f"{'='*50}"
        )

        # Update experiment
        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                exp.mean_tracking_ratio = mean_ratio
                exp.store_tracking_session(session)
                exp.store_frame_data(self._frame_data)
                self.main_window.exp_manager.update(
                    exp.exp_id, status="complete",
                    mean_tracking_ratio=mean_ratio,
                    n_frames=n_frames
                )
            self.main_window.set_status("Analysis complete", "ready")

        self.analysis_complete.emit(session)
        self._update_plots()

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.set_status("error", "Error")
        self._on_log(f"\n❌ ERROR: {msg}")
        QMessageBox.critical(self, "Analysis Error", msg)

        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                self.main_window.exp_manager.update(exp.exp_id, status="error")

    def _update_plots(self):
        """Update all live plots with current frame data."""
        if not self._frame_data:
            return

        frames = [d["frame"] for d in self._frame_data]
        ratios = [d["ratio"] for d in self._frame_data]
        tracked = [d["tracked"] for d in self._frame_data]
        detected = [d["detected"] for d in self._frame_data]
        rms_disp = [d["rms_disp"] for d in self._frame_data]
        times = [d["time"] for d in self._frame_data]

        # ── Tracking Ratio Plot ───────────────────────────────
        self.ratio_canvas.clear()
        ax1 = self.ratio_canvas.add_subplot(111)
        ax1.fill_between(frames, ratios, alpha=0.3, color="#bd93f9")
        ax1.plot(frames, ratios, color="#bd93f9", linewidth=2, marker='o',
                markersize=4, markerfacecolor="#ff79c6", markeredgecolor="#ff79c6")
        ax1.set_xlabel("Frame", fontsize=9)
        ax1.set_ylabel("Tracking Ratio", fontsize=9)
        ax1.set_title("Tracking Ratio", fontsize=10, color="#f8f8f2")
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=np.mean(ratios), color="#50fa7b", linestyle="--",
                    alpha=0.6, linewidth=1)
        ax1.grid(True, alpha=0.15, color="#44475a")
        self.ratio_canvas.draw()

        # ── Beads Tracked Plot ────────────────────────────────
        self.beads_canvas.clear()
        ax2 = self.beads_canvas.add_subplot(111)
        ax2.bar(frames, detected, alpha=0.3, color="#44475a",
               label="Detected", width=0.8)
        ax2.bar(frames, tracked, alpha=0.7, color="#8be9fd",
               label="Tracked", width=0.8)
        ax2.set_xlabel("Frame", fontsize=9)
        ax2.set_ylabel("Particles", fontsize=9)
        ax2.set_title("Particles Tracked", fontsize=10, color="#f8f8f2")
        ax2.legend(fontsize=8, framealpha=0.3, facecolor="#21252b",
                  edgecolor="#44475a", labelcolor="#f8f8f2")
        ax2.grid(True, alpha=0.15, color="#44475a")
        self.beads_canvas.draw()

        # ── Displacement / Timing Plot ────────────────────────
        self.disp_canvas.clear()
        ax3 = self.disp_canvas.add_subplot(111)
        ax3_twin = ax3.twinx()
        ax3_twin.set_facecolor("#21252b")

        l1 = ax3.plot(frames, rms_disp, color="#ffb86c", linewidth=2,
                      marker='s', markersize=3, label="RMS Disp")
        l2 = ax3_twin.plot(frames, times, color="#ff79c6", linewidth=1.5,
                           marker='^', markersize=3, label="Wall Time",
                           linestyle="--")

        ax3.set_xlabel("Frame", fontsize=9)
        ax3.set_ylabel("RMS Displacement (px)", fontsize=9, color="#ffb86c")
        ax3_twin.set_ylabel("Wall Time (s)", fontsize=9, color="#ff79c6")
        ax3.set_title("Displacement & Timing", fontsize=10, color="#f8f8f2")
        ax3.tick_params(axis='y', colors="#ffb86c")
        ax3_twin.tick_params(axis='y', colors="#ff79c6")
        for spine in ax3_twin.spines.values():
            spine.set_color("#44475a")

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, fontsize=8, framealpha=0.3,
                  facecolor="#21252b", edgecolor="#44475a", labelcolor="#f8f8f2")
        ax3.grid(True, alpha=0.15, color="#44475a")
        self.disp_canvas.draw()

    def get_session(self):
        return self._session

    # ── Experiment switching ──────────────────────────────────

    def save_to_experiment(self, exp):
        """Persist current state into experiment record cache."""
        exp.store_tracking_session(self._session)
        exp.store_frame_data(self._frame_data)
        if self._session and hasattr(self._session, 'frame_results'):
            exp.n_frames = len(self._session.frame_results)

    def load_from_experiment(self, exp):
        """Restore state from experiment record cache and refresh UI."""
        self._session = exp.get_tracking_session()
        self._frame_data = exp.get_frame_data() or []

        # Refresh live plots
        if self._frame_data:
            self._update_plots()
            n = len(self._frame_data)
            mean_ratio = np.mean([d.get("ratio", 0) for d in self._frame_data])
            self.status_label.set_status("ready", "Complete")
            self.eta_label.setText(f"{n} frames, ratio={mean_ratio:.3f}")
        else:
            # Clear plots for empty experiment
            self.ratio_canvas.clear()
            self.ratio_canvas.draw()
            self.beads_canvas.clear()
            self.beads_canvas.draw()
            self.disp_canvas.clear()
            self.disp_canvas.draw()
            self.status_label.set_status("idle", "No data")
            self.eta_label.setText("")
            self.log_edit.clear()

    def on_experiment_changed(self, exp_id: str):
        """Legacy hook — switching is now handled by save/load_from_experiment."""
        pass

    def on_activated(self):
        pass
    