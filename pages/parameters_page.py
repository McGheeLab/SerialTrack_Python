"""
Parameters Page — All SerialTrack tracking parameters with tooltips.

Includes auto-estimation based on bead density, bead size, etc.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox,
    QPushButton, QLabel, QTabWidget, QFrame, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from widgets.common import ParamEditor
from core.settings import Settings
from core.plugin_registry import ParamSpec


class ParametersPage(QWidget):
    """Tracking parameter configuration with tooltips and auto-estimation."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)

        # Auto-estimate banner
        auto_frame = QFrame()
        auto_frame.setStyleSheet(
            f"background-color: {Settings.BG_TERTIARY}; "
            f"border: 1px solid {Settings.ACCENT_CYAN}; border-radius: 8px; padding: 8px;"
        )
        auto_lay = QHBoxLayout(auto_frame)
        auto_lay.addWidget(QLabel("💡 Auto-estimate tracking parameters from known bead properties"))
        self.btn_auto = QPushButton("Auto-Estimate")
        self.btn_auto.setObjectName("primaryBtn")
        self.btn_auto.clicked.connect(self._auto_estimate)
        auto_lay.addWidget(self.btn_auto)
        content_layout.addWidget(auto_frame)

        # Tabs for parameter groups
        tabs = QTabWidget()

        # ── Tracking Mode & Search ────────────────────────────
        tracking_tab = QWidget()
        t_lay = QVBoxLayout(tracking_tab)

        self.tracking_params = ParamEditor()
        self.tracking_params.set_params([
            ParamSpec("mode", "Tracking Mode", "choice", "Incremental",
                      choices=["Incremental", "Cumulative", "Double Frame"],
                      tooltip="Incremental: frame-to-frame tracking (standard).\n"
                              "Cumulative: all frames tracked vs reference frame.\n"
                              "Double Frame: independent frame pairs (no trajectory linking)."),
            ParamSpec("f_o_s", "Field of Search (px)", "float", 60.0, 1.0, 500.0, 5.0,
                      tooltip="Maximum expected displacement between frames in pixels.\n"
                              "Should be ~3× the max bead displacement. Larger = slower but catches bigger motions."),
            ParamSpec("n_neighbors_max", "Max Neighbors", "int", 25, 3, 100, 1,
                      tooltip="Maximum number of neighbors for topology matching.\n"
                              "More neighbors = more robust matching but slower."),
            ParamSpec("n_neighbors_min", "Min Neighbors", "int", 1, 1, 20, 1,
                      tooltip="Minimum number of neighbors. Set low (1-3) for sparse fields."),
            ParamSpec("loc_solver", "Local Solver", "choice", "Topology",
                      choices=["Topology", "Histogram then Topology"],
                      tooltip="Topology: Uses neighborhood topology for matching (robust).\n"
                              "Histogram: Pre-filters by displacement histogram, then topology."),
        ])
        t_lay.addWidget(self.tracking_params)
        tabs.addTab(tracking_tab, "Tracking Mode")

        # ── Global Solver ─────────────────────────────────────
        solver_tab = QWidget()
        s_lay = QVBoxLayout(solver_tab)

        self.solver_params = ParamEditor()
        self.solver_params.set_params([
            ParamSpec("solver", "Global Solver", "choice", "Regularization",
                      choices=["MLS", "Regularization", "ADMM"],
                      tooltip="MLS: Moving least-squares (fast, less robust).\n"
                              "Regularization: Scatter→grid regularization (balanced).\n"
                              "ADMM: Augmented Lagrangian with L-curve (most robust, slowest)."),
            ParamSpec("smoothness", "Smoothness", "float", 0.1, 1e-6, 100.0, 0.01,
                      tooltip="Regularization strength. Higher = smoother displacement field.\n"
                              "Too high: over-smooths real deformation.\n"
                              "Too low: noisy displacement estimates."),
            ParamSpec("max_iter", "Max ADMM Iterations", "int", 20, 1, 100, 1,
                      tooltip="Maximum iterations for the ADMM loop per frame.\n"
                              "More iterations = better convergence but slower."),
            ParamSpec("iter_stop_threshold", "Convergence Threshold", "float", 1e-2, 1e-6, 1.0, 1e-3,
                      tooltip="Stop ADMM when tracking ratio change < this value.\n"
                              "Smaller = stricter convergence."),
        ])
        s_lay.addWidget(self.solver_params)
        tabs.addTab(solver_tab, "Global Solver")

        # ── Outlier & Filtering ───────────────────────────────
        outlier_tab = QWidget()
        o_lay = QVBoxLayout(outlier_tab)

        self.outlier_params = ParamEditor()
        self.outlier_params.set_params([
            ParamSpec("outlier_threshold", "Outlier Threshold", "float", 5.0, 1.0, 50.0, 0.5,
                      tooltip="Westerweel universal outlier threshold in units of median residual.\n"
                              "Lower = more aggressive outlier removal.\n"
                              "5 is typical; reduce to 3 for noisy data."),
            ParamSpec("dist_missing", "Missing Particle Distance", "float", 5.0, 0.5, 50.0, 0.5,
                      tooltip="Distance threshold to classify a particle as 'missing' (untracked).\n"
                              "Particles with displacement > this are flagged."),
        ])
        o_lay.addWidget(self.outlier_params)
        tabs.addTab(outlier_tab, "Outliers")

        # ── Strain Gauge ──────────────────────────────────────
        strain_tab = QWidget()
        st_lay = QVBoxLayout(strain_tab)

        self.strain_params = ParamEditor()
        self.strain_params.set_params([
            ParamSpec("strain_n_neighbors", "Strain Gauge Neighbors", "int", 20, 3, 100, 1,
                      tooltip="Number of neighbors for MLS strain gauge computation.\n"
                              "More = smoother but less local."),
            ParamSpec("strain_f_o_s", "Strain Field of Search", "float", 60.0, 1.0, 500.0, 5.0,
                      tooltip="Search radius for strain gauge neighbor lookup."),
            ParamSpec("use_prev_results", "Use Previous Results", "bool", False,
                      tooltip="Use displacement from previous frames as initial guess.\n"
                              "Can improve convergence for smooth time-varying deformations."),
        ])
        st_lay.addWidget(self.strain_params)
        tabs.addTab(strain_tab, "Strain / Prediction")

        # ── Physical Scaling ──────────────────────────────────
        scale_tab = QWidget()
        sc_lay = QVBoxLayout(scale_tab)

        self.scale_params = ParamEditor()
        self.scale_params.set_params([
            ParamSpec("xstep", "X Step (µm/px)", "float", 1.0, 0.001, 100.0, 0.01,
                      tooltip="Physical size of one pixel in X direction."),
            ParamSpec("ystep", "Y Step (µm/px)", "float", 1.0, 0.001, 100.0, 0.01,
                      tooltip="Physical size of one pixel in Y direction."),
            ParamSpec("zstep", "Z Step (µm/px)", "float", 1.0, 0.001, 100.0, 0.01,
                      tooltip="Physical size of one pixel/voxel in Z direction."),
            ParamSpec("tstep", "Time Step (s)", "float", 1.0, 0.001, 3600.0, 0.1,
                      tooltip="Time between frames in seconds."),
        ])
        sc_lay.addWidget(self.scale_params)
        tabs.addTab(scale_tab, "Physical Scales")

        # ── Trajectory Stitching ──────────────────────────────
        traj_tab = QWidget()
        tj_lay = QVBoxLayout(traj_tab)

        self.traj_params = ParamEditor()
        self.traj_params.set_params([
            ParamSpec("traj_dist_threshold", "Distance Threshold (px)", "float", 1.0, 0.1, 50.0, 0.1,
                      tooltip="Max distance to connect split trajectory segments."),
            ParamSpec("traj_extrap_method", "Extrapolation", "choice", "pchip",
                      choices=["pchip", "nearest"],
                      tooltip="pchip: Smooth motion extrapolation.\n"
                              "nearest: For Brownian/diffusive motion."),
            ParamSpec("traj_min_segment", "Min Segment Length", "int", 10, 1, 1000, 1,
                      tooltip="Minimum frames in a trajectory segment to attempt stitching."),
            ParamSpec("traj_max_gap", "Max Gap Length", "int", 0, 0, 20, 1,
                      tooltip="Maximum allowed frame gap between connected segments.\n"
                              "0 = segments must be adjacent."),
            ParamSpec("traj_merge_passes", "Merge Passes", "int", 4, 1, 10, 1,
                      tooltip="Number of iterative merge passes. More = more stitching."),
        ])
        tj_lay.addWidget(self.traj_params)
        tabs.addTab(traj_tab, "Trajectory Stitching")

        content_layout.addWidget(tabs)

        # Save/Apply buttons
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Experiment")
        self.btn_apply.setObjectName("primaryBtn")
        self.btn_apply.clicked.connect(self._apply_config)
        btn_row.addWidget(self.btn_apply)

        self.btn_reset = QPushButton("Reset Defaults")
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _auto_estimate(self):
        """Estimate tracking params from bead properties."""
        if self.main_window and "detection" in self.main_window.pages:
            det_page = self.main_window.pages["detection"]
            auto_vals = det_page.auto_params.get_values()
            bead_um = auto_vals.get("bead_diameter_um", 1.0)
            px_um = auto_vals.get("pixel_size_um", 0.5)
            density = auto_vals.get("bead_density", "medium")

            bead_px = bead_um / px_um

            # f_o_s ~ 10-20× bead diameter for typical TFM
            f_o_s = max(20, bead_px * 15)
            n_max = {"sparse": 15, "medium": 25, "dense": 40}.get(density, 25)

            self.tracking_params.set_values({
                "f_o_s": round(f_o_s, 1),
                "n_neighbors_max": n_max,
            })
            self.strain_params.set_values({
                "strain_f_o_s": round(f_o_s, 1),
            })
            self.scale_params.set_values({
                "xstep": px_um,
                "ystep": px_um,
            })

            if self.main_window:
                self.main_window.set_status("Parameters auto-estimated", "info")
        else:
            QMessageBox.information(
                self, "Auto-Estimate",
                "Set bead properties in Detection page first,\n"
                "then come back here to auto-estimate."
            )

    def _apply_config(self):
        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                exp.tracking_config = self.get_config()
                self.main_window.exp_manager.update(exp.exp_id)
                self.main_window.set_status("Parameters applied", "ready")

    def _reset_defaults(self):
        self.tracking_params.set_values({
            "mode": "Incremental", "f_o_s": 60.0,
            "n_neighbors_max": 25, "n_neighbors_min": 1,
            "loc_solver": "Topology",
        })
        self.solver_params.set_values({
            "solver": "Regularization", "smoothness": 0.1,
            "max_iter": 20, "iter_stop_threshold": 0.01,
        })

    def get_config(self) -> dict:
        """Return complete tracking configuration dict."""
        cfg = {}
        cfg.update(self.tracking_params.get_values())
        cfg.update(self.solver_params.get_values())
        cfg.update(self.outlier_params.get_values())
        cfg.update(self.strain_params.get_values())
        cfg.update(self.scale_params.get_values())
        cfg.update(self.traj_params.get_values())
        return cfg

    def on_experiment_changed(self, exp_id: str):
        exp = self.main_window.exp_manager.get(exp_id) if self.main_window else None
        if exp and exp.tracking_config:
            for editor in [self.tracking_params, self.solver_params,
                          self.outlier_params, self.strain_params,
                          self.scale_params, self.traj_params]:
                editor.set_values(exp.tracking_config)

    def on_activated(self):
        pass
