"""
Plots & Visualization Page — Publication-quality rendering of tracking results.

Supports:
  - Multiple data sources: displacement, velocity, strain, stress, von Mises
  - Plot types: heatmap, contour, quiver/vector, combined overlay
  - Colormap library with preview
  - 2D slice control for 3D datasets
  - Multi-component side-by-side or single-component focus
  - Export to PNG, SVG, PDF, TIFF at custom DPI
  - Customizable titles, labels, colorbar range
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QScrollArea, QFrame,
    QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTabWidget, QLineEdit, QSlider,
)
from PySide6.QtCore import Qt, Signal

from widgets.common import MplCanvas, ParamEditor, StatusIndicator
from core.settings import Settings
from core.plugin_registry import ParamSpec


# ═══════════════════════════════════════════════════════════════════
#  Colormap presets
# ═══════════════════════════════════════════════════════════════════

COLORMAPS = {
    "Sequential": [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "hot", "YlOrRd", "YlGnBu", "Blues", "Greens", "Reds",
    ],
    "Diverging": [
        "coolwarm", "RdBu_r", "seismic", "PiYG", "PRGn", "BrBG",
    ],
    "Perceptual": [
        "turbo", "twilight", "twilight_shifted",
    ],
}


# ═══════════════════════════════════════════════════════════════════
#  PlotsPage
# ═══════════════════════════════════════════════════════════════════

class PlotsPage(QWidget):
    """Publication-quality visualization of tracking & field results."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._cached_data = {}  # last loaded data arrays
        self._init_ui()

    # ───── UI ─────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: controls ──
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setMinimumWidth(300)
        left.setMaximumWidth(420)
        left_inner = QWidget()
        left_lay = QVBoxLayout(left_inner)
        left_lay.setSpacing(6)
        left.setWidget(left_inner)

        # --- Data Source ---
        src_grp = QGroupBox("Data Source")
        src_lay = QVBoxLayout(src_grp)

        src_lay.addWidget(QLabel("Source:"))
        self.cb_source = QComboBox()
        self.cb_source.addItems([
            "Displacement", "Velocity", "Strain (ε)",
            "Stress (σ)", "Von Mises Stress",
            "Deformation Gradient (F)", "Particle Trajectories",
        ])
        self.cb_source.currentTextChanged.connect(self._on_source_changed)
        src_lay.addWidget(self.cb_source)

        src_lay.addWidget(QLabel("Component:"))
        self.cb_component = QComboBox()
        self._update_components("Displacement")
        self.cb_component.currentIndexChanged.connect(self._refresh_plot)
        src_lay.addWidget(self.cb_component)

        # Frame selector
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self.sb_frame = QSpinBox()
        self.sb_frame.setMinimum(0)
        self.sb_frame.valueChanged.connect(self._refresh_plot)
        frame_row.addWidget(self.sb_frame)
        src_lay.addLayout(frame_row)

        # Z-slice for 3D
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z slice:"))
        self.sl_z = QSlider(Qt.Horizontal)
        self.sl_z.setMinimum(0)
        self.sl_z.valueChanged.connect(self._refresh_plot)
        z_row.addWidget(self.sl_z)
        self.lbl_z = QLabel("0")
        z_row.addWidget(self.lbl_z)
        src_lay.addLayout(z_row)

        left_lay.addWidget(src_grp)

        # --- Plot Type ---
        plot_grp = QGroupBox("Plot Type")
        plot_lay = QVBoxLayout(plot_grp)

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems([
            "Heatmap", "Filled Contour", "Line Contour",
            "Quiver (Vector Field)", "Heatmap + Quiver Overlay",
        ])
        self.cb_plot_type.currentTextChanged.connect(self._refresh_plot)
        plot_lay.addWidget(self.cb_plot_type)

        # Quiver options
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Quiver density:"))
        self.sb_quiver_skip = QSpinBox()
        self.sb_quiver_skip.setMinimum(1)
        self.sb_quiver_skip.setMaximum(20)
        self.sb_quiver_skip.setValue(3)
        self.sb_quiver_skip.setToolTip("Show every Nth arrow. Higher = sparser.")
        self.sb_quiver_skip.valueChanged.connect(self._refresh_plot)
        q_row.addWidget(self.sb_quiver_skip)
        plot_lay.addLayout(q_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Arrow scale:"))
        self.dsb_quiver_scale = QDoubleSpinBox()
        self.dsb_quiver_scale.setRange(0.01, 50.0)
        self.dsb_quiver_scale.setValue(1.0)
        self.dsb_quiver_scale.setSingleStep(0.1)
        self.dsb_quiver_scale.setDecimals(2)
        self.dsb_quiver_scale.setToolTip("Triangle size. Smaller = smaller triangles.")
        self.dsb_quiver_scale.valueChanged.connect(self._refresh_plot)
        scale_row.addWidget(self.dsb_quiver_scale)
        plot_lay.addLayout(scale_row)

        # Contour levels
        cnt_row = QHBoxLayout()
        cnt_row.addWidget(QLabel("Contour levels:"))
        self.sb_contour_levels = QSpinBox()
        self.sb_contour_levels.setRange(5, 100)
        self.sb_contour_levels.setValue(20)
        self.sb_contour_levels.valueChanged.connect(self._refresh_plot)
        cnt_row.addWidget(self.sb_contour_levels)
        plot_lay.addLayout(cnt_row)

        left_lay.addWidget(plot_grp)

        # --- Colormap ---
        cmap_grp = QGroupBox("Colormap")
        cmap_lay = QVBoxLayout(cmap_grp)

        self.cb_cmap_cat = QComboBox()
        self.cb_cmap_cat.addItems(list(COLORMAPS.keys()))
        self.cb_cmap_cat.currentTextChanged.connect(self._on_cmap_cat_changed)
        cmap_lay.addWidget(self.cb_cmap_cat)

        self.cb_cmap = QComboBox()
        self._on_cmap_cat_changed("Sequential")
        self.cb_cmap.currentTextChanged.connect(self._refresh_plot)
        cmap_lay.addWidget(self.cb_cmap)

        self.chk_reverse = QCheckBox("Reverse colormap")
        self.chk_reverse.stateChanged.connect(self._refresh_plot)
        cmap_lay.addWidget(self.chk_reverse)

        # Colorbar range
        range_row = QHBoxLayout()
        self.chk_auto_range = QCheckBox("Auto range")
        self.chk_auto_range.setChecked(True)
        self.chk_auto_range.stateChanged.connect(self._refresh_plot)
        range_row.addWidget(self.chk_auto_range)
        cmap_lay.addLayout(range_row)

        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("Min:"))
        self.dsb_vmin = QDoubleSpinBox()
        self.dsb_vmin.setRange(-1e9, 1e9)
        self.dsb_vmin.setDecimals(4)
        clim_row.addWidget(self.dsb_vmin)
        clim_row.addWidget(QLabel("Max:"))
        self.dsb_vmax = QDoubleSpinBox()
        self.dsb_vmax.setRange(-1e9, 1e9)
        self.dsb_vmax.setValue(1.0)
        self.dsb_vmax.setDecimals(4)
        clim_row.addWidget(self.dsb_vmax)
        cmap_lay.addLayout(clim_row)

        left_lay.addWidget(cmap_grp)

        # --- Labels ---
        label_grp = QGroupBox("Labels & Title")
        label_lay = QVBoxLayout(label_grp)

        self.le_title = QLineEdit()
        self.le_title.setPlaceholderText("Auto-generated title")
        label_lay.addWidget(QLabel("Title:"))
        label_lay.addWidget(self.le_title)

        self.le_xlabel = QLineEdit("x (px)")
        label_lay.addWidget(QLabel("X label:"))
        label_lay.addWidget(self.le_xlabel)

        self.le_ylabel = QLineEdit("y (px)")
        label_lay.addWidget(QLabel("Y label:"))
        label_lay.addWidget(self.le_ylabel)

        self.le_cbar_label = QLineEdit()
        self.le_cbar_label.setPlaceholderText("Auto")
        label_lay.addWidget(QLabel("Colorbar label:"))
        label_lay.addWidget(self.le_cbar_label)

        # Font controls
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font:"))
        self.cb_font_family = QComboBox()
        self.cb_font_family.addItems([
            "Helvetica Neue", "Arial", "Times New Roman",
            "Courier New", "DejaVu Sans", "Segoe UI",
        ])
        self.cb_font_family.currentTextChanged.connect(self._refresh_plot)
        font_row.addWidget(self.cb_font_family)
        label_lay.addLayout(font_row)

        fsize_row = QHBoxLayout()
        fsize_row.addWidget(QLabel("Size:"))
        self.sb_font_size = QSpinBox()
        self.sb_font_size.setRange(6, 24)
        self.sb_font_size.setValue(10)
        self.sb_font_size.valueChanged.connect(self._refresh_plot)
        fsize_row.addWidget(self.sb_font_size)
        label_lay.addLayout(fsize_row)

        self.chk_grid = QCheckBox("Show grid lines")
        label_lay.addWidget(self.chk_grid)
        self.chk_grid.stateChanged.connect(self._refresh_plot)

        left_lay.addWidget(label_grp)

        # --- Export ---
        export_grp = QGroupBox("Export")
        export_lay = QVBoxLayout(export_grp)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI:"))
        self.sb_dpi = QSpinBox()
        self.sb_dpi.setRange(72, 1200)
        self.sb_dpi.setValue(300)
        dpi_row.addWidget(self.sb_dpi)
        export_lay.addLayout(dpi_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.cb_format = QComboBox()
        self.cb_format.addItems(["PNG", "SVG", "PDF", "TIFF", "EPS"])
        fmt_row.addWidget(self.cb_format)
        export_lay.addLayout(fmt_row)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("💾  Export Current")
        btn_export.clicked.connect(self._export_current)
        btn_row.addWidget(btn_export)

        btn_export_all = QPushButton("📁  Export All Frames")
        btn_export_all.clicked.connect(self._export_all_frames)
        btn_row.addWidget(btn_export_all)
        export_lay.addLayout(btn_row)

        left_lay.addWidget(export_grp)

        # Quick-plot buttons
        quick_grp = QGroupBox("Quick Plots")
        quick_lay = QVBoxLayout(quick_grp)
        for label, src, comp, ptype in [
            ("Displacement Magnitude", "Displacement", "Magnitude", "Heatmap"),
            ("Von Mises Stress", "Von Mises Stress", "Von Mises", "Heatmap"),
            ("Strain εxx", "Strain (ε)", "ε_xx", "Heatmap"),
            ("Velocity Quiver", "Velocity", "Magnitude", "Quiver (Vector Field)"),
        ]:
            btn = QPushButton(f"⚡ {label}")
            btn.clicked.connect(
                lambda checked, s=src, c=comp, p=ptype:
                self._quick_plot(s, c, p))
            quick_lay.addWidget(btn)
        left_lay.addWidget(quick_grp)

        left_lay.addStretch()
        splitter.addWidget(left)

        # ── Right: canvas ──
        right = QWidget()
        right_lay = QVBoxLayout(right)

        # Refresh button
        top_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄  Refresh")
        btn_refresh.clicked.connect(self._refresh_plot)
        top_row.addWidget(btn_refresh)

        self.status = StatusIndicator()
        top_row.addWidget(self.status)
        top_row.addStretch()
        right_lay.addLayout(top_row)

        # Main canvas
        self.canvas = MplCanvas(figsize=(9, 7), toolbar=True)
        right_lay.addWidget(self.canvas)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    # ───── Component updates ──────────────────────────────────────

    def _update_components(self, source: str):
        self.cb_component.blockSignals(True)
        self.cb_component.clear()

        if source in ("Displacement", "Velocity"):
            self.cb_component.addItems([
                "Magnitude", "u_x / v_x", "u_y / v_y", "u_z / v_z",
                "All Components (side-by-side)",
            ])
        elif "Strain" in source:
            self.cb_component.addItems([
                "ε_xx", "ε_yy", "ε_zz", "ε_xy", "ε_xz", "ε_yz",
                "Effective Strain", "All Normal (side-by-side)",
            ])
        elif "Stress" in source and "Von Mises" not in source:
            self.cb_component.addItems([
                "σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_xz", "σ_yz",
                "Von Mises", "All Normal (side-by-side)",
            ])
        elif "Von Mises" in source:
            self.cb_component.addItems(["Von Mises"])
        elif "Deformation" in source:
            self.cb_component.addItems([
                "F_xx", "F_yy", "F_zz", "F_xy", "F_xz", "F_yz",
                "det(F) (Jacobian)",
            ])
        elif "Trajectories" in source:
            self.cb_component.addItems([
                "XY Projection", "XZ Projection", "YZ Projection",
                "Colored by Displacement", "Colored by Time",
            ])

        self.cb_component.blockSignals(False)

    def _on_source_changed(self, source: str):
        self._update_components(source)
        self._refresh_plot()

    def _on_cmap_cat_changed(self, cat: str):
        self.cb_cmap.blockSignals(True)
        self.cb_cmap.clear()
        self.cb_cmap.addItems(COLORMAPS.get(cat, ["viridis"]))
        self.cb_cmap.blockSignals(False)

    # ───── Data retrieval ─────────────────────────────────────────

    def _get_data(self) -> Optional[Dict[str, Any]]:
        """Fetch the appropriate data from upstream pages."""
        if not self.main_window:
            return None

        source = self.cb_source.currentText()
        frame_idx = self.sb_frame.value()

        if source in ("Displacement", "Velocity"):
            pp_page = self.main_window.pages.get("postprocess")
            pp_res = pp_page.get_results() if pp_page else {}
            frames = pp_res.get("frames", [])
            if frame_idx >= len(frames) or frames[frame_idx] is None:
                return None
            frame = frames[frame_idx]
            disp = np.array(frame["disp_components"])
            grids = [np.array(g) for g in frame["grids"]]

            if source == "Velocity":
                # Use tstep if available
                pp_config = pp_res.get("config", {})
                tstep = pp_config.get("tstep", 1.0)
                psteps = np.array([
                    pp_config.get("xstep", 1.0),
                    pp_config.get("ystep", 1.0),
                    pp_config.get("zstep", 1.0),
                ][:disp.shape[0]])
                vel = disp.copy()
                for d in range(disp.shape[0]):
                    vel[d] = disp[d] * psteps[d] / tstep
                return {"type": "vector", "data": vel, "grids": grids,
                        "ndim": disp.shape[0], "n_frames": len(frames)}
            return {"type": "vector", "data": disp, "grids": grids,
                    "ndim": disp.shape[0], "n_frames": len(frames)}

        elif "Strain" in source or "Deformation" in source:
            pp_page = self.main_window.pages.get("postprocess")
            pp_res = pp_page.get_results() if pp_page else {}
            frames = pp_res.get("frames", [])
            if frame_idx >= len(frames) or frames[frame_idx] is None:
                return None
            frame = frames[frame_idx]
            key = "eps_tensor" if "Strain" in source else "F_tensor"
            tensor = np.array(frame[key])
            grids = [np.array(g) for g in frame["grids"]]
            return {"type": "tensor", "data": tensor, "grids": grids,
                    "ndim": tensor.shape[0], "n_frames": len(frames)}

        elif "Stress" in source or "Von Mises" in source:
            stress_page = self.main_window.pages.get("stress")
            st_res = stress_page.get_results() if stress_page else {}
            frames = st_res.get("frames", [])
            if frame_idx >= len(frames) or frames[frame_idx] is None:
                return None
            frame = frames[frame_idx]
            if "Von Mises" in source:
                vm = np.array(frame["von_mises"])
                return {"type": "scalar", "data": vm,
                        "ndim": frame.get("n_dim", 3),
                        "n_frames": len(frames)}
            sigma = np.array(frame["sigma_tensor"])
            return {"type": "tensor", "data": sigma,
                    "ndim": sigma.shape[0], "n_frames": len(frames)}

        elif "Trajectories" in source:
            analysis_page = self.main_window.pages.get("analysis")
            session = analysis_page.get_session() if analysis_page else None
            if session is None:
                return None
            return {"type": "trajectories", "session": session,
                    "n_frames": len(session.frame_results) if hasattr(session, 'frame_results') else 0}

        return None

    def _extract_scalar(self, info: Dict, component: str) -> Optional[np.ndarray]:
        """Extract a single 2D/3D scalar field from data info."""
        dtype = info["type"]
        data = info["data"]
        ndim = info.get("ndim", 3)

        if dtype == "scalar":
            return data

        elif dtype == "vector":
            # data shape: (D, *grid_shape)
            if "Magnitude" in component:
                return np.sqrt(np.sum(data**2, axis=0))
            comp_map = {"x": 0, "y": 1, "z": 2}
            for key, idx in comp_map.items():
                if f"_{key}" in component and idx < ndim:
                    return data[idx]
            return data[0]  # fallback

        elif dtype == "tensor":
            # data shape: (D, D, *grid_shape)
            tensor_map = {
                "xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
                "xy": (0, 1), "xz": (0, 2), "yz": (1, 2),
            }
            for key, (i, j) in tensor_map.items():
                if key in component and i < ndim and j < ndim:
                    return data[i, j]

            if "Effective" in component:
                # Effective strain = sqrt(2/3 * eps:eps)
                inner = np.sum(data * data, axis=(0, 1))
                return np.sqrt(2.0 / 3.0 * inner)

            if "Von Mises" in component and ndim >= 2:
                s = data
                if ndim == 3:
                    return np.sqrt(0.5 * (
                        (s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2
                        + (s[2,2]-s[0,0])**2
                        + 6*(s[0,1]**2 + s[1,2]**2 + s[0,2]**2)))
                else:
                    return np.sqrt(s[0,0]**2 - s[0,0]*s[1,1]
                                   + s[1,1]**2 + 3*s[0,1]**2)

            if "det(F)" in component or "Jacobian" in component:
                # det(F) at each grid point (F includes displacement gradient)
                grid_shape = data.shape[2:]
                F_full = data.copy()
                for k in range(ndim):
                    F_full[k, k] += 1.0
                n_pts = int(np.prod(grid_shape))
                F_flat = F_full.reshape(ndim, ndim, n_pts)
                det = np.zeros(n_pts)
                for p in range(n_pts):
                    det[p] = np.linalg.det(F_flat[:, :, p])
                return det.reshape(grid_shape)

            return data[0, 0]  # fallback

        return None

    def _slice_2d(self, arr: np.ndarray) -> np.ndarray:
        """If 3D, take Z-slice. If 2D, return as-is."""
        if arr.ndim == 3:
            z = min(self.sl_z.value(), arr.shape[2] - 1)
            self.lbl_z.setText(str(z))
            return arr[:, :, z]
        elif arr.ndim == 2:
            return arr
        return arr

    # ───── Plot rendering ─────────────────────────────────────────

    def _refresh_plot(self):
        info = self._get_data()
        if info is None:
            self.status.set_status("warning", "No data available")
            return

        # Update frame range
        n_frames = info.get("n_frames", 0)
        if n_frames > 0:
            self.sb_frame.setMaximum(n_frames - 1)

        source = self.cb_source.currentText()
        component = self.cb_component.currentText()
        plot_type = self.cb_plot_type.currentText()

        # Handle trajectory plots separately
        if info["type"] == "trajectories":
            self._plot_trajectories(info, component)
            return

        # Side-by-side mode
        if "side-by-side" in component:
            self._plot_multi_component(info, source, plot_type)
            return

        # Single component
        scalar = self._extract_scalar(info, component)
        if scalar is None:
            self.status.set_status("warning", "Could not extract component")
            return

        # Update Z slider for 3D
        if scalar.ndim == 3:
            self.sl_z.setMaximum(scalar.shape[2] - 1)

        field_2d = self._slice_2d(scalar)
        self._render_single(field_2d, info, source, component, plot_type)
        self.status.set_status("ready", "Plot rendered")

    def _get_cmap(self) -> str:
        name = self.cb_cmap.currentText() or "viridis"
        if self.chk_reverse.isChecked():
            name += "_r"
        return name

    def _get_clim(self, data: np.ndarray):
        if self.chk_auto_range.isChecked():
            finite = data[np.isfinite(data)]
            if len(finite) == 0:
                return 0, 1
            return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
        return self.dsb_vmin.value(), self.dsb_vmax.value()

    def _auto_title(self, source: str, component: str) -> str:
        custom = self.le_title.text().strip()
        if custom:
            return custom
        frame = self.sb_frame.value()
        return f"{source} — {component} — Frame {frame}"

    def _get_font_props(self) -> dict:
        """Return font properties from controls."""
        return {
            "fontfamily": self.cb_font_family.currentText(),
            "fontsize": self.sb_font_size.value(),
        }

    def _add_colorbar(self, ax, mappable, label=""):
        """Add a properly-sized colorbar using axes_grid1 divider."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fp = self._get_font_props()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = self.canvas.figure.colorbar(mappable, cax=cax)
        if label:
            cbar.set_label(label, fontsize=fp["fontsize"] - 1,
                           color=Settings.FG_SECONDARY,
                           fontfamily=fp["fontfamily"])
        cbar.ax.tick_params(labelsize=fp["fontsize"] - 2, colors="#b0b0b0")
        return cbar

    def _render_single(self, data_2d: np.ndarray, info: Dict,
                       source: str, component: str, plot_type: str):
        """Render a single 2D field on the canvas."""
        self.canvas.clear()
        ax = self.canvas.add_subplot(1, 1, 1)

        cmap = self._get_cmap()
        vmin, vmax = self._get_clim(data_2d)
        title = self._auto_title(source, component)
        fp = self._get_font_props()
        cbar_label = self.le_cbar_label.text().strip() or component

        if "Quiver" in plot_type and "Overlay" not in plot_type:
            self._render_quiver(ax, info, data_2d)
        elif "Overlay" in plot_type:
            im = ax.imshow(data_2d.T, cmap=cmap, origin="lower",
                           vmin=vmin, vmax=vmax, aspect="equal")
            self._add_colorbar(ax, im, cbar_label)
            self._render_quiver(ax, info, data_2d)
        elif "Filled Contour" in plot_type:
            levels = self.sb_contour_levels.value()
            cf = ax.contourf(data_2d.T, levels=levels, cmap=cmap,
                             vmin=vmin, vmax=vmax, origin="lower")
            self._add_colorbar(ax, cf, cbar_label)
        elif "Line Contour" in plot_type:
            levels = self.sb_contour_levels.value()
            cs = ax.contour(data_2d.T, levels=levels, cmap=cmap,
                            vmin=vmin, vmax=vmax, origin="lower")
            ax.clabel(cs, inline=True, fontsize=max(6, fp["fontsize"] - 3))
        else:
            # Heatmap
            im = ax.imshow(data_2d.T, cmap=cmap, origin="lower",
                           vmin=vmin, vmax=vmax, aspect="equal")
            self._add_colorbar(ax, im, cbar_label)

        ax.set_title(title, color=Settings.FG_PRIMARY,
                     fontsize=fp["fontsize"] + 1, fontfamily=fp["fontfamily"])
        ax.set_xlabel(self.le_xlabel.text(), color=Settings.FG_SECONDARY,
                      fontsize=fp["fontsize"], fontfamily=fp["fontfamily"])
        ax.set_ylabel(self.le_ylabel.text(), color=Settings.FG_SECONDARY,
                      fontsize=fp["fontsize"], fontfamily=fp["fontfamily"])
        ax.tick_params(labelsize=fp["fontsize"] - 1)

        if self.chk_grid.isChecked():
            ax.grid(True, alpha=0.3, color=Settings.FG_SECONDARY)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _render_quiver(self, ax, info: Dict, bg_data: np.ndarray):
        """Overlay triangle markers pointing in displacement direction.

        Uses matplotlib PolyCollection for batch rendering (fast).
        Triangles are sized relative to skip spacing so they fit the grid.
        Scale control: larger value = larger triangles (intuitive).
        """
        if info["type"] != "vector":
            if not self.main_window:
                return
            pp_page = self.main_window.pages.get("postprocess")
            pp_res = pp_page.get_results() if pp_page else {}
            frames = pp_res.get("frames", [])
            fidx = self.sb_frame.value()
            if fidx >= len(frames) or frames[fidx] is None:
                return
            vec = np.array(frames[fidx]["disp_components"])
        else:
            vec = info["data"]

        ndim = vec.shape[0]
        skip = self.sb_quiver_skip.value()
        scale = self.dsb_quiver_scale.value()

        if ndim < 2:
            return

        ux = vec[0]
        uy = vec[1]
        if ux.ndim == 3:
            z = min(self.sl_z.value(), ux.shape[2] - 1)
            ux = ux[:, :, z]
            uy = uy[:, :, z]

        ny, nx = ux.shape
        Y, X = np.mgrid[0:ny, 0:nx]

        # Subsample
        X_s = X[::skip, ::skip].ravel()
        Y_s = Y[::skip, ::skip].ravel()
        U_s = ux[::skip, ::skip].ravel()
        V_s = uy[::skip, ::skip].ravel()

        mag = np.sqrt(U_s**2 + V_s**2)
        mask = mag > 1e-12
        if not np.any(mask):
            return

        X_s, Y_s, U_s, V_s, mag = X_s[mask], Y_s[mask], U_s[mask], V_s[mask], mag[mask]

        # Triangle size: fraction of skip spacing, scaled by user control
        # Default scale=1.0 gives triangles about 30% of skip spacing (small & clean)
        base_size = skip * 0.3 * scale
        # Clamp to reasonable range
        base_size = max(base_size, 0.5)
        base_size = min(base_size, skip * 2.0)

        # Compute triangle vertices for each point
        # Triangle points in displacement direction
        angles = np.arctan2(V_s, U_s)

        # Three vertices of equilateral triangle pointing along angle
        # Tip in the direction of flow, base perpendicular
        r = base_size * 0.5  # half-height
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Tip (forward)
        tx = X_s + r * cos_a
        ty = Y_s + r * sin_a
        # Base left
        lx = X_s - r * 0.5 * cos_a + r * 0.4 * sin_a
        ly = Y_s - r * 0.5 * sin_a - r * 0.4 * cos_a
        # Base right
        rx = X_s - r * 0.5 * cos_a - r * 0.4 * sin_a
        ry = Y_s - r * 0.5 * sin_a + r * 0.4 * cos_a

        # Build vertices array: (N, 3, 2)
        verts = np.stack([
            np.column_stack([tx, ty]),
            np.column_stack([lx, ly]),
            np.column_stack([rx, ry]),
        ], axis=1)

        # Color by magnitude
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        from matplotlib.collections import PolyCollection

        cmap_name = self._get_cmap()
        cmap_obj = cm.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=mag.min(), vmax=mag.max())
        colors = cmap_obj(norm(mag))

        pc = PolyCollection(verts, facecolors=colors, edgecolors='none',
                           alpha=0.85, zorder=5)
        ax.add_collection(pc)

    def _plot_multi_component(self, info: Dict, source: str, plot_type: str):
        """Plot multiple components side by side."""
        self.canvas.clear()
        ndim = info.get("ndim", 3)
        fp = self._get_font_props()

        if info["type"] == "vector":
            labels = ["u_x", "u_y", "u_z"][:ndim]
            arrays = [info["data"][d] for d in range(ndim)]
        elif info["type"] == "tensor":
            labels = ["xx", "yy", "zz"][:ndim]
            prefix = "ε" if "Strain" in source else "σ" if "Stress" in source else "F"
            labels = [f"{prefix}_{l}" for l in labels]
            arrays = [info["data"][d, d] for d in range(ndim)]
        else:
            return

        n = len(arrays)
        cmap = self._get_cmap()

        for i, (arr, label) in enumerate(zip(arrays, labels)):
            if arr.ndim == 3:
                arr = self._slice_2d(arr)
            ax = self.canvas.add_subplot(1, n, i + 1)
            vmin, vmax = self._get_clim(arr)
            im = ax.imshow(arr.T, cmap=cmap, origin="lower",
                           vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(label, color=Settings.FG_PRIMARY,
                         fontsize=fp["fontsize"], fontfamily=fp["fontfamily"])
            ax.tick_params(labelsize=fp["fontsize"] - 2)
            self._add_colorbar(ax, im)

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.status.set_status("ready", "Multi-component plot")

    def _plot_trajectories(self, info: Dict, component: str):
        """Plot particle trajectories from tracking session."""
        self.canvas.clear()
        session = info.get("session")
        if session is None:
            return

        ax = self.canvas.add_subplot(1, 1, 1)

        try:
            coords_ref = session.coords_ref
            frame_results = session.frame_results

            # Collect trajectories
            n_particles = len(coords_ref)
            n_frames = len(frame_results)

            # Build trajectory arrays (particle × frame × dim)
            ndim = coords_ref.shape[1]
            trajectories = np.full((n_particles, n_frames + 1, ndim), np.nan)
            trajectories[:, 0, :] = coords_ref

            for t, res in enumerate(frame_results):
                # track_b2a maps frame B particles → reference particles
                # Iterate B indices to avoid out-of-bounds on reference array
                n_b = len(res.track_b2a) if hasattr(res, 'track_b2a') else 0
                for b_idx in range(n_b):
                    a_idx = res.track_b2a[b_idx]
                    if 0 <= a_idx < n_particles:
                        if hasattr(res, 'coords_b') and b_idx < len(res.coords_b):
                            trajectories[a_idx, t+1, :] = res.coords_b[b_idx]
                        elif hasattr(res, 'disp_b2a') and a_idx < len(res.disp_b2a):
                            trajectories[a_idx, t+1, :] = (
                                coords_ref[a_idx] - res.disp_b2a[a_idx])

            # Color setup
            cmap_name = self._get_cmap()
            import matplotlib.cm as cm
            cmap_obj = cm.get_cmap(cmap_name)

            # Axis mapping
            if "XZ" in component:
                ax_x, ax_y, ax_label = 0, 2, ("x", "z")
            elif "YZ" in component:
                ax_x, ax_y, ax_label = 1, 2, ("y", "z")
            else:
                ax_x, ax_y, ax_label = 0, 1, ("x", "y")

            if ndim < 3 and (ax_x >= ndim or ax_y >= ndim):
                ax_x, ax_y, ax_label = 0, 1, ("x", "y")

            # Plot each trajectory
            max_plot = min(n_particles, 500)  # limit for performance
            for p in range(max_plot):
                traj = trajectories[p]
                valid = ~np.isnan(traj[:, ax_x])
                if np.sum(valid) < 2:
                    continue

                x = traj[valid, ax_x]
                y = traj[valid, ax_y]

                if "Displacement" in component:
                    disp = np.sqrt(np.sum(np.diff(traj[valid], axis=0)**2, axis=1))
                    total_disp = np.sum(disp)
                    color = cmap_obj(total_disp / (np.max(disp) * n_frames + 1e-9))
                elif "Time" in component:
                    color = cmap_obj(p / max_plot)
                else:
                    color = (Settings.ACCENT_CYAN,)

                ax.plot(x, y, '-', color=color, alpha=0.4, linewidth=0.7)
                ax.plot(x[0], y[0], 'o', color=Settings.ACCENT_GREEN,
                        markersize=2, alpha=0.5)

            fp = self._get_font_props()
            ax.set_xlabel(f"{ax_label[0]} (px)", color=Settings.FG_SECONDARY,
                          fontsize=fp["fontsize"], fontfamily=fp["fontfamily"])
            ax.set_ylabel(f"{ax_label[1]} (px)", color=Settings.FG_SECONDARY,
                          fontsize=fp["fontsize"], fontfamily=fp["fontfamily"])
            ax.set_title(f"Trajectories — {component}",
                         color=Settings.FG_PRIMARY,
                         fontsize=fp["fontsize"] + 1, fontfamily=fp["fontfamily"])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=fp["fontsize"] - 1)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", color=Settings.ACCENT_RED)

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.status.set_status("ready", "Trajectories plotted")

    # ───── Quick plots ────────────────────────────────────────────

    def _quick_plot(self, source: str, component: str, plot_type: str):
        idx_src = self.cb_source.findText(source)
        if idx_src >= 0:
            self.cb_source.setCurrentIndex(idx_src)

        idx_comp = self.cb_component.findText(component)
        if idx_comp >= 0:
            self.cb_component.setCurrentIndex(idx_comp)

        idx_pt = self.cb_plot_type.findText(plot_type)
        if idx_pt >= 0:
            self.cb_plot_type.setCurrentIndex(idx_pt)

        self._refresh_plot()

    # ───── Export ─────────────────────────────────────────────────

    def _export_current(self):
        fmt = self.cb_format.currentText().lower()
        dpi = self.sb_dpi.value()

        filter_map = {
            "png": "PNG (*.png)", "svg": "SVG (*.svg)",
            "pdf": "PDF (*.pdf)", "tiff": "TIFF (*.tiff)",
            "eps": "EPS (*.eps)",
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", f"plot.{fmt}",
            filter_map.get(fmt, "All (*)"))
        if not path:
            return

        try:
            self.canvas.figure.savefig(
                path, dpi=dpi, bbox_inches="tight",
                facecolor=Settings.BG_PRIMARY,
                edgecolor="none",
            )
            self.status.set_status("ready", f"Exported to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_all_frames(self):
        """Export current view for every frame to a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Export Folder")
        if not folder:
            return

        fmt = self.cb_format.currentText().lower()
        dpi = self.sb_dpi.value()
        source = self.cb_source.currentText()
        component = self.cb_component.currentText()

        n_frames = self.sb_frame.maximum() + 1
        if n_frames <= 0:
            QMessageBox.warning(self, "No Data", "No frames available.")
            return

        exported = 0
        for f in range(n_frames):
            self.sb_frame.setValue(f)
            self._refresh_plot()

            fname = f"{source}_{component}_frame{f:04d}.{fmt}".replace(
                " ", "_").replace("/", "_")
            path = os.path.join(folder, fname)
            try:
                self.canvas.figure.savefig(
                    path, dpi=dpi, bbox_inches="tight",
                    facecolor=Settings.BG_PRIMARY, edgecolor="none")
                exported += 1
            except Exception:
                pass

        self.status.set_status("ready", f"Exported {exported}/{n_frames} frames")

    # ───── Experiment lifecycle ────────────────────────────────────

    def on_experiment_changed(self, exp_id: str):
        pass  # Now handled by save/load_from_experiment

    def on_activated(self):
        self._refresh_plot()

    def save_to_experiment(self, exp):
        """Plots page has no internal state to save — it reads from other pages."""
        pass

    def load_from_experiment(self, exp):
        """Clear cache and refresh — data comes from other pages' results."""
        self._cached_data = {}
        self._refresh_plot()