"""
Mask Page — Draw shapes to define inclusion/exclusion masks.

Supports: rectangle, ellipse, freeform (2D), with 3D extensions planned.
Each shape can be added or subtracted from the mask.
"""
from __future__ import annotations

from typing import Optional, List
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QRadioButton, QButtonGroup,
    QSpinBox, QCheckBox, QScrollArea, QFrame, QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, Signal

from widgets.common import MplCanvas, ParamEditor
from core.settings import Settings

import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector
from matplotlib.path import Path as MplPath


class MaskPage(QWidget):
    """Mask definition page with interactive shape drawing."""
    mask_updated = Signal(object)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._mask: Optional[np.ndarray] = None
        self._shapes = []  # List of (shape_type, coords, mode)
        self._current_image = None
        self._selector = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── Controls ──────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(340)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(8)

        # Shape tools
        shape_grp = QGroupBox("Shape Tools")
        shape_lay = QVBoxLayout(shape_grp)

        shape_row = QHBoxLayout()
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Rectangle", "Ellipse", "Freeform"])
        self.shape_combo.currentTextChanged.connect(self._activate_selector)
        shape_row.addWidget(QLabel("Shape:"))
        shape_row.addWidget(self.shape_combo)
        shape_lay.addLayout(shape_row)

        # Add / Subtract mode
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rb_add = QRadioButton("Add to Mask")
        self.rb_add.setChecked(True)
        self.rb_sub = QRadioButton("Subtract")
        self.mode_group.addButton(self.rb_add, 0)
        self.mode_group.addButton(self.rb_sub, 1)
        mode_row.addWidget(self.rb_add)
        mode_row.addWidget(self.rb_sub)
        shape_lay.addLayout(mode_row)

        # Z range for 3D
        z_grp = QGroupBox("Z Range (3D)")
        z_lay = QHBoxLayout(z_grp)
        z_lay.addWidget(QLabel("From:"))
        self.z_from = QSpinBox()
        self.z_from.setRange(0, 9999)
        z_lay.addWidget(self.z_from)
        z_lay.addWidget(QLabel("To:"))
        self.z_to = QSpinBox()
        self.z_to.setRange(0, 9999)
        z_lay.addWidget(self.z_to)
        self.cb_all_z = QCheckBox("All Z")
        self.cb_all_z.setChecked(True)
        z_lay.addWidget(self.cb_all_z)
        shape_lay.addWidget(z_grp)

        # Instructions
        instr = QLabel(
            "Click and drag on the image to draw shapes.\n"
            "Rectangle/Ellipse: click-drag corners.\n"
            "Freeform: click points, right-click to close."
        )
        instr.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        instr.setWordWrap(True)
        shape_lay.addWidget(instr)

        ctrl_layout.addWidget(shape_grp)

        # Shapes list
        shapes_grp = QGroupBox("Mask Shapes")
        shapes_lay = QVBoxLayout(shapes_grp)
        self.shapes_list = QListWidget()
        self.shapes_list.setMaximumHeight(150)
        shapes_lay.addWidget(self.shapes_list)

        list_btns = QHBoxLayout()
        self.btn_remove_shape = QPushButton("Remove")
        self.btn_remove_shape.clicked.connect(self._remove_shape)
        list_btns.addWidget(self.btn_remove_shape)
        self.btn_clear_shapes = QPushButton("Clear All")
        self.btn_clear_shapes.clicked.connect(self._clear_shapes)
        list_btns.addWidget(self.btn_clear_shapes)
        shapes_lay.addLayout(list_btns)

        ctrl_layout.addWidget(shapes_grp)

        # Actions
        act_grp = QGroupBox("Actions")
        act_lay = QVBoxLayout(act_grp)

        self.btn_apply = QPushButton("Apply Mask")
        self.btn_apply.setObjectName("primaryBtn")
        self.btn_apply.clicked.connect(self._apply_mask)
        act_lay.addWidget(self.btn_apply)

        self.btn_invert = QPushButton("Invert Mask")
        self.btn_invert.clicked.connect(self._invert_mask)
        act_lay.addWidget(self.btn_invert)

        self.btn_fill_all = QPushButton("Fill All (Include Everything)")
        self.btn_fill_all.clicked.connect(self._fill_all)
        act_lay.addWidget(self.btn_fill_all)

        self.mask_info = QLabel("No mask defined")
        self.mask_info.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        act_lay.addWidget(self.mask_info)

        ctrl_layout.addWidget(act_grp)
        ctrl_layout.addStretch()

        ctrl_scroll.setWidget(ctrl_widget)
        splitter.addWidget(ctrl_scroll)

        # ── Image viewer ──────────────────────────────────────
        self.canvas = MplCanvas(figsize=(6, 5), toolbar=True)
        splitter.addWidget(self.canvas)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    def _activate_selector(self):
        """Activate the appropriate matplotlib selector."""
        # Selectors are created when image is displayed
        pass

    def _on_rect_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        mode = "add" if self.rb_add.isChecked() else "subtract"
        self._shapes.append(("rectangle", (x1, y1, x2, y2), mode))
        self.shapes_list.addItem(f"{'+ ' if mode == 'add' else '- '}Rect ({x1},{y1})-({x2},{y2})")
        self._redraw()

    def _on_ellipse_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        rx, ry = abs(x2-x1)//2, abs(y2-y1)//2
        mode = "add" if self.rb_add.isChecked() else "subtract"
        self._shapes.append(("ellipse", (cx, cy, rx, ry), mode))
        self.shapes_list.addItem(f"{'+ ' if mode == 'add' else '- '}Ellipse c=({cx},{cy}) r=({rx},{ry})")
        self._redraw()

    def _on_lasso_select(self, verts):
        mode = "add" if self.rb_add.isChecked() else "subtract"
        self._shapes.append(("freeform", verts, mode))
        self.shapes_list.addItem(f"{'+ ' if mode == 'add' else '- '}Freeform ({len(verts)} pts)")
        self._redraw()

    def _remove_shape(self):
        row = self.shapes_list.currentRow()
        if row >= 0:
            self.shapes_list.takeItem(row)
            self._shapes.pop(row)
            self._redraw()

    def _clear_shapes(self):
        self._shapes.clear()
        self.shapes_list.clear()
        self._mask = None
        self._redraw()

    def _apply_mask(self):
        """Build mask from shapes and apply."""
        if self._current_image is None:
            return

        shape = self._current_image.shape
        mask = np.zeros(shape[:2], dtype=bool)

        for stype, coords, mode in self._shapes:
            region = np.zeros(shape[:2], dtype=bool)

            if stype == "rectangle":
                x1, y1, x2, y2 = coords
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                region[x1:x2+1, y1:y2+1] = True

            elif stype == "ellipse":
                cx, cy, rx, ry = coords
                yy, xx = np.ogrid[:shape[0], :shape[1]]
                if rx > 0 and ry > 0:
                    region = ((xx - cx)**2 / rx**2 + (yy - cy)**2 / ry**2) <= 1

            elif stype == "freeform":
                verts = coords
                if len(verts) > 2:
                    path = MplPath(verts)
                    xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
                    points = np.column_stack([xx.ravel(), yy.ravel()])
                    region = path.contains_points(points).reshape(shape[:2])

            if mode == "add":
                mask |= region
            else:
                mask &= ~region

        # If no shapes added, default to all True
        if not self._shapes:
            mask = np.ones(shape[:2], dtype=bool)

        # Extend to 3D if needed
        if self._current_image.ndim == 3:
            nz = self._current_image.shape[2]
            if self.cb_all_z.isChecked():
                self._mask = np.stack([mask] * nz, axis=2)
            else:
                z1 = self.z_from.value()
                z2 = self.z_to.value()
                self._mask = np.zeros(shape, dtype=bool)
                self._mask[:, :, z1:z2+1] = mask[:, :, np.newaxis]
        else:
            self._mask = mask

        area_pct = 100.0 * mask.sum() / mask.size
        self.mask_info.setText(f"Mask: {area_pct:.1f}% coverage ({mask.sum()} px)")
        self.mask_updated.emit(self._mask)
        self._redraw()

    def _invert_mask(self):
        if self._mask is not None:
            self._mask = ~self._mask
            self._redraw()

    def _fill_all(self):
        if self._current_image is not None:
            self._mask = np.ones(self._current_image.shape, dtype=bool)
            self._shapes.clear()
            self.shapes_list.clear()
            self.mask_info.setText("Mask: 100% coverage (full)")
            self._redraw()

    def _redraw(self):
        """Redraw image with mask overlay and shape outlines."""
        if self._current_image is None:
            return

        self.canvas.clear()
        ax = self.canvas.add_subplot(111)

        img = self._current_image
        if img.ndim == 3:
            z = img.shape[2] // 2
            img = img[:, :, z]

        ax.imshow(img.T, cmap="gray", origin="lower", aspect="equal")

        # Overlay mask
        if self._mask is not None:
            mask_2d = self._mask
            if mask_2d.ndim == 3:
                mask_2d = mask_2d[:, :, mask_2d.shape[2]//2]
            overlay = np.zeros((*mask_2d.shape, 4))
            overlay[mask_2d, :] = [0.5, 0.3, 1.0, 0.15]
            overlay[~mask_2d, :] = [1.0, 0.0, 0.0, 0.25]
            ax.imshow(overlay.transpose(1, 0, 2), origin="lower", aspect="equal")

        # Draw shape outlines
        for stype, coords, mode in self._shapes:
            color = "#50fa7b" if mode == "add" else "#ff5555"
            if stype == "rectangle":
                x1, y1, x2, y2 = coords
                rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                          fill=False, edgecolor=color, linewidth=1.5)
                ax.add_patch(rect)
            elif stype == "ellipse":
                cx, cy, rx, ry = coords
                ell = mpatches.Ellipse((cx, cy), 2*rx, 2*ry,
                                       fill=False, edgecolor=color, linewidth=1.5)
                ax.add_patch(ell)
            elif stype == "freeform":
                verts = coords
                if len(verts) > 1:
                    xs, ys = zip(*verts)
                    ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]],
                           color=color, linewidth=1.5)

        # Set up selector
        shape_type = self.shape_combo.currentText()
        if shape_type == "Rectangle":
            self._selector = RectangleSelector(
                ax, self._on_rect_select,
                useblit=True, interactive=True,
                props=dict(facecolor='#bd93f9', alpha=0.2,
                          edgecolor='#bd93f9', linewidth=1.5),
            )
        elif shape_type == "Ellipse":
            self._selector = EllipseSelector(
                ax, self._on_ellipse_select,
                useblit=True, interactive=True,
                props=dict(facecolor='#bd93f9', alpha=0.2,
                          edgecolor='#bd93f9', linewidth=1.5),
            )
        elif shape_type == "Freeform":
            self._selector = LassoSelector(
                ax, self._on_lasso_select,
                props=dict(color='#bd93f9', linewidth=1.5),
            )

        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        self.canvas.draw()

    # ── Public API ────────────────────────────────────────────

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

    def on_experiment_changed(self, exp_id: str):
        pass

    def on_activated(self):
        """Load current image from Images page."""
        if self.main_window and "images" in self.main_window.pages:
            vols = self.main_window.pages["images"].get_volumes()
            if vols:
                self._current_image = vols[0]
                if self._current_image.ndim == 3:
                    nz = self._current_image.shape[2]
                    self.z_from.setRange(0, nz - 1)
                    self.z_to.setRange(0, nz - 1)
                    self.z_to.setValue(nz - 1)
                self._redraw()
