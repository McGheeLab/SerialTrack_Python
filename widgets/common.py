"""
Reusable custom widgets for SerialTrack GUI.

- MplCanvas: matplotlib figure embedded in Qt
- ParamEditor: auto-generates form from ParamSpec list
- ImageViewer: 2D/3D slice viewer with sliders
- ExperimentTimeline: sidebar list of experiments
- StatusIndicator: colored status badge
- TooltipBar: persistent tooltip display at bottom of right panel
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QGroupBox,
    QListWidget, QListWidgetItem, QFrame, QSizePolicy, QToolButton,
    QScrollArea, QMenu, QLineEdit, QProgressBar, QSplitter, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QSize, QEvent
from PySide6.QtGui import QFont, QColor, QIcon, QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from core.settings import Settings
from core.plugin_registry import ParamSpec


# ═══════════════════════════════════════════════════════════════
#  Tooltip Loader
# ═══════════════════════════════════════════════════════════════

_TOOLTIPS: Dict[str, Any] = {}


def load_tooltips(path: Optional[str] = None):
    """Load tooltip definitions from JSON file."""
    global _TOOLTIPS
    if path is None:
        # Try common locations
        for candidate in [
            Path(__file__).parent.parent / "tooltips.json",
            Path(__file__).parent / "tooltips.json",
            Path("tooltips.json"),
        ]:
            if candidate.exists():
                path = str(candidate)
                break
    if path and Path(path).exists():
        try:
            with open(path) as f:
                _TOOLTIPS = json.load(f)
        except Exception:
            _TOOLTIPS = {}


def get_tooltip(page: str, key: str) -> str:
    """Look up a tooltip string by page and key."""
    return _TOOLTIPS.get(page, {}).get(key, "")


# Load on import
load_tooltips()


# ═══════════════════════════════════════════════════════════════
#  Tooltip Bar — persistent tooltip display
# ═══════════════════════════════════════════════════════════════

class TooltipBar(QFrame):
    """A persistent tooltip display bar shown at the bottom of the right panel.

    Widgets register with this bar by installing an event filter.
    When hovered, the tooltip text is displayed in the bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tooltipBar")
        self.setFixedHeight(28)
        self.setStyleSheet(
            f"#tooltipBar {{ background-color: {Settings.BG_SECONDARY}; "
            f"border-top: 1px solid {Settings.BORDER_COLOR}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 2, 12, 2)

        icon_lbl = QLabel("💡")
        icon_lbl.setFixedWidth(20)
        icon_lbl.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(icon_lbl)

        self._label = QLabel("")
        self._label.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt 'Helvetica Neue'; "
            f"border: none; background: transparent;"
        )
        layout.addWidget(self._label)
        layout.addStretch()

        self._default_text = ""

    def set_default(self, text: str):
        self._default_text = text
        if not self._label.text():
            self._label.setText(text)

    def show_tooltip(self, text: str):
        if text:
            self._label.setText(text)
        else:
            self._label.setText(self._default_text)

    def clear_tooltip(self):
        self._label.setText(self._default_text)

    def register_widget(self, widget: QWidget, tooltip_text: str):
        """Register a widget so hovering it shows tooltip_text in this bar."""
        widget.setProperty("_tooltip_bar_text", tooltip_text)
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            text = obj.property("_tooltip_bar_text")
            if text:
                self.show_tooltip(text)
        elif event.type() == QEvent.Leave:
            self.clear_tooltip()
        return False


# ═══════════════════════════════════════════════════════════════
#  Matplotlib Canvas
# ═══════════════════════════════════════════════════════════════

class MplCanvas(QWidget):
    """Embeddable matplotlib canvas with toolbar."""

    def __init__(self, parent=None, figsize=(6, 4), dpi=100, toolbar=True):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi,
                             facecolor="#21252b", edgecolor="#44475a")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #21252b;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if toolbar:
            self.toolbar = NavToolbar(self.canvas, self)
            self.toolbar.setStyleSheet(
                "background-color: #2c313c; border: none; padding: 2px;"
            )
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas)
        self.ax = None

    def clear(self):
        self.figure.clear()
        self.ax = None

    def add_subplot(self, *args, **kwargs):
        self.ax = self.figure.add_subplot(*args, **kwargs)
        self._style_axes(self.ax)
        return self.ax

    def _style_axes(self, ax):
        """Apply Dracula theme to axes."""
        ax.set_facecolor("#21252b")
        ax.tick_params(colors="#b0b0b0", labelsize=8)
        ax.xaxis.label.set_color("#f8f8f2")
        ax.yaxis.label.set_color("#f8f8f2")
        ax.title.set_color("#f8f8f2")
        for spine in ax.spines.values():
            spine.set_color("#44475a")

    def draw(self):
        self.figure.tight_layout()
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
#  Parameter Editor — auto-generates form from ParamSpec
# ═══════════════════════════════════════════════════════════════

class ParamEditor(QWidget):
    """Auto-generates a parameter form from a list of ParamSpec."""
    params_changed = Signal(dict)

    def __init__(self, specs=None, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)
        self._widgets: Dict[str, QWidget] = {}
        self._specs: List[ParamSpec] = []
        if specs:
            self.set_params(specs)

    def set_params(self, specs: List[ParamSpec]):
        """Rebuild the form for new parameter specs."""
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()
        self._specs = specs

        for spec in specs:
            label = QLabel(spec.label)
            label.setToolTip(spec.tooltip)

            if spec.param_type == "float":
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(spec.min_val or -1e9, spec.max_val or 1e9)
                w.setSingleStep(spec.step or 0.1)
                w.setValue(spec.default)
                w.setToolTip(spec.tooltip)
                w.valueChanged.connect(self._emit_changed)

            elif spec.param_type == "int":
                w = QSpinBox()
                w.setRange(spec.min_val or 0, spec.max_val or 999999)
                w.setSingleStep(spec.step or 1)
                w.setValue(int(spec.default))
                w.setToolTip(spec.tooltip)
                w.valueChanged.connect(self._emit_changed)

            elif spec.param_type == "bool":
                w = QCheckBox()
                w.setChecked(bool(spec.default))
                w.setToolTip(spec.tooltip)
                w.stateChanged.connect(self._emit_changed)

            elif spec.param_type == "choice":
                w = QComboBox()
                w.addItems(spec.choices)
                if spec.default in spec.choices:
                    w.setCurrentText(str(spec.default))
                w.setToolTip(spec.tooltip)
                w.currentTextChanged.connect(self._emit_changed)

            elif spec.param_type == "str":
                w = QLineEdit(str(spec.default))
                w.setToolTip(spec.tooltip)
                w.textChanged.connect(self._emit_changed)
            else:
                continue

            self._widgets[spec.name] = w
            self._layout.addRow(label, w)

    def get_values(self) -> Dict[str, Any]:
        vals = {}
        for spec in self._specs:
            w = self._widgets.get(spec.name)
            if w is None:
                continue
            if isinstance(w, QDoubleSpinBox):
                vals[spec.name] = w.value()
            elif isinstance(w, QSpinBox):
                vals[spec.name] = w.value()
            elif isinstance(w, QCheckBox):
                vals[spec.name] = w.isChecked()
            elif isinstance(w, QComboBox):
                vals[spec.name] = w.currentText()
            elif isinstance(w, QLineEdit):
                vals[spec.name] = w.text()
        return vals

    def set_values(self, values: Dict[str, Any]):
        for name, val in values.items():
            w = self._widgets.get(name)
            if w is None:
                continue
            if isinstance(w, QDoubleSpinBox):
                w.setValue(float(val))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(val))
            elif isinstance(w, QLineEdit):
                w.setText(str(val))

    def _emit_changed(self, *_):
        self.params_changed.emit(self.get_values())


# ═══════════════════════════════════════════════════════════════
#  Image Viewer — 2D/3D slice viewer
# ═══════════════════════════════════════════════════════════════

class ImageViewer(QWidget):
    """2D/3D image viewer with Z-slice and time sliders."""
    slice_changed = Signal(int)
    time_changed = Signal(int)

    def __init__(self, parent=None, show_toolbar=True):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.canvas = MplCanvas(self, figsize=(5, 4), toolbar=show_toolbar)
        layout.addWidget(self.canvas)

        # Slider bar
        slider_frame = QWidget()
        slider_layout = QVBoxLayout(slider_frame)
        slider_layout.setContentsMargins(4, 0, 4, 4)
        slider_layout.setSpacing(2)

        # Z slice
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        z_row.addWidget(self.z_slider)
        self.z_label = QLabel("0/0")
        self.z_label.setMinimumWidth(50)
        z_row.addWidget(self.z_label)
        slider_layout.addLayout(z_row)

        # Time
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("T:"))
        self.t_slider = QSlider(Qt.Horizontal)
        self.t_slider.setRange(0, 0)
        self.t_slider.valueChanged.connect(self._on_t_changed)
        t_row.addWidget(self.t_slider)
        self.t_label = QLabel("0/0")
        self.t_label.setMinimumWidth(50)
        t_row.addWidget(self.t_label)
        slider_layout.addLayout(t_row)

        # Display controls
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "viridis", "plasma", "inferno",
                                   "magma", "hot", "bone", "coolwarm"])
        self.cmap_combo.currentTextChanged.connect(self._refresh)
        ctrl_row.addWidget(self.cmap_combo)
        ctrl_row.addStretch()
        slider_layout.addLayout(ctrl_row)

        layout.addWidget(slider_frame)

        # Data
        self._volumes: List[np.ndarray] = []  # Time series of 2D or 3D
        self._current_z = 0
        self._current_t = 0
        self._overlay_fn: Optional[Callable] = None

    def set_data(self, volumes: List[np.ndarray]):
        """Set image data. volumes is a list of 2D or 3D arrays (time series)."""
        self._volumes = volumes
        if not volumes:
            return

        nt = len(volumes)
        self.t_slider.setRange(0, max(0, nt - 1))
        self.t_slider.setValue(0)

        vol = volumes[0]
        if vol.ndim >= 3:
            nz = vol.shape[2] if vol.ndim == 3 else vol.shape[0]
            self.z_slider.setRange(0, nz - 1)
            self.z_slider.setValue(nz // 2)
            self.z_slider.setEnabled(True)
        else:
            self.z_slider.setRange(0, 0)
            self.z_slider.setEnabled(False)

        self._refresh()

    def set_overlay(self, fn: Optional[Callable]):
        """Set a function to draw overlays. Called with (ax, volume, z, t)."""
        self._overlay_fn = fn
        self._refresh()

    def _get_slice(self) -> Optional[np.ndarray]:
        if not self._volumes:
            return None
        t = min(self._current_t, len(self._volumes) - 1)
        vol = self._volumes[t]
        if vol.ndim == 2:
            return vol
        elif vol.ndim == 3:
            z = min(self._current_z, vol.shape[2] - 1)
            return vol[:, :, z]
        return vol

    def _on_z_changed(self, val):
        self._current_z = val
        nz = self.z_slider.maximum() + 1
        self.z_label.setText(f"{val}/{nz - 1}")
        self.slice_changed.emit(val)
        self._refresh()

    def _on_t_changed(self, val):
        self._current_t = val
        nt = self.t_slider.maximum() + 1
        self.t_label.setText(f"{val}/{nt - 1}")
        self.time_changed.emit(val)
        # Update Z range for new timepoint
        if self._volumes and val < len(self._volumes):
            vol = self._volumes[val]
            if vol.ndim >= 3:
                nz = vol.shape[2] if vol.ndim == 3 else vol.shape[0]
                self.z_slider.setRange(0, nz - 1)
        self._refresh()

    def _refresh(self):
        sl = self._get_slice()
        if sl is None:
            return
        self.canvas.clear()
        ax = self.canvas.add_subplot(111)
        cmap = self.cmap_combo.currentText()
        ax.imshow(sl.T, cmap=cmap, origin="lower", aspect="equal")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

        if self._overlay_fn:
            vol = self._volumes[self._current_t] if self._volumes else None
            self._overlay_fn(ax, vol, self._current_z, self._current_t)

        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════
#  Experiment Timeline sidebar widget
# ═══════════════════════════════════════════════════════════════

class ExperimentTimeline(QWidget):
    """Sidebar showing experiment history with selection."""
    experiment_selected = Signal(str)
    experiment_action = Signal(str, str)  # exp_id, action

    STATUS_COLORS = {
        "configured": "#ffb86c",
        "detecting": "#8be9fd",
        "tracking": "#bd93f9",
        "postprocessing": "#ff79c6",
        "complete": "#50fa7b",
        "error": "#ff5555",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("Experiments")
        header.setObjectName("sectionHeader")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("+ New")
        self.btn_new.setObjectName("primaryBtn")
        self.btn_new.setMaximumWidth(80)
        btn_row.addWidget(self.btn_new)

        self.btn_save = QPushButton("Save")
        self.btn_save.setMaximumWidth(60)
        btn_row.addWidget(self.btn_save)

        self.btn_load = QPushButton("Load")
        self.btn_load.setMaximumWidth(60)
        btn_row.addWidget(self.btn_load)
        layout.addLayout(btn_row)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("experimentList")
        self.list_widget.currentItemChanged.connect(self._on_selection)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.list_widget)

    def add_experiment(self, exp_id: str, name: str, status: str = "configured"):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, exp_id)
        color = self.STATUS_COLORS.get(status, "#b0b0b0")
        item.setText(f"● {name}")
        item.setForeground(QColor(color))
        item.setToolTip(f"ID: {exp_id}\nStatus: {status}")
        self.list_widget.addItem(item)

    def update_experiment(self, exp_id: str, name: str, status: str):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == exp_id:
                color = self.STATUS_COLORS.get(status, "#b0b0b0")
                item.setText(f"● {name}")
                item.setForeground(QColor(color))
                item.setToolTip(f"ID: {exp_id}\nStatus: {status}")
                break

    def remove_experiment(self, exp_id: str):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == exp_id:
                self.list_widget.takeItem(i)
                break

    def select_experiment(self, exp_id: str):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == exp_id:
                self.list_widget.setCurrentItem(item)
                break

    def _on_selection(self, current, previous):
        if current:
            self.experiment_selected.emit(current.data(Qt.UserRole))

    def _context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        exp_id = item.data(Qt.UserRole)
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #343b48; color: #f8f8f2; }"
                          "QMenu::item:selected { background-color: #44475a; }")
        menu.addAction("Duplicate", lambda: self.experiment_action.emit(exp_id, "duplicate"))
        menu.addAction("Rename", lambda: self.experiment_action.emit(exp_id, "rename"))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.experiment_action.emit(exp_id, "delete"))
        menu.exec(self.list_widget.mapToGlobal(pos))


# ═══════════════════════════════════════════════════════════════
#  Status Indicator
# ═══════════════════════════════════════════════════════════════

class StatusIndicator(QLabel):
    """Colored status badge."""
    def __init__(self, text="Ready", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusIndicator")
        self.set_status("ready")

    def set_status(self, status: str, text: Optional[str] = None):
        colors = {
            "ready": (Settings.ACCENT_GREEN, "#282a36"),
            "running": (Settings.ACCENT_PURPLE, "#f8f8f2"),
            "error": (Settings.ACCENT_RED, "#f8f8f2"),
            "warning": (Settings.ACCENT_ORANGE, "#282a36"),
            "info": (Settings.ACCENT_CYAN, "#282a36"),
        }
        bg, fg = colors.get(status, (Settings.BG_TERTIARY, Settings.FG_PRIMARY))
        self.setStyleSheet(
            f"background-color: {bg}; color: {fg}; "
            f"padding: 3px 10px; border-radius: 4px; font-weight: bold;"
        )
        if text:
            self.setText(text)