"""
SerialTrack GUI — Main Window
================================
PyDracula-styled main window with icon-only left sidebar navigation,
experiment timeline, tooltip bar, and stacked widget pages.

Experiment switching lifecycle:
  1. Pages save their state to the OLD experiment record's cache
  2. Manager emits active_changed(old_id, new_id)
  3. _on_active_changed tells each page to load from the NEW experiment
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QFrame, QPushButton, QLabel, QSizePolicy, QSplitter, QFileDialog,
    QMessageBox, QApplication, QInputDialog, QGraphicsDropShadowEffect,
    QProgressBar,
)
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QFont, QColor, QIcon

from core.settings import Settings
from core.theme import STYLESHEET
from core.experiment_manager import ExperimentManager
from widgets.common import ExperimentTimeline, StatusIndicator, TooltipBar

# Import plugins to register them
import plugins.enhancement.builtin  # noqa: F401


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{Settings.APP_NAME} v{Settings.APP_VERSION}")
        self.setMinimumSize(Settings.MIN_WIDTH, Settings.MIN_HEIGHT)
        self.resize(1440, 860)

        self.setStyleSheet(STYLESHEET)

        self.exp_manager = ExperimentManager(self)
        self.tooltip_bar = TooltipBar()
        self._switching = False  # Guard: prevents re-entrant experiment switching

        self._build_ui()
        self._connect_signals()
        self._register_tooltips()

        # Create a default experiment
        self.exp_manager.create_experiment("Untitled Experiment")

    # ═══════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("bgApp")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Left sidebar (icon-only) ───────────────────────────
        self.sidebar = QFrame()
        self.sidebar.setObjectName("leftMenuBg")
        self.sidebar.setFixedWidth(Settings.MENU_WIDTH)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(4, 12, 4, 8)
        title_layout.setAlignment(Qt.AlignCenter)

        logo_lbl = QLabel("ST")
        logo_lbl.setObjectName("titleLabel")
        logo_lbl.setAlignment(Qt.AlignCenter)
        logo_lbl.setStyleSheet(
            f"color: {Settings.ACCENT_PURPLE}; font: bold 14pt 'Helvetica Neue';"
        )
        logo_lbl.setToolTip(f"{Settings.APP_NAME}\n{Settings.APP_DESCRIPTION}")
        title_layout.addWidget(logo_lbl)
        sidebar_layout.addWidget(title_frame)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: {Settings.BORDER_COLOR};")
        sep.setMaximumHeight(1)
        sidebar_layout.addWidget(sep)

        nav_frame = QWidget()
        self.nav_layout = QVBoxLayout(nav_frame)
        self.nav_layout.setContentsMargins(0, 8, 0, 8)
        self.nav_layout.setSpacing(0)

        self.nav_buttons = {}
        pages_info = [
            ("images",      "🖼",  "Images — Load & Enhance"),
            ("mask",        "🎭",  "Mask — Define ROI"),
            ("detection",   "🔍",  "Detection — Particle Detection"),
            ("parameters",  "⚙",   "Parameters — Tracking Config"),
            ("analysis",    "▶",   "Analysis — Run Tracking"),
            ("postprocess", "📊",  "Post-Process — Fields"),
            ("stress",      "💪",  "Stress — Constitutive Models"),
            ("plots",       "📈",  "Plots — Visualization"),
        ]

        for key, icon, tooltip in pages_info:
            btn = QPushButton(icon)
            btn.setObjectName("navBtnIcon")
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(Settings.MENU_WIDTH, 44)
            btn.clicked.connect(lambda checked, k=key: self._navigate(k))
            self.nav_layout.addWidget(btn)
            self.nav_buttons[key] = btn

        self.nav_layout.addStretch()
        sidebar_layout.addWidget(nav_frame)
        main_layout.addWidget(self.sidebar)

        # ── Right content area ────────────────────────────────
        right_area = QWidget()
        right_area.setObjectName("contentArea")
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Top bar
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setMaximumHeight(40)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 4, 12, 4)

        self.page_title = QLabel("Images")
        self.page_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.page_title.setStyleSheet(f"color: {Settings.FG_PRIMARY};")
        top_layout.addWidget(self.page_title)
        top_layout.addStretch()

        self.status_indicator = StatusIndicator("Ready")
        top_layout.addWidget(self.status_indicator)
        right_layout.addWidget(top_bar)

        # Content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #44475a; width: 2px; }"
        )

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        self._create_pages()
        content_splitter.addWidget(self.page_stack)

        self.timeline = ExperimentTimeline()
        self.timeline.setMaximumWidth(280)
        self.timeline.setMinimumWidth(200)
        content_splitter.addWidget(self.timeline)

        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(content_splitter)

        right_layout.addWidget(self.tooltip_bar)

        # Bottom bar
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setMaximumHeight(28)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 2, 12, 2)

        version_lbl = QLabel(f"v{Settings.APP_VERSION}")
        version_lbl.setStyleSheet("color: #666; font: 8pt 'Segoe UI';")
        bottom_layout.addWidget(version_lbl)
        bottom_layout.addStretch()

        self.global_progress = QProgressBar()
        self.global_progress.setMaximumWidth(200)
        self.global_progress.setMaximumHeight(14)
        self.global_progress.setVisible(False)
        bottom_layout.addWidget(self.global_progress)

        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("color: #999; font: 8pt 'Segoe UI';")
        bottom_layout.addWidget(self.status_text)
        right_layout.addWidget(bottom_bar)

        main_layout.addWidget(right_area)
        self._navigate("images")

    def _create_pages(self):
        from pages.images_page import ImagesPage
        from pages.mask_page import MaskPage
        from pages.detection_page import DetectionPage
        from pages.parameters_page import ParametersPage
        from pages.analysis_page import AnalysisPage
        from pages.postprocess_page import PostProcessPage
        from pages.stress_page import StressPage
        from pages.plots_page import PlotsPage

        self.pages = {}
        page_classes = {
            "images": ImagesPage,
            "mask": MaskPage,
            "detection": DetectionPage,
            "parameters": ParametersPage,
            "analysis": AnalysisPage,
            "postprocess": PostProcessPage,
            "stress": StressPage,
            "plots": PlotsPage,
        }

        for key, cls in page_classes.items():
            page = cls(main_window=self)
            self.pages[key] = page
            self.page_stack.addWidget(page)

    # ═══════════════════════════════════════════════════════════
    #  Navigation
    # ═══════════════════════════════════════════════════════════

    def _navigate(self, page_key):
        if page_key not in self.pages:
            return
        for k, btn in self.nav_buttons.items():
            btn.setChecked(k == page_key)
        self.page_stack.setCurrentWidget(self.pages[page_key])

        titles = {
            "images": "Image Loading & Enhancement",
            "mask": "Mask Definition",
            "detection": "Particle Detection Preview",
            "parameters": "Tracking Parameters",
            "analysis": "Run Analysis",
            "postprocess": "Post-Processing",
            "stress": "Stress Analysis",
            "plots": "Visualization & Plots",
        }
        self.page_title.setText(titles.get(page_key, page_key.title()))

        page = self.pages[page_key]
        if hasattr(page, "on_activated"):
            page.on_activated()

    # ═══════════════════════════════════════════════════════════
    #  Experiment switching — the key orchestration
    # ═══════════════════════════════════════════════════════════

    def _connect_signals(self):
        self.exp_manager.experiment_added.connect(self._on_exp_added)
        self.exp_manager.experiment_updated.connect(self._on_exp_updated)
        self.exp_manager.experiment_removed.connect(self._on_exp_removed)
        self.exp_manager.active_changed.connect(self._on_active_changed)

        self.timeline.experiment_selected.connect(self._request_switch)
        self.timeline.experiment_action.connect(self._on_exp_action)
        self.timeline.btn_new.clicked.connect(self._new_experiment)
        self.timeline.btn_save.clicked.connect(self._save_session)
        self.timeline.btn_load.clicked.connect(self._load_session)

    def _request_switch(self, new_exp_id):
        """Called when user clicks an experiment in the timeline.

        Save current page states to old experiment BEFORE the switch.
        Guarded by _switching to prevent re-entrant calls from signal chains.
        """
        if self._switching:
            return
        old_exp = self.exp_manager.active
        if old_exp and old_exp.exp_id != new_exp_id:
            self._save_page_states(old_exp)
        self.exp_manager.set_active(new_exp_id)

    def _on_active_changed(self, old_id, new_id):
        """Respond to experiment switch: load new experiment state into all pages.

        Sets _switching guard to prevent the circular chain:
        select_experiment → currentItemChanged → experiment_selected →
        _request_switch → _save_page_states (would overwrite loaded data).

        If _switching is already True (e.g. during _load_session), we're being
        called from an outer guarded context — skip the load here since the
        outer caller will handle it.
        """
        if self._switching:
            # We're inside _load_session or another guarded block.
            # Just update the timeline selection; the caller will load pages.
            self.timeline.select_experiment(new_id)
            return

        self._switching = True
        try:
            self.timeline.select_experiment(new_id)
            new_exp = self.exp_manager.get(new_id)
            if new_exp:
                self._load_page_states(new_exp)
            self.status_text.setText(
                f"Experiment: {new_exp.name if new_exp else new_id}")
        finally:
            self._switching = False

    def _save_page_states(self, exp):
        """Ask every page to save its current state to the experiment record."""
        for key, page in self.pages.items():
            try:
                if hasattr(page, "save_to_experiment"):
                    page.save_to_experiment(exp)
                elif hasattr(page, "get_config"):
                    # Legacy pages that expose config via get_config()
                    cfg = page.get_config()
                    if key == "parameters":
                        exp.tracking_config = cfg
                    elif key == "detection":
                        exp.detection_config = cfg
            except Exception as e:
                print(f"[save_to_experiment] {page.__class__.__name__}: {e}")

    def _load_page_states(self, exp):
        """Ask every page to load state from the experiment record."""
        for key, page in self.pages.items():
            try:
                if hasattr(page, "load_from_experiment"):
                    page.load_from_experiment(exp)
                elif hasattr(page, "on_experiment_changed"):
                    page.on_experiment_changed(exp.exp_id)
            except Exception as e:
                print(f"[load_from_experiment] {page.__class__.__name__}: {e}")

    def _on_exp_added(self, exp_id):
        rec = self.exp_manager.get(exp_id)
        if rec:
            self.timeline.add_experiment(exp_id, rec.name, rec.status)

    def _on_exp_updated(self, exp_id):
        rec = self.exp_manager.get(exp_id)
        if rec:
            self.timeline.update_experiment(exp_id, rec.name, rec.status)

    def _on_exp_removed(self, exp_id):
        self.timeline.remove_experiment(exp_id)

    def _on_exp_action(self, exp_id, action):
        if action == "duplicate":
            # Make sure current page state is saved before duplicating
            src = self.exp_manager.get(exp_id)
            if src and self.exp_manager.active_id == exp_id:
                self._save_page_states(src)
            new_rec = self.exp_manager.duplicate(exp_id)
            if new_rec:
                self.exp_manager.set_active(new_rec.exp_id)
        elif action == "delete":
            reply = QMessageBox.question(
                self, "Delete Experiment",
                "Are you sure you want to delete this experiment?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.exp_manager.remove(exp_id)
        elif action == "rename":
            rec = self.exp_manager.get(exp_id)
            if rec:
                name, ok = QInputDialog.getText(
                    self, "Rename Experiment", "New name:", text=rec.name)
                if ok and name:
                    self.exp_manager.update(exp_id, name=name)

    def _new_experiment(self):
        # Save current experiment state first
        old_exp = self.exp_manager.active
        if old_exp:
            self._save_page_states(old_exp)

        name, ok = QInputDialog.getText(
            self, "New Experiment", "Experiment name:",
            text=f"Experiment {self.exp_manager.count + 1}")
        if ok and name:
            rec = self.exp_manager.create_experiment(name)
            # Always switch to the new (empty) experiment so pages reset
            self.exp_manager.set_active(rec.exp_id)

    def _save_session(self):
        # Save current page states before writing to disk
        old_exp = self.exp_manager.active
        if old_exp:
            self._save_page_states(old_exp)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "SerialTrack Session (*.stk);;All Files (*)")
        if path:
            try:
                self.exp_manager.save_session(path)
                self.status_text.setText(f"Session saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "SerialTrack Session (*.stk);;All Files (*)")
        if not path:
            return

        try:
            # Save current state before replacing everything
            old_exp = self.exp_manager.active
            if old_exp:
                self._save_page_states(old_exp)

            # Set guard: prevent ALL spurious switching during bulk load.
            # load_session adds experiments to timeline which fires
            # currentItemChanged → _request_switch. The guard blocks that.
            self._switching = True
            self.timeline.list_widget.clear()
            self.exp_manager.load_session(path)

            # Still under guard: select and load the active experiment.
            # Guard stays on so select_experiment's currentItemChanged
            # doesn't trigger _request_switch (which would save stale
            # page state over the freshly loaded experiment data).
            active_exp = self.exp_manager.active
            if active_exp:
                self.timeline.select_experiment(active_exp.exp_id)
                self._load_page_states(active_exp)
                self.status_text.setText(
                    f"Loaded: {Path(path).name} — {active_exp.name}")
            else:
                self.status_text.setText(f"Session loaded: {Path(path).name}")

            self._switching = False
        except Exception as e:
            self._switching = False
            QMessageBox.warning(self, "Load Error", str(e))

    # ═══════════════════════════════════════════════════════════
    #  Status helpers
    # ═══════════════════════════════════════════════════════════

    def set_status(self, text, level="ready"):
        self.status_text.setText(text)
        self.status_indicator.set_status(
            level, text.split(":")[0] if ":" in text else text[:20])

    def show_progress(self, visible, value=0, maximum=100):
        self.global_progress.setVisible(visible)
        self.global_progress.setMaximum(maximum)
        self.global_progress.setValue(value)

    # ═══════════════════════════════════════════════════════════
    #  Tooltip Registration
    # ═══════════════════════════════════════════════════════════

    def _register_tooltips(self):
        """Register page widgets with the tooltip bar for bottom-panel display."""
        tb = self.tooltip_bar

        # Nav buttons
        for key, btn in self.nav_buttons.items():
            from widgets.common import get_tooltip
            text = get_tooltip("nav", key)
            if text:
                tb.register_widget(btn, text)

        # Experiments timeline
        tb.register_page_widgets("experiments", {
            "btn_new": self.timeline.btn_new,
            "btn_save": self.timeline.btn_save,
            "btn_load": self.timeline.btn_load,
        })

        # Images page
        p = self.pages.get("images")
        if p:
            tb.register_page_widgets("images_page", {
                "btn_load_files": getattr(p, "btn_load_files", None),
                "btn_load_folder": getattr(p, "btn_load_folder", None),
                "method_combo": getattr(p, "method_combo", None),
                "btn_add_step": getattr(p, "btn_add_step", None),
                "btn_remove_step": getattr(p, "btn_remove_step", None),
                "btn_clear_pipeline": getattr(p, "btn_clear_pipeline", None),
                "btn_run_enhance": getattr(p, "btn_run_enhance", None),
                "btn_revert": getattr(p, "btn_revert", None),
            })

        # Detection page
        p = self.pages.get("detection")
        if p:
            tb.register_page_widgets("detection_page", {
                "btn_detect": getattr(p, "btn_detect", None),
                "btn_detect_all": getattr(p, "btn_detect_all", None),
            })

        # Analysis page
        p = self.pages.get("analysis")
        if p:
            tb.register_page_widgets("analysis_page", {
                "btn_run": getattr(p, "btn_run", None),
                "btn_cancel": getattr(p, "btn_cancel", None),
            })

        # Post-process page
        p = self.pages.get("postprocess")
        if p:
            tb.register_page_widgets("postprocess_page", {
                "btn_run": getattr(p, "btn_run", None),
                "sb_frame": getattr(p, "sb_frame", None),
                "cb_view": getattr(p, "cb_view", None),
            })

        # Stress page
        p = self.pages.get("stress")
        if p:
            tb.register_page_widgets("stress_page", {
                "cb_model": getattr(p, "cb_model", None),
                "btn_run": getattr(p, "btn_run", None),
                "sb_frame": getattr(p, "sb_frame", None),
                "cb_view": getattr(p, "cb_view", None),
            })

        # Plots page
        p = self.pages.get("plots")
        if p:
            tb.register_page_widgets("plots_page", {
                "cb_source": getattr(p, "cb_source", None),
                "cb_component": getattr(p, "cb_component", None),
                "sb_frame": getattr(p, "sb_frame", None),
                "sl_z": getattr(p, "sl_z", None),
                "cb_plot_type": getattr(p, "cb_plot_type", None),
                "sb_quiver_skip": getattr(p, "sb_quiver_skip", None),
                "dsb_quiver_scale": getattr(p, "dsb_quiver_scale", None),
                "sb_contour_levels": getattr(p, "sb_contour_levels", None),
                "cb_cmap_cat": getattr(p, "cb_cmap_cat", None),
                "cb_cmap": getattr(p, "cb_cmap", None),
                "chk_reverse": getattr(p, "chk_reverse", None),
                "chk_auto_range": getattr(p, "chk_auto_range", None),
                "dsb_vmin": getattr(p, "dsb_vmin", None),
                "dsb_vmax": getattr(p, "dsb_vmax", None),
                "chk_grid": getattr(p, "chk_grid", None),
                "sb_dpi": getattr(p, "sb_dpi", None),
                "cb_format": getattr(p, "cb_format", None),
                "sb_font_size": getattr(p, "sb_font_size", None),
                "cb_font_family": getattr(p, "cb_font_family", None),
            })