"""
SerialTrack GUI — Main Window
================================
PyDracula-styled main window with icon-only left sidebar navigation,
experiment timeline, tooltip bar, and stacked widget pages.
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

        # Apply theme
        self.setStyleSheet(STYLESHEET)

        # Experiment manager
        self.exp_manager = ExperimentManager(self)

        # Tooltip bar (created early so pages can reference it)
        self.tooltip_bar = TooltipBar()

        # Build UI
        self._build_ui()
        self._connect_signals()

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

        # Logo area — compact icon
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

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: {Settings.BORDER_COLOR};")
        sep.setMaximumHeight(1)
        sidebar_layout.addWidget(sep)

        # Navigation buttons — icon only
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

        # Content splitter: pages + experiment timeline
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #44475a; width: 2px; }"
        )

        # Stacked pages
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        self._create_pages()
        content_splitter.addWidget(self.page_stack)

        # Experiment timeline (right panel)
        self.timeline = ExperimentTimeline()
        self.timeline.setMaximumWidth(280)
        self.timeline.setMinimumWidth(200)
        content_splitter.addWidget(self.timeline)

        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(content_splitter)

        # Tooltip bar at bottom of right section
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

        # Select first page
        self._navigate("images")

    def _create_pages(self):
        """Create all workflow pages."""
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

    def _navigate(self, page_key: str):
        if page_key not in self.pages:
            return

        # Update nav buttons
        for k, btn in self.nav_buttons.items():
            btn.setChecked(k == page_key)

        # Switch page
        self.page_stack.setCurrentWidget(self.pages[page_key])

        # Update title
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

        # Notify page
        page = self.pages[page_key]
        if hasattr(page, "on_activated"):
            page.on_activated()

    # ═══════════════════════════════════════════════════════════
    #  Signals
    # ═══════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Experiment manager signals
        self.exp_manager.experiment_added.connect(self._on_exp_added)
        self.exp_manager.experiment_updated.connect(self._on_exp_updated)
        self.exp_manager.experiment_removed.connect(self._on_exp_removed)
        self.exp_manager.active_changed.connect(self._on_active_changed)

        # Timeline signals
        self.timeline.experiment_selected.connect(self.exp_manager.set_active)
        self.timeline.experiment_action.connect(self._on_exp_action)
        self.timeline.btn_new.clicked.connect(self._new_experiment)
        self.timeline.btn_save.clicked.connect(self._save_session)
        self.timeline.btn_load.clicked.connect(self._load_session)

    def _on_exp_added(self, exp_id: str):
        rec = self.exp_manager.get(exp_id)
        if rec:
            self.timeline.add_experiment(exp_id, rec.name, rec.status)

    def _on_exp_updated(self, exp_id: str):
        rec = self.exp_manager.get(exp_id)
        if rec:
            self.timeline.update_experiment(exp_id, rec.name, rec.status)

    def _on_exp_removed(self, exp_id: str):
        self.timeline.remove_experiment(exp_id)

    def _on_active_changed(self, exp_id: str):
        self.timeline.select_experiment(exp_id)
        # Notify all pages
        for page in self.pages.values():
            if hasattr(page, "on_experiment_changed"):
                page.on_experiment_changed(exp_id)

    def _on_exp_action(self, exp_id: str, action: str):
        if action == "duplicate":
            self.exp_manager.duplicate(exp_id)
        elif action == "delete":
            reply = QMessageBox.question(
                self, "Delete Experiment",
                "Are you sure you want to delete this experiment?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.exp_manager.remove(exp_id)
        elif action == "rename":
            rec = self.exp_manager.get(exp_id)
            if rec:
                name, ok = QInputDialog.getText(
                    self, "Rename Experiment",
                    "New name:", text=rec.name
                )
                if ok and name:
                    self.exp_manager.update(exp_id, name=name)

    def _new_experiment(self):
        name, ok = QInputDialog.getText(
            self, "New Experiment", "Experiment name:",
            text=f"Experiment {self.exp_manager.count + 1}"
        )
        if ok and name:
            self.exp_manager.create_experiment(name)

    def _save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "SerialTrack Session (*.stk);;All Files (*)"
        )
        if path:
            try:
                self.exp_manager.save_session(path)
                self.status_text.setText(f"Session saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "SerialTrack Session (*.stk);;All Files (*)"
        )
        if path:
            try:
                # Clear timeline
                self.timeline.list_widget.clear()
                self.exp_manager.load_session(path)
                self.status_text.setText(f"Session loaded: {Path(path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", str(e))

    # ═══════════════════════════════════════════════════════════
    #  Status helpers
    # ═══════════════════════════════════════════════════════════

    def set_status(self, text: str, level: str = "ready"):
        self.status_text.setText(text)
        self.status_indicator.set_status(level, text.split(":")[0] if ":" in text else text[:20])

    def show_progress(self, visible: bool, value: int = 0, maximum: int = 100):
        self.global_progress.setVisible(visible)
        self.global_progress.setMaximum(maximum)
        self.global_progress.setValue(value)