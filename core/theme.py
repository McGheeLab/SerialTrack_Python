"""
Dracula-based dark theme stylesheet for SerialTrack GUI.
"""

STYLESHEET = """
/* ═══════════════════════════════════════════════════════════
   GLOBAL
   ═══════════════════════════════════════════════════════════ */
QWidget {
    color: #f8f8f2;
    font: 10pt "Helvetica Neue";
    background-color: transparent;
}

QMainWindow {
    background-color: #282a36;
}

/* Tooltips */
QToolTip {
    color: #ffffff;
    background-color: rgba(33, 37, 43, 220);
    border: none;
    border-left: 2px solid #bd93f9;
    padding: 6px 10px;
    font: 9pt "Helvetica Neue";
}

/* ═══════════════════════════════════════════════════════════
   FRAMES & PANELS
   ═══════════════════════════════════════════════════════════ */
#bgApp {
    background-color: #282a36;
    border: 1px solid #44475a;
    border-radius: 10px;
}

#leftMenuBg {
    background-color: #21252b;
}

#contentArea {
    background-color: #282a36;
}

#topBar {
    background-color: #21252b;
    border-bottom: 1px solid #44475a;
}

#bottomBar {
    background-color: #21252b;
    border-top: 1px solid #44475a;
}

/* ═══════════════════════════════════════════════════════════
   LABELS
   ═══════════════════════════════════════════════════════════ */
QLabel {
    color: #f8f8f2;
    padding: 0px;
}

QLabel#titleLabel {
    font: bold 14pt "Helvetica Neue";
    color: #bd93f9;
}

QLabel#subtitleLabel {
    font: 9pt "Helvetica Neue";
    color: #b0b0b0;
}

QLabel#sectionHeader {
    font: bold 11pt "Helvetica Neue";
    color: #ff79c6;
    padding: 4px 0px;
}

QLabel#accentLabel {
    color: #8be9fd;
}

/* ═══════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════ */
QPushButton {
    background-color: #44475a;
    color: #f8f8f2;
    border: none;
    border-radius: 5px;
    padding: 6px 16px;
    font: 10pt "Helvetica Neue";
    min-height: 28px;
}
QPushButton:hover {
    background-color: #5a5e72;
}
QPushButton:pressed {
    background-color: #bd93f9;
    color: #282a36;
}
QPushButton:disabled {
    background-color: #363944;
    color: #666;
}

QPushButton#primaryBtn {
    background-color: #bd93f9;
    color: #282a36;
    font-weight: bold;
}
QPushButton#primaryBtn:hover {
    background-color: #caa4ff;
}
QPushButton#primaryBtn:pressed {
    background-color: #a678e8;
}

QPushButton#dangerBtn {
    background-color: #ff5555;
    color: #f8f8f2;
}
QPushButton#dangerBtn:hover {
    background-color: #ff7777;
}

QPushButton#successBtn {
    background-color: #50fa7b;
    color: #282a36;
    font-weight: bold;
}
QPushButton#successBtn:hover {
    background-color: #6dffa0;
}

/* Sidebar nav buttons — icon only */
QPushButton#navBtnIcon {
    background-color: transparent;
    color: #b0b0b0;
    border: none;
    border-radius: 0px;
    padding: 0px;
    font-size: 18pt;
    text-align: center;
    min-height: 44px;
    max-height: 44px;
}
QPushButton#navBtnIcon:hover {
    background-color: #343b48;
    color: #f8f8f2;
}
QPushButton#navBtnIcon:checked {
    border-left: 3px solid #bd93f9;
    background-color: #2c313c;
    color: #f8f8f2;
}

/* Legacy text nav button style (kept for compatibility) */
QPushButton#navBtn {
    background-color: transparent;
    color: #b0b0b0;
    border: none;
    border-radius: 0px;
    padding: 12px 20px;
    text-align: left;
    font: 10pt "Helvetica Neue";
    min-height: 38px;
}
QPushButton#navBtn:hover {
    background-color: #343b48;
    color: #f8f8f2;
}
QPushButton#navBtn:checked {
    border-left: 3px solid #bd93f9;
    background-color: #2c313c;
    color: #f8f8f2;
}

/* ═══════════════════════════════════════════════════════════
   INPUTS
   ═══════════════════════════════════════════════════════════ */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #343b48;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
    selection-background-color: #bd93f9;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #bd93f9;
}

QComboBox {
    background-color: #343b48;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
}
QComboBox:hover {
    border: 1px solid #bd93f9;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #bd93f9;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #343b48;
    color: #f8f8f2;
    border: 1px solid #44475a;
    selection-background-color: #44475a;
    outline: none;
}

QCheckBox {
    color: #f8f8f2;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #44475a;
    background-color: #343b48;
}
QCheckBox::indicator:checked {
    background-color: #bd93f9;
    border: 1px solid #bd93f9;
}

/* ═══════════════════════════════════════════════════════════
   SLIDERS
   ═══════════════════════════════════════════════════════════ */
QSlider::groove:horizontal {
    height: 6px;
    background: #44475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #bd93f9;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #bd93f9;
    border-radius: 3px;
}

QSlider::groove:vertical {
    width: 6px;
    background: #44475a;
    border-radius: 3px;
}
QSlider::handle:vertical {
    background: #bd93f9;
    width: 16px;
    height: 16px;
    margin: 0 -5px;
    border-radius: 8px;
}

/* ═══════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════ */
QProgressBar {
    background-color: #343b48;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #f8f8f2;
    font: bold 9pt "Helvetica Neue";
    min-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #bd93f9, stop:1 #ff79c6);
    border-radius: 4px;
}

/* ═══════════════════════════════════════════════════════════
   SCROLL BARS
   ═══════════════════════════════════════════════════════════ */
QScrollBar:vertical {
    background: #21252b;
    width: 10px;
    border: none;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #44475a;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #5a5e72;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #21252b;
    height: 10px;
    border: none;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #44475a;
    min-width: 30px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #5a5e72;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ═══════════════════════════════════════════════════════════
   TAB WIDGET
   ═══════════════════════════════════════════════════════════ */
QTabWidget::pane {
    border: 1px solid #44475a;
    border-radius: 5px;
    background-color: #282a36;
}
QTabBar::tab {
    background-color: #343b48;
    color: #b0b0b0;
    border: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background-color: #282a36;
    color: #bd93f9;
    border-bottom: 2px solid #bd93f9;
}
QTabBar::tab:hover:!selected {
    background-color: #3a3f4b;
    color: #f8f8f2;
}

/* ═══════════════════════════════════════════════════════════
   GROUP BOX
   ═══════════════════════════════════════════════════════════ */
QGroupBox {
    border: 1px solid #44475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font: bold 10pt "Helvetica Neue";
    color: #8be9fd;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #8be9fd;
}

/* ═══════════════════════════════════════════════════════════
   LISTS & TREES
   ═══════════════════════════════════════════════════════════ */
QListWidget, QTreeWidget, QTableWidget {
    background-color: #21252b;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 3px;
}
QListWidget::item:selected {
    background-color: #44475a;
    color: #bd93f9;
}
QListWidget::item:hover:!selected {
    background-color: #343b48;
}

QHeaderView::section {
    background-color: #343b48;
    color: #8be9fd;
    border: none;
    border-right: 1px solid #44475a;
    padding: 6px;
    font: bold 9pt "Helvetica Neue";
}

/* ═══════════════════════════════════════════════════════════
   TEXT EDIT / LOG
   ═══════════════════════════════════════════════════════════ */
QTextEdit, QPlainTextEdit {
    background-color: #21252b;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    padding: 4px;
    font: 9pt "Consolas";
}

/* ═══════════════════════════════════════════════════════════
   SPLITTER
   ═══════════════════════════════════════════════════════════ */
QSplitter::handle {
    background-color: #44475a;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}

/* ═══════════════════════════════════════════════════════════
   SCROLL AREA
   ═══════════════════════════════════════════════════════════ */
QScrollArea {
    border: none;
}

/* ═══════════════════════════════════════════════════════════
   SPECIFIC CUSTOM CLASSES
   ═══════════════════════════════════════════════════════════ */
#experimentList {
    background-color: #21252b;
    border: 1px solid #44475a;
    border-radius: 8px;
    padding: 4px;
}

#statusIndicator {
    font: bold 9pt "Helvetica Neue";
    padding: 4px 12px;
    border-radius: 4px;
}

#plotFrame {
    background-color: #21252b;
    border: 1px solid #44475a;
    border-radius: 8px;
}

#tooltipBar {
    background-color: #21252b;
    border-top: 1px solid #44475a;
}
"""