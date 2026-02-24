#!/usr/bin/env python3
"""
SerialTrack GUI — Main Entry Point
====================================
PyDracula-themed PySide6 interface for SerialTrack particle tracking.

Usage:
    python run.py
"""
import sys
import os
import importlib

# ── Locate and register the SerialTrack backend as a package ────
# The backend files (config.py, detection.py, tracking.py, etc.)
# use relative imports (from .config import ...) so they MUST be
# imported as a package, not as loose modules.
#
# This script lives inside the package directory alongside the
# backend files.  We add the PARENT directory to sys.path so
# Python can import the folder as a package by its directory name,
# then alias it as "serialtrack" for consistent imports.
#
# Folder layout expected:
#   SerialTrack_Python/      (or any name)
#     ├── __init__.py        ← backend package
#     ├── config.py
#     ├── detection.py
#     ├── tracking.py
#     ├── fields.py
#     ├── ...
#     ├── run.py             ← this file
#     ├── core/              ← GUI
#     ├── pages/             ← GUI
#     ├── widgets/           ← GUI
#     └── plugins/           ← GUI

_gui_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_gui_dir)
_pkg_name = os.path.basename(_gui_dir)

# Add parent so the folder is importable as a package
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
# Also add gui_dir so core/, pages/, widgets/ are importable
if _gui_dir not in sys.path:
    sys.path.insert(0, _gui_dir)

# Import the backend package by its actual folder name
_pkg = importlib.import_module(_pkg_name)

# Register it under the alias "serialtrack" so GUI pages can use
# consistent imports like: from serialtrack.config import ...
sys.modules["serialtrack"] = _pkg

# Also register each submodule so "from serialtrack.X import Y" works
for _submod_name in [
    "config", "detection", "tracking", "fields", "regularization",
    "matching", "outliers", "prediction", "trajectories", "io", "results",
]:
    _full = f"{_pkg_name}.{_submod_name}"
    if _full in sys.modules:
        sys.modules[f"serialtrack.{_submod_name}"] = sys.modules[_full]
    else:
        try:
            _sub = importlib.import_module(f".{_submod_name}", package=_pkg_name)
            sys.modules[f"serialtrack.{_submod_name}"] = _sub
        except ImportError:
            pass  # optional module, skip

os.environ["QT_FONT_DPI"] = "96"

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from core.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    import platform
    if platform.system() == "Darwin":
        font = QFont("Helvetica Neue", 10)
    elif platform.system() == "Windows":
        font = QFont("Segoe UI", 10)
    else:
        font = QFont("Sans Serif", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
