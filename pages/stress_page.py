"""
Stress Analysis Page — Transform strain fields to stress via constitutive models.

Supports:
  - Material property definitions (Young's modulus, Poisson's ratio, custom stiffness)
  - Constitutive models: linear elastic (isotropic & anisotropic), neo-Hookean, Mooney-Rivlin
  - JSON import/export for material libraries
  - Batch computation across all frames
  - Preview of stress tensor components
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List
import json
import numpy as np
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QScrollArea, QProgressBar,
    QTabWidget, QFrame, QMessageBox, QCheckBox, QListWidget,
    QListWidgetItem, QTextEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt, Signal, QThread

from widgets.common import ParamEditor, MplCanvas, StatusIndicator
from core.settings import Settings
from core.plugin_registry import ParamSpec
from core.experiment_manager import PostProcessRecord


# ═══════════════════════════════════════════════════════════════════
#  Constitutive model implementations
# ═══════════════════════════════════════════════════════════════════

def _stiffness_isotropic(E: float, nu: float, ndim: int = 3) -> np.ndarray:
    """Build isotropic linear elastic stiffness tensor (Voigt notation).

    Returns C as (6,6) for 3D or (3,3) for 2D plane-stress.
    """
    if ndim == 2:
        # Plane stress
        factor = E / (1.0 - nu**2)
        C = np.array([
            [1.0, nu,  0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]) * factor
    else:
        # 3D
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        C = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                C[i, j] = lam
            C[i, i] += 2.0 * mu
        C[3, 3] = mu
        C[4, 4] = mu
        C[5, 5] = mu
    return C


def strain_to_stress_linear(eps_tensor: np.ndarray, C: np.ndarray,
                            ndim: int = 3) -> np.ndarray:
    """Apply linear elastic C:ε → σ on gridded strain field.

    Parameters
    ----------
    eps_tensor : (D, D, *grid_shape) — strain tensor
    C : (voigt_size, voigt_size) — stiffness in Voigt notation
    ndim : 2 or 3

    Returns
    -------
    sigma_tensor : (D, D, *grid_shape) — stress tensor
    """
    grid_shape = eps_tensor.shape[2:]
    sigma = np.zeros_like(eps_tensor)

    if ndim == 3:
        # Convert eps to Voigt: [e11, e22, e33, 2*e23, 2*e13, 2*e12]
        voigt_eps = np.zeros((6, *grid_shape))
        voigt_eps[0] = eps_tensor[0, 0]
        voigt_eps[1] = eps_tensor[1, 1]
        voigt_eps[2] = eps_tensor[2, 2]
        voigt_eps[3] = 2.0 * eps_tensor[1, 2]
        voigt_eps[4] = 2.0 * eps_tensor[0, 2]
        voigt_eps[5] = 2.0 * eps_tensor[0, 1]

        flat = voigt_eps.reshape(6, -1)
        stress_flat = C @ flat  # (6, N)
        voigt_sig = stress_flat.reshape(6, *grid_shape)

        sigma[0, 0] = voigt_sig[0]
        sigma[1, 1] = voigt_sig[1]
        sigma[2, 2] = voigt_sig[2]
        sigma[1, 2] = sigma[2, 1] = voigt_sig[3]
        sigma[0, 2] = sigma[2, 0] = voigt_sig[4]
        sigma[0, 1] = sigma[1, 0] = voigt_sig[5]
    else:
        # 2D plane stress
        voigt_eps = np.zeros((3, *grid_shape))
        voigt_eps[0] = eps_tensor[0, 0]
        voigt_eps[1] = eps_tensor[1, 1]
        voigt_eps[2] = 2.0 * eps_tensor[0, 1]

        flat = voigt_eps.reshape(3, -1)
        stress_flat = C @ flat
        voigt_sig = stress_flat.reshape(3, *grid_shape)

        sigma[0, 0] = voigt_sig[0]
        sigma[1, 1] = voigt_sig[1]
        sigma[0, 1] = sigma[1, 0] = voigt_sig[2]

    return sigma


def strain_to_stress_neohookean(F_tensor: np.ndarray, mu: float,
                                 kappa: float, ndim: int = 3) -> np.ndarray:
    """Neo-Hookean hyperelastic: compute Cauchy stress from F.

    σ = (μ/J)(B - I) + κ(J-1)I

    Parameters
    ----------
    F_tensor : (D, D, *grid_shape) — deformation gradient
    mu : shear modulus
    kappa : bulk modulus

    Returns
    -------
    sigma : (D, D, *grid_shape)
    """
    grid_shape = F_tensor.shape[2:]
    D = ndim

    # Add identity to get full F = I + ∂u/∂x
    F_full = F_tensor.copy()
    for i in range(D):
        F_full[i, i] += 1.0

    sigma = np.zeros_like(F_tensor)

    # Vectorized computation
    n_pts = int(np.prod(grid_shape))
    F_flat = F_full.reshape(D, D, n_pts)
    sig_flat = np.zeros((D, D, n_pts))

    for p in range(n_pts):
        Fp = F_flat[:, :, p]
        J = np.linalg.det(Fp)
        if abs(J) < 1e-12:
            continue
        B = Fp @ Fp.T  # left Cauchy-Green
        I = np.eye(D)
        sig_flat[:, :, p] = (mu / J) * (B - I) + kappa * (J - 1.0) * I

    sigma = sig_flat.reshape(D, D, *grid_shape)
    return sigma


def strain_to_stress_mooney_rivlin(F_tensor: np.ndarray, C10: float,
                                    C01: float, kappa: float,
                                    ndim: int = 3) -> np.ndarray:
    """Mooney-Rivlin hyperelastic model.

    W = C10(I1-3) + C01(I2-3) + κ/2*(J-1)^2
    """
    grid_shape = F_tensor.shape[2:]
    D = ndim

    F_full = F_tensor.copy()
    for i in range(D):
        F_full[i, i] += 1.0

    n_pts = int(np.prod(grid_shape))
    F_flat = F_full.reshape(D, D, n_pts)
    sig_flat = np.zeros((D, D, n_pts))

    for p in range(n_pts):
        Fp = F_flat[:, :, p]
        J = np.linalg.det(Fp)
        if abs(J) < 1e-12:
            continue
        B = Fp @ Fp.T
        I = np.eye(D)
        I1 = np.trace(B)
        Binv = np.linalg.inv(B) if D == 3 else I  # fallback for 2D

        # Cauchy stress for Mooney-Rivlin
        sig = (2.0 / J) * (
            C10 * B
            - C01 * (I1 * I - B) @ np.linalg.inv(Fp).T @ Fp.T
        ) + kappa * (J - 1.0) * I

        # Simplified for near-incompressible:
        sig = (2.0 / J) * (
            (C10 + I1 * C01) * B - C01 * B @ B
        ) + kappa * (J - 1.0) * I

        sig_flat[:, :, p] = sig

    return sig_flat.reshape(D, D, *grid_shape)


# ═══════════════════════════════════════════════════════════════════
#  Worker thread
# ═══════════════════════════════════════════════════════════════════

class StressWorker(QThread):
    """Compute stress from post-processed strain data."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    log_msg = Signal(str)

    def __init__(self, pp_results: dict, material: dict, model: str,
                 parent=None):
        super().__init__(parent)
        self.pp_results = pp_results
        self.material = material
        self.model = model

    def run(self):
        try:
            mat = self.material
            model = self.model
            frames_in = self.pp_results.get("frames", [])
            n_frames = len(frames_in)
            results = {"frames": [], "material": mat, "model": model}

            E = mat.get("youngs_modulus", 1000.0)
            nu = mat.get("poisson_ratio", 0.3)
            mu = mat.get("shear_modulus", E / (2.0 * (1.0 + nu)))
            kappa = mat.get("bulk_modulus", E / (3.0 * (1.0 - 2.0 * nu)))

            for i, frame in enumerate(frames_in):
                self.log_msg.emit(f"Computing stress frame {i+1}/{n_frames}")

                if frame is None:
                    results["frames"].append(None)
                    self.progress.emit(int(100 * (i + 1) / n_frames))
                    continue

                eps = np.array(frame["eps_tensor"])
                F = np.array(frame["F_tensor"])
                ndim = eps.shape[0]

                if model == "Linear Elastic (Isotropic)":
                    if "custom_C" in mat and mat["custom_C"] is not None:
                        C = np.array(mat["custom_C"])
                    else:
                        C = _stiffness_isotropic(E, nu, ndim)
                    sigma = strain_to_stress_linear(eps, C, ndim)

                elif model == "Linear Elastic (Anisotropic)":
                    C = np.array(mat.get("custom_C", _stiffness_isotropic(E, nu, ndim)))
                    sigma = strain_to_stress_linear(eps, C, ndim)

                elif model == "Neo-Hookean":
                    sigma = strain_to_stress_neohookean(F, mu, kappa, ndim)

                elif model == "Mooney-Rivlin":
                    C10 = mat.get("C10", mu / 2.0)
                    C01 = mat.get("C01", 0.0)
                    sigma = strain_to_stress_mooney_rivlin(F, C10, C01, kappa, ndim)
                else:
                    sigma = strain_to_stress_linear(
                        eps, _stiffness_isotropic(E, nu, ndim), ndim)

                # Von Mises equivalent stress
                s = sigma
                if ndim == 3:
                    vm = np.sqrt(0.5 * (
                        (s[0,0] - s[1,1])**2
                        + (s[1,1] - s[2,2])**2
                        + (s[2,2] - s[0,0])**2
                        + 6.0 * (s[0,1]**2 + s[1,2]**2 + s[0,2]**2)
                    ))
                else:
                    vm = np.sqrt(
                        s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2
                        + 3.0 * s[0,1]**2
                    )

                # Principal stresses (at each grid point)
                # We'll store max/min principal for quick access
                results["frames"].append({
                    "sigma_tensor": sigma.tolist(),
                    "von_mises": vm.tolist(),
                    "n_dim": ndim,
                })

                self.progress.emit(int(100 * (i + 1) / n_frames))

            self.log_msg.emit(f"\n✓ Stress computation complete "
                              f"({n_frames} frames, model={model})")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════
#  StressPage
# ═══════════════════════════════════════════════════════════════════

class StressPage(QWidget):
    """Stress analysis: material properties → constitutive model → stress."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._results = {}
        self._worker = None
        self._custom_C = None  # anisotropic stiffness
        self._init_ui()

    # ───── UI ─────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel: material & model config ──
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setMinimumWidth(340)
        left.setMaximumWidth(460)
        left_inner = QWidget()
        left_lay = QVBoxLayout(left_inner)
        left_lay.setSpacing(8)
        left.setWidget(left_inner)

        # Model selection
        model_grp = QGroupBox("Constitutive Model")
        model_lay = QVBoxLayout(model_grp)
        self.cb_model = QComboBox()
        self.cb_model.addItems([
            "Linear Elastic (Isotropic)",
            "Linear Elastic (Anisotropic)",
            "Neo-Hookean",
            "Mooney-Rivlin",
        ])
        self.cb_model.currentTextChanged.connect(self._on_model_changed)
        model_lay.addWidget(self.cb_model)
        left_lay.addWidget(model_grp)

        # Material properties (common)
        self.mat_params = ParamEditor([
            ParamSpec(name="youngs_modulus", label="Young's Modulus E",
                      param_type="float", default=1000.0,
                      tooltip="Elastic modulus in Pa (e.g., 1000 for soft hydrogels, 200e9 for steel)"),
            ParamSpec(name="poisson_ratio", label="Poisson's Ratio ν",
                      param_type="float", default=0.45,
                      tooltip="Typically 0.3–0.5. Use ~0.45–0.5 for hydrogels, ~0.3 for metals"),
        ])
        mat_grp = QGroupBox("Material Properties (Linear)")
        mat_lay = QVBoxLayout(mat_grp)
        mat_lay.addWidget(self.mat_params)
        left_lay.addWidget(mat_grp)
        self.mat_grp = mat_grp

        # Hyperelastic properties
        self.hyper_params = ParamEditor([
            ParamSpec(name="shear_modulus", label="Shear Modulus μ",
                      param_type="float", default=350.0,
                      tooltip="μ = E / (2(1+ν)). Initial shear modulus for hyperelastic models."),
            ParamSpec(name="bulk_modulus", label="Bulk Modulus κ",
                      param_type="float", default=5000.0,
                      tooltip="κ = E / (3(1-2ν)). Resistance to volumetric change."),
        ])
        hyper_grp = QGroupBox("Hyperelastic Properties")
        hyper_lay = QVBoxLayout(hyper_grp)
        hyper_lay.addWidget(self.hyper_params)
        self.hyper_grp = hyper_grp
        hyper_grp.setVisible(False)
        left_lay.addWidget(hyper_grp)

        # Mooney-Rivlin specific
        self.mr_params = ParamEditor([
            ParamSpec(name="C10", label="C₁₀",
                      param_type="float", default=175.0,
                      tooltip="First Mooney-Rivlin constant (≈ μ/2)"),
            ParamSpec(name="C01", label="C₀₁",
                      param_type="float", default=0.0,
                      tooltip="Second Mooney-Rivlin constant. C01=0 reduces to neo-Hookean."),
        ])
        mr_grp = QGroupBox("Mooney-Rivlin Constants")
        mr_lay = QVBoxLayout(mr_grp)
        mr_lay.addWidget(self.mr_params)
        self.mr_grp = mr_grp
        mr_grp.setVisible(False)
        left_lay.addWidget(mr_grp)

        # Anisotropic stiffness table
        aniso_grp = QGroupBox("Custom Stiffness Matrix C (Voigt 6×6)")
        aniso_lay = QVBoxLayout(aniso_grp)

        self.stiffness_table = QTableWidget(6, 6)
        self.stiffness_table.setMaximumHeight(200)
        self.stiffness_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.stiffness_table.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        labels = ["11", "22", "33", "23", "13", "12"]
        self.stiffness_table.setHorizontalHeaderLabels(labels)
        self.stiffness_table.setVerticalHeaderLabels(labels)

        # Fill with isotropic defaults
        self._fill_stiffness_table(1000.0, 0.45)

        aniso_lay.addWidget(self.stiffness_table)

        btn_fill = QPushButton("Fill from E, ν")
        btn_fill.clicked.connect(self._fill_from_isotropic)
        aniso_lay.addWidget(btn_fill)

        self.aniso_grp = aniso_grp
        aniso_grp.setVisible(False)
        left_lay.addWidget(aniso_grp)

        # JSON import/export
        io_grp = QGroupBox("Material Library")
        io_lay = QHBoxLayout(io_grp)
        btn_load_mat = QPushButton("📂 Load JSON")
        btn_load_mat.clicked.connect(self._load_material_json)
        btn_save_mat = QPushButton("💾 Save JSON")
        btn_save_mat.clicked.connect(self._save_material_json)
        io_lay.addWidget(btn_load_mat)
        io_lay.addWidget(btn_save_mat)
        left_lay.addWidget(io_grp)

        # Run controls
        run_grp = QGroupBox("Compute")
        run_lay = QVBoxLayout(run_grp)

        pp_note = QLabel("Uses strain data from Post-Process tab")
        pp_note.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font-size: 11px;")
        run_lay.addWidget(pp_note)

        self.btn_run = QPushButton("▶  Compute Stress")
        self.btn_run.setStyleSheet(
            f"background-color: {Settings.ACCENT_GREEN}; color: #282a36; "
            f"font-weight: bold; padding: 8px; border-radius: 4px;")
        self.btn_run.clicked.connect(self._run_stress)
        run_lay.addWidget(self.btn_run)

        self.stress_progress = QProgressBar()
        self.stress_progress.setVisible(False)
        run_lay.addWidget(self.stress_progress)

        self.status = StatusIndicator()
        run_lay.addWidget(self.status)

        left_lay.addWidget(run_grp)

        # History
        hist_grp = QGroupBox("Stress Runs")
        hist_lay = QVBoxLayout(hist_grp)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        hist_lay.addWidget(self.history_list)
        left_lay.addWidget(hist_grp)

        left_lay.addStretch()
        splitter.addWidget(left)

        # ── Right panel: preview ──
        right = QWidget()
        right_lay = QVBoxLayout(right)

        # Frame selector
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self.sb_frame = QSpinBox()
        self.sb_frame.setMinimum(0)
        self.sb_frame.valueChanged.connect(self._update_preview)
        frame_row.addWidget(self.sb_frame)

        # Component selector
        frame_row.addWidget(QLabel("View:"))
        self.cb_view = QComboBox()
        self.cb_view.addItems([
            "Von Mises", "σ_xx", "σ_yy", "σ_zz",
            "σ_xy", "σ_xz", "σ_yz",
        ])
        self.cb_view.currentIndexChanged.connect(self._update_preview)
        frame_row.addWidget(self.cb_view)
        frame_row.addStretch()
        right_lay.addLayout(frame_row)

        # Canvas
        self.result_canvas = MplCanvas(figsize=(8, 5), toolbar=True)
        right_lay.addWidget(self.result_canvas)

        # Log
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        right_lay.addWidget(self.log_edit)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    # ───── Model switching ────────────────────────────────────────

    def _on_model_changed(self, model: str):
        is_linear = "Linear" in model
        is_aniso = "Anisotropic" in model
        is_nh = model == "Neo-Hookean"
        is_mr = model == "Mooney-Rivlin"

        self.mat_grp.setVisible(is_linear)
        self.aniso_grp.setVisible(is_aniso)
        self.hyper_grp.setVisible(is_nh or is_mr)
        self.mr_grp.setVisible(is_mr)

    # ───── Stiffness table helpers ────────────────────────────────

    def _fill_stiffness_table(self, E: float, nu: float):
        C = _stiffness_isotropic(E, nu, ndim=3)
        for i in range(6):
            for j in range(6):
                item = QTableWidgetItem(f"{C[i,j]:.4g}")
                item.setTextAlignment(Qt.AlignCenter)
                self.stiffness_table.setItem(i, j, item)

    def _fill_from_isotropic(self):
        vals = self.mat_params.get_values()
        self._fill_stiffness_table(vals["youngs_modulus"], vals["poisson_ratio"])

    def _read_stiffness_table(self) -> np.ndarray:
        C = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                item = self.stiffness_table.item(i, j)
                try:
                    C[i, j] = float(item.text()) if item else 0.0
                except ValueError:
                    C[i, j] = 0.0
        return C

    # ───── JSON I/O ───────────────────────────────────────────────

    def _gather_material(self) -> dict:
        model = self.cb_model.currentText()
        mat = {}
        mat["model"] = model

        if "Linear" in model:
            mat.update(self.mat_params.get_values())
            if "Anisotropic" in model:
                mat["custom_C"] = self._read_stiffness_table().tolist()
        if "Neo-Hookean" in model or "Mooney-Rivlin" in model:
            mat.update(self.hyper_params.get_values())
        if "Mooney-Rivlin" in model:
            mat.update(self.mr_params.get_values())
        return mat

    def _load_material_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Material", "", "JSON (*.json);;All (*)")
        if not path:
            return
        try:
            with open(path) as f:
                mat = json.load(f)

            # Set model
            model = mat.get("model", "Linear Elastic (Isotropic)")
            idx = self.cb_model.findText(model)
            if idx >= 0:
                self.cb_model.setCurrentIndex(idx)

            # Fill params
            if "youngs_modulus" in mat:
                self.mat_params.set_values({
                    "youngs_modulus": mat["youngs_modulus"],
                    "poisson_ratio": mat.get("poisson_ratio", 0.3),
                })
            if "shear_modulus" in mat:
                self.hyper_params.set_values({
                    "shear_modulus": mat["shear_modulus"],
                    "bulk_modulus": mat.get("bulk_modulus", 5000.0),
                })
            if "C10" in mat:
                self.mr_params.set_values({
                    "C10": mat["C10"],
                    "C01": mat.get("C01", 0.0),
                })
            if "custom_C" in mat:
                C = np.array(mat["custom_C"])
                for i in range(min(6, C.shape[0])):
                    for j in range(min(6, C.shape[1])):
                        item = QTableWidgetItem(f"{C[i,j]:.4g}")
                        item.setTextAlignment(Qt.AlignCenter)
                        self.stiffness_table.setItem(i, j, item)

            self.log_edit.append(f"Loaded material from {path}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _save_material_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Material", "material.json", "JSON (*.json);;All (*)")
        if not path:
            return
        try:
            mat = self._gather_material()
            with open(path, "w") as f:
                json.dump(mat, f, indent=2)
            self.log_edit.append(f"Saved material to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    # ───── Run ────────────────────────────────────────────────────

    def _run_stress(self):
        if not self.main_window:
            return

        # Get post-process results
        pp_page = self.main_window.pages.get("postprocess")
        pp_results = pp_page.get_results() if pp_page else {}

        if not pp_results or not pp_results.get("frames"):
            QMessageBox.warning(self, "No Data",
                                "Run post-processing first to generate strain fields.")
            return

        model = self.cb_model.currentText()
        material = self._gather_material()

        self.stress_progress.setVisible(True)
        self.stress_progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.status.set_status("running", "Computing...")
        self.log_edit.clear()
        self.log_edit.append(f"Model: {model}")
        self.log_edit.append(f"Material: E={material.get('youngs_modulus','N/A')}, "
                             f"ν={material.get('poisson_ratio','N/A')}")

        self._worker = StressWorker(pp_results, material, model)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.stress_progress.setValue)
        self._worker.log_msg.connect(lambda m: self.log_edit.append(m))
        self._worker.start()

    def _on_finished(self, results):
        self._results = results
        self.stress_progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.status.set_status("ready", "Complete")

        # Update frame selector
        n_frames = len(results.get("frames", []))
        self.sb_frame.setMaximum(max(0, n_frames - 1))

        # Add to history
        model = results.get("model", "?")
        mat = results.get("material", {})
        record = PostProcessRecord(
            description=f"{model}, E={mat.get('youngs_modulus','?')}",
            config={"material": mat, "model": model},
        )
        item = QListWidgetItem(f"[{record.timestamp}] {record.description}")
        item.setData(Qt.UserRole, record.record_id)
        self.history_list.addItem(item)

        # Save to experiment
        if self.main_window:
            exp = self.main_window.exp_manager.active
            if exp:
                exp.stress_runs.append(record)
                exp.store_stress_results(results)
                self.main_window.exp_manager.update(exp.exp_id)

        self._update_preview()

    def _on_error(self, msg):
        self.stress_progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.status.set_status("error", "Failed")
        self.log_edit.append(f"\n❌ Error: {msg}")
        QMessageBox.warning(self, "Stress Computation Error", msg)

    # ───── Preview ────────────────────────────────────────────────

    def _update_preview(self):
        if not self._results or not self._results.get("frames"):
            return

        frame_idx = self.sb_frame.value()
        frames = self._results["frames"]
        if frame_idx >= len(frames) or frames[frame_idx] is None:
            return

        frame = frames[frame_idx]
        view = self.cb_view.currentText()

        self.result_canvas.clear()

        if view == "Von Mises":
            data = np.array(frame["von_mises"])
            title = f"Von Mises Stress — Frame {frame_idx}"
            cmap = "hot"
        else:
            sigma = np.array(frame["sigma_tensor"])
            comp_map = {
                "σ_xx": (0, 0), "σ_yy": (1, 1), "σ_zz": (2, 2),
                "σ_xy": (0, 1), "σ_xz": (0, 2), "σ_yz": (1, 2),
            }
            i, j = comp_map.get(view, (0, 0))
            ndim = frame.get("n_dim", 3)
            if i >= ndim or j >= ndim:
                return
            data = sigma[i, j]
            title = f"{view} — Frame {frame_idx}"
            cmap = "coolwarm"

        # If 3D, take mid-slice
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]

        ax = self.result_canvas.add_subplot(1, 1, 1)
        im = ax.imshow(data.T, cmap=cmap, origin="lower")
        ax.set_title(title, color=Settings.FG_PRIMARY, fontsize=10)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        self.result_canvas.figure.colorbar(im, cax=cax)
        self.result_canvas.draw()

    # ───── Experiment lifecycle ────────────────────────────────────

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def on_experiment_changed(self, exp_id: str):
        pass  # Now handled by save/load_from_experiment

    def on_activated(self):
        pass

    def save_to_experiment(self, exp):
        """Persist stress results into experiment cache."""
        exp.store_stress_results(self._results)

    def load_from_experiment(self, exp):
        """Restore stress results from experiment cache."""
        self._results = exp.get_stress_results() or {}

        # Refresh history
        self.history_list.clear()
        for r in exp.stress_runs:
            item = QListWidgetItem(f"[{r.timestamp}] {r.description}")
            item.setData(Qt.UserRole, r.record_id)
            self.history_list.addItem(item)

        # Refresh preview
        frames = self._results.get("frames", [])
        if frames:
            self.sb_frame.setMaximum(max(0, len(frames) - 1))
            self._update_preview()
            self.status.set_status("ready", f"{len(frames)} frames loaded")
        else:
            self.sb_frame.setMaximum(0)
            self.result_canvas.clear()
            self.result_canvas.draw()
            self.status.set_status("idle", "No stress data")