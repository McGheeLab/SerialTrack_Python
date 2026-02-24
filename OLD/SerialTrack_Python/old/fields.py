
# ╔══════════════════════════════════════════════════════════════╗
# ║  FILE 2: serialtrack/fields.py                               ║
# ║  Displacement & strain field computation on grids             ║
# ║  Replaces: funDerivativeOp3 (post-processing usage),          ║
# ║  funCompDefGrad3 (strain gauge), postprocessing sections      ║
# ╚══════════════════════════════════════════════════════════════╝
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.sparse import eye as speye, diags as spdiags, kron as spkron, csc_matrix
from scipy.sparse.linalg import spsolve
import logging
from dataclasses import dataclass, field as dc_field
from typing import Dict

from .regularization import scatter_to_grid_multi

log = logging.getLogger("serialtrack.regularization")

@dataclass
class DisplacementField:
    """Gridded displacement field with metadata.

    Stores displacement components on a regular grid and provides
    gradient (strain) computation via ``np.gradient``.
    """
    grids: Tuple[np.ndarray, ...]           # (x_grid, y_grid, [z_grid])
    components: np.ndarray                   # (D, *grid_shape) displacement
    pixel_steps: np.ndarray                  # (D,) physical size per pixel
    time_step: float = 1.0                   # time between frames

    @property
    def ndim(self) -> int:
        return len(self.grids)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.grids[0].shape

    @property
    def velocity(self) -> np.ndarray:
        """Displacement / time_step → velocity field."""
        return self.components * self.pixel_steps[:, None, None, None][:self.ndim] / self.time_step

    def gradient(self) -> np.ndarray:
        """Compute deformation gradient tensor F = ∂u/∂x.

        Returns
        -------
        F : (D, D, *grid_shape) — F[i, j] = ∂u_i / ∂x_j
        """
        ndim = self.ndim
        F = np.zeros((ndim, ndim, *self.shape), dtype=np.float64)
        for i in range(ndim):
            grads = np.gradient(self.components[i], *[
                self.pixel_steps[d] * np.unique(np.round(np.diff(
                    self.grids[d].ravel()[:self.shape[d]+1]), 6))[0]
                if self.shape[d] > 1 else 1.0
                for d in range(ndim)
            ])
            # np.gradient returns list when ndim > 1
            if ndim == 1:
                grads = [grads]
            for j in range(ndim):
                F[i, j] = grads[j]
        return F

    def strain(self) -> np.ndarray:
        """Infinitesimal strain tensor ε = 0.5*(F + F^T).

        Returns (D, D, *grid_shape).
        """
        F = self.gradient()
        return 0.5 * (F + F.transpose(1, 0, *range(2, 2 + self.ndim)))

    def to_physical(self) -> 'DisplacementField':
        """Return a new field with displacements in physical units."""
        phys = self.components.copy()
        for d in range(self.ndim):
            sl = [None] * (self.ndim + 1)
            sl[0] = d
            phys[d] *= self.pixel_steps[d]
        return DisplacementField(
            grids=self.grids,
            components=phys,
            pixel_steps=self.pixel_steps,
            time_step=self.time_step,
        )


@dataclass
class StrainField:
    """Pre-computed strain field with components."""
    grids: Tuple[np.ndarray, ...]
    F_tensor: np.ndarray   # (D, D, *grid_shape) deformation gradient
    eps_tensor: np.ndarray # (D, D, *grid_shape) infinitesimal strain


def compute_strain_mls(
    disp: np.ndarray,
    coords: np.ndarray,
    f_o_s: float,
    n_neighbors: int,
    pixel_steps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving least-squares strain gauge at each particle.

    Replaces ``funCompDefGrad3.m`` for post-processing strain
    computation.  Fits u(x) = u0 + F·(x - x0) at each particle
    using its neighbors.

    Parameters
    ----------
    disp : (N, D) — displacement at each particle
    coords : (N, D) — particle positions
    f_o_s, n_neighbors : search parameters
    pixel_steps : optional physical scaling

    Returns
    -------
    U : (N, D) — fitted displacement (smoothed)
    F_tensor : (N, D, D) — deformation gradient at each particle
    valid : (N,) bool — which particles had a valid fit
    """
    ndim = coords.shape[1]
    N = len(coords)
    U = np.full((N, ndim), np.nan)
    F_tensor = np.full((N, ndim, ndim), np.nan)
    valid = np.zeros(N, dtype=bool)

    if N < ndim + 1:
        return U, F_tensor, valid

    tree = cKDTree(coords)
    K = min(n_neighbors, N - 1)
    radius = np.sqrt(ndim) * f_o_s
    dd, ii = tree.query(coords, k=K + 1)

    for p in range(N):
        # Neighbors within radius
        mask = dd[p] < radius
        idx = ii[p][mask]
        if len(idx) < ndim + 1:
            U[p] = disp[p]
            continue

        # Design matrix: [1, (x-x0), (y-y0), (z-z0)]
        dx = coords[idx] - coords[p]
        A = np.column_stack([np.ones(len(idx)), dx])

        try:
            for d in range(ndim):
                params, _, _, _ = np.linalg.lstsq(A, disp[idx, d], rcond=None)
                U[p, d] = params[0]
                F_tensor[p, d, :] = params[1:]  # ∂u_d/∂x_j
            valid[p] = True
        except np.linalg.LinAlgError:
            U[p] = disp[p]

    # Apply physical scaling if provided
    if pixel_steps is not None:
        ps = np.asarray(pixel_steps)
        for i in range(ndim):
            for j in range(ndim):
                F_tensor[valid, i, j] *= ps[i] / ps[j]

    return U, F_tensor, valid


def compute_gridded_strain(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 1e-3,
    pixel_steps: Optional[np.ndarray] = None,
) -> Tuple[DisplacementField, StrainField]:
    """Full post-processing: scatter → grid → gradient → strain.

    Replaces the postprocessing sections of
    ``run_Serial_MPT_3D_hardpar_accum.m``.

    Returns both the gridded displacement field and strain field.
    """
    ndim = coords.shape[1]
    ps = np.ones(ndim) if pixel_steps is None else np.asarray(pixel_steps)

    grids, disp_grid = scatter_to_grid_multi(
        coords, disp, grid_step, smoothness
    )

    dfield = DisplacementField(
        grids=grids,
        components=disp_grid,
        pixel_steps=ps,
    )

    F = dfield.gradient()
    eps = 0.5 * (F + F.transpose(1, 0, *range(2, 2 + ndim)))

    sfield = StrainField(grids=grids, F_tensor=F, eps_tensor=eps)

    return dfield, sfield