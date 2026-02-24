"""
SerialTrack Python — Image loading utilities
==============================================
    serialtrack/io.py

Replaces: funReadImage3.m, GenerateVolMatfile.m

Fixes from v1
-------------
- Removed unused imports (numba, scipy.ndimage) that added needless
  dependencies and slowed import time.
"""

from __future__ import annotations
from typing import Optional, Union, List
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger("serialtrack.io")


class ImageLoader:
    """Load 2-D image sequences or 3-D volumetric stacks.

    All loaders return arrays in **SerialTrack convention**:
        2-D → ``(x, y)``   (transposed from typical row-col)
        3-D → ``(x, y, z)`` (permuted from MATLAB image convention)

    Supported formats
    -----------------
    - Individual 2-D TIFFs / PNGs in a folder  →  :meth:`load_2d_sequence`
    - TIFF stack folder (one slice per file)    →  :meth:`load_3d_tiff_stack`
    - MATLAB ``.mat`` volumetric files          →  :meth:`load_3d_mat`
    - NumPy ``.npy`` volumes                    →  :meth:`load_npy`
    """

    # ---- 2-D ----
    @staticmethod
    def load_2d_sequence(
        folder: Union[str, Path],
        pattern: str = "*.tif*",
        dtype: type = np.float64,
    ) -> List[np.ndarray]:
        """Load a folder of 2-D images as a time sequence."""
        import tifffile

        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {folder}"
            )
        imgs: List[np.ndarray] = []
        for f in files:
            img = tifffile.imread(str(f))
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1)  # RGB → grey
            # Transpose to (x, y): image (row, col) → (col, row)
            imgs.append(np.ascontiguousarray(img.T, dtype=dtype))
        log.info("Loaded %d 2-D frames from %s", len(imgs), folder)
        return imgs

    # ---- 3-D from TIFF slices ----
    @staticmethod
    def load_3d_tiff_stack(
        folder: Union[str, Path],
        pattern: str = "*.tif*",
        dtype: type = np.float64,
    ) -> np.ndarray:
        """Build one 3-D volume from a folder of 2-D slice TIFFs."""
        import tifffile

        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {folder}"
            )
        slices = [tifffile.imread(str(f)) for f in files]
        vol = np.stack(slices, axis=-1)          # (row, col, z)
        vol = np.ascontiguousarray(
            vol.transpose(1, 0, 2), dtype=dtype  # → (x, y, z)
        )
        log.info("Built 3-D volume %s from %d slices", vol.shape, len(files))
        return vol

    # ---- 3-D from .mat ----
    @staticmethod
    def load_3d_mat(
        filepath: Union[str, Path],
        var_name: Optional[str] = None,
        dtype: type = np.float64,
    ) -> np.ndarray:
        """Load a 3-D volume from a MATLAB ``.mat`` file.

        Applies ``permute([2,1,3])`` to convert from MATLAB image
        convention ``(row, col, slice)`` to SerialTrack ``(x, y, z)``.
        """
        import scipy.io as sio

        filepath = Path(filepath)
        if filepath.suffix == ".mat":
            data = sio.loadmat(str(filepath), simplify_cells=True)
        else:
            import h5py  # v7.3 .mat → HDF5
            with h5py.File(str(filepath), "r") as f:
                keys = [k for k in f.keys() if not k.startswith("#")]
                var_name = var_name or keys[0]
                vol = np.array(f[var_name])
                return np.ascontiguousarray(vol.transpose(1, 0, 2), dtype=dtype)

        if var_name is None:
            keys = [k for k in data if not k.startswith("_")]
            var_name = keys[0]

        vol = np.asarray(data[var_name])
        # Unwrap MATLAB cell
        if vol.dtype == object:
            vol = vol.flat[0]
        vol = np.ascontiguousarray(vol.transpose(1, 0, 2), dtype=dtype)
        log.info("Loaded 3-D volume %s from %s['%s']",
                 vol.shape, filepath.name, var_name)
        return vol

    # ---- 3-D from .npy ----
    @staticmethod
    def load_npy(filepath: Union[str, Path], dtype: type = np.float64) -> np.ndarray:
        return np.load(str(filepath)).astype(dtype)

    # ---- convenience: load a time-series of 3-D volumes ----
    @staticmethod
    def load_3d_sequence(
        folder: Union[str, Path],
        pattern: str = "*.mat",
        var_name: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Load an ordered sequence of 3-D volumes from a folder."""
        folder = Path(folder)
        paths = sorted(folder.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {folder}"
            )
        vols: List[np.ndarray] = []
        for p in paths:
            if p.suffix == ".mat":
                vols.append(ImageLoader.load_3d_mat(p, var_name))
            elif p.suffix == ".npy":
                vols.append(ImageLoader.load_npy(p))
            else:
                raise ValueError(f"Unsupported volume format: {p.suffix}")
        log.info("Loaded %d volumes from %s", len(vols), folder)
        return vols
