"""
SerialTrack Python — Chunk 1
=============================
Split this file into three modules:
    serialtrack/config.py
    serialtrack/io.py
    serialtrack/detection.py

Dependencies:
    pip install numpy scipy scikit-image numba tifffile

Optional (for StarDist detection):
    pip install stardist tensorflow
"""
# ╔══════════════════════════════════════════════════════════════╗
# ║  FILE 3: serialtrack/detection.py — Particle detection      ║
# ╚══════════════════════════════════════════════════════════════╝
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path
from typing import List
import logging
from scipy import ndimage
import numba as nb
from .config import DetectionConfig, DetectionMethod

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  StarDist availability check
# ─────────────────────────────────────────────────────────────

_STARDIST_AVAILABLE = None  # lazy-checked on first use


def stardist_available() -> bool:
    """Check whether StarDist and its dependencies are importable.

    Result is cached after the first call.
    """
    global _STARDIST_AVAILABLE
    if _STARDIST_AVAILABLE is None:
        try:
            import stardist          # noqa: F401
            from csbdeep.utils import normalize  # noqa: F401
            _STARDIST_AVAILABLE = True
            logger.info("StarDist is available (version %s)", stardist.__version__)
        except ImportError:
            _STARDIST_AVAILABLE = False
            logger.info(
                "StarDist not installed. Install with: "
                "pip install stardist tensorflow"
            )
    return _STARDIST_AVAILABLE


# ─────────────────────────────────────────────────────────────
#  Numba-accelerated sub-pixel localization kernels
# ─────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _subpixel_poly_2d(log_img, xs, ys):
    """3-point parabola sub-pixel refinement for 2-D peaks.

    For each peak at integer coords (xs[i], ys[i]) in *log_img*,
    fit y = a + bx + cx² along each axis and return the shift to
    the parabola vertex.

    Returns (dx, dy) arrays — shifts to add to integer coords.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    h, w = log_img.shape  # note: (x-dim, y-dim) in our convention

    for i in range(n):
        xi, yi = xs[i], ys[i]
        # x-direction
        if 0 < xi < h - 1:
            a = log_img[xi - 1, yi]
            b = log_img[xi, yi]
            c = log_img[xi + 1, yi]
            d = 2.0 * (a - 2.0 * b + c)
            dx[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y-direction
        if 0 < yi < w - 1:
            a = log_img[xi, yi - 1]
            b = log_img[xi, yi]
            c = log_img[xi, yi + 1]
            d = 2.0 * (a - 2.0 * b + c)
            dy[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
    return dx, dy


@nb.njit(cache=True)
def _subpixel_poly_3d(log_img, xs, ys, zs):
    """3-point parabola sub-pixel refinement for 3-D peaks.

    Returns (dx, dy, dz) arrays.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)
    sx, sy, sz = log_img.shape

    for i in range(n):
        xi, yi, zi = xs[i], ys[i], zs[i]
        # x
        if 0 < xi < sx - 1:
            a, b, c = log_img[xi-1,yi,zi], log_img[xi,yi,zi], log_img[xi+1,yi,zi]
            d = 2.0*(a - 2.0*b + c)
            dx[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y
        if 0 < yi < sy - 1:
            a, b, c = log_img[xi,yi-1,zi], log_img[xi,yi,zi], log_img[xi,yi+1,zi]
            d = 2.0*(a - 2.0*b + c)
            dy[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
        # z
        if 0 < zi < sz - 1:
            a, b, c = log_img[xi,yi,zi-1], log_img[xi,yi,zi], log_img[xi,yi,zi+1]
            d = 2.0*(a - 2.0*b + c)
            dz[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dz[i] = 0.0
    return dx, dy, dz


@nb.njit(parallel=True, cache=True)
def _radial_symmetry_3d(patches, half_win, dccd, abc):
    """Radial-symmetry sub-voxel localization (Liu et al. 2013).

    Parameters
    ----------
    patches : (N, wx, wy, wz)  float64 array of image patches
    half_win : (3,) int array   half window sizes
    dccd : (3,) float array     pixel spacings
    abc  : (3,) float array     anisotropy factors

    Returns
    -------
    dx, dy, dz : (N,) sub-pixel shifts
    """
    N = patches.shape[0]
    wx, wy, wz = patches.shape[1], patches.shape[2], patches.shape[3]
    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)
    dz = np.zeros(N, dtype=np.float64)
    a, b, c = abc[0], abc[1], abc[2]
    dxc, dyc, dzc = dccd[0], dccd[1], dccd[2]

    for pi in nb.prange(N):
        # --- intensity-weighted centroid ---
        sx_ = 0.0; sy_ = 0.0; sz_ = 0.0; tot = 0.0
        for ix in range(wx):
            for iy in range(wy):
                for iz in range(wz):
                    v = patches[pi, ix, iy, iz]
                    sx_ += v * (ix - (wx-1)*0.5) * dxc
                    sy_ += v * (iy - (wy-1)*0.5) * dyc
                    sz_ += v * (iz - (wz-1)*0.5) * dzc
                    tot += v
        if tot < 1e-30:
            continue
        xm = sx_ / tot;  ym = sy_ / tot;  zm = sz_ / tot

        # --- build 3×3 normal system from gradient votes ---
        A00=0.;A01=0.;A02=0.;A11=0.;A12=0.;A22=0.
        B0=0.;B1=0.;B2=0.

        for ix in range(1, wx-1):
            for iy in range(1, wy-1):
                for iz in range(1, wz-1):
                    gu = (patches[pi,ix+1,iy,iz] - patches[pi,ix-1,iy,iz])/(2*dxc)
                    gv = (patches[pi,ix,iy+1,iz] - patches[pi,ix,iy-1,iz])/(2*dyc)
                    gw = (patches[pi,ix,iy,iz+1] - patches[pi,ix,iy,iz-1])/(2*dzc)
                    gm = np.sqrt(gu*gu + gv*gv + gw*gw)
                    if gm < 1e-10:
                        continue
                    gu /= gm; gv /= gm; gw /= gm

                    xp = (ix-(wx-1)*0.5)*dxc/a - xm/a
                    yp = (iy-(wy-1)*0.5)*dyc/b - ym/b
                    zp = (iz-(wz-1)*0.5)*dzc/c - zm/c
                    dd = np.sqrt(xp*xp + yp*yp + zp*zp)
                    if dd < 1e-10:
                        continue
                    q = gm*gm / dd

                    A00 += q*(1-gu*gu); A01 += q*(-gu*gv); A02 += q*(-gu*gw)
                    A11 += q*(1-gv*gv); A12 += q*(-gv*gw); A22 += q*(1-gw*gw)
                    dot = gu*xp + gv*yp + gw*zp
                    B0 += q*(xp - gu*dot)
                    B1 += q*(yp - gv*dot)
                    B2 += q*(zp - gw*dot)

        # --- solve 3×3 symmetric system via Cramer ---
        det = (A00*(A11*A22 - A12*A12)
             - A01*(A01*A22 - A02*A12)
             + A02*(A01*A12 - A02*A11))
        if abs(det) < 1e-30:
            continue
        inv = 1.0 / det
        dx[pi] = ((A11*A22-A12*A12)*B0 + (A02*A12-A01*A22)*B1 + (A01*A12-A02*A11)*B2)*inv*a
        dy[pi] = ((A02*A12-A01*A22)*B0 + (A00*A22-A02*A02)*B1 + (A01*A02-A00*A12)*B2)*inv*b
        dz[pi] = ((A01*A12-A02*A11)*B0 + (A01*A02-A00*A12)*B1 + (A00*A11-A01*A01)*B2)*inv*c

    return dx, dy, dz


# ─────────────────────────────────────────────────────────────
#  Main detector class
# ─────────────────────────────────────────────────────────────

class ParticleDetector:
    """Detect and localise particles in 2-D or 3-D images.

    Methods
    -------
    detect(img)
        Full pipeline: threshold → filter → detect → sub-pixel.
        Returns ``(N, ndim)`` coordinate array.

    Examples
    --------
    >>> cfg = DetectionConfig(threshold=0.3, bead_radius=4)
    >>> det = ParticleDetector(cfg)
    >>> coords = det.detect(image_3d)       # shape (N, 3)
    >>> coords = det.detect(image_2d)       # shape (N, 2)

    StarDist example:
    >>> from serialtrack.config import DetectionMethod, StarDistConfig
    >>> cfg = DetectionConfig(
    ...     method=DetectionMethod.STARDIST,
    ...     stardist=StarDistConfig(
    ...         model_name='2D_versatile_fluo',
    ...         prob_thresh=0.5,
    ...         nms_thresh=0.4,
    ...     ),
    ... )
    >>> det = ParticleDetector(cfg)
    >>> coords = det.detect(image_2d)       # shape (N, 2)
    """

    def __init__(self, config: DetectionConfig):
        self.cfg = config
        self._stardist_model = None  # lazy-loaded

    # ── public API ──────────────────────────────────────────

    def detect(
        self,
        img: np.ndarray,
        roi_slices: Optional[Tuple[slice, ...]] = None,
    ) -> np.ndarray:
        """Run full detection pipeline. Returns (N, ndim) coords."""
        ndim = img.ndim

        # ROI crop
        offset = np.zeros(ndim, dtype=np.float64)
        if roi_slices is not None:
            offset = np.array([s.start or 0 for s in roi_slices], dtype=np.float64)
            img = img[roi_slices].copy()

        # StarDist handles its own preprocessing, so dispatch early
        if self.cfg.method == DetectionMethod.STARDIST:
            coords = self._detect_stardist(img)
            if coords.size:
                coords += offset
            return coords

        # Deconvolution
        if self.cfg.psf is not None:
            from skimage.restoration import richardson_lucy
            img = richardson_lucy(
                img.astype(np.float64), self.cfg.psf,
                num_iter=self.cfg.deconv_iters, clip=False,
            )

        # Invert for dark particles
        if self.cfg.color == "black":
            img = img.max() - img

        # Normalise to [0, 1]
        img = img.astype(np.float64)
        vmax = img.max()
        if vmax > 0:
            img_n = img / vmax
        else:
            return np.empty((0, ndim))

        # Dispatch
        if self.cfg.method == DetectionMethod.TRACTRAC:
            coords = self._detect_tractrac(img_n)
        else:
            coords = self._detect_tpt(img_n, img)

        # Offset back to full-image coords & clip
        if coords.size:
            coords += offset
        return coords

    # ── StarDist method ────────────────────────────────────

    def _detect_stardist(self, img: np.ndarray) -> np.ndarray:
        """StarDist deep-learning instance segmentation.

        Uses pretrained or custom StarDist models to detect fluorescent
        beads via star-convex polygon/polyhedra prediction. Returns
        intensity-weighted centroids for each detected instance.

        The method automatically selects StarDist2D or StarDist3D based
        on input dimensionality, and picks an appropriate pretrained
        model if the configured model doesn't match the dimensionality.

        Parameters
        ----------
        img : np.ndarray
            Input image, 2D (H, W) or 3D (Z, H, W) / (X, Y, Z).

        Returns
        -------
        coords : np.ndarray, shape (N, ndim)
            Detected particle centroids in pixel coordinates.
        """
        if not stardist_available():
            raise ImportError(
                "StarDist is not installed. Install with:\n"
                "  pip install stardist tensorflow\n"
                "Or select a different detection method."
            )

        from stardist.models import StarDist2D, StarDist3D
        from csbdeep.utils import normalize

        ndim = img.ndim
        sd_cfg = self.cfg.stardist

        # ── Load or reuse model ─────────────────────────────
        model = self._get_stardist_model(ndim, sd_cfg)

        # ── Prepare input ───────────────────────────────────
        img_f = img.astype(np.float32)

        # Invert for dark particles (StarDist expects bright objects)
        if self.cfg.color == "black":
            img_f = img_f.max() - img_f

        # Normalize input intensity
        if sd_cfg.normalize_input:
            img_f = normalize(img_f, sd_cfg.norm_pmin, sd_cfg.norm_pmax)

        # ── Build prediction kwargs ─────────────────────────
        predict_kwargs = {}

        if sd_cfg.prob_thresh is not None:
            predict_kwargs['prob_thresh'] = sd_cfg.prob_thresh
        if sd_cfg.nms_thresh is not None:
            predict_kwargs['nms_thresh'] = sd_cfg.nms_thresh
        if sd_cfg.n_tiles is not None:
            predict_kwargs['n_tiles'] = sd_cfg.n_tiles

        # 3D anisotropic scaling
        if ndim == 3 and sd_cfg.scale is not None:
            predict_kwargs['scale'] = sd_cfg.scale

        # ── Run prediction ──────────────────────────────────
        labels, details = model.predict_instances(img_f, **predict_kwargs)

        n_detected = labels.max()
        if n_detected == 0:
            logger.info("StarDist: no objects detected")
            return np.empty((0, ndim))

        logger.info("StarDist: detected %d instances", n_detected)

        # ── Extract centroids ───────────────────────────────
        # Use 'points' from details if available (these are the
        # star-convex polygon centers, which are very accurate).
        # Otherwise fall back to intensity-weighted center-of-mass.
        if 'points' in details and len(details['points']) > 0:
            coords = np.array(details['points'], dtype=np.float64)
        else:
            # Fallback: intensity-weighted center-of-mass per label
            coords = self._centroids_from_labels(labels, img_f)

        # ── Apply size filter ───────────────────────────────
        # StarDist labels give us instance areas/volumes for filtering
        if self.cfg.min_size > 1 or self.cfg.max_size < np.inf:
            coords = self._filter_by_label_size(
                labels, coords, n_detected,
                self.cfg.min_size, self.cfg.max_size,
            )

        return coords

    # ── SSL / download helpers ─────────────────────────────

    @staticmethod
    def _load_pretrained_ssl(ModelClass, model_name: str):
        """Load a pretrained StarDist model with SSL error fallback.

        Tries the normal ``from_pretrained()`` first. If it fails with
        an SSL certificate error (common on macOS, corporate networks,
        and conda environments), retries with an unverified SSL context
        as a temporary monkey-patch on ``urllib.request.urlopen``.

        Parameters
        ----------
        ModelClass : StarDist2D or StarDist3D
        model_name : str
            Pretrained model identifier (e.g. '2D_versatile_fluo').

        Returns
        -------
        model : StarDist model instance

        Raises
        ------
        RuntimeError
            If the model cannot be loaded even with the SSL fallback.
        """
        import ssl
        import urllib.request

        # ── Attempt 1: normal download ──────────────────────
        try:
            return ModelClass.from_pretrained(model_name)
        except Exception as e:
            err_str = str(e).lower()
            is_ssl = (
                'ssl' in err_str
                or 'certificate' in err_str
                or 'certificate_verify_failed' in err_str
                or 'urlopen error' in err_str
            )
            if not is_ssl:
                raise  # not an SSL problem — propagate as-is

        logger.warning(
            "SSL certificate error downloading StarDist model '%s'. "
            "Retrying with unverified SSL context...",
            model_name,
        )

        # ── Attempt 2: temporarily bypass SSL verification ──
        # Save the original urlopen so we can restore it
        _orig_urlopen = urllib.request.urlopen

        def _unverified_urlopen(url, *args, **kwargs):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return _orig_urlopen(url, *args, context=ctx, **kwargs)

        try:
            urllib.request.urlopen = _unverified_urlopen
            model = ModelClass.from_pretrained(model_name)
            logger.info(
                "Successfully loaded '%s' with SSL verification disabled. "
                "Consider fixing your SSL certificates:\n"
                "  • macOS: run 'Install Certificates.command' from your "
                "Python installation\n"
                "  • conda: conda install -c conda-forge certifi\n"
                "  • pip:   pip install --upgrade certifi",
                model_name,
            )
            return model
        except Exception as e2:
            raise RuntimeError(
                f"Failed to download StarDist model '{model_name}' even "
                f"with SSL verification disabled.\n\n"
                f"Original error: {e}\n"
                f"Retry error: {e2}\n\n"
                f"You can manually download the model:\n"
                f"  1. Download from: https://github.com/stardist/"
                f"stardist-models/releases/\n"
                f"  2. Extract to a local folder\n"
                f"  3. Set 'Model Base Dir' in StarDist parameters to "
                f"the parent folder"
            ) from e2
        finally:
            # Always restore the original urlopen
            urllib.request.urlopen = _orig_urlopen

    @staticmethod
    def download_pretrained_model(
        model_name: str = "2D_versatile_fluo",
        target_dir: Optional[str] = None,
    ) -> Path:
        """Download a pretrained StarDist model to a local directory.

        Convenience function for users who need to pre-download models
        (e.g. for offline use or to work around SSL issues). The
        downloaded model can then be used by setting
        ``StarDistConfig.model_basedir`` to the target directory.

        Parameters
        ----------
        model_name : str
            One of: '2D_versatile_fluo', '2D_paper_dsb2018',
            '2D_versatile_he', '3D_demo'.
        target_dir : str, optional
            Directory to save the model. Defaults to
            ``~/.serialtrack/stardist_models/``.

        Returns
        -------
        model_path : Path
            Path to the downloaded model directory.

        Examples
        --------
        >>> from serialtrack.detection import ParticleDetector
        >>> path = ParticleDetector.download_pretrained_model('3D_demo')
        >>> print(f"Model saved to: {path}")
        >>> # Then in config:
        >>> cfg = StarDistConfig(
        ...     model_name='3D_demo',
        ...     model_basedir=str(path.parent),
        ... )
        """
        if target_dir is None:
            target_dir = str(
                Path.home() / ".serialtrack" / "stardist_models"
            )

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / model_name

        if model_path.exists():
            logger.info("Model already exists at %s", model_path)
            return model_path

        # Determine 2D vs 3D
        _3d_models = {'3D_demo'}
        if model_name in _3d_models:
            from stardist.models import StarDist3D as MC
        else:
            from stardist.models import StarDist2D as MC

        logger.info("Downloading StarDist model '%s' to %s", model_name, target)

        try:
            # from_pretrained downloads to its own cache; we load then
            # re-export the config to our target dir
            model = ParticleDetector._load_pretrained_ssl(MC, model_name)
            # Copy the model directory to our target
            import shutil
            src = Path(model.logdir)
            if src.exists():
                shutil.copytree(src, model_path, dirs_exist_ok=True)
                logger.info("Model copied to %s", model_path)
            return model_path
        except Exception as e:
            logger.error("Failed to download model: %s", e)
            raise

    # ── Model loading ──────────────────────────────────────

    def _get_stardist_model(self, ndim: int, sd_cfg):
        """Load or return cached StarDist model.

        Handles automatic model class selection (2D vs 3D),
        fallback for mismatched model/dimension combinations,
        and SSL certificate errors during model download.
        """
        from stardist.models import StarDist2D, StarDist3D

        # Check if cached model is still valid
        if self._stardist_model is not None:
            cached_ndim = getattr(self._stardist_model, '_serialtrack_ndim', None)
            cached_name = getattr(self._stardist_model, '_serialtrack_name', None)
            if cached_ndim == ndim and cached_name == sd_cfg.model_name:
                return self._stardist_model

        model_name = sd_cfg.model_name
        model_basedir = sd_cfg.model_basedir

        # Known pretrained model names
        _2d_pretrained = {
            '2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he',
        }
        _3d_pretrained = {
            '3D_demo',
        }

        # Configure GPU usage early (before model loading triggers TF init)
        if not sd_cfg.use_gpu:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        if model_basedir is not None:
            # Custom / locally-downloaded model
            ModelClass = StarDist3D if ndim == 3 else StarDist2D
            logger.info(
                "Loading custom StarDist%dD model '%s' from '%s'",
                ndim, model_name, model_basedir,
            )
            model = ModelClass(None, name=model_name, basedir=model_basedir)
        else:
            # Pretrained model — with SSL fallback
            model = self._resolve_pretrained_model(
                ndim, model_name,
                _2d_pretrained, _3d_pretrained,
                StarDist2D, StarDist3D,
            )

        # Cache with metadata
        model._serialtrack_ndim = ndim
        model._serialtrack_name = sd_cfg.model_name
        self._stardist_model = model
        return model

    def _resolve_pretrained_model(
        self, ndim, model_name, _2d_pretrained, _3d_pretrained,
        StarDist2D, StarDist3D,
    ):
        """Resolve the correct pretrained model with dimension matching.

        Falls back to default models when there's a 2D/3D mismatch,
        and handles SSL errors transparently.
        """
        if ndim == 3:
            if model_name in _3d_pretrained:
                logger.info("Loading pretrained StarDist3D: %s", model_name)
                return self._load_pretrained_ssl(StarDist3D, model_name)
            elif model_name in _2d_pretrained:
                logger.warning(
                    "Model '%s' is 2D but input is 3D. "
                    "Falling back to '3D_demo'.", model_name,
                )
                return self._load_pretrained_ssl(StarDist3D, '3D_demo')
            else:
                try:
                    return self._load_pretrained_ssl(StarDist3D, model_name)
                except Exception:
                    logger.warning(
                        "Could not load '%s' as 3D model; "
                        "falling back to '3D_demo'.", model_name,
                    )
                    return self._load_pretrained_ssl(StarDist3D, '3D_demo')
        else:
            if model_name in _2d_pretrained:
                logger.info("Loading pretrained StarDist2D: %s", model_name)
                return self._load_pretrained_ssl(StarDist2D, model_name)
            elif model_name in _3d_pretrained:
                logger.warning(
                    "Model '%s' is 3D but input is 2D. "
                    "Falling back to '2D_versatile_fluo'.", model_name,
                )
                return self._load_pretrained_ssl(
                    StarDist2D, '2D_versatile_fluo',
                )
            else:
                try:
                    return self._load_pretrained_ssl(StarDist2D, model_name)
                except Exception:
                    logger.warning(
                        "Could not load '%s' as 2D model; "
                        "falling back to '2D_versatile_fluo'.",
                        model_name,
                    )
                    return self._load_pretrained_ssl(
                        StarDist2D, '2D_versatile_fluo',
                    )

    @staticmethod
    def _centroids_from_labels(
        labels: np.ndarray,
        img: np.ndarray,
    ) -> np.ndarray:
        """Compute intensity-weighted centroids from a label image.

        Parameters
        ----------
        labels : np.ndarray
            Integer label image from StarDist (0 = background).
        img : np.ndarray
            Original image for intensity weighting.

        Returns
        -------
        coords : np.ndarray, shape (N, ndim)
        """
        n_labels = labels.max()
        if n_labels == 0:
            return np.empty((0, labels.ndim))

        label_ids = range(1, n_labels + 1)
        centroids = ndimage.center_of_mass(img, labels, label_ids)
        return np.array(centroids, dtype=np.float64)

    @staticmethod
    def _filter_by_label_size(
        labels: np.ndarray,
        coords: np.ndarray,
        n_labels: int,
        min_size: int,
        max_size: int,
    ) -> np.ndarray:
        """Filter detections by instance size (area in 2D, volume in 3D).

        Parameters
        ----------
        labels : np.ndarray
            Label image from StarDist.
        coords : np.ndarray, shape (N, ndim)
            Centroid coordinates (one per label).
        n_labels : int
            Number of detected instances.
        min_size, max_size : int
            Size bounds in pixels^d.

        Returns
        -------
        filtered : np.ndarray, shape (M, ndim) where M <= N
        """
        if len(coords) == 0:
            return coords

        # Compute sizes for each label
        label_ids = np.arange(1, n_labels + 1)
        sizes = ndimage.sum_labels(
            np.ones_like(labels), labels, label_ids,
        )

        # Build mask — coords and sizes are aligned (label 1 → index 0)
        n_coords = min(len(coords), len(sizes))
        mask = np.ones(n_coords, dtype=bool)
        for i in range(n_coords):
            s = sizes[i]
            if s < min_size or s > max_size:
                mask[i] = False

        filtered = coords[:n_coords][mask]
        n_removed = n_coords - filtered.shape[0]
        if n_removed > 0:
            logger.info(
                "StarDist size filter: kept %d / %d (removed %d outside [%d, %d])",
                len(filtered), n_coords, n_removed, min_size, max_size,
            )
        return filtered

    # ── TracTrac method ─────────────────────────────────────

    def _detect_tractrac(self, img_n: np.ndarray) -> np.ndarray:
        """LoG → local-maximum → sub-pixel polynomial fit."""
        ndim = img_n.ndim
        bw = self._size_filtered_mask(img_n)
        img_m = img_n * bw  # masked image

        if self.cfg.bead_radius > 0:
            return self._log_detect(img_m, img_n, ndim)
        else:
            return self._centroid_detect(img_n, ndim)

    def _log_detect(self, img_m, img_n, ndim):
        sigma = self.cfg.bead_radius

        # Laplacian of Gaussian
        log_img = -ndimage.gaussian_laplace(img_m, sigma=sigma)

        # Local-maximum filter
        fp_size = int(2 * sigma * 2) + 1
        fp = np.ones((fp_size,) * ndim)
        rng = np.random.default_rng(42)
        noise = rng.random(log_img.shape) * 1e-5
        dilated = ndimage.maximum_filter(log_img + noise, footprint=fp)
        peaks = ((log_img + noise) == dilated) & (img_n > self.cfg.threshold)

        coords_int = np.asarray(np.nonzero(peaks), dtype=np.int64).T  # (N, ndim)
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Trim border
        nb_ = max(int((sigma + 2) / 2), 1)
        mask = np.ones(len(coords_int), dtype=np.bool_)
        for d in range(ndim):
            mask &= (coords_int[:, d] >= nb_) & (coords_int[:, d] < img_n.shape[d] - nb_)
        coords_int = coords_int[mask]
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Sub-pixel via parabola on log(LoG)
        log_safe = np.log(np.clip(log_img - log_img.min() + 1e-8, 1e-12, None))

        if ndim == 2:
            dx, dy = _subpixel_poly_2d(log_safe, coords_int[:, 0], coords_int[:, 1])
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
        else:
            dx, dy, dz = _subpixel_poly_3d(
                log_safe, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
            )
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5) & (np.abs(dz) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
            out[:, 2] += dz[valid]
        return out

    # ── TPT method ──────────────────────────────────────────

    def _detect_tpt(self, img_n: np.ndarray, img_raw: np.ndarray) -> np.ndarray:
        """Blob centroid → radial-symmetry sub-voxel refinement."""
        ndim = img_n.ndim
        coords = self._centroid_detect(img_n, ndim)
        if len(coords) == 0 or ndim != 3:
            return coords  # radial symmetry only for 3-D

        # Extract patches for radial-symmetry
        ws = np.array(self.cfg.win_size[:3], dtype=np.int64)
        half = ws // 2
        ci = np.round(coords).astype(np.int64)

        # Pad with reflected noise
        img_f = img_raw.astype(np.float64)
        img_f += self.cfg.rand_noise * np.random.default_rng(0).random(img_f.shape)
        img_p = np.pad(img_f, [(h, h) for h in half], mode="reflect")

        patches = np.empty((len(ci), ws[0], ws[1], ws[2]), dtype=np.float64)
        for i, c in enumerate(ci):
            cp = c + half  # padded coords
            patches[i] = img_p[
                cp[0]-half[0]:cp[0]+half[0]+1,
                cp[1]-half[1]:cp[1]+half[1]+1,
                cp[2]-half[2]:cp[2]+half[2]+1,
            ]

        dccd = np.array(self.cfg.dccd[:3], dtype=np.float64)
        abc = np.array(self.cfg.abc[:3], dtype=np.float64)

        dx, dy, dz = _radial_symmetry_3d(patches, half, dccd, abc)

        # Apply only well-behaved shifts
        ok = (np.abs(dx) < half[0]) & (np.abs(dy) < half[1]) & (np.abs(dz) < half[2])
        out = coords.copy()
        out[ok, 0] += dx[ok]
        out[ok, 1] += dy[ok]
        out[ok, 2] += dz[ok]
        return out

    # ── shared helpers ──────────────────────────────────────

    def _size_filtered_mask(self, img_n: np.ndarray) -> np.ndarray:
        """Threshold → label → keep blobs within [min_size, max_size]."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return bw
        sizes = ndimage.sum_labels(bw, labeled, range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        for i, s in enumerate(sizes, 1):
            if self.cfg.min_size <= s <= self.cfg.max_size:
                keep[i] = True
        return keep[labeled]

    def _centroid_detect(self, img_n: np.ndarray, ndim: int) -> np.ndarray:
        """Connected-component centroids, filtered by size."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return np.empty((0, ndim))
        sizes = ndimage.sum_labels(bw, labeled, range(1, n + 1))
        centroids = ndimage.center_of_mass(img_n, labeled, range(1, n + 1))
        out = np.array([
            c for c, s in zip(centroids, sizes)
            if self.cfg.min_size <= s <= self.cfg.max_size
        ])
        return out if out.size else np.empty((0, ndim))

    @staticmethod
    def clip_to_bounds(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Remove coords outside [0, shape) for each dimension."""
        if coords.size == 0:
            return coords
        mask = np.ones(len(coords), dtype=bool)
        for d in range(coords.shape[1]):
            mask &= (coords[:, d] >= 0) & (coords[:, d] < shape[d])
        return coords[mask]
