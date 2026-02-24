"""Built-in image enhancement plugins."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from core.plugin_registry import EnhancementPlugin, ParamSpec, PluginBase


@PluginBase.register
class NormalizeEnhancement(EnhancementPlugin):
    name = "Normalize"
    description = "Normalize intensity to [0, 1] range"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("clip_low", "Clip Low %", "float", 0.0, 0.0, 49.0, 0.5,
                      tooltip="Percentile to clip at the low end"),
            ParamSpec("clip_high", "Clip High %", "float", 100.0, 51.0, 100.0, 0.5,
                      tooltip="Percentile to clip at the high end"),
        ]

    def execute(self, volume, params, progress_cb=None):
        lo = np.percentile(volume, params.get("clip_low", 0))
        hi = np.percentile(volume, params.get("clip_high", 100))
        out = np.clip(volume, lo, hi).astype(np.float32)
        if hi > lo:
            out = (out - lo) / (hi - lo)
        if progress_cb:
            progress_cb(100)
        return out


@PluginBase.register
class CLAHEEnhancement(EnhancementPlugin):
    name = "CLAHE"
    description = "Contrast Limited Adaptive Histogram Equalization"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("kernel_size", "Kernel Size", "int", 64, 8, 512, 8,
                      tooltip="Size of the local region for histogram equalization"),
            ParamSpec("clip_limit", "Clip Limit", "float", 0.01, 0.001, 0.1, 0.005,
                      tooltip="Clipping limit for contrast limiting (lower = less enhancement)"),
        ]

    def execute(self, volume, params, progress_cb=None):
        from skimage import exposure
        ks = params.get("kernel_size", 64)
        cl = params.get("clip_limit", 0.01)
        if volume.ndim == 3:
            out = np.empty_like(volume, dtype=np.float32)
            for z in range(volume.shape[0]):
                sl = volume[z].astype(np.float32)
                if sl.max() > sl.min():
                    sl = (sl - sl.min()) / (sl.max() - sl.min())
                out[z] = exposure.equalize_adapthist(
                    sl, kernel_size=min(ks, min(sl.shape)), clip_limit=cl
                ).astype(np.float32)
                if progress_cb:
                    progress_cb(int(100 * (z + 1) / volume.shape[0]))
            return out
        else:
            sl = volume.astype(np.float32)
            if sl.max() > sl.min():
                sl = (sl - sl.min()) / (sl.max() - sl.min())
            result = exposure.equalize_adapthist(
                sl, kernel_size=min(ks, min(sl.shape)), clip_limit=cl
            ).astype(np.float32)
            if progress_cb:
                progress_cb(100)
            return result


@PluginBase.register
class GaussianBlurEnhancement(EnhancementPlugin):
    name = "Gaussian Blur"
    description = "Gaussian smoothing filter to reduce noise"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma", "Sigma", "float", 1.0, 0.1, 20.0, 0.1,
                      tooltip="Standard deviation of Gaussian kernel (larger = more blur)"),
        ]

    def execute(self, volume, params, progress_cb=None):
        from scipy.ndimage import gaussian_filter
        sigma = params.get("sigma", 1.0)
        out = gaussian_filter(volume.astype(np.float32), sigma=sigma)
        if progress_cb:
            progress_cb(100)
        return out


@PluginBase.register
class MedianFilterEnhancement(EnhancementPlugin):
    name = "Median Filter"
    description = "Median filter for salt-and-pepper noise removal"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("size", "Kernel Size", "int", 3, 3, 15, 2,
                      tooltip="Size of median filter window (must be odd)"),
        ]

    def execute(self, volume, params, progress_cb=None):
        from scipy.ndimage import median_filter
        size = params.get("size", 3)
        if size % 2 == 0:
            size += 1
        out = median_filter(volume.astype(np.float32), size=size)
        if progress_cb:
            progress_cb(100)
        return out


@PluginBase.register
class BackgroundSubtractEnhancement(EnhancementPlugin):
    name = "Background Subtract"
    description = "Subtract a smoothed background (rolling ball-like)"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("radius", "Background Radius", "float", 50.0, 5.0, 500.0, 5.0,
                      tooltip="Radius for background estimation (larger = less aggressive)"),
        ]

    def execute(self, volume, params, progress_cb=None):
        from scipy.ndimage import gaussian_filter
        radius = params.get("radius", 50.0)
        bg = gaussian_filter(volume.astype(np.float64), sigma=radius)
        out = np.clip(volume.astype(np.float64) - bg, 0, None)
        mx = out.max()
        if mx > 0:
            out /= mx
        if progress_cb:
            progress_cb(100)
        return out.astype(np.float32)


@PluginBase.register
class GammaEnhancement(EnhancementPlugin):
    name = "Gamma Correction"
    description = "Adjust image gamma (brightness curve)"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("gamma", "Gamma", "float", 1.0, 0.1, 5.0, 0.1,
                      tooltip="Gamma < 1 brightens, gamma > 1 darkens"),
        ]

    def execute(self, volume, params, progress_cb=None):
        gamma = params.get("gamma", 1.0)
        v = volume.astype(np.float32)
        mn, mx = v.min(), v.max()
        if mx > mn:
            v = (v - mn) / (mx - mn)
        out = np.power(v, gamma)
        if progress_cb:
            progress_cb(100)
        return out


@PluginBase.register
class InvertEnhancement(EnhancementPlugin):
    name = "Invert"
    description = "Invert intensity values"

    def get_params(self) -> List[ParamSpec]:
        return []

    def execute(self, volume, params, progress_cb=None):
        mx = volume.max()
        if progress_cb:
            progress_cb(100)
        return (mx - volume).astype(np.float32)
