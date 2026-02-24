"""
Plugin Registry — extensible system for image enhancement, mask generation,
post-processing, and visualization methods.

Each plugin category has a base class. New methods simply subclass and register.

Usage:
    @EnhancementPlugin.register
    class CLAHEEnhancement(EnhancementPlugin):
        name = "CLAHE"
        description = "Contrast Limited Adaptive Histogram Equalization"
        ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Base plugin & parameter spec
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParamSpec:
    """Describes a single parameter for a plugin."""
    name: str
    label: str
    param_type: str = "float"   # float, int, bool, choice, str
    default: Any = 0.0
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    choices: List[str] = field(default_factory=list)
    tooltip: str = ""


class PluginBase(ABC):
    """Base class for all SerialTrack GUI plugins."""
    name: str = "Unnamed"
    description: str = ""
    category: str = "general"
    icon: str = ""  # Optional icon name

    _registry: Dict[str, Type['PluginBase']] = {}

    @classmethod
    def register(cls, plugin_cls):
        """Decorator to register a plugin subclass."""
        key = f"{plugin_cls.category}:{plugin_cls.name}"
        PluginBase._registry[key] = plugin_cls
        return plugin_cls

    @classmethod
    def get_plugins(cls, category: str) -> List[Type['PluginBase']]:
        return [v for k, v in PluginBase._registry.items()
                if k.startswith(f"{category}:")]

    @classmethod
    def get_plugin(cls, category: str, name: str) -> Optional[Type['PluginBase']]:
        return PluginBase._registry.get(f"{category}:{name}")

    @abstractmethod
    def get_params(self) -> List[ParamSpec]:
        """Return parameter specifications."""
        ...

    @abstractmethod
    def execute(self, data: Any, params: Dict[str, Any],
                progress_cb=None) -> Any:
        """Execute the plugin operation."""
        ...


# ═══════════════════════════════════════════════════════════════
#  Category: Image Enhancement
# ═══════════════════════════════════════════════════════════════

class EnhancementPlugin(PluginBase):
    category = "enhancement"

    @abstractmethod
    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        ...


# ═══════════════════════════════════════════════════════════════
#  Category: Mask Generation
# ═══════════════════════════════════════════════════════════════

class MaskPlugin(PluginBase):
    category = "mask"

    @abstractmethod
    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        """Return a boolean mask array."""
        ...


# ═══════════════════════════════════════════════════════════════
#  Category: Post-Processing
# ═══════════════════════════════════════════════════════════════

class PostProcessPlugin(PluginBase):
    category = "postprocess"

    @abstractmethod
    def execute(self, data: Dict[str, Any], params: Dict[str, Any],
                progress_cb=None) -> Dict[str, Any]:
        """Process tracking results. Returns computed fields."""
        ...


# ═══════════════════════════════════════════════════════════════
#  Category: Visualization
# ═══════════════════════════════════════════════════════════════

class VisualizationPlugin(PluginBase):
    category = "visualization"

    @abstractmethod
    def execute(self, data: Dict[str, Any], params: Dict[str, Any],
                figure=None, progress_cb=None) -> Any:
        """Render visualization. May draw on a provided matplotlib figure."""
        ...
