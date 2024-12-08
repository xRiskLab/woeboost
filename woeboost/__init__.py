# -*- coding: utf-8 -*-
"""
__init__.py.

This module is the entry point for the package. It imports the main classes and functions
from the package and sets the version number.
"""

from .classifier import WoeBoostClassifier, WoeBoostConfig
from .explainer import EvidenceAnalyzer, PDPAnalyzer, WoeInferenceMaker
from .learner import WoeLearner

__all__ = [
    "WoeBoostClassifier",
    "WoeBoostConfig",
    "EvidenceAnalyzer",
    "PDPAnalyzer",
    "WoeInferenceMaker",
    "WoeLearner"
]

# Add dynamic version retrieval
try:
    from importlib.metadata import version
    __version__ = version("woeboost")
except ImportError:
    __version__ = "unknown"
