"""
Model architectures for Nested Learning.

This module provides complete model implementations using nested layers and CMS.
"""

from .nested_mlp import NestedMLP
from .hope import HopeModel

__all__ = [
    'NestedMLP',
    'HopeModel',
]
