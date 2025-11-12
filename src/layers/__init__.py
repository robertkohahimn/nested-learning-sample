"""
Neural network layers for Nested Learning.

This module provides layer implementations with multi-frequency parameter updates.
"""

from .nested_layer import (
    NestedLayer,
    NestedLinear,
    NestedEmbedding,
    NestedLayerNorm,
    FrequencyLevel,
    get_frequency_aware_param_groups
)
from .cms_block import CMSBlock, CMSAttentionBlock

__all__ = [
    # Base classes
    'NestedLayer',
    'FrequencyLevel',
    
    # Layer implementations
    'NestedLinear',
    'NestedEmbedding',
    'NestedLayerNorm',
    
    # Blocks
    'CMSBlock',
    'CMSAttentionBlock',
    
    # Utilities
    'get_frequency_aware_param_groups',
]
