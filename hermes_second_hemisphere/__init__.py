"""Hermes Second Hemisphere — Asymmetric Dual-Core Context Compression Engine."""

from .aux_client import AuxCompressionClient
from .core import ContinuousCompactor
from .hardwall import HardWallGuard
from .models import AuxServerProps, CompactionResult, CompactionTier
from .utils import estimate_tokens, extract_hard_facts

__version__ = "1.0.0"

__all__ = [
    "ContinuousCompactor",
    "AuxCompressionClient",
    "HardWallGuard",
    "CompactionTier",
    "CompactionResult",
    "AuxServerProps",
    "estimate_tokens",
    "extract_hard_facts",
]
