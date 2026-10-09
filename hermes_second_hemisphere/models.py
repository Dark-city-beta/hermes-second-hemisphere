"""Data models for Hermes Second Hemisphere context compression."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class CompactionTier(enum.Enum):
    """Execution tiers for context compaction."""
    LAYER1_AUX_LLM = "layer1_aux_llm"
    LAYER2_MAIN_FALLBACK = "layer2_main_fallback"
    LAYER3_EMERGENCY = "layer3_emergency_compact"
    PROACTIVE_PRUNE = "proactive_prune"


@dataclass
class AuxServerProps:
    """Live telemetry discovered from auxiliary server /props endpoint."""
    n_ctx: int = 16384
    total_slots: int = 1
    model_alias: str = ""
    is_live: bool = False
    error: Optional[str] = None


@dataclass
class CompactionResult:
    """Outcome of a compaction operation."""
    messages: List[Dict[str, Any]]
    initial_tokens: int
    final_tokens: int
    tier: CompactionTier
    elapsed_seconds: float
    success: bool
    summary: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def reduction_percentage(self) -> float:
        if self.initial_tokens <= 0:
            return 0.0
        return max(0.0, (1.0 - (self.final_tokens / self.initial_tokens)) * 100.0)
