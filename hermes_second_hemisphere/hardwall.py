"""Hard-Wall Interceptors to prevent HTTP 400 Context Overflow and Lockout loops."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .core import ContinuousCompactor
from .utils import estimate_tokens

logger = logging.getLogger(__name__)


class HardWallGuard:
    """Preflight and post-failure guard against context limit crashes.
    
    Ensures an AI agent never sends a request larger than the LLM's physical context window
    and never gets permanently blocked by cooldown or attempts-exhausted lockouts.
    """

    def __init__(
        self,
        compactor: ContinuousCompactor,
        hard_limit_tokens: int = 65536,
        danger_threshold_ratio: float = 0.75,
    ) -> None:
        self.compactor = compactor
        self.hard_limit_tokens = hard_limit_tokens
        self.danger_threshold_ratio = danger_threshold_ratio

    def inspect_and_defend_preflight(
        self,
        messages: List[Dict[str, Any]],
        is_blocked_by_cooldown: bool = False,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Pre-turn check.
        
        If context is at dangerous levels (e.g. >= 75% of window), triggers emergency compaction
        even if regular LLM compression is in cooldown or attempts are exhausted.
        """
        current_tokens = estimate_tokens(messages)
        ceiling = int(self.hard_limit_tokens * self.danger_threshold_ratio)

        if current_tokens >= ceiling:
            logger.warning(
                "HardWallGuard PREFLIGHT: context (%d tokens) >= %d (%.0f%% limit) [blocked=%s]. "
                "Triggering Layer 3 Emergency Compaction to prevent HTTP 400!",
                current_tokens, ceiling, self.danger_threshold_ratio * 100, is_blocked_by_cooldown,
            )
            compacted = self.compactor.emergency_compact(messages)
            new_tokens = estimate_tokens(compacted)
            logger.info("HardWallGuard: context trimmed from %d to %d tokens.", current_tokens, new_tokens)
            return compacted, True

        return messages, False

    def recover_from_overflow_exception(
        self,
        messages: List[Dict[str, Any]],
        error_message: str,
        attempts_exhausted: bool = False,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Post-failure interceptor.
        
        Catches HTTP 400 context overflow rejections or max-attempt lockouts and recovers history.
        """
        is_overflow = any(
            phrase in error_message.lower()
            for phrase in (
                "exceeds the available context size",
                "context window exceeded",
                "maximum context length",
                "prompt is too long",
                "400",
            )
        )

        if is_overflow or attempts_exhausted:
            logger.warning(
                "HardWallGuard RECOVERY: Intercepted fatal context error or lockout (attempts_exhausted=%s). "
                "Applying deterministic emergency reduction.",
                attempts_exhausted,
            )
            compacted = self.compactor.emergency_compact(
                messages, target_tokens=int(self.hard_limit_tokens * 0.50)
            )
            return compacted, True

        return messages, False
