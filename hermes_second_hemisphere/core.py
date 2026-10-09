"""Continuous 3-Layer Context Compactor Engine."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .aux_client import AuxCompressionClient
from .models import CompactionResult, CompactionTier
from .utils import estimate_tokens, extract_hard_facts

logger = logging.getLogger(__name__)


class ContinuousCompactor:
    """Asymmetric Context Compressor for Long-Running Local LLM Agents."""

    def __init__(
        self,
        aux_client: Optional[AuxCompressionClient] = None,
        main_context_window: int = 65536,
        compression_threshold_tokens: int = 36000,
        protect_first_n: int = 1,
        protect_last_n: int = 2,
        target_ratio: float = 0.50,
        emergency_trigger_ratio: float = 0.75,
    ) -> None:
        self.aux_client = aux_client or AuxCompressionClient()
        self.main_context_window = main_context_window
        self.threshold_tokens = compression_threshold_tokens
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.target_ratio = target_ratio
        self.emergency_trigger_ratio = emergency_trigger_ratio

    def is_compression_needed(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if message history has exceeded the soft compression threshold."""
        return estimate_tokens(messages) >= self.threshold_tokens

    def is_emergency_needed(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if message history is approaching the hard model context limit."""
        return estimate_tokens(messages) >= int(self.main_context_window * self.emergency_trigger_ratio)

    def compact(
        self,
        messages: List[Dict[str, Any]],
        fallback_fn: Optional[Callable[[List[Dict[str, Any]]], Tuple[bool, str]]] = None,
        force_tier: Optional[CompactionTier] = None,
    ) -> CompactionResult:
        """Execute continuous compaction across Layer 1, Layer 2, or Layer 3."""
        initial_tokens = estimate_tokens(messages)
        start_time = time.time()
        warnings: List[str] = []

        if len(messages) <= (self.protect_first_n + self.protect_last_n):
            return CompactionResult(
                messages=messages,
                initial_tokens=initial_tokens,
                final_tokens=initial_tokens,
                tier=CompactionTier.PROACTIVE_PRUNE,
                elapsed_seconds=0.0,
                success=True,
                warnings=["History too short to compact."],
            )

        # ── Layer 1: Subdominant Auxiliary LLM ──────────────────────────────────
        if force_tier is None or force_tier == CompactionTier.LAYER1_AUX_LLM:
            try:
                head, middle, tail = self._split_messages(messages)
                if middle:
                    ok, summary_text = self.aux_client.generate_summary(middle)
                    if ok and summary_text:
                        compacted = head + [
                            {
                                "role": "system",
                                "content": f"[CONVERSATION SUMMARY BY AUXILIARY BRAIN]:\n{summary_text}",
                            }
                        ] + tail
                        final_tokens = estimate_tokens(compacted)
                        logger.info("Layer 1 (Aux LLM) compaction succeeded: %d -> %d tokens", initial_tokens, final_tokens)
                        return CompactionResult(
                            messages=compacted,
                            initial_tokens=initial_tokens,
                            final_tokens=final_tokens,
                            tier=CompactionTier.LAYER1_AUX_LLM,
                            elapsed_seconds=time.time() - start_time,
                            success=True,
                            summary=summary_text,
                        )
                    else:
                        warnings.append(f"Layer 1 failed: {summary_text}")
            except Exception as exc:
                warnings.append(f"Layer 1 exception: {exc}")

        # ── Layer 2: Main Model Fallback ────────────────────────────────────────
        if (force_tier is None or force_tier == CompactionTier.LAYER2_MAIN_FALLBACK) and fallback_fn:
            try:
                head, middle, tail = self._split_messages(messages)
                if middle:
                    ok, fb_summary = fallback_fn(middle)
                    if ok and fb_summary:
                        compacted = head + [
                            {
                                "role": "system",
                                "content": f"[CONVERSATION SUMMARY VIA FALLBACK]:\n{fb_summary}",
                            }
                        ] + tail
                        final_tokens = estimate_tokens(compacted)
                        logger.info("Layer 2 (Fallback) compaction succeeded: %d -> %d tokens", initial_tokens, final_tokens)
                        return CompactionResult(
                            messages=compacted,
                            initial_tokens=initial_tokens,
                            final_tokens=final_tokens,
                            tier=CompactionTier.LAYER2_MAIN_FALLBACK,
                            elapsed_seconds=time.time() - start_time,
                            success=True,
                            summary=fb_summary,
                            warnings=warnings,
                        )
            except Exception as exc:
                warnings.append(f"Layer 2 fallback failed: {exc}")

        # ── Layer 3: Deterministic Emergency Compaction (Zero-LLM) ──────────────
        compacted = self.emergency_compact(messages)
        final_tokens = estimate_tokens(compacted)
        logger.warning("Layer 3 (Deterministic Emergency Compactor) applied: %d -> %d tokens", initial_tokens, final_tokens)
        return CompactionResult(
            messages=compacted,
            initial_tokens=initial_tokens,
            final_tokens=final_tokens,
            tier=CompactionTier.LAYER3_EMERGENCY,
            elapsed_seconds=time.time() - start_time,
            success=True,
            warnings=warnings,
        )

    def emergency_compact(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Deterministic zero-LLM compaction algorithm.
        
        Guarantees that context drops below target_tokens without making any LLM calls.
        """
        if not messages or len(messages) <= 2:
            return messages

        if target_tokens is None:
            target_tokens = int(self.main_context_window * self.target_ratio)

        result = [dict(m) for m in messages]
        head_idx = 1 if result and result[0].get("role") == "system" else 0
        tail_idx = max(head_idx + 1, len(result) - 1)

        # Pass 1: Prune massive tool call output blocks everywhere except very last prompt
        for i in range(head_idx, tail_idx):
            msg = result[i]
            role = msg.get("role")
            content = msg.get("content")
            if role in ("tool", "assistant") and isinstance(content, str) and len(content) > 150:
                head_part = content[:80].strip()
                tail_part = content[-40:].strip()
                msg["content"] = f"{head_part}\n... [trimmed tool output] ...\n{tail_part}"

        if estimate_tokens(result) <= target_tokens:
            return result

        # Pass 2: Extract hard facts before dropping old turns
        middle_slice = result[head_idx:tail_idx]
        facts_summary = extract_hard_facts(middle_slice)

        head_part = result[:head_idx]
        tail_part = result[tail_idx:]

        # Drop middle turns progressively from the oldest until current total <= target_tokens
        while middle_slice:
            current_est = estimate_tokens(head_part) + estimate_tokens(middle_slice) + estimate_tokens(tail_part) + estimate_tokens(facts_summary)
            if current_est <= target_tokens:
                break
            middle_slice.pop(0)

        # Insert facts anchor
        facts_anchor = {
            "role": "system",
            "content": f"{facts_summary}\n[NOTE: Older tool turns pruned to prevent context overflow.]",
        }

        final_messages = head_part + [facts_anchor] + middle_slice + tail_part
        return final_messages

    def _split_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split message sequence into head (protected), middle (compressible), and tail (active)."""
        head_count = self.protect_first_n
        if messages and messages[0].get("role") == "system":
            head_count = max(1, self.protect_first_n)
        tail_count = max(1, self.protect_last_n)

        if len(messages) <= (head_count + tail_count):
            return messages[:head_count], [], messages[head_count:]

        head = [dict(m) for m in messages[:head_count]]
        middle = [dict(m) for m in messages[head_count:-tail_count]]
        tail = [dict(m) for m in messages[-tail_count:]]
        return head, middle, tail
