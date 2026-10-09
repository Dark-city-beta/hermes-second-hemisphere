"""Auxiliary LLM client with truthful live /props probing and size-error classification."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .models import AuxServerProps
from .utils import estimate_tokens

logger = logging.getLogger(__name__)


class AuxCompressionClient:
    """Manages asynchronous interaction with the subdominant (auxiliary) LLM."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        timeout: float = 180.0,
        max_summary_tokens: int = 350,
        input_cap_tokens: int = 3000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_summary_tokens = max_summary_tokens
        self.input_cap_tokens = input_cap_tokens
        self._cached_props: Optional[AuxServerProps] = None
        self._props_last_checked: float = 0.0

    def get_server_props(self, force_refresh: bool = False) -> AuxServerProps:
        """Fetch truthful live generation settings from llama.cpp /props."""
        now = time.time()
        if not force_refresh and self._cached_props and (now - self._props_last_checked < 60.0):
            return self._cached_props

        url = f"{self.base_url}/props"
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-Second-Hemisphere/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                gen_settings = data.get("default_generation_settings", {})
                props = AuxServerProps(
                    n_ctx=int(gen_settings.get("n_ctx", 16384)),
                    total_slots=int(data.get("total_slots", 1)),
                    model_alias=str(data.get("model_alias", self.model)),
                    is_live=True,
                )
                self._cached_props = props
                self._props_last_checked = now
                return props
        except Exception as exc:
            logger.debug("Failed to fetch /props from %s: %s", url, exc)
            props = AuxServerProps(
                n_ctx=16384,
                total_slots=1,
                model_alias=self.model,
                is_live=False,
                error=str(exc),
            )
            self._cached_props = props
            self._props_last_checked = now
            return props

    def generate_summary(
        self,
        text_or_messages: str | List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Request compact factual summary from the auxiliary LLM."""
        url = f"{self.base_url}/v1/chat/completions"
        if not system_prompt:
            system_prompt = (
                "You are an expert factual summarizer for an AI agent. "
                "Condense the following conversation history into key decisions, "
                "file modifications, user goals, and active tasks. Be concise and precise."
            )

        if isinstance(text_or_messages, str):
            prompt_content = text_or_messages
        else:
            # Format message array into linear string bounded by input_cap_tokens
            lines: List[str] = []
            for m in text_or_messages:
                role = m.get("role", "unknown")
                c = m.get("content") or ""
                lines.append(f"[{role.upper()}]: {c}")
            prompt_content = "\n".join(lines)

        # Enforce input degradation ladder
        if estimate_tokens(prompt_content) > self.input_cap_tokens:
            prompt_content = prompt_content[: int(self.input_cap_tokens * 3.2)]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize:\n\n{prompt_content}"},
            ],
            "max_tokens": self.max_summary_tokens,
            "temperature": 0.2,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    summary = choices[0].get("message", {}).get("content", "").strip()
                    return True, summary
                return False, "Empty response from auxiliary model"
        except urllib.error.HTTPError as http_err:
            body = ""
            try:
                body = http_err.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            # Size-error classification: 400 context overflow must not mark aux as down
            if http_err.code == 400 and ("exceed" in body.lower() or "too long" in body.lower()):
                logger.warning("Aux model rejected prompt size (input overflow): %s", body)
                return False, f"SIZE_ERROR: {body}"
            logger.error("HTTP error during auxiliary summary: %d %s", http_err.code, body)
            return False, f"HTTP_{http_err.code}: {body}"
        except Exception as exc:
            logger.error("Failed to connect to auxiliary compression model: %s", exc)
            return False, str(exc)
