"""Utility functions for token estimation and hard fact extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


def estimate_tokens(item: Any, chars_per_token: float = 3.2) -> int:
    """Fast, robust token estimation for messages or strings.
    
    Uses 3.2 chars/token for Russian/multilingual mixed code/text.
    """
    if isinstance(item, str):
        return max(1, int(len(item) / chars_per_token))
    elif isinstance(item, dict):
        content = item.get("content") or ""
        total_chars = len(str(content))
        if "tool_calls" in item:
            total_chars += len(str(item["tool_calls"]))
        return max(1, int(total_chars / chars_per_token))
    elif isinstance(item, list):
        return sum(estimate_tokens(m, chars_per_token) for m in item)
    return 0


def extract_hard_facts(messages: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """Deterministic factual extractor.
    
    Scans past conversation turns to retain critical file paths, shell commands,
    git branches, and error traces before dropping raw message history.
    """
    found_paths: Set[str] = set()
    found_commands: Set[str] = set()
    found_errors: Set[str] = set()

    # Regex patterns for high-value technical entities
    path_regex = re.compile(r'(?:/[\w\.\-]+)+|(?:[a-zA-Z]:\\[\w\.\-\\]+)')
    cmd_regex = re.compile(r'(?:`{1,3}|\$ )([a-zA-Z0-9_\-\./]+(?:\s+[\w\-\./]+){1,5})')
    err_regex = re.compile(r'(?:Error|Exception|HTTP \d{3}|FAILED|Fatal):?\s*([^\n\r`]{1,80})', re.IGNORECASE)

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        
        # Collect paths
        for p in path_regex.findall(content):
            if len(p) > 4 and not p.endswith("."):
                found_paths.add(p.strip())

        # Collect commands
        for cmd in cmd_regex.findall(content):
            cmd_clean = cmd.strip()
            if 3 < len(cmd_clean) < 60:
                found_commands.add(cmd_clean)

        # Collect errors
        for err in err_regex.findall(content):
            err_clean = err.strip()
            if 3 < len(err_clean) < 80:
                found_errors.add(err_clean)

    parts: List[str] = ["[ARCHIVED_FACTS_RESTORED]"]
    if found_paths:
        parts.append("Paths: " + ", ".join(sorted(list(found_paths))[:10]))
    if found_commands:
        parts.append("Commands: " + " | ".join(sorted(list(found_commands))[:6]))
    if found_errors:
        parts.append("Errors: " + " | ".join(sorted(list(found_errors))[:4]))

    result = "\n".join(parts)
    return result[:max_chars] if len(result) > max_chars else result
