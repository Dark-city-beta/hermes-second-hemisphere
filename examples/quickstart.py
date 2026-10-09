"""Quickstart demonstration of Hermes Second Hemisphere context compression."""

import os
import sys

# Ensure local package is importable directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hermes_second_hemisphere import (
    ContinuousCompactor,
    AuxCompressionClient,
    HardWallGuard,
    estimate_tokens,
)


def run_demo():
    print("=" * 60)
    print(" Hermes Second Hemisphere: Asymmetric Context Engine Demo")
    print("=" * 60)

    # 1. Initialize compactor with a 16k soft threshold and 32k hard limit for demo
    compactor = ContinuousCompactor(
        aux_client=AuxCompressionClient(base_url="http://127.0.0.1:8081"),
        main_context_window=32768,
        compression_threshold_tokens=8000,
        protect_first_n=2,
        protect_last_n=4,
    )
    guard = HardWallGuard(compactor=compactor, hard_limit_tokens=32768)

    # 2. Synthesize a long-running developer session with tools
    messages = [
        {"role": "system", "content": "You are Hermes, an autonomous engineering agent."},
        {"role": "user", "content": "Analyze the backend repository and fix bug in authentication."},
    ]

    # Add tool loops (cat, grep, sed, test outputs)
    for i in range(1, 25):
        messages.extend([
            {
                "role": "assistant",
                "content": f"Running diagnostic command: grep -rn 'auth_token' /var/log/app/trace_{i}.log",
                "tool_calls": [{"id": f"call_{i}", "function": {"name": "run_shell"}}],
            },
            {
                "role": "tool",
                "name": "run_shell",
                "content": (
                    f"2026-10-09 12:{i:02d}:00 [INFO] auth_token validated for session_{i} in /etc/security/keys.json.\n"
                    f"Command executed: sed -i 's/retry=3/retry=5/' /var/app/config.yaml\n"
                    f"Error encountered: Exception: Connection timeout on database port 5432 after {i*10}ms\n"
                ) * 8,
            },
        ])

    initial_tokens = estimate_tokens(messages)
    print(f"\nInitial message count: {len(messages)}")
    print(f"Initial estimated tokens: {initial_tokens}")

    # 3. Test Layer 3 (Deterministic Emergency Compactor) directly
    print("\n[Testing Layer 3: Deterministic Emergency Compaction]")
    compacted_l3 = compactor.emergency_compact(messages, target_tokens=4000)
    l3_tokens = estimate_tokens(compacted_l3)
    print(f"Messages after Layer 3: {len(compacted_l3)}")
    print(f"Tokens after Layer 3: {l3_tokens} ({((1 - l3_tokens/initial_tokens)*100):.1f}% reduction)")

    # Inspect retained hard facts
    for m in compacted_l3:
        if "[ARCHIVED_FACTS_RESTORED]" in m.get("content", ""):
            print("\nExtracted Hard Facts preserved in history:")
            print(m["content"])
            break

    # 4. Test Preflight Hard-Wall Guard
    print("\n[Testing HardWallGuard Preflight]")
    safe_msgs, triggered = guard.inspect_and_defend_preflight(messages, is_blocked_by_cooldown=True)
    print(f"Guard triggered: {triggered}")
    print(f"Safe tokens ready for model: {estimate_tokens(safe_msgs)}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
