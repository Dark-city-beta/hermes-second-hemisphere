"""Unit tests for Hermes Second Hemisphere context compression."""

import unittest
from hermes_second_hemisphere import (
    ContinuousCompactor,
    AuxCompressionClient,
    HardWallGuard,
    CompactionTier,
    estimate_tokens,
    extract_hard_facts,
)


class TestSecondHemisphere(unittest.TestCase):

    def test_estimate_tokens(self):
        text = "Hello world from Hermes"
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 0)

        msg = {"role": "user", "content": "A" * 320}
        tokens_msg = estimate_tokens(msg)
        self.assertTrue(90 <= tokens_msg <= 110)

        msg_list = [msg, msg]
        self.assertEqual(estimate_tokens(msg_list), tokens_msg * 2)

    def test_extract_hard_facts(self):
        messages = [
            {"role": "user", "content": "Please inspect /etc/nginx/sites-available/default and /var/log/syslog"},
            {"role": "assistant", "content": "I will run `systemctl status nginx` and check error: Exception: Port 80 already in use"},
        ]
        facts = extract_hard_facts(messages)
        self.assertIn("[ARCHIVED_FACTS_RESTORED]", facts)
        self.assertIn("/etc/nginx/sites-available/default", facts)
        self.assertIn("systemctl status nginx", facts)

    def test_emergency_compact(self):
        compactor = ContinuousCompactor(
            main_context_window=10000,
            compression_threshold_tokens=5000,
            target_ratio=0.50,
        )
        messages = [
            {"role": "system", "content": "System prompt initial."},
            {"role": "user", "content": "Task start."},
        ]
        for i in range(15):
            messages.append({
                "role": "tool",
                "name": "shell",
                "content": f"Output from command {i}: " + ("data block with trace and details. " * 50),
            })
        messages.append({"role": "user", "content": "Final user query."})

        initial_tokens = estimate_tokens(messages)
        compacted = compactor.emergency_compact(messages, target_tokens=2000)
        final_tokens = estimate_tokens(compacted)

        self.assertLess(final_tokens, initial_tokens)
        self.assertLessEqual(final_tokens, 2500)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[-1]["role"], "user")

    def test_fallback_tier(self):
        compactor = ContinuousCompactor(
            aux_client=AuxCompressionClient(base_url="http://127.0.0.1:9999"),  # Unreachable aux
            main_context_window=10000,
            compression_threshold_tokens=1000,
        )

        def mock_fallback(messages):
            return True, "Mock summary produced by fallback tier."

        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1" * 500},
            {"role": "user", "content": "U2" * 500},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Active question"},
        ]

        result = compactor.compact(messages, fallback_fn=mock_fallback)
        self.assertTrue(result.success)
        self.assertEqual(result.tier, CompactionTier.LAYER2_MAIN_FALLBACK)
        self.assertEqual(result.summary, "Mock summary produced by fallback tier.")

    def test_hardwall_guard_preflight_and_recovery(self):
        compactor = ContinuousCompactor(main_context_window=1000, target_ratio=0.40)
        guard = HardWallGuard(compactor=compactor, hard_limit_tokens=1000, danger_threshold_ratio=0.70)

        # Context with 800 tokens (>= 700 danger threshold)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message " + ("token_bloat " * 100)},
            {"role": "tool", "content": "Large tool output " * 200},
            {"role": "user", "content": "Last question"},
        ]

        safe_msgs, triggered = guard.inspect_and_defend_preflight(messages, is_blocked_by_cooldown=True)
        self.assertTrue(triggered)
        self.assertLessEqual(estimate_tokens(safe_msgs), 500)

        recovered_msgs, recovered = guard.recover_from_overflow_exception(
            messages, "HTTP 400: request exceeds the available context size", attempts_exhausted=True
        )
        self.assertTrue(recovered)
        self.assertLessEqual(estimate_tokens(recovered_msgs), 600)


if __name__ == "__main__":
    unittest.main()
