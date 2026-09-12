"""
runtime/test_native_agent.py
Automated unit and mock integration test suite for the Native Ollama Agent.
"""

import unittest
from unittest.mock import patch, MagicMock
from runtime.native_agent import NativeOllamaAgent


class TestNativeOllamaAgent(unittest.TestCase):

    def test_conversational_turn(self):
        """Pure conversational turn without tool calls."""
        agent = NativeOllamaAgent(verbose=False)
        mock_resp = {"role": "assistant", "content": "Hello! How can I help you today?"}

        with patch.object(agent, "_call_ollama", return_value=mock_resp):
            reply = agent.run("Hello there!")

        self.assertEqual(reply, "Hello! How can I help you today?")
        self.assertEqual(len(agent.messages), 3)  # system + user + assistant

    def test_single_tool_execution(self):
        """Single tool invocation and observation synthesis."""
        agent = NativeOllamaAgent(verbose=False)
        tool_call_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "system_health",
                        "arguments": {"target": "cpu"}
                    }
                }
            ]
        }
        final_msg = {
            "role": "assistant",
            "content": "Your CPU usage is currently 22.5%."
        }

        with patch.object(agent, "_call_ollama", side_effect=[tool_call_msg, final_msg]):
            with patch("runtime.native_agent.TOOL_DISPATCHER", {"system_health": lambda target: "CPU Utilization: 22.5%"}):
                reply = agent.run("Check my CPU")

        self.assertEqual(reply, "Your CPU usage is currently 22.5%.")
        # messages: system + user + asst(tool_call) + tool_obs + asst(final)
        self.assertEqual(len(agent.messages), 5)
        self.assertEqual(agent.messages[3]["role"], "tool")
        self.assertIn("22.5%", agent.messages[3]["content"])

    def test_parallel_tool_execution(self):
        """Parallel tool invocations in a single turn."""
        agent = NativeOllamaAgent(verbose=False)
        parallel_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "system_health", "arguments": {"target": "cpu"}}},
                {"function": {"name": "system_health", "arguments": {"target": "ram"}}},
            ]
        }
        final_msg = {
            "role": "assistant",
            "content": "CPU is 20% and RAM is 45%."
        }

        def mock_health(target):
            return f"{target.upper()}: OK"

        with patch.object(agent, "_call_ollama", side_effect=[parallel_msg, final_msg]):
            with patch("runtime.native_agent.TOOL_DISPATCHER", {"system_health": mock_health}):
                reply = agent.run("Check CPU and RAM")

        self.assertEqual(reply, "CPU is 20% and RAM is 45%.")
        # messages: system + user + asst + tool_1 + tool_2 + asst(final)
        self.assertEqual(len(agent.messages), 6)
        self.assertEqual(agent.messages[3]["role"], "tool")
        self.assertEqual(agent.messages[4]["role"], "tool")

    def test_unknown_tool_graceful_handling(self):
        """Graceful error reporting on hallucinated tool names."""
        agent = NativeOllamaAgent(verbose=False)
        bad_tool_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "non_existent_tool", "arguments": {"arg": "val"}}}
            ]
        }
        final_msg = {
            "role": "assistant",
            "content": "I couldn't find that tool."
        }

        with patch.object(agent, "_call_ollama", side_effect=[bad_tool_msg, final_msg]):
            reply = agent.run("Do something impossible")

        self.assertEqual(reply, "I couldn't find that tool.")
        self.assertIn("Unknown tool", agent.messages[3]["content"])

    def test_reset_conversation(self):
        """Resetting conversation restores baseline system prompt."""
        agent = NativeOllamaAgent(verbose=False)
        agent.messages.append({"role": "user", "content": "test"})
        agent.messages.append({"role": "assistant", "content": "response"})
        self.assertEqual(len(agent.messages), 3)

        agent.reset_conversation()
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
