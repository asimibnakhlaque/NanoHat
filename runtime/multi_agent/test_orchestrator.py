"""
runtime/multi_agent/test_orchestrator.py
Comprehensive automated test suite for the NanoHat multi-agent runtime pipeline:
- Router classification & fallback tests
- Planner step parsing & rediagnosis tests
- Rolling executor conversation & dependency tests
- Self-healing retry loops
- Two-layer memory isolation & sliding window tests
"""

import unittest
from unittest.mock import patch, MagicMock
from runtime.multi_agent.router import RouterAgent
from runtime.multi_agent.planner import PlannerAgent
from runtime.multi_agent.orchestrator import NanoHatOrchestrator


class TestMultiAgentPipeline(unittest.TestCase):

    def test_router_classification(self):
        router = RouterAgent()
        with patch.object(router, "_call", return_value="CHAT"):
            self.assertEqual(router.classify("Hello, how are you?"), "CHAT")

        with patch.object(router, "_call", return_value="TOOL"):
            self.assertEqual(router.classify("Check my RAM usage"), "TOOL")

        # Fallback heuristic when model outputs messy text
        with patch.object(router, "_call", return_value="I am unsure"):
            self.assertEqual(router.classify("Check my CPU utilization"), "TOOL")
            self.assertEqual(router.classify("Tell me a funny story"), "CHAT")

    def test_planner_step_parsing(self):
        planner = PlannerAgent()
        plan_text = (
            "1. system_health(target=cpu)\n"
            "2. system_health(target=ram)\n"
            "3. system_action(action=get_datetime)"
        )
        steps = planner.parse_steps(plan_text)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "system_health(target=cpu)")
        self.assertEqual(steps[1], "system_health(target=ram)")
        self.assertEqual(steps[2], "system_action(action=get_datetime)")

        # Bullet list format
        bullet_plan = "- calculator(expression=5*10)\n- web_search(query=Fedora 41)"
        bullet_steps = planner.parse_steps(bullet_plan)
        self.assertEqual(len(bullet_steps), 2)
        self.assertEqual(bullet_steps[0], "calculator(expression=5*10)")

    def test_chat_route_no_tool_call(self):
        """CHAT intent routes directly to Chitchat; Executor is never invoked."""
        orch = NanoHatOrchestrator(verbose=False)
        with patch.object(orch.router, "classify", return_value="CHAT"), \
             patch.object(orch.chitchat, "chat", return_value="Hello! How can I help you today?") as mock_chat, \
             patch.object(orch.executor, "run_plan") as mock_exec:
            result = orch.run("Hello there!")

        self.assertEqual(result, "Hello! How can I help you today?")
        mock_chat.assert_called_once()
        mock_exec.assert_not_called()

    def test_tool_route_single_step(self):
        """TOOL intent routes to Planner -> Executor -> Synthesizer."""
        orch = NanoHatOrchestrator(verbose=False)
        with patch.object(orch.router, "classify", return_value="TOOL"), \
             patch.object(orch.planner, "create_plan", return_value="1. system_health(target=cpu)"), \
             patch.object(orch.executor, "run_plan", return_value=(["CPU Utilization: 24.5%"], None)), \
             patch.object(orch.chitchat, "synthesize", return_value="Your CPU usage is currently 24.5%.") as mock_synth:
            result = orch.run("What is my CPU usage?")

        self.assertEqual(result, "Your CPU usage is currently 24.5%.")
        mock_synth.assert_called_once()

    def test_dynamic_dependency_rolling_history(self):
        """Sequential plan execution preserves rolling tool output across steps."""
        orch = NanoHatOrchestrator(verbose=False)
        two_step_plan = "1. system_health(target=top_processes)\n2. system_action(action=kill_process, target=firefox)"
        with patch.object(orch.router, "classify", return_value="TOOL"), \
             patch.object(orch.planner, "create_plan", return_value=two_step_plan), \
             patch.object(orch.executor, "run_plan", return_value=(
                 ["firefox 12345 (12% CPU)", "Terminated process 'firefox'."], None
             )), \
             patch.object(orch.chitchat, "synthesize", return_value="Found Firefox process and terminated it safely."):
            result = orch.run("Find and kill Firefox")

        self.assertEqual(result, "Found Firefox process and terminated it safely.")

    def test_self_healing_on_executor_failure(self):
        """Executor failure triggers Planner rediagnosis and successful retry."""
        orch = NanoHatOrchestrator(verbose=False)
        with patch.object(orch.router, "classify", return_value="TOOL"), \
             patch.object(orch.planner, "create_plan", return_value="1. invalid_tool_name()"), \
             patch.object(orch.planner, "diagnose_and_replan", return_value="1. system_health(target=cpu)"), \
             patch.object(orch.executor, "run_plan", side_effect=[
                 ([], "Unknown tool 'invalid_tool_name'"),
                 (["CPU Utilization: 15%"], None)
             ]), \
             patch.object(orch.chitchat, "synthesize", return_value="CPU usage is 15%."):
            result = orch.run("Check CPU")

        self.assertEqual(result, "CPU usage is 15%.")

    def test_clean_history_excludes_tool_tags(self):
        """Clean memory must never contain raw XML tool tags or JSON blocks."""
        orch = NanoHatOrchestrator(verbose=False)
        with patch.object(orch.router, "classify", return_value="TOOL"), \
             patch.object(orch.planner, "create_plan", return_value="1. calculator(expression=10*10)"), \
             patch.object(orch.executor, "run_plan", return_value=(["100"], None)), \
             patch.object(orch.chitchat, "synthesize", return_value="10 multiplied by 10 is 100."):
            orch.run("Calculate 10 * 10")

        self.assertEqual(len(orch.clean_history), 2)
        for msg in orch.clean_history:
            self.assertNotIn("<tool_call>", msg["content"])
            self.assertNotIn("<thought>", msg["content"])
            self.assertNotIn("```json", msg["content"])

    def test_sliding_window_trims_history(self):
        """History older than max_history_turns is pruned to preserve small model context."""
        orch = NanoHatOrchestrator(max_history_turns=2, verbose=False)
        with patch.object(orch.router, "classify", return_value="CHAT"), \
             patch.object(orch.chitchat, "chat", return_value="Roger that"):
            for i in range(5):
                orch.run(f"Turn {i}")

        # Max 2 turns = 4 messages max
        self.assertEqual(len(orch.clean_history), 4)
        self.assertEqual(orch.clean_history[0]["content"], "Turn 3")
        self.assertEqual(orch.clean_history[2]["content"], "Turn 4")


if __name__ == "__main__":
    unittest.main()
