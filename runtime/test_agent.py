"""
runtime/test_agent.py
Comprehensive automated test suite for the NanoHat Fedora Agent:
- Tool execution unit tests (V2 contract: OK: / ERROR[reason]: prefixes)
- Fail-closed destructive-action confirmation gate (regression: headless bypass)
- Calculator DoS/AST guard tests (regression: eval bombs)
- Scheduler persistence, due parsing, temporal ranges, lifecycle
- Context-window management (regression: unbounded history growth)
- JSON repair (nested braces), loop-guard finalize
- Schema checker & canonicalizer tests
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

from runtime import tools as T
from runtime.agent import (
    NanoHatAgent,
    _repair_and_parse_json,
    _normalize_tool_call,
    _execute_tool,
    _cap_tool_output,
    MAX_HISTORY_CHARS,
)
from validator.schema_checker import validate_conversation, validate_tool_call
from validator.canonicalizer import canonicalize_conversation
from generator.schemas import CANONICAL_SYSTEM_PROMPT


def _ok(msg):
    return f"OK: {msg}"


def _err(reason, msg):
    return f"ERROR[{reason}]: {msg}"


class IsolatedHomeTestCase(unittest.TestCase):
    """Redirects HOME to a temp dir so tests never touch real user data."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="nanohat_test_")
        self._old_home = os.environ.get("HOME")
        os.environ["HOME"] = self._tmp

    def tearDown(self):
        if self._old_home is not None:
            os.environ["HOME"] = self._old_home


# ---------------------------------------------------------------------------
# SAFETY GATE (the regression that started Phase H)
# ---------------------------------------------------------------------------
class TestDestructiveGate(IsolatedHomeTestCase):

    def test_headless_stdin_refuses_fail_closed(self):
        """Regression: previously input() raised EOFError -> except:pass -> EXECUTED."""
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False
        with mock.patch.object(T.sys, "stdin", fake_stdin), \
             mock.patch.object(T.sys, "stdout", fake_stdin):
            allowed = T._confirm_destructive("kill_process", "firefox")
        self.assertFalse(allowed)
        result = T.system_action(action="kill_process", target="definitely_not_running_xyz")
        self.assertTrue(result.startswith(_err("not_confirmed", "")), result)

    def test_interactive_decline_refuses(self):
        fake = mock.Mock()
        fake.isatty.return_value = True
        with mock.patch.object(T.sys, "stdin", fake), \
             mock.patch.object(T.sys, "stdout", fake), \
             mock.patch("builtins.input", return_value="n"):
            allowed = T._confirm_destructive("empty_trash", "")
        self.assertFalse(allowed)

    def test_interactive_accept_allows(self):
        fake = mock.Mock()
        fake.isatty.return_value = True
        with mock.patch.object(T.sys, "stdin", fake), \
             mock.patch.object(T.sys, "stdout", fake), \
             mock.patch("builtins.input", return_value="y"):
            allowed = T._confirm_destructive("restart_service", "nginx")
        self.assertTrue(allowed)

    def test_env_override_for_daemons(self):
        with mock.patch.dict(os.environ, {"NANOHAT_AUTO_APPROVE_DESTRUCTIVE": "1"}):
            self.assertTrue(T._confirm_destructive("kill_process", "x"))


# ---------------------------------------------------------------------------
# CALCULATOR (no-eval AST guard)
# ---------------------------------------------------------------------------
class TestCalculator(unittest.TestCase):

    def test_basic_arithmetic(self):
        self.assertEqual(T.calculator("120 * 0.15"), _ok("18"))
        self.assertEqual(T.calculator("(45 + 78) * 2"), _ok("246"))
        self.assertEqual(T.calculator("2 ** 10"), _ok("1024"))

    def test_injection_rejected(self):
        res = T.calculator("__import__('os').system('ls')")
        self.assertTrue(res.startswith("ERROR[syntax]"), res)

    def test_dos_bomb_rejected_fast(self):
        """Regression: '9**(9**9)' previously hung the CPU via eval()."""
        res = T.calculator("9 ** (9 ** 9)")
        self.assertTrue(res.startswith("ERROR[range]"), res)
        res2 = T.calculator("2 ** 99999999999999999999")
        self.assertTrue(res2.startswith("ERROR["), res2)

    def test_division_by_zero(self):
        self.assertEqual(T.calculator("5 / 0"), _err("math", "division by zero."))


# ---------------------------------------------------------------------------
# SCHEDULER (persistence + lifecycle + temporal queries)
# ---------------------------------------------------------------------------
class TestScheduler(IsolatedHomeTestCase):

    def _set(self, task, due):
        return T.scheduler(action="set", task=task, due=due)

    def test_set_relative_returns_minted_id(self):
        res = self._set("Call dentist", "10m")
        self.assertTrue(res.startswith(_ok(f"Task scheduled: [t_")), res)
        self.assertIn("(in 10m)", res)
        tid = res.split("[")[1].split("]")[0]
        tasks = T._load_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], tid)
        # Due must be absolute ISO ~10 minutes from now
        due_dt = datetime.fromisoformat(tasks[0]["due_at"])
        delta = (due_dt - datetime.now(due_dt.tzinfo)).total_seconds()
        self.assertAlmostEqual(delta, 600, delta=15)

    def test_natural_language_delay(self):
        """Regression: legacy parser turned '20 minutes' into 60 seconds."""
        for spec, expected_secs in [("20 minutes", 1200), ("1h30m", 5400), ("45s", 45)]:
            res = self._set(f"task {spec}", spec)
            self.assertTrue(res.startswith(_ok("Task scheduled")), res)
            entry = T._load_tasks()[-1]
            due_dt = datetime.fromisoformat(entry["due_at"])
            delta = (due_dt - datetime.now(due_dt.tzinfo)).total_seconds()
            self.assertAlmostEqual(delta, expected_secs, delta=15,
                                   msg=f"spec '{spec}' parsed wrong")

    def test_absolute_due_and_bad_due(self):
        tomorrow = datetime.now().astimezone() + timedelta(days=1)
        res = self._set("Team standup", tomorrow.strftime("%Y-%m-%d %H:%M"))
        self.assertTrue(res.startswith(_ok("Task scheduled")), res)
        res_bad = self._set("Vague task", "sometime soon")
        self.assertTrue(res_bad.startswith(_err("bad_due", "")), res_bad)
        res_empty = self._set("No time", "")
        self.assertTrue(res_empty.startswith(_err("bad_due", "")), res_empty)

    def test_list_ranges_and_status_lifecycle(self):
        yesterday = datetime.now().astimezone() - timedelta(days=1)
        today = datetime.now().astimezone() + timedelta(hours=2)
        self._set("Old task", "5m")
        # Inject a done task dated yesterday directly for range testing
        tasks = T._load_tasks()
        tid = tasks[-1]["id"]
        tasks.append({
            "id": "t_yesterdy", "task": "Yesterday thing",
            "due_at": yesterday.isoformat(timespec="minutes"),
            "created_at": yesterday.isoformat(timespec="seconds"),
            "status": "done", "completed_at": yesterday.isoformat(timespec="seconds"),
        })
        tasks.append({
            "id": "t_tomorrow", "task": "Tomorrow thing",
            "due_at": today.isoformat(timespec="minutes"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending", "completed_at": None,
        })
        T._save_tasks(tasks)

        lst_today = T.scheduler(action="list", range="today")
        self.assertIn(tid, lst_today)
        self.assertNotIn("t_yesterdy", lst_today)  # done filtered by default status=pending

        lst_yest_done = T.scheduler(action="list", range="yesterday", status="done")
        self.assertIn("t_yesterdy", lst_yest_done)
        self.assertIn("Yesterday thing", lst_yest_done)

        lst_any = T.scheduler(action="list", status="any", range="all")
        self.assertIn("t_tomorrow", lst_any)

    def test_update_and_delete_by_id(self):
        self._set("Review PR", "1h")
        tasks = T._load_tasks()
        tid = tasks[-1]["id"]

        upd = T.scheduler(action="update", id=tid, status="done")
        self.assertTrue(upd.startswith(_ok(f"Task [{tid}] updated")), upd)
        self.assertEqual(T._load_tasks()[0]["status"], "done")

        re_up = T.scheduler(action="update", id=tid, due="30m")
        self.assertIn("due=", re_up)

        dele = T.scheduler(action="delete", id=tid)
        self.assertEqual(dele, _ok(f"Task [{tid}] deleted."))
        gone = T.scheduler(action="delete", id=tid)
        self.assertTrue(gone.startswith(_err("not_found", "")), gone)

    def test_update_requires_id_or_changes(self):
        self.assertTrue(
            T.scheduler(action="update").startswith(_err("missing_id", "")))
        self.assertTrue(
            T.scheduler(action="update", id="t_nope").startswith(_err("not_found", "")))


# ---------------------------------------------------------------------------
# USER MEMORY (list + uniform errors + timestamps)
# ---------------------------------------------------------------------------
class TestUserMemory(IsolatedHomeTestCase):

    def test_store_get_delete_roundtrip(self):
        store_res = T.user_memory(action="store", key="test_key", value="test_val")
        self.assertEqual(store_res, _ok("Memory stored: test_key = 'test_val'"))
        get_res = T.user_memory(action="get", key="test_key")
        self.assertIn("test_val", get_res)
        del_res = T.user_memory(action="delete", key="test_key")
        self.assertIn("deleted", del_res)
        missing = T.user_memory(action="get", key="test_key")
        self.assertTrue(missing.startswith(_err("not_found", "")))

    def test_list_action(self):
        empty = T.user_memory(action="list")
        self.assertIn("empty", empty)
        T.user_memory(action="store", key="editor", value="neovim")
        listing = T.user_memory(action="list")
        self.assertIn("editor='neovim'", listing)
        self.assertIn("memories stored", listing)

    def test_legacy_flat_file_compat(self):
        """Old flat memory.json entries still readable."""
        db_path = os.path.join(os.environ["HOME"], ".config/smollm2-fedora-agent/memory.json")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, "w") as f:
            json.dump({"legacy_key": "legacy_value"}, f)
        res = T.user_memory(action="get", key="legacy_key")
        self.assertEqual(res, _ok("legacy_key = 'legacy_value'"))


# ---------------------------------------------------------------------------
# SAFE KILL (exact match only; ambiguity surfaced, not slaughtered)
# ---------------------------------------------------------------------------
class TestSafeKill(IsolatedHomeTestCase):

    def test_nonexistent_process_not_found(self):
        res = T._safe_kill("definitely_no_such_proc_xyz123")
        self.assertTrue(res.startswith(_err("not_found", "")), res)

    def test_invalid_names_rejected(self):
        self.assertTrue(T._safe_kill("firefox; rm -rf /").startswith(_err("invalid_name", "")))
        self.assertTrue(T.system_action(action="kill_process",
                                        target="app && curl evil.com").startswith(_err("not_confirmed", "")) or
                        T._safe_kill("app && curl evil.com").startswith(_err("invalid_name", "")))


# ---------------------------------------------------------------------------
# NOTIFICATION WORKER (persistent tasks must actually ring)
# ---------------------------------------------------------------------------
class TestNotificationWorker(IsolatedHomeTestCase):

    def _insert_due_task(self, due_dt, notified=False, status="pending"):
        tasks = T._load_tasks()
        tasks.append({
            "id": "t_due0001", "task": "Take the cake out",
            "due_at": due_dt.isoformat(timespec="seconds"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": status, "completed_at": None,
            "notified": notified,
        })
        T._save_tasks(tasks)

    def test_overdue_pending_task_fires_once(self):
        past = datetime.now().astimezone() - timedelta(minutes=30)
        self._insert_due_task(past)
        fired_calls = []
        with mock.patch.object(T.subprocess, "run",
                               side_effect=lambda cmd, **kw: fired_calls.append(cmd) or mock.Mock(returncode=0)):
            fired = T._poll_notifications_once()
        self.assertEqual(fired, ["t_due0001"])
        self.assertTrue(fired_calls and fired_calls[0][0] == "notify-send")
        self.assertIn("Take the cake out", fired_calls[0][-1])
        # idempotent: second poll does not re-fire
        with mock.patch.object(T.subprocess, "run") as spy:
            fired2 = T._poll_notifications_once()
        self.assertEqual(fired2, [])
        spy.assert_not_called()
        self.assertTrue(T._load_tasks()[0]["notified"])

    def test_future_and_done_tasks_never_fire(self):
        future = datetime.now().astimezone() + timedelta(hours=1)
        self._insert_due_task(future)  # pending, not yet due
        tasks = T._load_tasks()
        tasks.append({**tasks[0], "id": "t_done001", "status": "done"})  # done but overdue
        T._save_tasks(tasks)
        with mock.patch.object(T.subprocess, "run") as spy:
            fired = T._poll_notifications_once()
        self.assertEqual(fired, [])
        spy.assert_not_called()

    def test_worker_start_is_idempotent(self):
        with mock.patch.object(T.threading, "Thread") as thread_spy:
            first = T.start_notification_worker(interval_seconds=60)
            second = T.start_notification_worker(interval_seconds=60)
        # only one real thread spawn; second call short-circuits
        self.assertTrue(first)
        if first:
            self.assertFalse(second)


# ---------------------------------------------------------------------------
# LAUNCH_APP URL SUPPORT + MEMORY CAPS
# ---------------------------------------------------------------------------
class TestLaunchAndCaps(IsolatedHomeTestCase):

    def test_url_routes_to_xdg_open(self):
        calls = []
        with mock.patch.object(T.subprocess, "run",
                               side_effect=lambda cmd, **kw: calls.append((cmd, kw)) or mock.Mock(returncode=0)):
            res = T.system_action(action="launch_app", target="https://youtube.com")
        self.assertEqual(res, _ok("Opened https://youtube.com in your browser."))
        self.assertEqual(calls[0][0], ["xdg-open", "https://youtube.com"])

    def test_url_scheme_validated(self):
        res = T.system_action(action="launch_app", target="file:///etc/shadow")
        self.assertTrue(res.startswith(_err("invalid_name", "")), res)

    def test_memory_value_cap(self):
        res = T.user_memory(action="store", key="big", value="x" * 600)
        self.assertTrue(res.startswith(_err("too_long", "")), res)
        res2 = T.user_memory(action="store", key="k" * 100, value="v")
        self.assertTrue(res2.startswith(_err("too_long", "")), res2)

    def test_scheduler_set_initializes_notified_false(self):
        T.scheduler(action="set", task="ring me", due="30m")
        self.assertFalse(T._load_tasks()[0]["notified"])


# ---------------------------------------------------------------------------
# SYSTEM HEALTH & ACTIONS (contract conformance)
# ---------------------------------------------------------------------------
class TestSystemHealth(unittest.TestCase):

    def test_cpu_ram_disk_all_prefixed(self):
        for target, marker in [("cpu", "CPU Utilization:"), ("ram", "RAM:"),
                               ("disk", "Disk /:"), ("all", "System Overview:")]:
            res = T.system_health(target)
            self.assertTrue(res.startswith(_ok(marker)), f"{target}: {res}")

    def test_top_processes_excludes_ps_artifact(self):
        res = T.system_health("top_processes")
        self.assertNotIn("--sort=-%cpu", res)  # regression: ps lists itself

    def test_datetime_has_offset(self):
        res = T.system_action(action="get_datetime")
        self.assertTrue(res.startswith(_ok("")))
        digits = "".join(c for c in res if c.isdigit())
        self.assertGreaterEqual(len(digits), 12)


# ---------------------------------------------------------------------------
# AGENT CONTEXT MANAGEMENT (unbounded growth regression)
# ---------------------------------------------------------------------------
class TestContextManagement(unittest.TestCase):

    def _agent(self):
        agent = NanoHatAgent.__new__(NanoHatAgent)  # skip backend init
        agent.backend = "ollama"
        agent.model_name = "test"
        agent.max_steps = 5
        agent.messages = [{"role": "system", "content": CANONICAL_SYSTEM_PROMPT}]
        agent.transformers_model = None
        agent.transformers_tokenizer = None
        return agent

    def test_history_pruned_under_budget(self):
        agent = self._agent()
        for i in range(50):
            agent.messages.append({"role": "user", "content": f"question number {i} " * 40})
            agent.messages.append({"role": "assistant", "content": "answer " * 60})
        total = sum(len(m["content"]) for m in agent.messages)
        self.assertGreater(total, MAX_HISTORY_CHARS)
        agent._prune_history()
        total_after = sum(len(m["content"]) for m in agent.messages)
        self.assertLessEqual(total_after, MAX_HISTORY_CHARS + 200)
        self.assertEqual(agent.messages[0]["role"], "system")

    def test_view_starts_on_user_boundary_and_keeps_system(self):
        agent = self._agent()
        big = "x" * (MAX_HISTORY_CHARS // 3)
        agent.messages += [
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "tool", "content": big},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "working on it"},
        ]
        view = agent._build_context_view(agent.messages)
        self.assertEqual(view[0]["role"], "system")
        self.assertEqual(view[1]["role"], "user")
        self.assertEqual(view[-1]["content"], "working on it")

    def test_oversized_tool_output_capped(self):
        blob = "line of process output\n" * 500
        capped = _cap_tool_output(blob)
        self.assertLess(len(capped), 900)
        self.assertIn("[output truncated]", capped)


# ---------------------------------------------------------------------------
# JSON REPAIR & NORMALIZATION
# ---------------------------------------------------------------------------
class TestJsonRepairAndNormalization(unittest.TestCase):

    def test_nested_braces_extracted_fully(self):
        raw = '{"name": "scheduler", "arguments": {"action": "set", "task": "hi", "meta": {"a": {"b": 1}}}, junk'
        parsed = _repair_and_parse_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["arguments"]["meta"]["a"]["b"], 1)

    def test_single_quotes_and_fences(self):
        fenced = '```json\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n```'
        self.assertEqual(_repair_and_parse_json(fenced)["name"], "calculator")

    def test_legacy_reminder_call_routes_to_scheduler(self):
        name, args = _normalize_tool_call("reminder", {"task": "Tea", "time_or_delay": "10m"})
        self.assertEqual(name, "scheduler")
        self.assertEqual(args["action"], "set")
        self.assertEqual(args["due"], "10m")

    def test_health_action_auto_route(self):
        name, args = _normalize_tool_call("system_action", {"action": "check_ram"})
        self.assertEqual(name, "system_health")
        self.assertEqual(args, {"target": "ram"})

    def test_unknown_tool_contract(self):
        res = _execute_tool("bash_terminal", {"cmd": "ls"})
        self.assertTrue(res.startswith("ERROR[unknown_tool]"), res)


# ---------------------------------------------------------------------------
# VALIDATOR & CANONICALIZER (V2 schemas)
# ---------------------------------------------------------------------------
class TestValidationAndCanonicalization(unittest.TestCase):

    def _conv(self, tool_content):
        return {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": "Do the thing."},
                {"role": "assistant", "content": tool_content},
            ]
        }

    def test_valid_scheduler_conversation_passes(self):
        conv = self._conv('<thought>Schedule it.</thought>\n<tool_call>{"name":"scheduler","arguments":{"action":"set","task":"Check oven","due":"10m"}}</tool_call>')
        ok, reason = validate_conversation(conv)
        self.assertTrue(ok, reason)

    def test_scheduler_missing_due_rejected(self):
        conv = self._conv('<thought>Schedule.</thought>\n<tool_call>{"name":"scheduler","arguments":{"action":"set","task":"No time"}}</tool_call>')
        ok, reason = validate_conversation(conv)
        self.assertFalse(ok)
        self.assertIn("due", reason)

    def test_memory_list_without_key_passes(self):
        tc = {"name": "user_memory", "arguments": {"action": "list"}}
        ok, reason = validate_tool_call(tc)
        self.assertTrue(ok, reason)

    def test_memory_get_with_value_rejected(self):
        tc = {"name": "user_memory", "arguments": {"action": "get", "key": "k", "value": "v"}}
        ok, reason = validate_tool_call(tc)
        self.assertFalse(ok)

    def test_hallucinated_tool_rejected(self):
        conv = self._conv('<thought>Run bash.</thought>\n<tool_call>{"name": "bash_terminal", "arguments": {"cmd": "ls"}}</tool_call>')
        ok, reason = validate_conversation(conv)
        self.assertFalse(ok)
        self.assertIn("Hallucinated tool", reason)

    def test_consecutive_assistant_turns_rejected(self):
        invalid_conv = {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "assistant", "content": "Hello!"},
                {"role": "assistant", "content": "How can I help?"},
            ]
        }
        ok, reason = validate_conversation(invalid_conv)
        self.assertFalse(ok)
        self.assertIn("consecutive assistant turns", reason)

    def test_system_prompt_must_be_canonical_nanohat(self):
        bad_conv = {
            "messages": [
                {"role": "system", "content": "You are some other agent."},
                {"role": "user", "content": "Hi"},
            ]
        }
        ok, reason = validate_conversation(bad_conv)
        self.assertFalse(ok)

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


# ---------------------------------------------------------------------------
# CLI ENTRYPOINT TESTS
# ---------------------------------------------------------------------------
class TestCLIEntrypoint(IsolatedHomeTestCase):

    @mock.patch("runtime.agent.NanoHatAgent")
    def test_single_shot_positional_string(self, mock_agent_cls):
        mock_instance = mock.MagicMock()
        mock_instance.run.return_value = "System is healthy."
        mock_agent_cls.return_value = mock_instance

        from runtime.agent import main
        test_argv = ["agent.py", "What is my current system health?"]

        with mock.patch("sys.argv", test_argv), mock.patch("builtins.print") as mock_print:
            main()
            mock_instance.run.assert_called_once_with("What is my current system health?")
            mock_print.assert_called_with("System is healthy.")

    @mock.patch("runtime.agent.NanoHatAgent")
    def test_single_shot_multi_word_tokens(self, mock_agent_cls):
        mock_instance = mock.MagicMock()
        mock_instance.run.return_value = "RAM is 40% used."
        mock_agent_cls.return_value = mock_instance

        from runtime.agent import main
        test_argv = ["agent.py", "check", "ram", "usage"]

        with mock.patch("sys.argv", test_argv), mock.patch("builtins.print") as mock_print:
            main()
            mock_instance.run.assert_called_once_with("check ram usage")
            mock_print.assert_called_with("RAM is 40% used.")

    @mock.patch("runtime.agent.NanoHatAgent")
    def test_single_shot_named_query(self, mock_agent_cls):
        mock_instance = mock.MagicMock()
        mock_instance.run.return_value = "Result: 4"
        mock_agent_cls.return_value = mock_instance

        from runtime.agent import main
        test_argv = ["agent.py", "-q", "calculate 2 + 2"]

        with mock.patch("sys.argv", test_argv), mock.patch("builtins.print") as mock_print:
            main()
            mock_instance.run.assert_called_once_with("calculate 2 + 2")
            mock_print.assert_called_with("Result: 4")

    @mock.patch("runtime.agent.NanoHatAgent")
    def test_interactive_mode_exits_cleanly(self, mock_agent_cls):
        mock_instance = mock.MagicMock()
        mock_agent_cls.return_value = mock_instance

        from runtime.agent import main
        test_argv = ["agent.py"]

        with mock.patch("sys.argv", test_argv), mock.patch("builtins.input", return_value="exit"):
            main()
            mock_instance.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
