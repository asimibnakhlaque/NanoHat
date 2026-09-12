"""
runtime/tools.py
Consolidated Fedora desktop & system tools for the NanoHat agent.

Hardening (Phase H):
- Fail-closed confirmation gate for destructive actions (headless => refuse,
  explicit NANOHAT_AUTO_APPROVE_DESTRUCTIVE=1 override for daemons).
- Persistent scheduler replacing the fire-and-forget reminder thread
  (absolute due times, backend-minted ids, status lifecycle, temporal ranges).
- Safe kill via psutil enumeration: exact-name kills only; ambiguous partial
  matches are reported back for disambiguation instead of being slaughtered.
- AST-walked calculator (no eval), operand caps, no DoS expressions.
- Uniform machine-checkable output contract: "OK: ..." / "ERROR[reason]: ...".
- Zero shell interpolation: subprocess.run(shell=False) everywhere.
"""

import os
import re
import sys
import json
import ast
import time
import secrets
import threading
import subprocess
import datetime
import psutil

# Point rustls and OpenSSL directly to the Fedora public CA bundle
os.environ["SSL_CERT_FILE"] = "/etc/pki/tls/certs/ca-bundle.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"

CONFIG_DIR_NAME = "~/.config/smollm2-fedora-agent"


def _memory_db() -> str:
    """Resolved at CALL time so tests can redirect HOME safely."""
    return os.path.expanduser(os.path.join(CONFIG_DIR_NAME, "memory.json"))


def _tasks_db() -> str:
    return os.path.expanduser(os.path.join(CONFIG_DIR_NAME, "tasks.json"))

# Flexible import for smolagents @tool decorator (or fallback dummy decorator for testing)
try:
    from smolagents import tool
except ImportError:
    def tool(func):
        """Fallback decorator when smolagents is not yet installed in local environment."""
        func.is_tool = True
        return func


# ---------------------------------------------------------------------------
# OUTPUT CONTRACT HELPERS
# ---------------------------------------------------------------------------
def _ok(msg: str) -> str:
    return f"OK: {msg}"


def _err(reason: str, msg: str) -> str:
    return f"ERROR[{reason}]: {msg}"


def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. CALCULATOR TOOL (AST-walked, no eval, DoS-proof)
# ---------------------------------------------------------------------------
_CALC_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_MAX_EXPR_LEN = 200
_MAX_NODES = 60
_MAX_INT_DIGITS = 15
_MAX_EXPONENT = 100
_MAX_RESULT_BITS = 4096   # ~1200 decimal digits; kills runaway int growth


class _CalcRangeError(ValueError):
    """Magnitude violation (too large / would hang). Distinct from syntax errors."""


class _CalcGuard(ast.NodeVisitor):
    def __init__(self):
        self.nodes = 0

    def generic_visit(self, node):
        self.nodes += 1
        if self.nodes > _MAX_NODES:
            raise ValueError("expression too complex")
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("only numeric literals allowed")
            if isinstance(node.value, int) and len(str(abs(node.value))) > _MAX_INT_DIGITS:
                raise _CalcRangeError("numeric literal too large")
            if isinstance(node.value, float) and not (abs(node.value) < 1e15):
                raise _CalcRangeError("float literal too large")
        super().generic_visit(node)


def _eval_calc_node(node):
    if isinstance(node, ast.Expression):
        return _eval_calc_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_calc_node(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _CALC_OPS:
            raise ValueError("operator not allowed")
        left = _eval_calc_node(node.left)
        right = _eval_calc_node(node.right)
        if op_type is ast.Pow:
            # Eval-time cap: blocks ALL exponent bombs, however nested.
            if not isinstance(right, (int, float)) or abs(right) > _MAX_EXPONENT:
                raise _CalcRangeError("exponent too large")
            if isinstance(left, float) and (left != left or abs(left) == float("inf")):
                raise ValueError("invalid base")
        result = _CALC_OPS[op_type](left, right)
        if isinstance(result, int) and result.bit_length() > _MAX_RESULT_BITS:
            raise _CalcRangeError("result too large")
        if isinstance(result, complex) or (isinstance(result, float) and result != result):
            raise ValueError("invalid result")
        return result
    raise ValueError("unsupported syntax")


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression safely.

    Args:
        expression: Mathematical expression string to calculate (e.g. '120 * 0.15').
    """
    expression = str(expression).strip()
    if not expression:
        return _err("empty", "no expression provided.")
    if len(expression) > _MAX_EXPR_LEN:
        return _err("too_long", "expression exceeds 200 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
        _CalcGuard().visit(tree)
        result = _eval_calc_node(tree)
        if isinstance(result, float):
            if result != result or result in (float("inf"), float("-inf")):
                return _err("overflow", "result is out of representable range.")
            result = round(result, 10)
            if result == int(result) and "e" not in str(result).lower():
                result = int(result)
        return _ok(str(result))
    except ZeroDivisionError:
        return _err("math", "division by zero.")
    except OverflowError:
        return _err("overflow", "result too large to compute.")
    except _CalcRangeError as e:
        return _err("range", str(e) + ".")
    except (ValueError, SyntaxError, TypeError) as e:
        return _err("syntax", f"invalid expression ({e}). Allowed: numbers + - * / // % ** ( ).")


# ---------------------------------------------------------------------------
# 2. WEB SEARCH TOOL
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Searches the web for real-time information.

    Args:
        query: Search query keywords.
    """
    query = str(query).strip()
    if not query:
        return _err("empty", "no search query provided.")
    try:
        try:
            from ddgs import DDGS  # modern package name
        except ImportError:
            from duckduckgo_search import DDGS  # legacy fallback
        results = list(DDGS().text(query, max_results=3, timeout=10))
    except Exception as e:
        return _err("search_failed", f"web search is unavailable right now ({e}).")
    if not results:
        return _err("no_results", f"no results found for '{query}'.")
    lines = [f"- {r.get('title', '').strip()}: {r.get('body', '').strip()}".rstrip(": ")
             for r in results]
    return _ok("\n".join(lines))


# ---------------------------------------------------------------------------
# 3. USER MEMORY TOOL (persistent KV + list; auto-timestamped storage)
# ---------------------------------------------------------------------------
_memory_lock = threading.Lock()


def _load_memory() -> dict:
    data = _read_json(_memory_db())
    return data if isinstance(data, dict) else {}


def _memory_get_value(entry):
    """Entries may be legacy plain strings or {'value':...,'created_at':...}."""
    if isinstance(entry, dict):
        return entry.get("value", "")
    return entry


@tool
def user_memory(action: str, key: str = "", value: str = "") -> str:
    """Retains, retrieves, lists, or deletes persistent user preferences and notes.

    Args:
        action: The operation to perform ('store', 'get', 'list', 'delete').
        key: The memory identifier key (not required for 'list').
        value: The value string to store (leave empty for get/list/delete).
    """
    action = str(action).strip().lower()
    key = str(key).strip()

    if action == "list":
        with _memory_lock:
            data = _load_memory()
        if not data:
            return _ok("Memory is empty. Nothing stored yet.")
        items = []
        for k in sorted(data.keys()):
            v = str(_memory_get_value(data[k]))
            when = ""
            if isinstance(data[k], dict) and data[k].get("created_at"):
                when = f" (saved {data[k]['created_at'][:10]})"
            preview = v if len(v) <= 40 else v[:37] + "..."
            items.append(f"{k}='{preview}'{when}")
        return _ok(f"{len(data)} memories stored:\n" + "\n".join(items))

    if action == "store":
        if not key:
            return _err("missing_key", "a memory key is required to store a value.")
        if len(key) > 64:
            return _err("too_long", "memory key exceeds 64 characters.")
        if not str(value).strip():
            return _err("missing_value", "a non-empty value is required to store a memory.")
        if len(str(value)) > 512:
            return _err("too_long", "memory value exceeds 512 characters.")
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with _memory_lock:
            data = _load_memory()
            prev = data.get(key)
            entry = {
                "value": str(value),
                "created_at": (prev.get("created_at") if isinstance(prev, dict) else now) or now,
                "updated_at": now,
            }
            data[key] = entry
            _atomic_write_json(_memory_db(), data)
        return _ok(f"Memory stored: {key} = '{value}'")

    if action == "get":
        if not key:
            return _err("missing_key", "a memory key is required to retrieve a value.")
        with _memory_lock:
            data = _load_memory()
        if key in data:
            return _ok(f"{key} = '{_memory_get_value(data[key])}'")
        return _err("not_found", f"Memory key '{key}' not found.")

    if action == "delete":
        if not key:
            return _err("missing_key", "a memory key is required to delete an entry.")
        with _memory_lock:
            data = _load_memory()
            if key in data:
                del data[key]
                _atomic_write_json(_memory_db(), data)
                return _ok(f"Memory key '{key}' deleted.")
        return _err("not_found", f"Key '{key}' not found.")

    return _err("bad_action", "Invalid action. Choose 'store', 'get', 'list', or 'delete'.")


# ---------------------------------------------------------------------------
# 4. SCHEDULER TOOL (persistent tasks, absolute due times, full lifecycle)
# ---------------------------------------------------------------------------
_tasks_lock = threading.Lock()

_REL_CHUNK_RE = re.compile(
    r"(\d+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)", re.I)
# Whole-spec validation: one or more <number><unit> chunks and nothing else
# (no trailing \b inside chunks so "1h30m" parses both parts).
_REL_FULL_RE = re.compile(
    r"^\s*(\d+\s*(?:hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\s*)+$", re.I)
_UNIT_SECONDS = {
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
}
_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def parse_due(spec: str):
    """Parses a due spec into an absolute local datetime.

    Accepts ISO datetimes ('2026-08-24 15:00', '2026-08-24T09:30') or relative
    delays ('10m', '1h30m', '45s', 'in 20 minutes').
    Returns (datetime, human_relative_suffix or None); raises ValueError.
    """
    spec = str(spec).strip()
    if not spec:
        raise ValueError("due date/time is required")

    try:
        dt = datetime.datetime.fromisoformat(spec.replace("T", " "))
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt, None
    except ValueError:
        pass

    cleaned = re.sub(r"^\s*in\s+", "", spec, flags=re.I).strip().lower()
    if not _REL_FULL_RE.match(cleaned):
        raise ValueError(f"cannot understand due time '{spec}'. Use 'YYYY-MM-DD HH:MM' or a delay like '10m'.")
    chunks = _REL_CHUNK_RE.findall(cleaned)
    if not chunks:
        raise ValueError(f"cannot understand due time '{spec}'. Use 'YYYY-MM-DD HH:MM' or a delay like '10m'.")
    total = 0
    parts = []
    for num, unit in chunks:
        u = unit.lower()
        secs = None
        for key, mult in _UNIT_SECONDS.items():
            if u == key or (len(key) > 1 and u.startswith(key)):
                secs = int(num) * mult
                break
        if secs is None:
            secs = int(num) * _UNIT_SECONDS[u[0]]
        total += secs
        parts.append(f"{num}{u[0]}")
    if total <= 0:
        raise ValueError("delay must be positive")
    dt = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=total)
    return dt, "".join(parts)


def _load_tasks() -> list:
    data = _read_json(_tasks_db())
    return data if isinstance(data, list) else []


def _save_tasks(tasks: list) -> None:
    _atomic_write_json(_tasks_db(), tasks)


def _mint_task_id(tasks: list) -> str:
    existing = {t["id"] for t in tasks}
    while True:
        tid = "t_" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(6))
        if tid not in existing:
            return tid


# ---------------------------------------------------------------------------
# NOTIFICATION WORKER - persistent tasks must actually RING.
# Polls due pending tasks, fires notify-send once each (idempotent via
# 'notified' flag), survives restarts (overdue items fire immediately).
# ---------------------------------------------------------------------------
_notify_lock = threading.Lock()
_worker_started = False


def _poll_notifications_once(now=None):
    """Fires notifications for every due, unnotified, pending task. Returns fired ids."""
    now = now or datetime.datetime.now().astimezone()
    fired = []
    with _notify_lock:
        tasks = _load_tasks()
        dirty = False
        for t in tasks:
            if t.get("status") != "pending" or t.get("notified"):
                continue
            try:
                due_dt = datetime.datetime.fromisoformat(t["due_at"])
            except Exception:
                continue
            if due_dt <= now:
                overdue_min = int((now - due_dt).total_seconds() // 60)
                body = t["task"] + (f"  ({overdue_min}m overdue)" if overdue_min >= 10 else "")
                try:
                    subprocess.run(["notify-send", "-a", "NanoHat", "Reminder", body],
                                   shell=False, capture_output=True, check=False, timeout=10)
                    fired.append(t["id"])
                except Exception:
                    continue  # transient failure: retry next poll
                t["notified"] = True
                dirty = True
        if dirty:
            _save_tasks(tasks)
    return fired


def start_notification_worker(interval_seconds: int = 20):
    """Starts the background polling loop (idempotent). Safe to call from any runtime."""
    global _worker_started
    with _notify_lock:
        if _worker_started:
            return False
        _worker_started = True

    def _loop():
        while True:
            try:
                _poll_notifications_once()
            except Exception:
                pass
            time.sleep(max(5, int(interval_seconds)))

    threading.Thread(target=_loop, daemon=True, name="nanohat-notifier").start()
    return True


def _fmt_task(t: dict, now=None) -> str:
    line = f"[{t['id']}] {t['status']} {t['due_at']} - {t['task']}"
    if now and t.get("status") == "pending":
        try:
            if datetime.datetime.fromisoformat(t["due_at"]) < now:
                line += " (overdue)"
        except Exception:
            pass
    return line


@tool
def scheduler(action: str, task: str = "", due: str = "", id: str = "",
              status: str = "", range: str = "") -> str:
    """Creates, lists, updates, or deletes persistent scheduled tasks.

    Args:
        action: Task operation ('set', 'list', 'update', 'delete').
        task: Task description (required for 'set').
        due: When it is due ('YYYY-MM-DD HH:MM' or relative like '10m', 'in 20 minutes'). Required for 'set'.
        id: Backend-provided task id (required for 'update'/'delete').
        status: Filter for 'list' ('pending','done','cancelled','any'); new state for 'update'.
        range: Time window for 'list' ('today','tomorrow','yesterday','week','all').
    """
    action = str(action).strip().lower()

    if action == "set":
        if not str(task).strip():
            return _err("missing_task", "a task description is required.")
        try:
            due_dt, rel = parse_due(due)
        except ValueError as e:
            return _err("bad_due", str(e))
        with _tasks_lock:
            tasks = _load_tasks()
            tid = _mint_task_id(tasks)
            entry = {
                "id": tid,
                "task": str(task).strip(),
                "due_at": due_dt.isoformat(timespec="seconds"),
                "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "status": "pending",
                "completed_at": None,
                "notified": False,
            }
            tasks.append(entry)
            _save_tasks(tasks)
        when = f"due {entry['due_at']}"
        if rel:
            when += f" (in {rel})"
        return _ok(f"Task scheduled: [{tid}] '{entry['task']}' {when}.")

    if action == "list":
        status_filter = (status or "").strip().lower() or "pending"
        range_name = (range or "all").strip().lower()
        now = datetime.datetime.now().astimezone()
        with _tasks_lock:
            tasks = sorted(_load_tasks(), key=lambda t: t.get("due_at", ""))

        lo = hi = None
        today0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if range_name == "today":
            lo, hi = today0, today0 + datetime.timedelta(days=1)
        elif range_name == "tomorrow":
            lo, hi = today0 + datetime.timedelta(days=1), today0 + datetime.timedelta(days=2)
        elif range_name == "yesterday":
            lo, hi = today0 - datetime.timedelta(days=1), today0
        elif range_name == "week":
            lo, hi = now, now + datetime.timedelta(days=7)

        shown = []
        for t in tasks:
            if status_filter != "any" and t.get("status", "pending") != status_filter:
                continue
            try:
                due_dt = datetime.datetime.fromisoformat(t["due_at"])
            except Exception:
                continue
            if lo and not (lo <= due_dt < hi):
                continue
            shown.append(_fmt_task(t, now))

        scope = "all pending" if range_name == "all" else f"{range_name} ({status_filter})"
        if not shown:
            return _ok(f"No {scope} tasks found.")
        return _ok(f"{len(shown)} task(s) found:\n" + "\n".join(shown))

    if action == "update":
        tid = str(id).strip()
        if not tid:
            return _err("missing_id", "a task id is required to update a task.")
        new_status = (status or "").strip().lower()
        if new_status and new_status not in ("pending", "done", "cancelled"):
            return _err("bad_status", "status must be 'pending', 'done', or 'cancelled'.")
        new_due = None
        if str(due).strip():
            try:
                new_due, _rel = parse_due(due)
            except ValueError as e:
                return _err("bad_due", str(e))
        new_task = str(task).strip()
        with _tasks_lock:
            tasks = _load_tasks()
            target = next((t for t in tasks if t["id"] == tid), None)
            if not target:
                return _err("not_found", f"No task with id '{tid}'.")
            if not new_status and not new_due and not new_task:
                return _err("nothing_to_update", "provide a new 'task', 'due', or 'status'.")
            changes = []
            if new_task:
                target["task"] = new_task
                changes.append("task text")
            if new_due:
                target["due_at"] = new_due.isoformat(timespec="seconds")
                changes.append(f"due={target['due_at']}")
            if new_status:
                target["status"] = new_status
                target["completed_at"] = (
                    datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                    if new_status == "done" else None)
                if new_status != "pending":
                    target["notified"] = True
                changes.append(f"status={new_status}")
            _save_tasks(tasks)
        return _ok(f"Task [{tid}] updated ({', '.join(changes)}).")

    if action == "delete":
        tid = str(id).strip()
        if not tid:
            return _err("missing_id", "a task id is required to delete a task.")
        with _tasks_lock:
            tasks = _load_tasks()
            remaining = [t for t in tasks if t["id"] != tid]
            if len(remaining) == len(tasks):
                return _err("not_found", f"No task with id '{tid}'.")
            _save_tasks(remaining)
        return _ok(f"Task [{tid}] deleted.")

    return _err("bad_action", "Invalid action. Choose 'set', 'list', 'update', or 'delete'.")


# ---------------------------------------------------------------------------
# 5. SYSTEM HEALTH TOOL
# ---------------------------------------------------------------------------
@tool
def system_health(target: str) -> str:
    """Inspects Fedora Linux subsystem metrics and resource consumption.

    Args:
        target: Subsystem to check ('cpu', 'ram', 'disk', 'network', 'battery', 'top_processes', 'all').
    """
    target = str(target).strip().lower()
    try:
        if target == "cpu":
            cpu_pct = psutil.cpu_percent(interval=0.3)
            return _ok(f"CPU Utilization: {cpu_pct}%")
        if target == "ram":
            mem = psutil.virtual_memory()
            return _ok(f"RAM: {mem.used / (1024**3):.1f}GB used / {mem.total / (1024**3):.1f}GB total ({mem.percent}% utilization)")
        if target == "disk":
            disk = psutil.disk_usage('/')
            return _ok(f"Disk /: {disk.used / (1024**3):.1f}GB used / {disk.total / (1024**3):.1f}GB total ({disk.percent}% used)")
        if target == "network":
            res = subprocess.run(["nmcli", "device", "status"], shell=False, capture_output=True,
                                 text=True, check=False, timeout=10)
            body = res.stdout.strip() if res.stdout else "Network: Interface online."
            return _ok(body)
        if target == "battery":
            bat = getattr(psutil, "sensors_battery", lambda: None)()
            if bat:
                return _ok(f"Battery: {bat.percent}%, {'Charging' if bat.power_plugged else 'Discharging'}")
            return _ok("Battery: AC Connected (No battery detected)")
        if target == "top_processes":
            res = subprocess.run(["ps", "aux", "--sort=-%cpu"], shell=False, capture_output=True,
                                 text=True, check=False, timeout=10)
            # Drop the ps command's own listing artifact from the output
            lines = [l for l in res.stdout.splitlines()
                     if "--sort=-%cpu" not in l and l.strip()]
            return _ok("\n".join(lines[:6]))
        if target == "all":
            cpu = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            return _ok(f"System Overview: CPU {cpu}%, RAM {mem.percent}%, Disk {disk.percent}%")
    except subprocess.TimeoutExpired:
        return _err("timeout", f"'{target}' inspection timed out.")
    except Exception as e:
        return _err("health_check_failed", f"could not inspect '{target}' ({e}).")
    return _err("bad_target", f"Unknown target '{target}'. Allowed: cpu, ram, disk, network, battery, top_processes, all.")


# ---------------------------------------------------------------------------
# 6. SYSTEM ACTION TOOL (fail-closed gate + safe execution)
# ---------------------------------------------------------------------------
DESTRUCTIVE_ACTIONS = {"kill_process", "restart_service", "empty_trash"}
_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _confirm_destructive(action: str, target: str) -> bool:
    """Fail-closed confirmation for destructive actions.

    - Interactive TTY: prompts the user ([y/N]).
    - Non-interactive/headless: REFUSES unless NANOHAT_AUTO_APPROVE_DESTRUCTIVE
      is explicitly set to 1/true/yes (for trusted daemon deployments).
    """
    if os.environ.get("NANOHAT_AUTO_APPROVE_DESTRUCTIVE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
        answer = input(f"\n⚠️  AGENT REQUEST: {action}"
                       f"{f' on {chr(39)}{target}{chr(39)}' if target else ''}. Confirm? [y/N]: ").strip().lower()
        return answer == "y"
    except Exception:
        return False


def _find_process_matches(target: str):
    """Returns (exact, partial) process lists for the given name query."""
    q = target.lower()
    exact, partial = [], []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            if not name:
                continue
            if name == q:
                exact.append(proc)
                continue
            if q in name:
                partial.append(proc)
                continue
            cmdline = " ".join(info.get("cmdline") or []).lower()
            if cmdline and q in cmdline:
                partial.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return exact, partial


def _safe_kill(target: str) -> str:
    target = str(target).strip()
    if not target or not _NAME_RE.match(target):
        return _err("invalid_name", "Invalid or dangerous process name.")
    exact, partial = _find_process_matches(target)

    victims = exact if exact else []
    if not victims:
        if partial:
            lines = []
            for p in partial[:5]:
                try:
                    lines.append(f"PID {p.pid}: {p.name()} - {' '.join(p.cmdline())[:80]}")
                except Exception:
                    continue
            hint = "\n".join(lines)
            more = f"\n(+{len(partial) - 5} more)" if len(partial) > 5 else ""
            return _err("ambiguous",
                        f"No process exactly named '{target}', but {len(partial)} partial match(es) found:\n"
                        f"{hint}{more}\nSpecify the exact process name to kill.")
        return _err("not_found", f"No process matching '{target}' found.")

    killed = []
    for p in victims:
        try:
            p.terminate()
            p.wait(timeout=3)
            killed.append(f"PID {p.pid}")
        except psutil.TimeoutExpired:
            try:
                p.kill()
                killed.append(f"PID {p.pid} (force killed)")
            except Exception:
                pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        return _ok(f"Terminated process '{target}' ({', '.join(killed)}).")
    return _err("kill_failed", f"Could not terminate '{target}'.")


def _safe_restart_service(target: str) -> str:
    target = str(target).strip()
    if not target or not _NAME_RE.match(target):
        return _err("invalid_name", "Invalid service name.")
    res = subprocess.run(["systemctl", "--user", "restart", target], shell=False,
                         capture_output=True, text=True, check=False, timeout=30)
    if res.returncode == 0:
        return _ok(f"Service '{target}' restarted.")
    detail = (res.stderr or "").strip().splitlines()
    reason = detail[-1] if detail else "systemctl reported failure"
    return _err("service_error", f"Could not restart '{target}': {reason}")


def _toggle_bluetooth() -> str:
    res = subprocess.run(["bluetoothctl", "show"], shell=False, capture_output=True,
                         text=True, check=False, timeout=10)
    if "Powered: yes" in res.stdout:
        subprocess.run(["bluetoothctl", "power", "off"], shell=False, capture_output=True,
                       check=False, timeout=10)
        return _ok("Bluetooth disabled.")
    subprocess.run(["bluetoothctl", "power", "on"], shell=False, capture_output=True,
                   check=False, timeout=10)
    return _ok("Bluetooth enabled.")


def _toggle_wifi() -> str:
    subprocess.run(["nmcli", "radio", "wifi", "toggle"], shell=False, capture_output=True,
                   check=False, timeout=10)
    status_res = subprocess.run(["nmcli", "radio", "wifi"], shell=False, capture_output=True,
                                text=True, check=False, timeout=10)
    state = status_res.stdout.strip() or "unknown"
    return _ok(f"WiFi is now {state}.")


@tool
def system_action(action: str, target: str = "") -> str:
    """Safely executes predefined OS desktop actions, launches apps, or queries system state.

    Args:
        action: Safe keyword ('kill_process', 'restart_service', 'toggle_wifi', 'toggle_bluetooth', 'empty_trash', 'lock_screen', 'take_screenshot', 'launch_app', 'get_datetime').
        target: Target process name, service name, or app name (e.g. 'firefox', 'NetworkManager').
    """
    action = str(action).strip().lower()
    target = str(target).strip()

    # Fail-closed destructive confirmation gate
    if action in DESTRUCTIVE_ACTIONS:
        if not _confirm_destructive(action, target):
            return _err("not_confirmed",
                        f"Destructive action '{action}' requires interactive user confirmation "
                        f"(or NANOHAT_AUTO_APPROVE_DESTRUCTIVE=1 for headless deployments). "
                        f"Nothing was changed.")

    try:
        if action == "get_datetime":
            now = datetime.datetime.now().astimezone()
            stamp = now.strftime("%Y-%m-%d %H:%M:%S %Z%z").rstrip() or now.isoformat(timespec="seconds")
            return _ok(stamp)

        if action == "launch_app":
            url_match = re.match(r"^https?://[^\s]+$", target)
            if url_match:
                res = subprocess.run(["xdg-open", target], shell=False, capture_output=True,
                                     check=False, timeout=15)
                if res.returncode == 0:
                    return _ok(f"Opened {target} in your browser.")
                return _err("launch_failed", f"Could not open '{target}'.")
            if not target or not _NAME_RE.match(target):
                return _err("invalid_name", "Invalid application name.")
            res = subprocess.run(["gtk-launch", target], shell=False, capture_output=True,
                                 check=False, timeout=15)
            if res.returncode == 0:
                return _ok(f"Launched application '{target}'.")
            return _err("launch_failed", f"Application '{target}' not found in desktop database.")

        if action == "kill_process":
            return _safe_kill(target)

        if action == "restart_service":
            return _safe_restart_service(target)

        if action == "toggle_wifi":
            return _toggle_wifi()

        if action == "toggle_bluetooth":
            return _toggle_bluetooth()

        if action == "empty_trash":
            res = subprocess.run(["gio", "trash", "--empty"], shell=False, capture_output=True,
                                 check=False, timeout=60)
            if res.returncode == 0:
                return _ok("Trash emptied.")
            return _err("trash_error", "gio failed to empty the trash.")

        if action == "lock_screen":
            res = subprocess.run(["loginctl", "lock-session"], shell=False, capture_output=True,
                                 check=False, timeout=10)
            if res.returncode == 0:
                return _ok("Desktop session locked.")
            return _err("lock_failed", "Could not lock the session (loginctl failed).")

        if action == "take_screenshot":
            pics = os.path.expanduser("~/Pictures")
            os.makedirs(pics, exist_ok=True)
            shot_path = os.path.join(
                pics,
                "Screenshot_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
            backends = (
                [["gnome-screenshot", "-f", shot_path]],
                [["spectacle", "-b", "-n", "-o", shot_path]],
                [["grim", shot_path]],
            )
            for cmd in backends:
                try:
                    res = subprocess.run(cmd[0], shell=False, capture_output=True,
                                         check=False, timeout=20)
                    if res.returncode == 0 and os.path.exists(shot_path):
                        return _ok(f"Screenshot saved to {shot_path}.")
                except FileNotFoundError:
                    continue
            return _err("backend_unavailable",
                        "No screenshot backend found (gnome-screenshot/spectacle/grim).")

    except subprocess.TimeoutExpired:
        return _err("timeout", f"Action '{action}' timed out.")
    except Exception as e:
        return _err("execution_failed", f"Action '{action}' failed unexpectedly ({e}).")

    return _err("bad_action", f"Unknown system action '{action}'. Allowed: kill_process, restart_service, "
                              f"toggle_wifi, toggle_bluetooth, empty_trash, lock_screen, take_screenshot, "
                              f"launch_app, get_datetime.")
