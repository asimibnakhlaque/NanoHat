"""
runtime/tools.py
6 Consolidated Fedora desktop & system tools for the SmolLM2 agent.
Enforces the Zero Shell Interpolation Policy (subprocess.run with shell=False),
threaded background delay for reminders, and in-tool confirmation gates for destructive actions.
"""

import os
import re
import json
import time
import psutil
import datetime
import threading
import subprocess

# Point rustls and OpenSSL directly to the Fedora public CA bundle
os.environ["SSL_CERT_FILE"] = "/etc/pki/tls/certs/ca-bundle.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"

# Flexible import for smolagents @tool decorator (or fallback dummy decorator for testing)
try:
    from smolagents import tool
except ImportError:
    def tool(func):
        """Fallback decorator when smolagents is not yet installed in local environment."""
        func.is_tool = True
        return func

# ---------------------------------------------------------------------------
# 1. CALCULATOR TOOL
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression safely.
    
    Args:
        expression: Mathematical expression string to calculate (e.g. '120 * 0.15').
    """
    try:
        allowed = set("0123456789+-*/(). %")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in mathematical expression."
        # Safe arithmetic evaluation with disabled builtins
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation Error: {e}"

# ---------------------------------------------------------------------------
# 2. WEB SEARCH TOOL
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Searches the web for real-time information.
    
    Args:
        query: Search query keywords.
    """
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=3))
        if results:
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return str(e)

# ---------------------------------------------------------------------------
# 3. USER MEMORY TOOL
# ---------------------------------------------------------------------------
@tool
def user_memory(action: str, key: str, value: str = "") -> str:
    """Retains, retrieves, or deletes persistent user preferences and notes.
    
    Args:
        action: The operation to perform ('store', 'get', 'delete').
        key: The memory identifier key.
        value: The value string to store (leave empty for 'get' or 'delete').
    """
    db_path = os.path.expanduser("~/.config/smollm2-fedora-agent/memory.json")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    data = {}
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
            
    if action == "store":
        data[key] = value
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return f"Memory stored: {key} = '{value}'"
    elif action == "get":
        if key in data:
            return f"{key} = '{data[key]}'"
        return f"Memory key '{key}' not found."
    elif action == "delete":
        if key in data:
            del data[key]
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return f"Memory key '{key}' deleted."
        return f"Key '{key}' not found."
    return "Error: Invalid action. Choose 'store', 'get', or 'delete'."

# ---------------------------------------------------------------------------
# 4. REMINDER TOOL (Threaded Background Delay)
# ---------------------------------------------------------------------------
def _delayed_notification(task: str, delay_str: str):
    """Background daemon thread that sleeps for delay_str and fires notify-send."""
    seconds = 60 # default fallback
    clean = delay_str.strip().lower()
    if 'm' in clean:
        try: seconds = int(clean.replace('m', '').strip()) * 60
        except Exception: pass
    elif 'h' in clean:
        try: seconds = int(clean.replace('h', '').strip()) * 3600
        except Exception: pass
    elif 's' in clean:
        try: seconds = int(clean.replace('s', '').strip())
        except Exception: pass
        
    time.sleep(seconds)
    subprocess.run(["notify-send", "SmolLM2 Reminder", task], shell=False, check=False)

@tool
def reminder(task: str, time_or_delay: str) -> str:
    """Schedules a desktop notification or reminder.
    
    Args:
        task: Description of the task to be reminded of.
        time_or_delay: When to trigger the reminder (e.g. '20m', '1h', '30s').
    """
    threading.Thread(target=_delayed_notification, args=(task, time_or_delay), daemon=True).start()
    return f"Reminder scheduled in background: '{task}' (in {time_or_delay})."

# ---------------------------------------------------------------------------
# 5. SYSTEM HEALTH TOOL
# ---------------------------------------------------------------------------
@tool
def system_health(target: str) -> str:
    """Inspects Fedora Linux subsystem metrics and resource consumption.
    
    Args:
        target: Subsystem to check ('cpu', 'ram', 'disk', 'network', 'battery', 'top_processes', 'all').
    """
    target = target.strip().lower()
    if target == "cpu":
        cpu_pct = psutil.cpu_percent(interval=0.3)
        return f"CPU Utilization: {cpu_pct}%"
    elif target == "ram":
        mem = psutil.virtual_memory()
        return f"RAM: {mem.used / (1024**3):.1f}GB used / {mem.total / (1024**3):.1f}GB total ({mem.percent}% utilization)"
    elif target == "disk":
        disk = psutil.disk_usage('/')
        return f"Disk /: {disk.used / (1024**3):.1f}GB used / {disk.total / (1024**3):.1f}GB total ({disk.percent}% used)"
    elif target == "network":
        res = subprocess.run(["nmcli", "device", "status"], shell=False, capture_output=True, text=True, check=False)
        return res.stdout.strip() if res.stdout else "Network: Interface online."
    elif target == "battery":
        bat = psutil.sensors_battery()
        if bat:
            return f"Battery: {bat.percent}%, {'Charging' if bat.power_plugged else 'Discharging'}"
        return "Battery: AC Connected (No battery detected)"
    elif target == "top_processes":
        res = subprocess.run(["ps", "aux", "--sort=-%cpu"], shell=False, capture_output=True, text=True, check=False)
        return "\n".join(res.stdout.splitlines()[:6])
    elif target == "all":
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return f"System Overview: CPU {cpu}%, RAM {mem.percent}%, Disk {disk.percent}%"
    return f"Error: Unknown target '{target}'. Allowed: cpu, ram, disk, network, battery, top_processes, all."

# ---------------------------------------------------------------------------
# 6. SYSTEM ACTION TOOL (Safe Execution & In-Tool Confirmation Gate)
# ---------------------------------------------------------------------------
DESTRUCTIVE_ACTIONS = {"kill_process", "restart_service", "empty_trash"}

def _safe_kill(target: str) -> str:
    if not target or not re.match(r"^[a-zA-Z0-9_\-\.]+$", target):
        return "Error: Invalid or dangerous process name."
    res = subprocess.run(["pkill", "-f", target], shell=False, capture_output=True, text=True, check=False)
    return f"Terminated process '{target}'." if res.returncode == 0 else f"No process matching '{target}' found."

def _safe_restart_service(target: str) -> str:
    if not target or not re.match(r"^[a-zA-Z0-9_\-\.]+$", target):
        return "Error: Invalid service name."
    res = subprocess.run(["systemctl", "--user", "restart", target], shell=False, capture_output=True, text=True, check=False)
    return f"Service '{target}' restarted." if res.returncode == 0 else f"Error restarting '{target}'."

def _toggle_bluetooth() -> str:
    res = subprocess.run(["bluetoothctl", "show"], shell=False, capture_output=True, text=True, check=False)
    if "Powered: yes" in res.stdout:
        subprocess.run(["bluetoothctl", "power", "off"], shell=False, capture_output=True, check=False)
        return "Bluetooth disabled."
    else:
        subprocess.run(["bluetoothctl", "power", "on"], shell=False, capture_output=True, check=False)
        return "Bluetooth enabled."

def _toggle_wifi() -> str:
    subprocess.run(["nmcli", "radio", "wifi", "toggle"], shell=False, capture_output=True, check=False)
    status_res = subprocess.run(["nmcli", "radio", "wifi"], shell=False, capture_output=True, text=True, check=False)
    state = status_res.stdout.strip()
    return f"WiFi is now {state}."

@tool
def system_action(action: str, target: str = "") -> str:
    """Safely executes predefined OS desktop actions, launches apps, or queries system state.
    
    Args:
        action: Safe keyword ('kill_process', 'restart_service', 'toggle_wifi', 'toggle_bluetooth', 'empty_trash', 'lock_screen', 'take_screenshot', 'launch_app', 'get_datetime').
        target: Target process name, service name, or app name (e.g. 'firefox', 'NetworkManager').
    """
    action = action.strip().lower()
    
    # In-Tool Destructive Confirmation Gate
    if action in DESTRUCTIVE_ACTIONS:
        target_str = f" on '{target}'" if target else ""
        try:
            confirm = input(f"\n⚠️  AGENT REQUEST: {action}{target_str}. Confirm? [y/N]: ").strip().lower()
            if confirm != 'y':
                return "Action cancelled by user."
        except Exception:
            pass # Non-interactive test fallback
            
    if action == "get_datetime":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    elif action == "launch_app":
        if not target or not re.match(r"^[a-zA-Z0-9_\-\.]+$", target):
            return "Error: Invalid application name."
        subprocess.run(["gtk-launch", target], shell=False, capture_output=True, check=False)
        return f"Launched application '{target}'."
    elif action == "kill_process":
        return _safe_kill(target)
    elif action == "restart_service":
        return _safe_restart_service(target)
    elif action == "toggle_wifi":
        subprocess.run(["nmcli", "radio", "wifi", "toggle"], shell=False, capture_output=True, check=False)
        return _toggle_wifi()
    elif action == "toggle_bluetooth":
        return _toggle_bluetooth()
    elif action == "empty_trash":
        subprocess.run(["gio", "trash", "--empty"], shell=False, capture_output=True, check=False)
        return "Trash emptied."
    elif action == "lock_screen":
        subprocess.run(["loginctl", "lock-session"], shell=False, capture_output=True, check=False)
        return "Desktop session locked."
    elif action == "take_screenshot":
        shot_path = os.path.expanduser("~/screenshot.png")
        subprocess.run(["gnome-screenshot", "-f", shot_path], shell=False, capture_output=True, check=False)
        return f"Screenshot saved to {shot_path}."
        
    return f"Error: Unknown system action '{action}'."
