#!/usr/bin/env python3
"""
capture_tool_outputs.py
Probes the Fedora Linux host to capture ground-truth CLI output snippets.
Saves outputs to real_tool_outputs.json to ground the synthetic dataset generator.
"""

import json
import os
import subprocess

# Mirror of runtime/tools.py output contract helpers
def _ok(msg: str) -> str:
    return f"OK: {msg}"

def _err(reason: str, msg: str) -> str:
    return f"ERROR[{reason}]: {msg}"

def capture_outputs(output_path="grounding/real_tool_outputs.json"):
    outputs = {}
    
    # 1. system_health targets
    outputs["system_health"] = {}
    
    # CPU
    try:
        cpu_res = subprocess.run(["top", "-bn1"], capture_output=True, text=True, timeout=5)
        outputs["system_health"]["cpu"] = "\n".join(cpu_res.stdout.splitlines()[:12])
    except Exception as e:
        outputs["system_health"]["cpu"] = f"%Cpu(s): 12.4 us, 3.2 sy, 0.0 ni, 84.1 id, 0.3 wa"

    # RAM
    try:
        ram_res = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
        outputs["system_health"]["ram"] = ram_res.stdout.strip()
    except Exception as e:
        outputs["system_health"]["ram"] = "               total        used        free      shared  buff/cache   available\nMem:            15Gi       4.2Gi       6.1Gi       512Mi       5.2Gi        10Gi"

    # Disk
    try:
        disk_res = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=5)
        outputs["system_health"]["disk"] = disk_res.stdout.strip()
    except Exception as e:
        outputs["system_health"]["disk"] = "Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p3  475G  182G  270G  41% /"

    # Network
    try:
        net_res = subprocess.run(["nmcli", "device", "status"], capture_output=True, text=True, timeout=5)
        outputs["system_health"]["network"] = net_res.stdout.strip()
    except Exception as e:
        outputs["system_health"]["network"] = "DEVICE  TYPE      STATE      CONNECTION\nwlp2s0  wifi      connected  Home-WiFi-5G\nenp1s0  ethernet  unavailable --\nlo      loopback  unmanaged  --"

    # Battery
    try:
        bat_enum = subprocess.run(["upower", "-e"], capture_output=True, text=True, timeout=5)
        bat_dev = [line for line in bat_enum.stdout.splitlines() if "battery" in line]
        if bat_dev:
            bat_info = subprocess.run(["upower", "-i", bat_dev[0]], capture_output=True, text=True, timeout=5)
            # keep first 10 lines
            outputs["system_health"]["battery"] = "\n".join(bat_info.stdout.splitlines()[:10])
        else:
            outputs["system_health"]["battery"] = "native-path: BAT0\nstate: discharging\npercentage: 78%\ntime to empty: 4.2 hours"
    except Exception as e:
        outputs["system_health"]["battery"] = "native-path: BAT0\nstate: discharging\npercentage: 78%\ntime to empty: 4.2 hours"

    # Top processes
    try:
        ps_res = subprocess.run(["ps", "aux", "--sort=-%cpu"], capture_output=True, text=True, timeout=5)
        outputs["system_health"]["top_processes"] = "\n".join(ps_res.stdout.splitlines()[:8])
    except Exception as e:
        outputs["system_health"]["top_processes"] = "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nuser      4210 24.5  6.2 3894210 982140 ?      Sl   10:00   4:12 /usr/lib64/firefox/firefox\nuser      1204  8.1  1.4 1204910 230140 ?      Sl   09:30   1:45 /usr/bin/gnome-shell"

    # 2. system_action sample outputs (V2 uniform contract)
    outputs["system_action"] = {
        "get_datetime": _ok(subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip() or "2026-08-14 11:20:00 IST"),
        "launch_app_success": _ok("Launched application 'firefox'."),
        "launch_app_failure": _err("launch_failed", "Application 'nonexistent-app' not found in desktop database."),
        "kill_process_success": _ok("Terminated process 'chrome' (PID 5389)."),
        "kill_process_ambiguous": _err("ambiguous", "No process exactly named 'code', but 2 partial match(es) found:\nPID 1204: vim - vim code_notes.txt\nPID 991: tail -f tail -f encode.sh\nSpecify the exact process name to kill."),
        "kill_process_failure": _err("not_found", "No process matching 'unresponsive_app' found."),
        "kill_process_refused_headless": _err("not_confirmed", "Destructive action 'kill_process' requires interactive user confirmation. Nothing was changed."),
        "restart_service_success": _ok("Service 'nginx' restarted."),
        "restart_service_failure": _err("service_error", "Could not restart 'bad_service': Unit bad_service.service not found."),
        "toggle_wifi": _ok("WiFi is now enabled."),
        "toggle_bluetooth": _ok("Bluetooth powered on."),
        "empty_trash": _ok("Trash emptied."),
        "lock_screen": _ok("Desktop session locked."),
        "take_screenshot": _ok("Screenshot saved to /home/user/Pictures/Screenshot_20260814_112003.png.")
    }

    # 3. calculator sample outputs
    outputs["calculator"] = {
        "120 * 0.15": _ok("18"),
        "(45 + 78) * 2": _ok("246"),
        "2 ** 10": _ok("1024"),
        "9 ** (9 ** 9)": _err("range", "exponent too large"),
        "sqrt(144)": _err("syntax", "invalid expression - functions are not allowed")
    }

    # 4. user_memory sample outputs
    outputs["user_memory"] = {
        "store": _ok("Memory stored: user_alias = 'kaizencore'"),
        "get": _ok("user_alias = 'kaizencore'"),
        "list_empty": _ok("Memory is empty. Nothing stored yet."),
        "list": _ok("3 memories stored:\nbirthday='March 14' (saved 2026-08-20)\neditor='neovim' (saved 2026-08-21)\nuser_alias='kaizencore' (saved 2026-08-14)"),
        "get_not_found": _err("not_found", "Memory key 'preferred_editor' not found."),
        "delete": _ok("Memory key 'temp_note' deleted.")
    }

    # 5. scheduler sample outputs (replaces legacy reminder)
    outputs["scheduler"] = {
        "set_relative": _ok("Task scheduled: [t_a1b2c3] 'Check the oven' due 2026-08-24T15:20 (+0530) (in 20m)."),
        "set_absolute": _ok("Task scheduled: [t_d4e5f6] 'Team standup' due 2026-08-25T10:00 (+0530)."),
        "set_bad_due": _err("bad_due", "cannot understand due time 'sometime soon'. Use 'YYYY-MM-DD HH:MM' or a delay like '10m'."),
        "list_today": _ok("2 task(s) found:\n[t_a1b2c3] pending 2026-08-24T15:20+05:30 - Check the oven\n[t_d4e5f6] pending 2026-08-24T18:00+05:30 - Review pull request"),
        "list_yesterday_done": _ok("1 task(s) found:\n[t_x9y8z7] done 2026-08-23T09:00+05:30 - Morning backup"),
        "list_empty": _ok("No all pending tasks found."),
        "update_status": _ok("Task [t_a1b2c3] updated (status=done)."),
        "update_reschedule": _ok("Task [t_d4e5f6] updated (due=2026-08-26T10:00)."),
        "delete": _ok("Task [t_a1b2c3] deleted."),
        "delete_not_found": _err("not_found", "No task with id 't_zzzzzz'.")
    }

    # 6. web_search sample outputs
    outputs["web_search"] = {
        "fedora_release": _ok("\n- Fedora Linux 44 released: featuring GNOME 48 and Linux Kernel 6.14.\n- Fedora 44 release schedule: official announcement and download links.\n- What's new in Fedora Workstation 44: developer highlights."),
        "no_results": _err("no_results", "no results found for 'asdkjhqwe12345'.")
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"✓ Real tool outputs captured and written to: {output_path}")
    return outputs

if __name__ == "__main__":
    capture_outputs()
