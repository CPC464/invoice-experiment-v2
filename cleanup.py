#!/usr/bin/env python3
"""
Cleanup script to terminate any hanging Flask processes on port 5002.
Run this if you see multiple processes running on the same port.
"""

import os
import subprocess
import signal
import sys


def find_processes_by_port(port):
    """Find processes using the specified port"""
    try:
        # Different commands based on OS
        if sys.platform == "darwin":  # macOS
            cmd = ["lsof", "-i", f":{port}"]
        elif sys.platform.startswith("linux"):  # Linux
            cmd = ["lsof", "-i", f":{port}"]
        elif sys.platform == "win32":  # Windows
            cmd = ["netstat", "-ano", "|", "findstr", f":{port}"]
        else:
            print(f"Unsupported platform: {sys.platform}")
            return []

        output = subprocess.check_output(cmd, universal_newlines=True)
        lines = output.strip().split("\n")

        # Skip header line
        if len(lines) <= 1:
            return []

        processes = []

        # Parse output based on OS
        if sys.platform in ("darwin", "linux"):
            # Skip the header line
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    process = {"command": parts[0], "pid": int(parts[1])}
                    processes.append(process)
        elif sys.platform == "win32":
            for line in lines:
                if f":{port}" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        process = {"command": "Unknown", "pid": int(parts[4])}
                        processes.append(process)

        return processes
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        print(f"Error while finding processes: {e}")
        return []


def terminate_process(pid):
    """Terminate a process by PID"""
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Successfully terminated process {pid}")
        return True
    except Exception as e:
        print(f"Failed to terminate process {pid}: {e}")
        return False


def main():
    port = 5002
    print(f"Searching for processes using port {port}...")

    processes = find_processes_by_port(port)

    if not processes:
        print(f"No processes found using port {port}")
        return

    print(f"Found {len(processes)} processes using port {port}:")
    for i, process in enumerate(processes):
        print(f"{i+1}. PID: {process['pid']}, Command: {process['command']}")

    confirm = input("\nDo you want to terminate all these processes? (y/n): ")
    if confirm.lower() != "y":
        print("Operation cancelled.")
        return

    success_count = 0
    for process in processes:
        if terminate_process(process["pid"]):
            success_count += 1

    print(f"Terminated {success_count} out of {len(processes)} processes.")


if __name__ == "__main__":
    main()
