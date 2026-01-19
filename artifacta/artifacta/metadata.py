"""Metadata capture (git, environment, system)."""

import getpass
import os
import platform
import socket
import subprocess
import sys

import psutil


def capture_metadata():
    """Capture all metadata like W&B does."""
    return {
        "git": _capture_git(),
        "environment": _capture_environment(),
        "system": _capture_system(),
        "dependencies": _capture_dependencies(),
    }


def _capture_git():
    """Git information."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
            )
            .decode()
            .strip()
        )

        remote = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )

        dirty = (
            len(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5
                )
                .decode()
                .strip()
            )
            > 0
        )

        return {"commit": commit, "remote": remote, "dirty": dirty}
    except Exception:
        return None


def _capture_environment():
    """Environment information."""
    return {
        "hostname": socket.gethostname(),
        "username": getpass.getuser(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv) if sys.argv else None,
    }


def _capture_system():
    """System hardware."""
    system = {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }

    # GPU info (optional)
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        system["gpus"] = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            system["gpus"].append(
                {
                    "id": i,
                    "name": pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                    if isinstance(pynvml.nvmlDeviceGetName(handle), bytes)
                    else pynvml.nvmlDeviceGetName(handle),
                    "memory_total_gb": round(
                        pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3, 2
                    ),
                }
            )
        pynvml.nvmlShutdown()
    except Exception:
        pass  # No GPU or pynvml not installed

    return system


def _capture_dependencies():
    """Installed packages."""
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        return {"pip_freeze": pip_freeze}
    except Exception:
        return None
