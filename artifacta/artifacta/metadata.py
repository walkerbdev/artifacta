"""Comprehensive metadata capture for run reproducibility.

This module captures a complete snapshot of the execution environment at run
initialization, similar to Weights & Biases. The metadata enables reproducibility
by recording exactly what system, dependencies, and git state were used for a run.

Captured Metadata Categories:
    1. Git: Commit hash, remote URL, branch, dirty status, diff
    2. Environment: Hostname, username, Python version, platform, CWD, command
    3. System: CPU count, memory, GPU info (name, memory per GPU)
    4. Dependencies: pip freeze output (all installed packages with versions)

Metadata Structure:
    All metadata is organized in a hierarchical dictionary:
    {
        "git": {...},
        "environment": {...},
        "system": {...},
        "dependencies": {...}
    }

Git Metadata Algorithm:
    1. Run 'git rev-parse HEAD' to get commit hash
    2. Run 'git config --get remote.origin.url' to get remote URL
    3. Run 'git status --porcelain' to check for uncommitted changes
    4. If uncommitted changes exist (dirty=True), capture full diff
    5. All commands use 5-second timeout and redirect stderr to DEVNULL
    6. If any command fails, return None (not in git repo or git not available)

    Why capture diff for dirty repos:
        If there are uncommitted changes, the commit hash alone isn't enough
        for reproducibility. We capture the full diff so users can see exactly
        what modifications were present during the run.

Environment Metadata:
    - hostname: Identifies which machine ran the experiment
    - username: Tracks who ran the experiment (useful in shared environments)
    - python_version: sys.version (e.g., "3.10.4 (main, ...)")
    - platform: platform.platform() (e.g., "Darwin-21.4.0-arm64-arm-64bit")
    - cwd: Current working directory (helps reproduce relative paths)
    - command: Full command-line invocation (e.g., "python train.py --lr 0.01")

System Metadata Algorithm:
    1. CPU: psutil.cpu_count() with logical=False (physical cores) and True (threads)
    2. Memory: psutil.virtual_memory().total converted to GB
    3. GPU detection:
       a. Try to import pynvml and initialize NVML
       b. Get device count via nvmlDeviceGetCount()
       c. For each GPU: Get handle, extract name and total memory
       d. Decode name from bytes if necessary (NVML returns bytes on some platforms)
       e. Convert memory from bytes to GB (divide by 1024^3)
       f. Call nvmlShutdown() to cleanup
       g. If any step fails, skip GPU metadata (no GPU or pynvml not installed)

Dependencies Metadata:
    - Run 'pip freeze' to get all installed packages with exact versions
    - Uses sys.executable to ensure correct Python interpreter
    - 10-second timeout to avoid hangs on slow systems
    - Redirect stderr to suppress warnings
    - If fails, return None (pip not available or command failed)

    Why pip freeze vs requirements.txt:
        requirements.txt only lists direct dependencies. pip freeze captures
        the complete dependency tree with exact versions, which is needed for
        perfect reproducibility (transitive dependencies can change behavior).

Design Philosophy:
    - Comprehensive: Capture everything needed for reproducibility
    - Fail-safe: All capture functions return None on error, never crash
    - Zero configuration: Works automatically, no user setup required
    - Storage efficient: Text-based, compresses well in database
"""

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
