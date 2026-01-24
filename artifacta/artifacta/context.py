"""Context providers for automatic environment detection and tagging.

This module implements a plugin-based system for detecting the execution context
(git repository, Jupyter notebook, Docker container) and automatically adding
relevant metadata tags to runs. This enables run reproducibility and environment
tracking without requiring manual configuration.

Architecture:
    The module follows a provider pattern with a registry:

    1. Base class (ContextProvider): Defines interface (in_context, tags)
    2. Concrete providers: GitContext, NotebookContext, DockerContext
    3. Registry (CONTEXT_PROVIDERS): List of enabled providers
    4. Collector (collect_context_tags): Iterates over providers and aggregates tags

Detection Strategies:

    Git Detection:
        - Run 'git rev-parse HEAD' command to check if in git repository
        - If succeeds, we're in a git repo; if fails, we're not
        - Extract: commit hash, branch name, remote URL, dirty status
        - Capture diff if dirty (uncommitted changes exist)
        - All git commands use 5-second timeout to avoid hangs
        - stderr redirected to DEVNULL to suppress error messages

    Notebook Detection:
        - Check if 'ipykernel' or 'IPython' modules are loaded in sys.modules
        - This works for Jupyter, JupyterLab, Google Colab, VSCode notebooks
        - Tag: source.type = "NOTEBOOK"
        - Future: Could extract cell number, notebook path via IPython API

    Docker Detection:
        - Check if /.dockerenv file exists (created by Docker runtime)
        - This is the most reliable cross-platform Docker detection method
        - Tag: docker.container = "true"
        - Future: Read DOCKER_IMAGE, DOCKER_TAG from environment variables

Tag Aggregation:
    collect_context_tags() iterates over all registered providers:
    1. Call provider.in_context() to check if context is active
    2. If True, call provider.tags() to get tag dictionary
    3. Merge tags into aggregated dictionary (later providers override earlier)
    4. If any provider raises exception, silently continue (graceful degradation)
    5. Return merged tags dictionary

Why only Git is enabled by default:
    The CONTEXT_PROVIDERS registry only includes GitContext by default.
    NotebookContext and DockerContext are defined but not registered to avoid
    polluting tags unnecessarily. Users running in notebooks/Docker likely
    want to track this explicitly, not automatically.

Design Philosophy:
    - Zero configuration: Automatic detection, no setup required
    - Fail-safe: All detection wrapped in try/except, never crash user code
    - Extensible: Easy to add new providers (CI/CD, Kubernetes, Slurm, etc.)
    - Minimal overhead: Only run detection once at run initialization
"""

import os
import subprocess
import sys


class ContextProvider:
    """Base class for context providers."""

    def in_context(self):
        """Returns True if this context is active."""
        raise NotImplementedError

    def tags(self):
        """Returns dict of tags for this context."""
        raise NotImplementedError


class GitContext(ContextProvider):
    """Git context - detects git repo and adds git tags."""

    def in_context(self):
        """Check if running in a git repository."""
        try:
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
            )
            return True
        except Exception:
            return False

    def tags(self):
        """Extract git metadata tags."""
        tags = {}
        try:
            # Commit hash
            tags["git.commit"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
                )
                .decode()
                .strip()
            )

            # Branch name
            tags["git.branch"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .decode()
                .strip()
            )

            # Remote URL
            tags["git.remote"] = (
                subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .decode()
                .strip()
            )

            # Check if dirty (uncommitted changes)
            status_output = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5
                )
                .decode()
                .strip()
            )
            dirty = len(status_output) > 0
            tags["git.dirty"] = str(dirty).lower()

            # Capture diff if dirty
            if dirty:
                diff = subprocess.check_output(
                    ["git", "diff"], stderr=subprocess.DEVNULL, timeout=5
                ).decode()
                tags["git.diff"] = diff
        except Exception:
            pass
        return tags


class NotebookContext(ContextProvider):
    """Jupyter notebook context."""

    def in_context(self):
        """Check if running in Jupyter notebook."""
        # Check if running in Jupyter
        try:
            return "ipykernel" in sys.modules or "IPython" in sys.modules
        except Exception:
            return False

    def tags(self):
        """Extract notebook context tags."""
        tags = {"source.type": "NOTEBOOK"}
        # Could add notebook path, cell number, etc. later
        return tags


class DockerContext(ContextProvider):
    """Docker container context."""

    def in_context(self):
        """Check if running in Docker container."""
        # Check for .dockerenv file (exists in Docker containers)
        return os.path.exists("/.dockerenv")

    def tags(self):
        """Extract Docker context tags."""
        tags = {"docker.container": "true"}
        # Could read docker image info from env vars
        # e.g., tags['docker.image'] = os.getenv('DOCKER_IMAGE')
        return tags


# Registry of all context providers (only git - no pollution from docker/notebook)
CONTEXT_PROVIDERS = [
    GitContext(),
]


def collect_context_tags():
    """Collect tags from all active context providers."""
    tags = {}

    # Collect from all active providers
    for provider in CONTEXT_PROVIDERS:
        try:
            if provider.in_context():
                provider_tags = provider.tags()
                tags.update(provider_tags)
        except Exception:
            # Silently continue if provider fails
            pass

    return tags
