"""Context providers - automatically detect environment and add tags."""

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
