"""HTTP emitter for metrics - MLflow pattern."""

import os
from typing import Any, Dict, Optional

import requests


class HTTPEmitter:
    """Emit metrics directly to API Gateway (MLflow/W&B pattern).

    Metrics are sent in real-time to the API server, enabling:
    - Immediate visualization in UI (via WebSocket)
    - No file watching needed on backend
    - Simpler architecture with fewer moving parts

    Falls back gracefully if server is unavailable (local JSONL still written)
    """

    def __init__(self, run_id: str, api_url: Optional[str] = None):
        """Initialize HTTP emitter.

        Args:
            run_id: Run ID for this emitter.
            api_url: Base URL of tracking server. If not provided, uses ARTIFACTA_API_URL environment variable.
        """
        self.run_id = run_id
        self.api_url = api_url or os.getenv("ARTIFACTA_API_URL")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.enabled = True
        # In strict mode (for tests), raise exceptions instead of gracefully degrading
        self.strict_mode = os.getenv("ARTIFACTA_STRICT_MODE", "").lower() in ("1", "true", "yes")

        # Test connection
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=2)
            if response.status_code != 200:
                msg = "⚠️  API Gateway health check failed, disabling HTTP emission"
                if self.strict_mode:
                    raise RuntimeError(msg)
                print(msg)
                self.enabled = False
        except requests.RequestException as e:
            msg = f"⚠️  API Gateway not reachable, disabling HTTP emission: {e}"
            if self.strict_mode:
                raise RuntimeError(msg) from e
            print(msg)
            self.enabled = False

    def emit_init(self, metadata: Dict[str, Any]) -> bool:
        """Emit run initialization to API Gateway.

        Creates the run entry in database.
        """
        if not self.enabled:
            if self.strict_mode:
                raise RuntimeError("HTTP emitter is disabled but strict mode is enabled")
            return False

        try:
            response = self.session.post(f"{self.api_url}/api/runs", json=metadata, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            if self.strict_mode:
                raise RuntimeError(f"Failed to emit init to API Gateway: {e}") from e
            print(f"⚠️  Failed to emit init to API Gateway: {e}")
            return False

    def emit_structured_data(self, data: Dict[str, Any]) -> bool:
        """Emit structured data primitive to API Gateway.

        Broadcasted to UI in real-time via WebSocket.
        """
        if not self.enabled:
            return False

        try:
            response = self.session.post(
                f"{self.api_url}/api/runs/{self.run_id}/data", json=data, timeout=2
            )
            response.raise_for_status()
            return True
        except Exception:
            # Fail silently - don't block training if API is down
            return False

    def emit_artifact(
        self,
        artifact_data: Dict[str, Any],
        content: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[str]:
        """Emit artifact metadata to API Gateway.

        Registers artifact reference in database.

        Args:
            artifact_data: Artifact metadata dict
            content: Optional file content for code artifacts
            role: Optional role ("input" or "output") - defaults to "output"

        Returns:
            artifact_id if successful, None otherwise
        """
        if not self.enabled:
            if self.strict_mode:
                raise RuntimeError("HTTP emitter is disabled but strict mode is enabled")
            return None

        try:
            # Add content and role to payload if provided
            payload = artifact_data.copy()
            if content is not None:
                payload["content"] = content
            if role is not None:
                payload["role"] = role

            response = self.session.post(f"{self.api_url}/api/artifacts", json=payload, timeout=5)
            response.raise_for_status()
            return response.json().get("artifact_id")
        except Exception as e:
            if self.strict_mode:
                raise RuntimeError(f"Failed to emit artifact to API Gateway: {e}") from e
            print(f"⚠️  Failed to emit artifact to API Gateway: {e}")
            return None

    def update_run_config_artifact(self, artifact_id: str) -> bool:
        """Update run to link config artifact.

        Args:
            artifact_id: The artifact ID to link as config.
        """
        if not self.enabled:
            return False

        try:
            response = self.session.patch(
                f"{self.api_url}/api/runs/{self.run_id}/config-artifact",
                json={"config_artifact_id": artifact_id},
                timeout=5,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️  Failed to update config artifact link: {e}")
            return False

    def emit_note(self, project_id: str, title: str, content: str) -> Optional[int]:
        """Create a notebook entry/note for a project.

        Args:
            project_id: Project identifier.
            title: Note title.
            content: HTML content.

        Returns:
            Note ID if successful, None otherwise.
        """
        if not self.enabled:
            return None

        try:
            response = self.session.post(
                f"{self.api_url}/api/projects/{project_id}/notes",
                json={
                    "title": title,
                    "content": content,
                },
                timeout=5,
            )
            response.raise_for_status()
            return response.json().get("id")
        except Exception as e:
            print(f"⚠️  Failed to create note: {e}")
            return None

    def close(self):
        """Close HTTP session."""
        self.session.close()
