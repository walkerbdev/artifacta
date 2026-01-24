"""HTTP emitter for real-time metrics and artifact transmission.

This module implements the client-side HTTP emitter that sends run data, metrics,
and artifacts to the tracking server in real-time. The push-based design enables
immediate visualization and reduces architectural complexity compared to
file-watching approaches.

Architecture:
    The emitter acts as a reliable HTTP client with graceful degradation:

    1. Initialization: Health check against tracking server
    2. Run Creation: POST to /api/runs to create database entry
    3. Data Emission: POST to /api/runs/{run_id}/data for real-time metrics
    4. Artifact Registration: POST to /api/artifacts for file metadata
    5. WebSocket Integration: Server broadcasts emissions to connected clients

Graceful Degradation:
    The emitter handles network failures gracefully to avoid blocking training:

    - If health check fails -> disable HTTP emission, warn user, continue locally
    - If data emission fails -> fail silently, don't block training loop
    - If artifact emission fails -> warn user but continue
    - Strict mode (for tests) -> raise exceptions instead of degrading

    This ensures that network issues never crash the user's training job.

Connection Management:
    - Uses requests.Session for connection pooling (HTTP keep-alive)
    - Configurable timeouts (2s for health/data, 5s for init/artifacts)
    - Session headers set once at initialization
    - Explicit close() method for cleanup

Environment Variables:
    - ARTIFACTA_API_URL: Base URL of tracking server (e.g., http://localhost:8000)
    - ARTIFACTA_STRICT_MODE: Enable strict mode for testing (raise exceptions on errors)
"""

import os
from typing import Any, Dict, Optional

import requests


class HTTPEmitter:
    """Emit metrics directly to API Gateway for real-time tracking.

    Metrics are sent in real-time to the API server, enabling:
    - Immediate visualization in UI (via WebSocket)
    - No file watching needed on backend
    - Simpler architecture with fewer moving parts

    Falls back gracefully if server is unavailable (local JSONL still written)
    """

    def __init__(self, run_id: str, api_url: Optional[str] = None):
        """Initialize HTTP emitter with health check and connection setup.

        Initialization algorithm:
            1. Store run_id for all subsequent API calls
            2. Resolve api_url from parameter or ARTIFACTA_API_URL environment variable
            3. Create persistent requests.Session for connection pooling (HTTP keep-alive)
            4. Set Content-Type header once for all requests
            5. Check strict_mode from environment (for testing vs production behavior)
            6. Perform health check via GET /health with 2-second timeout
            7. If health check fails:
               - Strict mode: Raise RuntimeError (tests should fail fast)
               - Normal mode: Print warning, disable emitter, continue locally

        The health check ensures the tracking server is available before attempting
        any data emissions. This avoids timeout delays on every emit call if the
        server is down.

        Connection pooling via Session:
            - Reuses TCP connections across multiple requests
            - Reduces latency (no handshake overhead per request)
            - Automatically handles keep-alive headers

        Args:
            run_id: Run ID for this emitter (links all emissions to this run)
            api_url: Base URL of tracking server (e.g., http://localhost:8000).
                     Defaults to ARTIFACTA_API_URL env var or http://localhost:8000.

        Raises:
            RuntimeError: If health check fails and ARTIFACTA_STRICT_MODE is enabled
        """
        self.run_id = run_id
        self.api_url = api_url or os.getenv("ARTIFACTA_API_URL", "http://localhost:8000")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.enabled = True
        # In strict mode (for tests), raise exceptions instead of gracefully degrading
        self.strict_mode = os.getenv("ARTIFACTA_STRICT_MODE", "").lower() in ("1", "true", "yes")

        # Test connection
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=2)
            if response.status_code != 200:
                msg = "API Gateway health check failed, disabling HTTP emission"
                if self.strict_mode:
                    raise RuntimeError(msg)
                print(msg)
                self.enabled = False
        except requests.RequestException as e:
            msg = f"API Gateway not reachable, disabling HTTP emission: {e}"
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
            print(f"Failed to emit init to API Gateway: {e}")
            return False

    def emit_structured_data(self, data: Dict[str, Any]) -> bool:
        """Emit structured data primitive to API Gateway with real-time WebSocket broadcast.

        Data flow:
            1. Check if emitter is enabled (skip if health check failed)
            2. POST to /api/runs/{run_id}/data with JSON payload
            3. Server receives data and stores in database
            4. Server broadcasts data to all WebSocket clients subscribed to this run
            5. UI updates in real-time (live charts, metrics tables)

        Failure handling:
            - Fails silently (returns False) without raising exceptions
            - This is intentional: network issues shouldn't crash training loops
            - Training continues, local JSONL still written
            - User can still view data after run completes

        Performance considerations:
            - 2-second timeout to avoid blocking training
            - No retry logic (fire-and-forget for performance)
            - Session reuse minimizes connection overhead

        Args:
            data: Dictionary containing primitive type data (Series, Distribution, etc.)
                  Must include 'name' and 'data' keys at minimum

        Returns:
            True if emission successful, False otherwise (including when disabled)
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
            print(f"Failed to emit artifact to API Gateway: {e}")
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
            print(f"Failed to update config artifact link: {e}")
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
            print(f"Failed to create note: {e}")
            return None

    def close(self):
        """Close HTTP session."""
        self.session.close()
