# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""WebSocket endpoint for real-time metrics and run updates.

This module implements the WebSocket server endpoint that clients connect to for
receiving real-time updates about runs, metrics, and artifacts. It's the push
notification mechanism that enables live UI updates without polling.

Architecture:

    The WebSocket system follows a pub-sub pattern:

    1. **Connection Management** (this module):
       - Accept WebSocket connections at /ws/metrics
       - Add connections to global websocket_clients set
       - Keep connections alive (receive ping/pong messages)
       - Remove disconnected clients from set

    2. **Broadcasting** (runs.py and other routes):
       - When data is logged, broadcast to all connected clients
       - Iterate over websocket_clients set
       - Send JSON message to each client
       - Remove failed clients (disconnected, errors)

    3. **Client Consumption** (frontend JavaScript):
       - Connect WebSocket to /ws/metrics
       - Listen for JSON messages
       - Update UI components based on message type
       - Reconnect on disconnect

Connection Lifecycle:

    Connect:
        1. Client initiates WebSocket connection
        2. Server calls websocket.accept()
        3. Server adds websocket to global set
        4. Log connection count
        5. Enter receive loop (keep-alive)

    Alive:
        - Client sends periodic ping messages (heartbeat)
        - Server receives via websocket.receive_text()
        - Connection stays open indefinitely
        - No response needed (WebSocket protocol handles pong)

    Disconnect:
        - Client closes connection (normal shutdown)
        - WebSocketDisconnect exception raised
        - Server removes client from set
        - Log remaining connection count

    Error:
        - Network error, protocol error, etc.
        - Generic Exception caught
        - Server removes client from set (discard for safety)
        - Log error message

Message Format:

    Broadcast messages from server to clients:
    {
        "type": "run_created" | "data_logged" | "artifact_uploaded",
        "payload": {
            "run_id": "...",
            "name": "loss",
            "primitive_type": "series",
            ...
        }
    }

    Message types:
        - run_created: New run started
        - data_logged: Metric/primitive logged to run
        - artifact_uploaded: Artifact added to run
        (more types can be added as needed)

Global State Management:

    websocket_clients_global:
        - Module-level variable (shared across all requests)
        - Set[WebSocket] type (O(1) add/remove)
        - Initialized by main.py via set_websocket_clients()
        - Thread-safe for async (FastAPI handles concurrency)

    Why global set:
        - Needs to be shared across all route handlers
        - Routes need to broadcast to all connected clients
        - FastAPI doesn't have built-in pub-sub (unlike Socket.io)
        - Simple and efficient for this use case

Scalability Considerations:

    Current design (in-memory set):
        - Works well for single-server deployment
        - All WebSocket connections on same server
        - Simple, fast, no external dependencies

    Future scaling (if needed):
        - Redis Pub/Sub for multi-server deployment
        - Each server maintains local websocket_clients
        - Broadcasts go through Redis channel
        - All servers receive and forward to their clients

Keep-Alive Strategy:

    The receive loop keeps connection open:
    - await websocket.receive_text() blocks until message
    - Client sends ping periodically (e.g., every 30 seconds)
    - Server doesn't need to respond (WebSocket handles it)
    - Prevents connection timeout from load balancers/proxies

    Why receive-based keep-alive:
        - Simpler than server-side ping/pong
        - Client controls heartbeat frequency
        - Works with all WebSocket client libraries
        - No need for asyncio.sleep() polling loop

Error Recovery:

    Client-side should implement reconnection:
    - Detect WebSocket close event
    - Wait with exponential backoff (1s, 2s, 4s, ...)
    - Reconnect to /ws/metrics
    - Resume receiving updates

    Server-side is stateless:
        - No per-client state to restore
        - Clients immediately receive future broadcasts
        - Past messages not replayed (use HTTP API for history)
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


# Global websocket_clients will be injected by main.py
websocket_clients_global = None


def set_websocket_clients(clients):  # type: ignore[no-untyped-def]
    """Set the global websocket clients set."""
    global websocket_clients_global
    websocket_clients_global = clients


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket) -> None:
    """Websocket endpoint for real-time metrics streaming.

    Clients connect here to receive metrics as they're emitted from training scripts
    """
    await websocket.accept()
    websocket_clients_global.add(websocket)
    logger.info(f"WebSocket client connected (total: {len(websocket_clients_global)})")

    try:
        # Keep connection alive
        while True:
            # Wait for ping from client (or any message)
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_clients_global.remove(websocket)
        logger.info(f"WebSocket client disconnected (remaining: {len(websocket_clients_global)})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_clients_global.discard(websocket)
