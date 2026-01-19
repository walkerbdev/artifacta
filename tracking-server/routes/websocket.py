# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""WebSocket endpoints for real-time metrics streaming."""

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
