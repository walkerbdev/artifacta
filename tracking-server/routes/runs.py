# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Run management endpoints for tracking server.

This module implements HTTP endpoints for run creation, retrieval, and data emission.
It serves as the primary interface between the Artifacta SDK (client) and the
tracking server (backend), enabling real-time metrics logging and WebSocket broadcasts.

Endpoint Overview:

    POST /api/runs
        - Create new run entry in database
        - Called by SDK when run.start() is invoked
        - Generates unique run name if not provided
        - Broadcasts run creation to WebSocket clients
        - Returns: {"status": "created", "run_id": "..."}

    GET /api/runs
        - List all runs with structured data
        - Supports filtering by project, pagination via limit
        - Optionally includes tags and params
        - Groups structured data by metric name
        - Returns: List[RunDict] with structured_data field

    GET /api/runs/{run_id}
        - Retrieve single run by ID
        - Includes all structured data grouped by name
        - Returns: RunDict with structured_data field

    PATCH /api/runs/{run_id}/config-artifact
        - Update run to link config artifact (single source of truth)
        - Called by SDK after config artifact is logged
        - Returns: {"status": "updated"}

    POST /api/runs/{run_id}/data
        - Log structured data primitive (Series, Distribution, etc.)
        - Called by SDK when run.log() is invoked
        - Stores data as JSON in database
        - Broadcasts to WebSocket clients for real-time UI updates
        - Returns: {"status": "logged"}

Data Flow (SDK to UI):

    1. SDK calls run.log(name="loss", data={"epoch": [1,2,3], "loss": [0.5,0.3,0.2]})
    2. SDK sends POST to /api/runs/{run_id}/data with payload
    3. Server validates and stores in StructuredData table
    4. Server broadcasts to all connected WebSocket clients
    5. UI receives WebSocket message and updates chart in real-time

Config Artifact Linking:

    The config artifact pattern ensures configuration is stored as an artifact
    (for provenance) and linked to the run (for easy access):

    1. SDK logs config as artifact via log_artifact()
    2. SDK calls PATCH /api/runs/{run_id}/config-artifact with artifact_id
    3. Server updates run.config_artifact_id foreign key
    4. GET /api/runs fetches artifact content and parses config JSON
    5. UI displays config in run details

Structured Data Grouping:

    Multiple emissions of the same metric name are grouped together:
    - run.log("loss", ...) at step 1
    - run.log("loss", ...) at step 2
    - run.log("loss", ...) at step 3
    - Result: structured_data["loss"] = [step1_data, step2_data, step3_data]

    This enables:
    - Line charts with multiple points
    - Confusion matrices that update over epochs
    - A/B test results with progressive accumulation

WebSocket Broadcasting:

    broadcast_to_websockets() helper:
        1. Iterate over all connected WebSocket clients (global set)
        2. Send JSON message via client.send_json()
        3. Catch exceptions (client disconnected, network error)
        4. Track disconnected clients in set
        5. Remove disconnected clients from global set (cleanup)

    Message format:
        {
            "type": "data_logged" | "run_created",
            "payload": {
                "run_id": "...",
                "name": "loss",
                "primitive_type": "series"
            }
        }

Error Handling:

    - HTTPException for client errors (404 Not Found, 400 Bad Request)
    - Generic Exception caught and logged with exc_info=True
    - Raised as HTTPException(500) with error details
    - This prevents stack traces leaking to client

Database Session Management:

    - SQLAlchemy session injected via Depends(get_session)
    - Session automatically committed at end of request (FastAPI middleware)
    - Session rolled back on exception (automatic cleanup)
    - No manual commit() needed in route handlers

Performance Considerations:

    - Queries use indexes (run_id, created_at, etc.) for fast lookups
    - Limit parameter prevents unbounded result sets
    - JSON parsing done lazily (only for requested runs)
    - Structured data grouped in Python (not SQL) for flexibility
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from database import Artifact, Run, StructuredData, Tag
from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .dependencies import get_db, get_session, get_websocket_clients

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])


# Request/Response models
class RunCreateRequest(BaseModel):
    """Run creation request model."""

    run_id: str
    project: Optional[str] = None
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, Any]] = None


class ConfigArtifactRequest(BaseModel):
    """Config artifact link request model."""

    config_artifact_id: str


# WebSocket broadcast helper
async def broadcast_to_websockets(
    message: Dict[str, Any], websocket_clients: Set[WebSocket]
) -> None:
    """Broadcast message to all connected WebSocket clients."""
    if not websocket_clients:
        return

    disconnected: Set[WebSocket] = set()
    for client in websocket_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket client: {e}")
            disconnected.add(client)

    # Remove disconnected clients
    websocket_clients.difference_update(disconnected)


@router.post("")
async def create_run(
    request: RunCreateRequest,
    session: Session = Depends(get_session),
    websocket_clients: Set[WebSocket] = Depends(get_websocket_clients),
    db: Any = Depends(get_db),
) -> Dict[str, str]:
    """Create a new run.

    Called by artifacta SDK when run.start() is invoked.
    """
    try:
        logger.info(f"Creating run: {request.run_id}")

        # Generate unique name if not provided
        run_name = request.name or db.generate_run_name(request.run_id)

        # Create run using SQLAlchemy
        run = Run(
            run_id=request.run_id,
            project=request.project,
            name=run_name,
            config_artifact_id=None,  # Will be set later when config artifact is logged
            created_at=int(time.time() * 1000),
        )
        session.add(run)
        session.flush()  # Get any auto-generated values

        # Broadcast to WebSocket clients
        await broadcast_to_websockets(
            {
                "type": "run_created",
                "payload": {
                    "run_id": request.run_id,
                    "project": request.project,
                    "name": request.name,
                },
            },
            websocket_clients,
        )

        return {"status": "created", "run_id": request.run_id}

    except Exception as e:
        logger.error(f"Failed to create run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("")
async def list_runs(
    limit: int = 100,
    project: Optional[str] = None,
    include_tags: bool = False,
    include_params: bool = False,
    session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    """List all runs with structured data.

    Called by frontend to populate run history.
    Returns runs with structured_data field containing all logged primitives.
    """
    try:
        # Build query
        query = session.query(Run)

        if project:
            query = query.filter(Run.project == project)

        query = query.order_by(Run.created_at.desc()).limit(limit)
        runs = query.all()

        result: List[Dict[str, Any]] = []

        for run in runs:
            run_dict: Dict[str, Any] = {
                "run_id": run.run_id,
                "name": run.name,
                "project": run.project,
                "config_artifact_id": run.config_artifact_id,
                "created_at": run.created_at,
            }

            # Fetch config from artifact if it exists
            if run.config_artifact_id:
                artifact = (
                    session.query(Artifact)
                    .filter(Artifact.artifact_id == run.config_artifact_id)
                    .first()
                )

                if artifact and artifact.content:
                    try:
                        # Parse file collection JSON
                        content_data = json.loads(artifact.content)
                        if content_data.get("files") and len(content_data["files"]) > 0:
                            config_json = content_data["files"][0].get("content")
                            if config_json:
                                config = json.loads(config_json)
                                run_dict["config"] = config

                                # Extract params if requested
                                if include_params:
                                    run_dict["params"] = {
                                        k: v for k, v in config.items() if not k.startswith("_")
                                    }
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

            # Set config to None if not found
            if "config" not in run_dict:
                run_dict["config"] = None

            # Fetch tags if requested
            if include_tags:
                tags = session.query(Tag).filter(Tag.run_id == run.run_id).all()
                run_dict["tags"] = {tag.key: tag.value for tag in tags}

            # Fetch structured data
            structured_data_rows = (
                session.query(StructuredData)
                .filter(StructuredData.run_id == run.run_id)
                .order_by(StructuredData.timestamp.asc())
                .all()
            )

            # Group by name
            structured_data: Dict[str, List[Dict[str, Any]]] = {}
            for data_row in structured_data_rows:
                name_key = str(data_row.name)
                if name_key not in structured_data:
                    structured_data[name_key] = []

                structured_data[name_key].append(
                    {
                        "primitive_type": data_row.primitive_type,
                        "section": data_row.section,
                        "data": json.loads(data_row.data),
                        "metadata": json.loads(data_row.meta) if data_row.meta else None,
                        "timestamp": data_row.timestamp,
                    }
                )

            run_dict["structured_data"] = structured_data
            result.append(run_dict)

        return result

    except Exception as e:
        logger.error(f"Failed to list runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Get a single run by ID.

    Called by frontend when viewing run details.
    """
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        run_dict: Dict[str, Any] = {
            "run_id": run.run_id,
            "name": run.name,
            "project": run.project,
            "config_artifact_id": run.config_artifact_id,
            "created_at": run.created_at,
        }

        # Fetch structured data
        structured_data_rows = (
            session.query(StructuredData)
            .filter(StructuredData.run_id == run_id)
            .order_by(StructuredData.timestamp.asc())
            .all()
        )

        # Group by name
        structured_data: Dict[str, List[Dict[str, Any]]] = {}
        for data_row in structured_data_rows:
            name_key = str(data_row.name)
            if name_key not in structured_data:
                structured_data[name_key] = []

            structured_data[name_key].append(
                {
                    "primitive_type": data_row.primitive_type,
                    "section": data_row.section,
                    "data": json.loads(data_row.data),
                    "metadata": json.loads(data_row.meta) if data_row.meta else None,
                    "timestamp": data_row.timestamp,
                }
            )

        run_dict["structured_data"] = structured_data

        return run_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/{run_id}/config-artifact")
async def update_config_artifact(
    run_id: str,
    request: ConfigArtifactRequest,
    session: Session = Depends(get_session),
) -> Dict[str, str]:
    """Update run to link config artifact."""
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        run.config_artifact_id = request.config_artifact_id  # type: ignore[assignment]
        session.flush()

        return {"status": "updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config artifact: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{run_id}/data")
async def log_structured_data(
    run_id: str,
    data: Dict[str, Any],
    session: Session = Depends(get_session),
    websocket_clients: Set[WebSocket] = Depends(get_websocket_clients),
) -> Dict[str, str]:
    """Log structured data primitive to a run.

    Called by artifacta SDK when run.log() is invoked.
    """
    try:
        # Create structured data entry
        structured_data = StructuredData(
            run_id=run_id,
            name=data["name"],
            primitive_type=data["primitive_type"],
            section=data.get("section"),
            data=json.dumps(data["data"]),
            meta=json.dumps(data.get("metadata")) if data.get("metadata") else None,
            timestamp=int(time.time() * 1000),
        )
        session.add(structured_data)
        session.flush()

        # Broadcast to WebSocket clients
        await broadcast_to_websockets(
            {
                "type": "data_logged",
                "payload": {
                    "run_id": run_id,
                    "name": data["name"],
                    "primitive_type": data["primitive_type"],
                },
            },
            websocket_clients,
        )

        return {"status": "logged"}

    except Exception as e:
        logger.error(f"Failed to log structured data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
