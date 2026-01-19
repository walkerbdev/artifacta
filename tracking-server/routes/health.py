# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Health check and debug endpoints."""

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Project root for file storage
PROJECT_ROOT = Path(__file__).parent.parent.parent


# Response models
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    database_connected: bool


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    db = request.state.db
    db_ok = db is not None
    status = "healthy" if db_ok else "unhealthy"

    return HealthResponse(status=status, database_connected=db_ok)


@router.post("/api/debug-logs")
async def write_debug_logs(request: Dict[str, Any]) -> Dict[str, str]:
    """Write frontend debug logs to disk."""
    try:
        filename = request.get("filename", "debug.txt")
        content = request.get("content", "")

        # Write to logs directory
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / filename
        log_file.write_text(content, encoding="utf-8")

        return {"status": "ok", "path": str(log_file)}
    except Exception as e:
        logger.error(f"Error writing debug logs: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
