"""Tracking Server (MLflow-style).

HTTP API with direct SDK emission support:
- Receives metrics directly from artifacta Python SDK
- Stores in SQLite database
- Broadcasts to WebSocket clients in real-time
"""
# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,no-untyped-call"

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Set

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import artifacts, chat, health, projects, runs, websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration
API_PORT = int(os.getenv("TRACKING_SERVER_PORT", os.getenv("API_PORT", "8000")))

# Database path - resolve to project root regardless of where script is run from
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "runs.db")))

# Ensure database directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Global state
db: Optional[Any] = None  # SQLAlchemy Database instance
websocket_clients: Set[WebSocket] = set()


# Database initialization
def init_database() -> None:
    """Initialize SQLite database with schema using SQLAlchemy."""
    logger.info(f"Initializing database at {DB_PATH}")

    # Use SQLAlchemy to create tables (single source of truth!)
    from database import init_db

    init_db()

    logger.info("✅ Database initialized")


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Initialize and cleanup application resources."""
    # Startup
    global db
    init_database()

    # Initialize SQLAlchemy Database for project notes
    from database import Database

    db_uri = f"sqlite:///{DB_PATH}"
    db = Database(db_uri=db_uri)

    logger.info("✅ Tracking server ready")

    yield


# FastAPI app
app = FastAPI(
    title="Artifacta API Gateway",
    description="HTTP API for Artifacta backend services",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Rows", "X-Offset", "X-Limit", "X-Has-More"],
)


# Dependency injection middleware to provide db and websocket_clients to routes
@app.middleware("http")
async def inject_dependencies(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Inject global dependencies into request state."""
    request.state.db = db
    request.state.websocket_clients = websocket_clients
    response = await call_next(request)
    return response


# Initialize websocket clients in the websocket module
websocket.set_websocket_clients(websocket_clients)

# Health and debug endpoints
app.include_router(health.router)

# WebSocket endpoints
app.include_router(websocket.router)

# Run management endpoints
app.include_router(runs.router)

# Artifact management endpoints
app.include_router(artifacts.router)

# Project notes endpoints
app.include_router(projects.router)

# LLM chat proxy endpoints
app.include_router(chat.router)


# Serve static UI files if dist folder exists
from config import UI_DIST_PATH

if UI_DIST_PATH.exists():
    # Mount static assets (JS, CSS, images, etc.)
    app.mount("/assets", StaticFiles(directory=UI_DIST_PATH / "assets"), name="assets")

    # Serve index.html for root and any unmatched routes (SPA fallback)
    @app.get("/{full_path:path}")
    async def serve_ui(full_path: str):
        """Serve the UI for all non-API routes."""
        # Don't intercept API routes
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return None

        # Serve index.html from dist folder for all other routes (SPA routing)
        return FileResponse(UI_DIST_PATH / "index.html")
else:
    logger.warning(
        f"⚠️  UI dist folder not found at {UI_DIST_PATH}. "
        "Run 'npm install && npm run build' to build the UI, "
        "or use 'artifacta server' for API-only mode."
    )


if __name__ == "__main__":
    import uvicorn

    from config import SERVER_BIND_HOST, get_port

    uvicorn.run("main:app", host=SERVER_BIND_HOST, port=get_port(), log_level="info")
