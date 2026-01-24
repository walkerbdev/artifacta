"""Artifacta Tracking Server - FastAPI application with real-time WebSocket support.

This is the main entry point for the tracking server, which provides an HTTP API
for receiving run data from the Artifacta SDK and a WebSocket interface for
real-time updates to connected UIs.

Architecture:

    The server is built on FastAPI and organized into layers:

    1. **Application Layer** (this file):
       - FastAPI app initialization with lifespan management
       - CORS middleware configuration for web frontend
       - Dependency injection middleware for database and WebSocket clients
       - Route registration and static file serving
       - Global state management (database, WebSocket clients)

    2. **Database Layer** (database.py):
       - SQLAlchemy ORM models and schema
       - Database connection management
       - Session creation and cleanup

    3. **Routes Layer** (routes/*.py):
       - Individual endpoint implementations
       - Request/response validation via Pydantic
       - Business logic for runs, artifacts, projects, chat

    4. **WebSocket Layer** (routes/websocket.py):
       - WebSocket connection management
       - Real-time broadcast to connected clients
       - Connection lifecycle (connect, disconnect, error handling)

Lifespan Management:

    FastAPI's lifespan context manager handles startup and shutdown:

    Startup:
        1. Initialize SQLite database via init_database()
        2. Create Database instance with connection string
        3. Store in global 'db' variable for dependency injection
        4. Log ready message

    Shutdown:
        - Automatic cleanup (context manager exit)
        - Database connections closed by SQLAlchemy
        - WebSocket clients disconnected automatically

    Why lifespan:
        - Replaces deprecated @app.on_event("startup") / @app.on_event("shutdown")
        - Provides cleaner resource management with context manager pattern
        - Ensures cleanup happens even on crashes (finally block semantics)

Dependency Injection:

    Two mechanisms provide dependencies to route handlers:

    1. **HTTP Middleware** (inject_dependencies):
       - Intercepts every HTTP request
       - Attaches db and websocket_clients to request.state
       - Routes access via request.state.db, request.state.websocket_clients

    2. **FastAPI Depends** (routes/dependencies.py):
       - get_db(), get_session(), get_websocket_clients() extractors
       - Type-safe dependency injection in route signatures
       - Cleaner than accessing request.state directly

CORS Configuration:

    Permissive CORS for development (allow all origins):
        - allow_origins=["*"]: Any domain can call the API
        - allow_methods=["*"]: All HTTP methods (GET, POST, PATCH, DELETE)
        - allow_headers=["*"]: All custom headers
        - expose_headers: Pagination headers exposed to browser JavaScript

    Why permissive:
        - Local development: UI runs on localhost:5173, API on localhost:8000
        - Production: Should restrict origins to deployed frontend domain

    Security consideration:
        For production, set allow_origins to specific domain list

WebSocket Client Management:

    Global set of connected WebSocket clients:
        - websocket_clients: Set[WebSocket] - thread-safe for async
        - Shared across all route handlers via dependency injection
        - Modified by websocket route (add on connect, remove on disconnect)
        - Used by data emission routes to broadcast updates

    Broadcasting strategy:
        - When SDK emits data, server stores in database
        - Server then broadcasts to all connected WebSocket clients
        - Clients receive real-time updates without polling
        - Disconnected clients removed automatically on send failure

Static File Serving:

    Single-page application (SPA) routing:
        1. Mount /assets directory for static files (JS, CSS, images)
        2. Catch-all route serves index.html for non-API paths
        3. API routes (/api/*, /ws/*) not intercepted
        4. Enables client-side routing (React Router, etc.)

    Fallback behavior:
        - If UI dist folder missing, log warning
        - Server continues in API-only mode
        - Useful for headless deployments or during UI development

Environment Configuration:

    - TRACKING_SERVER_PORT: Port to bind (default 8000)
    - API_PORT: Alternative env var name (backwards compatibility)
    - DB_PATH: Database file location (default: ./data/runs.db)
    - UI_DIST_PATH: Location of built UI files (from config.py)

Design Philosophy:

    - Real-time first: WebSocket broadcasts enable live UI updates
    - Database-backed: SQLite for development, PostgreSQL for production
    - Stateless routes: All state in database or request context
    - Graceful degradation: API works without UI, UI works without WebSocket
"""
# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,no-untyped-call"

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Set

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from routes import artifacts, chat, health, projects, runs, websocket

from config import UI_DIST_PATH

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

    logger.info("Database initialized")


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

    logger.info("Tracking server ready")

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
if UI_DIST_PATH.exists():
    # Mount static assets (JS, CSS, images, etc.)
    app.mount("/assets", StaticFiles(directory=UI_DIST_PATH / "assets"), name="assets")

    # Serve index.html for root and any unmatched routes (SPA fallback)
    # Note: This catch-all route is registered LAST so API routes take precedence
    @app.get("/{full_path:path}", response_model=None, include_in_schema=False)
    async def serve_ui(full_path: str) -> FileResponse:
        """Serve the UI for all non-API routes (SPA fallback)."""
        # This should never match API routes since they're registered first,
        # but check anyway as a safety measure
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="API route not found")

        # Serve index.html for all other paths (enables client-side routing)
        return FileResponse(UI_DIST_PATH / "index.html")
else:
    logger.warning(
        f"UI dist folder not found at {UI_DIST_PATH}. "
        "Run 'npm install && npm run build' to build the UI, "
        "or use 'artifacta server' for API-only mode."
    )


if __name__ == "__main__":
    import uvicorn

    from config import SERVER_BIND_HOST, get_port

    uvicorn.run("main:app", host=SERVER_BIND_HOST, port=get_port(), log_level="info")
