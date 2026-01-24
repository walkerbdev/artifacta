"""Configuration constants for Artifacta tracking server.

Centralized configuration management for:
- Server host/port settings
- Database paths and timeouts
- File upload limits
- CORS origins
- Pagination defaults
- UI static file locations

Configuration sources (priority order):
1. Environment variables (highest priority)
2. Default constants defined here
3. Hard-coded fallbacks

Environment variables:
- TRACKING_SERVER_HOST: Server bind host
- TRACKING_SERVER_PORT: Server port
- DATABASE_PATH: SQLite database file path
- UI_PORT: UI development server port
- CORS_ORIGINS: Comma-separated allowed origins

Design philosophy:
- Sane defaults for local development
- Override via env vars for production
- Explicit constants prevent magic numbers throughout codebase
"""

import os
from pathlib import Path

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_UI_PORT",
    "SERVER_BIND_HOST",
    "PROJECT_ROOT",
    "DEFAULT_DB_PATH",
    "UI_DIST_PATH",
    "MAX_UPLOAD_SIZE_MB",
    "MAX_UPLOAD_SIZE_BYTES",
    "THUMBNAIL_SIZE",
    "SUPPORTED_IMAGE_FORMATS",
    "API_PREFIX",
    "CORS_ORIGINS",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "REQUEST_TIMEOUT",
    "DB_TIMEOUT",
    "get_host",
    "get_port",
    "get_ui_port",
    "get_api_url",
]

# Server Configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_UI_PORT = 5173
SERVER_BIND_HOST = "0.0.0.0"  # Bind to all interfaces for server

# Database Configuration
# Get project root (parent of tracking-server directory)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "runs.db"))

# UI Static Files Configuration
try:
    # When installed from pip
    from artifacta_ui import UI_DIST_PATH
except ImportError:
    # When running in development
    UI_DIST_PATH = PROJECT_ROOT / "artifacta_ui" / "dist"

# File Upload Configuration
MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Image Processing Configuration
THUMBNAIL_SIZE = (200, 200)
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# API Configuration
API_PREFIX = "/api"
# CORS origins should be configured via environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]

# Pagination
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000

# Timeouts (seconds)
REQUEST_TIMEOUT = 30
DB_TIMEOUT = 10


# Environment Variables
def get_host() -> str:
    """Get server host from environment or default."""
    return os.getenv("TRACKING_SERVER_HOST", DEFAULT_HOST)


def get_port() -> int:
    """Get server port from environment or default."""
    return int(os.getenv("TRACKING_SERVER_PORT", str(DEFAULT_PORT)))


def get_ui_port() -> int:
    """Get UI port from environment or default."""
    return int(os.getenv("UI_PORT", str(DEFAULT_UI_PORT)))


def get_api_url() -> str:
    """Get full API URL."""
    host = get_host()
    port = get_port()
    return f"http://{host}:{port}"
