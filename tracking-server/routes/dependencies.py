# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Dependency injection helpers for route modules.

FastAPI dependency injection pattern for accessing global state:
- Database connection (SQLAlchemy)
- Database session (with automatic commit/rollback)
- WebSocket clients set (for real-time broadcasting)

Architecture:
- Middleware injects global state into request.state (see main.py)
- These functions extract state and provide it to route handlers
- Enables testing by mocking request.state
- Follows FastAPI Depends() pattern for clean separation

Session lifecycle:
- get_session() yields a session (context manager pattern)
- Automatically commits on success
- Automatically rolls back on exception
- Always closes session in finally block
- Prevents connection leaks and ensures transaction integrity
"""

from typing import Any, Generator, Set

from fastapi import Request, WebSocket
from sqlalchemy.orm import Session


def get_db(request: Request) -> Any:
    """Get SQLAlchemy database from request state."""
    return request.state.db


def get_session(request: Request) -> Generator[Session, None, None]:
    """Get SQLAlchemy session for database operations.

    Yields a session and ensures it's properly closed after request.
    """
    db = request.state.db
    session = db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_websocket_clients(request: Request) -> Set[WebSocket]:
    """Get websocket clients set from request state."""
    return request.state.websocket_clients
