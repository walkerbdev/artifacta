# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Dependency injection helpers for route modules."""

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
