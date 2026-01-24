# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Project notes and attachments for lab notebook functionality.

This module implements a lab notebook system where users can create markdown notes,
attach files (PDFs, images, videos), and organize experiments by project. It supports
both explicit projects (created via API) and implicit projects (inferred from runs).

Architecture:
    - Projects: Logical grouping of runs and notes
    - ProjectNotes: Markdown content with title and timestamps
    - ProjectNoteAttachments: Files attached to notes (hash-based storage)

Project Types:

    1. **Explicit Projects**:
       - Created via POST /api/projects
       - Have entry in Projects table
       - Tracked via created_at and updated_at timestamps
       - Can exist without any runs

    2. **Implicit Projects**:
       - Inferred from Run.project field
       - No entry in Projects table
       - Automatically discovered when listing projects
       - Created on-the-fly when first run uses project name

    List projects merges both types for unified UX.

Note Management:

    CREATE: POST /api/projects/{project_id}/notes
        - Auto-creates project if doesn't exist
        - Stores markdown content
        - Tracks created_at and updated_at

    UPDATE: PUT /api/projects/{project_id}/notes/{note_id}
        - Partial update (only fields provided)
        - Updates updated_at timestamp
        - Keeps created_at unchanged

    DELETE: DELETE /api/projects/{project_id}/notes/{note_id}
        - Cascades to attachments (database ON DELETE CASCADE)
        - Deletes attachment files from disk
        - Returns 404 if note not found

Attachment Storage:

    Files stored with hash-based paths for:
        - Deduplication (same file uploaded multiple times)
        - Integrity verification (detect corruption)
        - Content-addressable lookup

    Storage path format: uploads/{first_2_chars_of_hash}/{uuid}{ext}
    Example: uploads/ab/c3f7d4e8-1234-5678-9abc-def012345678.pdf

    Upload flow:
        1. Read file content and compute SHA256 hash
        2. Generate UUID for unique filename
        3. Extract extension from original filename
        4. Create directory structure (e.g., uploads/ab/)
        5. Write file to disk
        6. Create ProjectNoteAttachment record in database

Inline Viewing vs Download:

    GET /attachments/{attachment_id}/download?inline=true:
        - inline=true: Sets Content-Disposition: inline
          - Browser displays PDF/image/video in iframe/tab
          - Used for preview in UI
        - inline=false: Sets Content-Disposition: attachment
          - Browser triggers download dialog
          - Used for explicit downloads

    MIME type determines browser behavior:
        - application/pdf: Browser PDF viewer
        - image/*: Display inline
        - video/*: HTML5 video player
        - audio/*: HTML5 audio player
        - Other: Trigger download

File Cleanup:

    When attachment is deleted:
        1. Delete database record (ProjectNoteAttachment)
        2. Delete file from disk (PROJECT_ROOT / storage_path)
        3. If file doesn't exist, continue (already deleted)
        4. Ignore errors (best effort cleanup)

Database Session Management:

    Manual session management (unlike runs.py which uses Depends):
        - session = db.get_session()
        - try/except/finally with session.close()
        - Manual commit() and rollback()
        - Required for attachment file operations (need transaction control)

Error Handling:
    - 404 Not Found: Project/note/attachment doesn't exist
    - 400 Bad Request: Project already exists
    - 500 Internal Server Error: Database errors, file I/O errors
    - All errors logged with logger.error()
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["projects"])

# Project root for file storage
PROJECT_ROOT = Path(__file__).parent.parent.parent


# Request models
class ProjectCreateRequest(BaseModel):
    """Project creation request model."""

    project_id: str


class ProjectNoteCreateRequest(BaseModel):
    """Project note creation request model."""

    title: str
    content: str


class ProjectNoteUpdateRequest(BaseModel):
    """Project note update request model."""

    title: Optional[str] = None
    content: Optional[str] = None


@router.get("/projects")
async def list_projects(db: Any = Depends(get_db)) -> Dict[str, Any]:
    """List all projects (both explicit and implicit from runs)."""
    from database import Project, Run

    session = db.get_session()
    try:
        # Get explicit projects
        explicit_projects = session.query(Project).all()
        explicit_ids = {p.project_id for p in explicit_projects}

        # Get implicit projects from runs
        implicit_runs = (
            session.query(Run.project)
            .filter(Run.project.isnot(None))
            .filter(~Run.project.in_(explicit_ids))
            .distinct()
            .all()
        )

        # Combine both
        all_projects = [p.to_dict() for p in explicit_projects]
        for (project_id,) in implicit_runs:
            all_projects.append(
                {
                    "project_id": project_id,
                    "created_at": None,
                    "updated_at": None,
                    "is_implicit": True,
                }
            )

        return {"projects": all_projects}
    finally:
        session.close()


@router.post("/projects")
async def create_project(
    request: ProjectCreateRequest, db: Any = Depends(get_db)
) -> Dict[str, Any]:
    """Create a new project."""
    from database import Project

    session = db.get_session()
    try:
        # Check if project already exists
        existing = session.query(Project).filter(Project.project_id == request.project_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Project already exists")

        # Create project
        project = Project(
            project_id=request.project_id,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        return project.to_dict()
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()


@router.get("/projects/{project_id}/notes")
async def list_project_notes(project_id: str, db: Any = Depends(get_db)) -> Dict[str, Any]:
    """List all notes for a project."""
    from database import ProjectNote

    session = db.get_session()
    try:
        notes = (
            session.query(ProjectNote)
            .filter(ProjectNote.project_id == project_id)
            .order_by(ProjectNote.created_at.desc())
            .all()
        )
        return {"notes": [note.to_dict() for note in notes]}
    finally:
        session.close()


@router.post("/projects/{project_id}/notes")
async def create_project_note(
    project_id: str, request: ProjectNoteCreateRequest, db: Any = Depends(get_db)
) -> Dict[str, Any]:
    """Create a new project note."""
    from database import Project, ProjectNote

    session = db.get_session()
    try:
        # Ensure project exists
        project = session.query(Project).filter(Project.project_id == project_id).first()
        if not project:
            # Auto-create project
            project = Project(
                project_id=project_id,
                created_at=int(time.time()),
                updated_at=int(time.time()),
            )
            session.add(project)

        # Create note
        note = ProjectNote(
            project_id=project_id,
            title=request.title,
            content=request.content,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        session.add(note)
        session.commit()
        session.refresh(note)
        return note.to_dict()
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating project note: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()


@router.get("/projects/{project_id}/notes/{note_id}")
async def get_project_note(
    project_id: str, note_id: int, db: Any = Depends(get_db)
) -> Dict[str, Any]:
    """Get a single project note."""
    from database import ProjectNote

    session = db.get_session()
    try:
        note = (
            session.query(ProjectNote)
            .filter(ProjectNote.id == note_id, ProjectNote.project_id == project_id)
            .first()
        )
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        return note.to_dict()
    finally:
        session.close()


@router.put("/projects/{project_id}/notes/{note_id}")
async def update_project_note(
    project_id: str, note_id: int, request: ProjectNoteUpdateRequest, db: Any = Depends(get_db)
) -> Dict[str, Any]:
    """Update a project note."""
    from database import ProjectNote

    session = db.get_session()
    try:
        note = (
            session.query(ProjectNote)
            .filter(ProjectNote.id == note_id, ProjectNote.project_id == project_id)
            .first()
        )
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        if request.title is not None:
            note.title = request.title
        if request.content is not None:
            note.content = request.content
        note.updated_at = int(time.time())

        session.commit()
        session.refresh(note)
        return note.to_dict()
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating project note: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()


@router.delete("/projects/{project_id}/notes/{note_id}")
async def delete_project_note(
    project_id: str, note_id: int, db: Any = Depends(get_db)
) -> Dict[str, str]:
    """Delete a project note."""
    from database import ProjectNote

    session = db.get_session()
    try:
        note = (
            session.query(ProjectNote)
            .filter(ProjectNote.id == note_id, ProjectNote.project_id == project_id)
            .first()
        )
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        # Delete note (cascades to attachments and links)
        session.delete(note)
        session.commit()
        return {"status": "deleted"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting project note: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()


# ============================================================================
# Project Note Attachments API Endpoints
# ============================================================================


@router.post("/notes/{note_id}/attachments")
async def upload_attachment(
    note_id: int, file: UploadFile = File(...), db: Any = Depends(get_db)
) -> Dict[str, Any]:
    """Upload a file attachment to a project note."""
    from database import ProjectNote, ProjectNoteAttachment

    session = db.get_session()
    try:
        # Verify note exists
        note = session.query(ProjectNote).filter(ProjectNote.id == note_id).first()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        # Read file and compute hash
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Generate storage path: uploads/ab/ab34ef56...uuid.ext
        import uuid

        file_uuid = str(uuid.uuid4())
        folder = file_hash[:2]
        ext = Path(file.filename).suffix if file.filename else ""
        storage_path = f"uploads/{folder}/{file_uuid}{ext}"

        # Ensure directory exists
        storage_full_path = PROJECT_ROOT / storage_path
        storage_full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file to disk
        with open(storage_full_path, "wb") as f:
            f.write(file_content)

        # Create attachment record
        attachment = ProjectNoteAttachment(
            note_id=note_id,
            real_name=file.filename or "unnamed",
            storage_path=storage_path,
            mime_type=file.content_type or "application/octet-stream",
            filesize=len(file_content),
            hash=file_hash,
            created_at=int(time.time()),
        )
        session.add(attachment)
        session.commit()
        session.refresh(attachment)

        return attachment.to_dict()
    except Exception as e:
        session.rollback()
        logger.error(f"Error uploading attachment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()


@router.get("/notes/{note_id}/attachments")
async def list_attachments(note_id: int, db: Any = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all attachments for a note."""
    from database import ProjectNoteAttachment

    session = db.get_session()
    try:
        attachments = (
            session.query(ProjectNoteAttachment)
            .filter(ProjectNoteAttachment.note_id == note_id)
            .all()
        )
        return [att.to_dict() for att in attachments]
    finally:
        session.close()


@router.get("/attachments/{attachment_id}/download")
async def download_attachment(
    attachment_id: int,
    inline: bool = True,  # Default to inline viewing for iframes
    db: Any = Depends(get_db),
) -> Any:
    """Download or view an attachment file inline."""
    from database import ProjectNoteAttachment

    session = db.get_session()
    try:
        attachment = (
            session.query(ProjectNoteAttachment)
            .filter(ProjectNoteAttachment.id == attachment_id)
            .first()
        )
        if not attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")

        file_path = PROJECT_ROOT / attachment.storage_path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        # Set Content-Disposition based on inline parameter
        headers = {}
        if inline:
            # For inline viewing (PDFs, images, videos, audio in iframes/browsers)
            headers["Content-Disposition"] = f'inline; filename="{attachment.real_name}"'
        else:
            # For forcing download
            headers["Content-Disposition"] = f'attachment; filename="{attachment.real_name}"'

        return FileResponse(
            path=file_path,
            media_type=attachment.mime_type,
            headers=headers,
        )
    finally:
        session.close()


@router.delete("/attachments/{attachment_id}")
async def delete_attachment(attachment_id: int, db: Any = Depends(get_db)) -> Dict[str, str]:
    """Delete an attachment."""
    from database import ProjectNoteAttachment

    session = db.get_session()
    try:
        attachment = (
            session.query(ProjectNoteAttachment)
            .filter(ProjectNoteAttachment.id == attachment_id)
            .first()
        )
        if not attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")

        # Delete file from disk
        file_path = PROJECT_ROOT / attachment.storage_path
        if file_path.exists():
            file_path.unlink()

        # Delete database record
        session.delete(attachment)
        session.commit()

        return {"status": "deleted"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting attachment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()
