# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Artifact storage and retrieval endpoints for tracking server.

This module implements content-addressable artifact storage with deduplication,
inline content support, and file serving capabilities. It bridges between the
SDK's artifact logging and the UI's artifact viewing/downloading needs.

Key Features:
    - Content-addressable storage via SHA256 hashing
    - Automatic deduplication (same hash reuses existing artifact)
    - Support for both filesystem and virtual (inline) artifacts
    - File preview with pagination for large artifacts
    - Individual file serving from multi-file artifacts
    - Download endpoints for artifact export
    - Artifact role tracking (input vs output)
    - Provenance tracking via hash.code tags

Artifact Storage Models:

    1. **Filesystem Artifacts** (storage_path: actual file path):
       - Large files (model checkpoints, datasets, videos)
       - Files remain on user's filesystem
       - storage_path points to actual location
       - metadata stored in database
       - content field is NULL

    2. **Virtual Artifacts** (storage_path: virtual://...):
       - Small text files (code, configs, logs)
       - Content inlined in database (content field)
       - No actual filesystem storage
       - Faster access (no disk I/O)
       - Better for version control

Content-Addressable Storage Algorithm:

    When SDK logs artifact:
    1. Compute SHA256 hash of artifact content
    2. Query database for existing artifact with same hash
    3. If exists:
       a. Reuse existing artifact_id
       b. Create new ArtifactLink with current run_id
       c. Skip storage (file already exists)
    4. If not exists:
       a. Generate new artifact_id (art_XXXXXXXX)
       b. Create Artifact record with hash and storage_path
       c. Create ArtifactLink with current run_id
       d. For virtual artifacts, store content in database

    Benefits:
        - Deduplication saves storage (checkpoints reused across runs)
        - Hash ensures integrity (detect file corruption)
        - Provenance tracking (which runs used which artifacts)

Artifact Links (Many-to-Many):

    The ArtifactLink table enables artifact reuse:
    - One artifact can be linked to multiple runs
    - One run can link to multiple artifacts
    - Role field distinguishes "input" vs "output"
    - created_at tracks when link was established

    Use cases:
        - Pretrained model used as input by multiple fine-tuning runs
        - Dataset artifact shared across experiment runs
        - Best checkpoint from run A used as input to run B

File Serving Strategy:

    GET /artifact/{artifact_id}/files/{filename}:
        1. Check if virtual artifact (storage_path starts with "virtual://")
        2. If virtual:
           a. Parse content JSON
           b. Find file by filename in files array
           c. Return file content with appropriate MIME type
        3. If filesystem:
           a. Resolve file path (storage_path + filename)
           b. Determine MIME type from extension
           c. Return FileResponse with streaming

    MIME type handling:
        - Explicit map for common types (mp4, png, pdf, etc.)
        - Fallback to application/octet-stream
        - text/* and application/json served inline
        - Other types trigger download

Provenance Tracking:

    Code artifacts automatically update hash.code tag:
    1. Parse artifact content JSON
    2. Check if any file has metadata.type == "code"
    3. If yes, create/update Tag with key="hash.code", value=artifact_hash
    4. UI can show code hash for reproducibility
    5. Users can verify code hasn't changed across runs

Preview Pagination:

    For large artifacts with many files (e.g., dataset with 10K images):
    - offset, limit parameters control pagination
    - Returns only requested slice of files
    - has_more flag indicates more files available
    - Frontend can load files incrementally

Error Handling:
    - 404 Not Found: Artifact ID doesn't exist, file not found on disk
    - 500 Internal Server Error: Database errors, JSON parsing errors
    - All exceptions logged with exc_info=True for debugging
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from database import Artifact, ArtifactLink, Tag
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .dependencies import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["artifacts"])


# Request/Response models
class ArtifactCreateRequest(BaseModel):
    """Artifact creation request model."""

    run_id: str
    name: str
    hash: str
    storage_path: str
    size_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[str] = None  # JSON file collection structure
    role: Optional[str] = None  # "input" or "output" - None means output for backwards compat


class ArtifactCreateResponse(BaseModel):
    """Artifact creation response model."""

    artifact_id: str


@router.post("/artifacts", response_model=ArtifactCreateResponse)
async def create_artifact(
    request: ArtifactCreateRequest,
    session: Session = Depends(get_session),
) -> ArtifactCreateResponse:
    """Register an artifact for a run.

    Called by artifacta SDK when run.log_artifact() is invoked.
    Stores metadata + file collection JSON - actual files stay on user's filesystem.
    """
    try:
        # Check if artifact with this hash already exists
        existing = session.query(Artifact).filter(Artifact.hash == request.hash).first()

        if existing:
            artifact_id = str(existing.artifact_id)
            logger.info(f"Reusing existing artifact: {artifact_id} (hash={request.hash[:8]}...)")
        else:
            artifact_id = f"art_{uuid.uuid4().hex[:16]}"
            logger.info(f"Creating artifact: {artifact_id} for run {request.run_id}")

            artifact = Artifact(
                artifact_id=artifact_id,
                run_id=request.run_id,
                name=request.name,
                hash=request.hash,
                storage_path=request.storage_path,
                size_bytes=request.size_bytes,
                meta=json.dumps(request.metadata) if request.metadata else None,
                content=request.content,
                created_at=int(time.time() * 1000),
            )
            session.add(artifact)
            session.flush()

        # Create artifact link with role (default to 'output' for backwards compatibility)
        role = request.role if request.role else "output"
        link_id = f"link_{uuid.uuid4().hex[:16]}"

        artifact_link = ArtifactLink(
            link_id=link_id,
            artifact_id=artifact_id,
            run_id=request.run_id,
            role=role,
            created_at=int(time.time() * 1000),
        )
        session.add(artifact_link)
        session.flush()

        # If content has code files, update the run's hash.code tag for provenance
        if request.content:
            try:
                content_data = json.loads(request.content)
                has_code = any(
                    f.get("metadata", {}).get("type") == "code"
                    for f in content_data.get("files", [])
                )
            except (json.JSONDecodeError, KeyError):
                has_code = False

            if has_code:
                # Create or update hash.code tag
                code_tag = (
                    session.query(Tag)
                    .filter(Tag.run_id == request.run_id, Tag.key == "hash.code")
                    .first()
                )

                if code_tag:
                    code_tag.value = request.hash  # type: ignore[assignment]
                else:
                    code_tag = Tag(
                        run_id=request.run_id,
                        key="hash.code",
                        value=request.hash,
                    )
                    session.add(code_tag)

                session.flush()

        return ArtifactCreateResponse(artifact_id=artifact_id)

    except Exception as e:
        logger.error(f"Failed to create artifact: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/artifacts/{run_id}")
async def list_artifacts(
    run_id: str,
    session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    """List all artifacts for a run.

    Returns artifact metadata without content (content is fetched separately).
    """
    try:
        artifacts = session.query(Artifact).filter(Artifact.run_id == run_id).all()

        result = []
        for artifact in artifacts:
            result.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "run_id": artifact.run_id,
                    "name": artifact.name,
                    "hash": artifact.hash,
                    "storage_path": artifact.storage_path,
                    "size_bytes": artifact.size_bytes,
                    "metadata": json.loads(artifact.meta) if artifact.meta else {},
                    "created_at": artifact.created_at,
                }
            )

        return result

    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/artifact/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Get a single artifact by ID.

    Returns artifact metadata and content (if inline).
    """
    try:
        artifact = session.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        result = {
            "artifact_id": artifact.artifact_id,
            "run_id": artifact.run_id,
            "name": artifact.name,
            "hash": artifact.hash,
            "storage_path": artifact.storage_path,
            "size_bytes": artifact.size_bytes,
            "metadata": json.loads(artifact.meta) if artifact.meta else {},
            "created_at": artifact.created_at,
        }

        # Include content if it exists (for virtual artifacts)
        if artifact.content:
            try:
                result["content"] = json.loads(artifact.content)
            except json.JSONDecodeError:
                result["content"] = artifact.content

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get artifact: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/artifact/{artifact_id}/preview")
async def get_artifact_preview(
    artifact_id: str,
    offset: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Get preview of artifact content.

    For code artifacts, returns paginated file list with content.
    """
    try:
        artifact = session.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if not artifact.content:
            return {"files": [], "total_files": 0, "has_more": False}

        try:
            content_data = json.loads(artifact.content)
            all_files = content_data.get("files", [])
            total_files = len(all_files)

            # Paginate files
            paginated_files = all_files[offset : offset + limit]
            has_more = (offset + limit) < total_files

            return {
                "files": paginated_files,
                "total_files": total_files,
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
            }

        except json.JSONDecodeError:
            return {"error": "Invalid content format"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get artifact preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/artifact/{artifact_id}/files/{filename:path}", response_model=None)
async def get_artifact_file(
    artifact_id: str,
    filename: str,
    session: Session = Depends(get_session),
) -> Union[FileResponse, JSONResponse, Response]:
    """Get a specific file from an artifact.

    For inline artifacts (text/code), returns the content directly.
    For filesystem artifacts (media), streams the file with appropriate MIME type.
    """
    try:
        from fastapi.responses import Response

        artifact = session.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Check if it's a filesystem artifact
        if not artifact.storage_path.startswith("virtual://"):
            # Real filesystem artifact - look for the file in storage_path
            storage_path = Path(artifact.storage_path)

            # If storage_path is a directory, look for the file inside it
            file_path = storage_path / filename if storage_path.is_dir() else storage_path

            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {filename}")

            # Return file with appropriate MIME type
            # Determine MIME type from file extension
            mime_type = "application/octet-stream"
            ext = file_path.suffix.lower()
            mime_map = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
            }
            mime_type = mime_map.get(ext, mime_type)

            return FileResponse(path=str(file_path), media_type=mime_type, filename=file_path.name)

        # Virtual artifact - return inline content
        if not artifact.content:
            raise HTTPException(status_code=404, detail="No inline content available")

        try:
            content_data = json.loads(artifact.content)
            files = content_data.get("files", [])

            # Find the requested file
            for file_obj in files:
                if file_obj.get("path") == filename:
                    # For inline files, return the content with appropriate MIME type
                    content = file_obj.get("content", "")
                    mime_type = file_obj.get("mime_type", "text/plain")

                    # Return as appropriate response type
                    if mime_type.startswith("text/") or mime_type in [
                        "application/json",
                        "application/x-yaml",
                    ]:
                        return Response(content=content, media_type=mime_type)
                    else:
                        return JSONResponse(content=file_obj)

            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail="Invalid content format") from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get artifact file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/artifact/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    session: Session = Depends(get_session),
) -> FileResponse:
    """Download an artifact file.

    For filesystem artifacts, returns the actual file.
    For virtual artifacts, returns synthesized file from content.
    """
    try:
        artifact = session.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Check if it's a filesystem artifact
        if not artifact.storage_path.startswith("virtual://"):
            path = Path(artifact.storage_path)
            if path.exists() and path.is_file():
                return FileResponse(
                    path=str(path),
                    filename=artifact.name,
                    media_type="application/octet-stream",
                )
            else:
                raise HTTPException(status_code=404, detail="Artifact file not found on disk")

        # Virtual artifact - return content as file
        if artifact.content:
            # Create temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=f"_{artifact.name}"
            ) as tmp:
                tmp.write(str(artifact.content))
                tmp_path = tmp.name

            return FileResponse(
                path=tmp_path,
                filename=artifact.name,
                media_type="application/octet-stream",
            )
        else:
            raise HTTPException(status_code=404, detail="No content available for download")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download artifact: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


def flatten_metadata(metadata: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten nested metadata dictionary using dot notation.

    Example: {"build": {"version": "1.2.3"}} -> {"build.version": "1.2.3"}.
    """
    result = {}
    for key, value in metadata.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metadata(value, new_key))
        else:
            result[new_key] = str(value)
    return result


@router.get("/runs/{run_id}/artifact-links")
async def get_artifact_links(
    run_id: str,
    session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    """Get all artifact links for a run with their role (input/output).

    Used by lineage visualization and Files & Artifacts panel.
    Returns flattened artifact properties for UI compatibility.
    """
    try:
        links = session.query(ArtifactLink).filter(ArtifactLink.run_id == run_id).all()

        result = []
        for link in links:
            # Fetch artifact details
            artifact = (
                session.query(Artifact).filter(Artifact.artifact_id == link.artifact_id).first()
            )

            if artifact:
                # Parse and flatten metadata
                metadata = {}
                if artifact.meta:
                    try:
                        parsed_meta = json.loads(artifact.meta)
                        metadata = flatten_metadata(parsed_meta)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Flatten artifact properties to match UI expectations
                result.append(
                    {
                        "link_id": link.link_id,
                        "artifact_id": link.artifact_id,
                        "run_id": link.run_id,
                        "role": link.role,
                        "created_at": link.created_at,
                        "name": artifact.name,
                        "hash": artifact.hash,
                        "storage_path": artifact.storage_path,
                        "size_bytes": artifact.size_bytes,
                        "content": artifact.content,
                        "metadata": metadata,  # Flattened metadata
                    }
                )

        return result

    except Exception as e:
        logger.error(f"Failed to get artifact links: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
