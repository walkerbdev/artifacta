# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""Artifact management endpoints - SQLAlchemy ORM version."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from database import Artifact, ArtifactLink, Tag
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
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
            artifact_id = existing.artifact_id
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
                    code_tag.value = request.hash
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
) -> Union[FileResponse, JSONResponse]:
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
                tmp.write(artifact.content)
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
