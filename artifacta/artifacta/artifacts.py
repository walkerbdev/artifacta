"""Artifact file collection utilities."""

import json
import mimetypes
from pathlib import Path
from typing import Any

# File extensions that indicate code files (for hash.code tag detection)
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".sql",
    ".r",
    ".R",
    ".m",
    ".lua",
}


def collect_files(
    path: str | Path, include_content: bool = False, max_inline_size: int = 100_000
) -> dict[str, Any]:
    """Collect file metadata from a path (file or directory).

    Returns unified structure for all artifact types - agnostic to content.

    Args:
        path: Path to file or directory
        include_content: Whether to inline text file content
        max_inline_size: Maximum file size (bytes) to inline

    Returns:
        dict with:
            - files: List of file dicts with path, mime_type, content, metadata
            - total_files: Count of files
            - total_size: Total size in bytes
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Get list of file paths to process
    if path_obj.is_file():
        file_paths = [(path_obj, path_obj.name)]  # (abs_path, rel_path)
    elif path_obj.is_dir():
        file_paths = [
            (f, str(f.relative_to(path_obj)))
            for f in sorted(path_obj.rglob("*"))
            if f.is_file() and not f.name.startswith(".")  # Skip hidden files
        ]
    else:
        raise ValueError(f"Path is neither file nor directory: {path}")

    files = []
    total_size = 0

    for abs_path, rel_path in file_paths:
        file_info = _extract_file_info(abs_path, rel_path, include_content, max_inline_size)
        files.append(file_info)
        total_size += file_info["size"]

    return {
        "files": files,
        "total_files": len(files),
        "total_size": total_size,
    }


def _extract_file_info(
    abs_path: Path, rel_path: str, include_content: bool, max_inline_size: int
) -> dict[str, Any]:
    """Extract metadata and optional content from a single file."""
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(abs_path))
    if mime_type is None:
        # Try to detect if it's text
        try:
            with open(abs_path, encoding="utf-8") as f:
                f.read(1024)  # Try reading first KB
            mime_type = "text/plain"
        except (UnicodeDecodeError, PermissionError):
            mime_type = "application/octet-stream"

    # Determine if text
    is_text = mime_type.startswith("text/") or mime_type in [
        "application/json",
        "application/yaml",
        "application/x-yaml",
        "application/xml",
        "application/javascript",
    ]

    file_size = abs_path.stat().st_size

    file_info = {
        "path": rel_path,
        "size": file_size,
        "mime_type": mime_type,
        "is_text": is_text,
        "content": None,
        "metadata": {},
    }

    # Include content for small text files
    if include_content and is_text and file_size <= max_inline_size:
        try:
            with open(abs_path, encoding="utf-8") as f:
                file_info["content"] = f.read()
        except (UnicodeDecodeError, PermissionError):
            # Not actually text or can't read
            file_info["is_text"] = False
            file_info["content"] = None

    # Add file-specific metadata
    ext = abs_path.suffix.lower()
    if mime_type == "text/csv" or ext == ".csv":
        file_info["metadata"]["type"] = "tabular"
    elif mime_type.startswith("image/"):
        file_info["metadata"]["type"] = "image"
    elif ext in CODE_EXTENSIONS:
        file_info["metadata"]["type"] = "code"

    return file_info


def files_to_json(files_data: dict[str, Any]) -> str:
    """Convert files data structure to JSON string for storage."""
    return json.dumps(files_data, indent=None, separators=(",", ":"))


def json_to_files(json_str: str) -> dict[str, Any]:
    """Parse JSON string back to files data structure."""
    return json.loads(json_str)
