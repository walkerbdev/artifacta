"""Artifact file collection and metadata extraction utilities.

This module handles the discovery, analysis, and metadata extraction of artifact files.
It provides a unified interface for collecting both individual files and entire directories,
with intelligent MIME type detection and optional content inlining for small text files.

Architecture:
    The module operates in two main modes:

    1. Single file collection: Extract metadata from one file
    2. Directory collection: Recursively discover and process all files

    Both modes produce a consistent data structure that's agnostic to the artifact type,
    allowing downstream systems to handle any file uniformly.

Key Features:
    - MIME type detection: Uses Python's mimetypes library with fallback heuristics
    - Content inlining: Optionally embeds small text files (< 100KB by default)
    - Text detection: Multi-stage approach (MIME type + read test + encoding detection)
    - Metadata tagging: Automatic classification (code, image, tabular, etc.)
    - Directory traversal: Recursive with hidden file filtering

MIME Detection Algorithm:
    1. Try mimetypes.guess_type() based on file extension
    2. If None, attempt to read first 1KB as UTF-8 text
    3. If successful -> "text/plain", if fails -> "application/octet-stream"
    4. Check if MIME type is text-like (text/*, application/json, etc.)

Content Inlining Strategy:
    - Only inline text files (not binary)
    - Only if file size <= max_inline_size (default 100KB)
    - If read fails (encoding errors), mark as non-text
    - This avoids loading large files or binary data into memory

File Type Classification:
    - Code: Based on CODE_EXTENSIONS set (40+ programming languages)
    - Image: Based on MIME type starting with "image/"
    - Tabular: Based on MIME type or .csv extension
    - This metadata helps the UI render appropriate previews
"""

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Union

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
    path: Union[str, Path], include_content: bool = False, max_inline_size: int = 100_000
) -> Dict[str, Any]:
    """Collect file metadata from a path (file or directory) with recursive traversal.

    This is the main entry point for artifact collection. It handles both single files
    and entire directory trees, producing a unified data structure for storage in the
    tracking server.

    Collection Algorithm:
        1. Validate path exists (raise FileNotFoundError if not)
        2. Determine collection mode:
           - File mode: Process single file with its name as relative path
           - Directory mode: Recursively discover all files via rglob("*")
        3. For each file:
           - Skip hidden files (starting with ".")
           - Extract full metadata via _extract_file_info()
           - Accumulate total size
        4. Return unified structure with files list and summary statistics

    Directory traversal:
        - Uses Path.rglob("*") for recursive globbing (depth-first)
        - Filters out directories (only collects actual files)
        - Sorts file paths for deterministic ordering
        - Computes relative paths from directory root for portability

    Why unified structure:
        - Same format for single file vs directory artifacts
        - Downstream code doesn't need to special-case different artifact types
        - Easy to serialize to JSON for database storage
        - Frontend can render both cases with same component

    Args:
        path: Path to file or directory to collect
        include_content: Whether to inline text file content (default False)
                         Set True for code artifacts, False for large model checkpoints
        max_inline_size: Maximum file size in bytes to inline (default 100KB)
                         Files larger than this are never inlined, even if text

    Returns:
        Dictionary with:
            - files: List of file dictionaries with path, mime_type, content, metadata
            - total_files: Count of files collected (int)
            - total_size: Total size in bytes across all files (int)

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If path is neither file nor directory (e.g., socket, device)
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
) -> Dict[str, Any]:
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


def files_to_json(files_data: Dict[str, Any]) -> str:
    """Convert files data structure to JSON string for storage."""
    return json.dumps(files_data, indent=None, separators=(",", ":"))


def json_to_files(json_str: str) -> Dict[str, Any]:
    """Parse JSON string back to files data structure."""
    return json.loads(json_str)
