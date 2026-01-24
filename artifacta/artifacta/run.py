"""Run management with provenance tracking and artifact logging.

The Run class is the core abstraction for experiment tracking in Artifacta.
It orchestrates metric logging, artifact management, and system monitoring
while maintaining full reproducibility through content-addressed storage.

Key components:
- HTTPEmitter: Real-time communication with tracking server (MLflow/W&B pattern)
- SystemMonitor: Background thread for CPU/memory/GPU metrics
- Artifact hashing: SHA256-based content addressing for reproducibility
- Auto-provenance: Automatic capture of config, dependencies, environment
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

from .artifacts import collect_files, files_to_json
from .emitter import HTTPEmitter
from .monitor import SystemMonitor
from .primitives import (
    PRIMITIVE_TYPES,
    auto_convert,
)


class Run:
    """A single training run with provenance tracking and artifact management.

    The Run class is the core abstraction for experiment tracking in Artifacta.
    It handles:
    - Automatic provenance capture (config, dependencies, environment, code)
    - Real-time metric emission to tracking server via HTTP
    - Artifact logging with content-addressed storage (SHA256 hashing)
    - System monitoring (CPU, memory, GPU) in background thread
    - Graceful degradation when tracking server is unavailable

    Architecture:
    - HTTPEmitter: Sends data to tracking server in real-time (MLflow/W&B pattern)
    - SystemMonitor: Background thread that captures system metrics every N seconds
    - Artifact hashing: SHA256 of file contents ensures reproducibility tracking

    Lifecycle:
    1. __init__: Create run object with metadata
    2. start(): Auto-log provenance artifacts, start system monitoring
    3. log()/log_artifact(): Log data during training
    4. finish(): Stop monitoring, close connections

    Example:
        >>> import artifacta as ds
        >>> run = ds.init(project="mnist", name="exp-1", config={"lr": 0.001})
        >>> run.log("metrics", {"epoch": [1,2,3], "loss": [0.5, 0.3, 0.2]})
        >>> run.log_output("model.pt")
        >>> run.finish()
    """

    def __init__(self, project, name, config, code_dir=None):
        """Initialize a Run instance with metadata.

        Args:
            project: Project name for grouping related experiments.
            name: Human-readable run name (auto-generated if None).
            config: Configuration dictionary (hyperparameters, settings).
            code_dir: Optional code directory path for artifact logging.
        """
        self.id = f"run_{int(time.time() * 1000)}"
        self.project = project
        self.name = name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.config = config
        self.code_dir = code_dir

        # HTTP emitter (API Gateway is source of truth)
        self.http_emitter = HTTPEmitter(self.id)

        # System monitor
        self.monitor = None

        # State
        self.step = 0
        self.started_at = None
        self.finished = False
        self.summary = {}  # Latest metrics
        self.code_artifact_hash = None  # Hash of logged code artifact (if any)

    def start(self):
        """Start the run and auto-log provenance artifacts.

        Initialization sequence:
        1. Emit run creation to tracking server (creates database entry)
        2. Auto-log provenance artifacts as inputs:
           - config.json: Hyperparameter configuration (JSON)
           - requirements.txt: pip freeze output (dependencies)
           - environment.json: Platform info (Python version, OS, CUDA)
        3. Start system monitoring background thread (1-second interval)

        Why 1-second interval for system monitoring:
        - Short training runs (< 30s) need faster sampling to capture metrics
        - Balances data granularity with overhead
        - Can be adjusted for longer experiments

        Graceful degradation:
        - If tracking server is unavailable, HTTPEmitter disables itself
        - Run continues normally, but metrics aren't persisted
        - Useful for offline development
        """
        self.started_at = datetime.utcnow()

        # Emit run creation to API Gateway (simplified - no config/tags)
        self.http_emitter.emit_init({"run_id": self.id, "project": self.project, "name": self.name})

        # Auto-log provenance artifacts as inputs
        self._log_config_artifact()  # Config as artifact
        self._log_dependencies_artifact()  # pip freeze as artifact
        self._log_environment_artifact()  # Platform info as artifact

        # Start system monitoring (emits to API Gateway)
        # Use 1 second interval for faster sampling (important for short training runs)
        self.monitor = SystemMonitor(interval=1, http_emitter=self.http_emitter)
        self.monitor.start()

        print(f"Run started: {self.name}")
        print(f"   ID: {self.id}")
        print(f"   Project: {self.project}")

    def log(self, name: str, data, section: str = None):
        """Log structured data - accepts primitives OR plain Python types (auto-converted).

        How it works:
        1. Auto-convert plain Python types to primitives (see primitives.py:auto_convert)
        2. Extract primitive type from PRIMITIVE_TYPES mapping
        3. Convert primitive to dict representation
        4. Emit to tracking server via HTTP (real-time)
        5. Server broadcasts to WebSocket clients for live UI updates

        Auto-conversion rules:
        - dict with list values → Series primitive
        - 1D list/array → Distribution primitive
        - 2D list/array → Matrix primitive
        - numpy arrays → Distribution or Matrix based on dimensions

        Args:
            name: Name for this data object (e.g., "training_metrics", "confusion_matrix")
            data: Can be:
                  - Data primitive (Series, Distribution, Matrix, etc.) - used as-is
                  - Plain dict (auto-converted to Series) - e.g., {"epoch": [1,2,3], "loss": [0.5,0.3,0.2]}
                  - Plain list/array (auto-converted to Distribution or Matrix based on shape)
                  - Numpy array (auto-converted based on dimensions)
            section: Optional section/group name for organizing plots in UI (e.g., "Training Metrics", "Validation")

        Examples:
            >>> # Easy way - just log dicts/lists (auto-detected):
            >>> run.log("loss_curve", {"epoch": [1,2,3], "loss": [0.5, 0.3, 0.2]})
            >>> run.log("confusion_matrix", [[10, 2], [1, 15]])
            >>> run.log("weights", np.array([0.1, 0.2, 0.15, ...]))
            >>>
            >>> # Advanced way - explicit primitives for fine control:
            >>> run.log("training_metrics", Series(
            ...     index="epoch",
            ...     fields={"train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
            ... ), section="Training Metrics")
        """
        if self.finished:
            raise RuntimeError("Cannot log to finished run")

        # Auto-convert plain types to primitives
        data = auto_convert(data)

        # Determine primitive type
        primitive_type = PRIMITIVE_TYPES.get(type(data))
        if not primitive_type:
            raise ValueError(
                f"Unknown data type: {type(data)}. Must be one of {list(PRIMITIVE_TYPES.keys())}"
            )

        # Convert to dict
        data_dict = data.to_dict()

        # Extract metadata if present
        metadata = data_dict.pop("metadata", None)

        # Emit to API Gateway
        self.http_emitter.emit_structured_data(
            {
                "name": name,
                "primitive_type": primitive_type,
                "section": section,
                "data": data_dict,
                "metadata": metadata,
            }
        )

    def log_artifact(self, name, path, include_content=True, metadata=None, role="output"):
        """Log an artifact (file or directory) for this run with content-addressed storage.

        How it works:
        1. Collect file metadata (MIME type, size, path) using artifacts.collect_files
        2. Compute SHA256 hash of file contents for content addressing
        3. Inline text file contents if include_content=True (useful for code files)
        4. Auto-extract metadata from model files (PyTorch, TensorFlow, ONNX)
        5. Emit artifact to tracking server with metadata + content
        6. Track code artifact hash for provenance

        Content addressing (SHA256):
        - Single file: hash of file contents
        - Directory: hash of all file contents + filenames (sorted)
        - Enables reproducibility tracking and deduplication
        - Used for lineage graph and experiment comparison

        Works with both single files and directories containing multiple files.
        Each file in the collection retains its own type information (MIME type, language, etc).

        Args:
            name: Artifact name (e.g., "training_code", "model_checkpoint", "experiment_results")
            path: File or directory path
            include_content: If True, inline text file contents (useful for code/config files)
            metadata: Optional metadata dict for the artifact
            role: 'input' or 'output' (default: 'output')

        Examples:
            >>> # Log code directory with contents as input
            >>> run.log_artifact("src", "src/", include_content=True, role="input")
            >>>
            >>> # Log model checkpoint as output
            >>> run.log_artifact("checkpoint", "model.pt", role="output")
            >>>
            >>> # Log mixed experiment results (code + plots + data)
            >>> run.log_artifact("experiment_results", "results/", include_content=True)
        """
        if self.finished:
            raise RuntimeError("Cannot log to finished run")

        from pathlib import Path

        path_obj = Path(path)

        # Collect file metadata and optional content
        files_data = collect_files(path, include_content=include_content)

        # Compute hash
        artifact_hash = self._compute_artifact_hash(path_obj)

        # Convert files data to JSON for storage
        content_json = files_to_json(files_data)

        # Check if this contains code files - store hash for provenance
        has_code = any(f.get("metadata", {}).get("type") == "code" for f in files_data["files"])
        if has_code:
            self.code_artifact_hash = artifact_hash

        # Auto-extract metadata from model files
        extracted_metadata = {}
        if path_obj.is_file():
            from .metadata_extractors import extract_metadata

            extracted_metadata = extract_metadata(str(path_obj.absolute()))

        # Merge extracted metadata with user-provided metadata
        final_metadata = {**extracted_metadata, **(metadata or {})}

        # Emit to API Gateway
        self.http_emitter.emit_artifact(
            {
                "run_id": self.id,
                "name": name,
                "hash": artifact_hash,
                "storage_path": str(path_obj.absolute()),
                "size_bytes": files_data["total_size"],
                "metadata": final_metadata,
            },
            content=content_json,
            role=role,
        )

        # Print summary
        file_count = files_data["total_files"]
        size_mb = files_data["total_size"] / 1024 / 1024

        print(f"Artifact logged: {name}")
        print(f"   Path: {path}")
        print(f"   Files: {file_count}")
        print(f"   Size: {size_mb:.2f} MB")

    def log_input(self, path, name=None, include_content=True, metadata=None):
        """Log an input artifact (file or directory) for this run.

        Convenience method that calls log_artifact with role="input".

        Args:
            path: File or directory path
            name: Optional artifact name (defaults to basename of path)
            include_content: If True, inline text file contents
            metadata: Optional metadata dict for the artifact

        Examples:
            >>> # Log training data as input (name auto-detected)
            >>> run.log_input("data/train.csv")
            >>>
            >>> # Log with custom name
            >>> run.log_input("data/train.csv", name="training_data")
            >>>
            >>> # Log source code directory as input
            >>> run.log_input("src/", include_content=True)
        """
        import os

        if name is None:
            name = os.path.basename(path.rstrip("/"))
        self.log_artifact(
            name, path, include_content=include_content, metadata=metadata, role="input"
        )

    def update_config(self, new_config: dict):
        """Update run configuration with new parameters.

        Merges new_config into existing config and re-logs the config artifact.
        Useful for autolog scenarios where parameters are discovered during training
        (e.g., optimizer config, framework-specific params).

        Args:
            new_config: Dictionary of new configuration parameters to merge

        Examples:
            >>> # User initializes with their hyperparameters
            >>> run = init(project="mnist", config={"batch_size": 32, "epochs": 10})
            >>>
            >>> # Autolog discovers optimizer config during training
            >>> run.update_config({"optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-5})
            >>>
            >>> # Final config contains both user and auto-discovered params
            >>> print(run.config)
            >>> # {"batch_size": 32, "epochs": 10, "optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-5}
        """
        if new_config:
            self.config.update(new_config)
            self._log_config_artifact()  # Re-log config artifact with updated values

    def log_output(self, path, name=None, include_content=True, metadata=None):
        """Log an output artifact (file or directory) for this run.

        Convenience method that calls log_artifact with role="output".

        Args:
            path: File or directory path
            name: Optional artifact name (defaults to basename of path)
            include_content: If True, inline text file contents
            metadata: Optional metadata dict for the artifact

        Examples:
            >>> # Log trained model as output (name auto-detected)
            >>> run.log_output("checkpoints/model.pt")
            >>>
            >>> # Log with custom name
            >>> run.log_output("checkpoints/model.pt", name="final_model")
            >>>
            >>> # Log experiment results as output
            >>> run.log_output("results/", include_content=True)
        """
        import os

        if name is None:
            name = os.path.basename(path.rstrip("/"))
        self.log_artifact(
            name, path, include_content=include_content, metadata=metadata, role="output"
        )

    def _compute_artifact_hash(self, path):
        """Compute SHA256 hash of file(s) at path for content addressing.

        Hashing strategy:
        - Single file: SHA256 of raw file contents (read in 8KB chunks)
        - Directory: SHA256 of all files + filenames (sorted for determinism)

        Why include filenames in directory hash:
        - Renaming a file changes the artifact identity
        - Moving files between directories changes identity
        - Ensures that directory structure matters, not just file contents

        Performance:
        - Chunk-based reading (8KB) handles large files without memory issues
        - Recursive glob sorted by path ensures deterministic ordering

        Args:
            path: Path object (file or directory).

        Returns:
            SHA256 hash string (hex digest).
        """
        import hashlib

        if path.is_file():
            # Single file - hash content
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                # Read in chunks for large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        elif path.is_dir():
            # Directory - hash all files sorted
            hasher = hashlib.sha256()
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    # Include filename in hash
                    hasher.update(file_path.name.encode())
                    # Hash file content
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
            return hasher.hexdigest()

        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

    def _log_virtual_artifact(self, name, type, content_str, mime_type="text/plain"):
        """Log a virtual artifact (content stored in DB, no filesystem path).

        Args:
            name: Artifact name (e.g., "config.json", "requirements.txt").
            type: Artifact type (e.g., "config", "dependencies", "environment").
            content_str: String content.
            mime_type: MIME type of content.
        """
        artifact_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Create file collection structure
        content_json = json.dumps(
            {
                "files": [
                    {
                        "path": name,
                        "content": content_str,
                        "mime_type": mime_type,
                        "size_bytes": len(content_str.encode()),
                    }
                ],
                "total_files": 1,
                "total_size": len(content_str.encode()),
            }
        )

        # Emit as input artifact and return artifact_id
        artifact_id = self.http_emitter.emit_artifact(
            {
                "run_id": self.id,
                "name": name,
                "type": type,
                "hash": artifact_hash,
                "storage_path": f"virtual://{type}",
                "size_bytes": len(content_str.encode()),
                "metadata": {},
            },
            content=content_json,
            role="input",
        )
        return artifact_id

    def _log_config_artifact(self):
        """Auto-log config as artifact and link to run."""
        if not self.config or len(self.config) == 0:
            return

        config_json = json.dumps(self.config, indent=2, sort_keys=True)
        artifact_id = self._log_virtual_artifact(
            "config.json", "config", config_json, "application/json"
        )

        # Update run to link config artifact
        if artifact_id:
            self.http_emitter.update_run_config_artifact(artifact_id)

    def _log_dependencies_artifact(self):
        """Auto-log pip freeze as artifact."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            deps_content = result.stdout.strip()
            if deps_content:
                self._log_virtual_artifact(
                    "requirements.txt", "dependencies", deps_content, "text/plain"
                )
        except Exception:
            pass  # Non-critical, silently skip

    def _log_environment_artifact(self):
        """Auto-log platform/environment info as artifact."""
        env_info = {
            "python_version": sys.version,
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda_version": os.environ.get("CUDA_VERSION", None),
        }

        env_json = json.dumps(env_info, indent=2, sort_keys=True)
        self._log_virtual_artifact("environment.json", "environment", env_json, "application/json")

    def create_note(self, title, content):
        """Create a notebook entry/note for this project.

        Args:
            title: Note title
            content: HTML content for the note

        Returns:
            Note ID

        Example:
            >>> run = ds.init(project="my-project", name="exp-1")
            >>> run.create_note(
            ...     "Experiment Results",
            ...     "<h1>Results</h1><p>Accuracy: 95%</p>"
            ... )
        """
        return self.http_emitter.emit_note(
            project_id=self.project,
            title=title,
            content=content,
        )

    def finish(self):
        """Mark run as finished."""
        if self.finished:
            return  # Already finished

        self.finished = True

        # Stop system monitoring
        if self.monitor:
            self.monitor.stop()

        # Close HTTP emitter
        self.http_emitter.close()

        print(f"Run finished: {self.name}")
        print(f"   Summary: {self.summary}")
