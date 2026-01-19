"""artifacta - Universal experiment tracking."""

__version__ = "2.0.0"

import atexit

from .autolog import autolog
from .autolog import disable as disable_autolog
from .primitives import (
    BarChart,
    Curve,
    Distribution,
    Matrix,
    Scatter,
    Series,
    Table,
)
from .run import Run

_current_run = None
_atexit_registered = False


def _auto_finish():
    """Auto-finish current run on script exit."""
    global _current_run
    if _current_run and not _current_run.finished:
        _current_run.finish()
        _current_run = None


def init(project=None, name=None, config=None, code_dir=None):
    """Initialize a training run.

    The run will automatically finish when your script exits.
    You don't need to call finish() manually.

    Args:
        project: Project name (default: "default")
        name: Run name (auto-generated if not provided)
        config: Configuration dict (hyperparameters, etc.)
        code_dir: Optional code directory for hash computation fallback (default: auto-detect via git)

    Returns:
        Run object

    Example:
        >>> import artifacta as ds
        >>> run = ds.init(project="mnist", config={"lr": 0.001})
        >>> # Train your model...
        >>> # Run auto-finishes when script exits!
    """
    global _current_run, _atexit_registered

    # Finish previous run if exists
    if _current_run and not _current_run.finished:
        _current_run.finish()

    _current_run = Run(project or "default", name, config or {}, code_dir=code_dir)
    _current_run.start()

    # Register auto-finish on first init
    if not _atexit_registered:
        atexit.register(_auto_finish)
        _atexit_registered = True

    return _current_run


def log(name, data):
    """Log structured data primitive.

    Args:
        name: Name for this data object
        data: One of the primitives (Series, Distribution, Matrix, etc.)

    Example:
        >>> ns.log("training_metrics", Series(
        ...     index="epoch",
        ...     fields={"train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
        ... ))
    """
    if not _current_run:
        raise RuntimeError("Call ns.init() first")
    _current_run.log(name, data)


def get_run():
    """Get current run object."""
    return _current_run
