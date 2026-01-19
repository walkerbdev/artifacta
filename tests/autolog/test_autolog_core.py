"""Core autolog API tests"""

import sys

import pytest


def test_autolog_import():
    """Test that autolog can be imported from main artifacta package"""
    import artifacta as ds

    assert hasattr(ds, "autolog")
    assert hasattr(ds, "disable_autolog")


def test_autolog_framework_detection():
    """Test automatic framework detection"""
    import artifacta as ds

    # With pytorch_lightning installed, should auto-detect and succeed
    try:
        import pytorch_lightning  # noqa: F401

        ds.autolog()  # Should succeed
        ds.disable_autolog()
    except ImportError:
        # Without pytorch_lightning installed, should raise error
        with pytest.raises(RuntimeError, match="Could not detect ML framework"):
            ds.autolog()


def test_autolog_explicit_framework():
    """Test explicit framework specification"""
    import artifacta as ds

    # PyTorch should work if installed
    try:
        import pytorch_lightning  # noqa: F401

        ds.autolog(framework="pytorch")
        ds.disable_autolog()
    except ImportError:
        pass

    # TensorFlow should work if installed
    try:
        import tensorflow  # noqa: F401

        ds.autolog(framework="tensorflow")
        ds.disable_autolog()
    except ImportError:
        pass


def test_autolog_invalid_framework():
    """Test invalid framework raises error"""
    import artifacta as ds

    with pytest.raises(ValueError, match="Unsupported framework"):
        ds.autolog(framework="invalid_framework")


def test_disable_autolog():
    """Test disabling autolog"""
    import artifacta as ds

    # Should not raise even if not enabled
    ds.disable_autolog()

    # Enable and disable should work
    if "pytorch_lightning" in sys.modules:
        ds.autolog(framework="pytorch")
        ds.disable_autolog()
