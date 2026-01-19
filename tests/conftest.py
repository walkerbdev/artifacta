"""
Pytest configuration and fixtures for Artifacta tests
"""

import os
import tempfile
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent

# Test constants from environment with defaults matching artifacta ui defaults
# Default to localhost:8000 if not set (matches DEFAULT_HOST and DEFAULT_PORT in config.py)
TRACKING_HOST = os.getenv("TRACKING_SERVER_HOST", "localhost")
TRACKING_PORT = os.getenv("TRACKING_SERVER_PORT", "8000")

API_URL = f"http://{TRACKING_HOST}:{TRACKING_PORT}"

TEST_DB_PATH = project_root / "data" / "test_runs.db"


@pytest.fixture(scope="session", autouse=True)
def set_api_url_env():
    """Set ARTIFACTA_API_URL environment variable for all tests"""
    os.environ["ARTIFACTA_API_URL"] = API_URL
    # Enable strict mode so tests fail loudly on API errors
    os.environ["ARTIFACTA_STRICT_MODE"] = "1"
    yield
    # Cleanup
    if "ARTIFACTA_API_URL" in os.environ:
        del os.environ["ARTIFACTA_API_URL"]
    if "ARTIFACTA_STRICT_MODE" in os.environ:
        del os.environ["ARTIFACTA_STRICT_MODE"]


@pytest.fixture(scope="session")
def api_url():
    """Base API URL for testing"""
    return API_URL


@pytest.fixture(scope="function")
def clean_database():
    """Clean test database before each test"""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    yield
    # Cleanup after test
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture(scope="function")
def temp_dir():
    """Provide a temporary directory for test artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def tracking_server_running():
    """
    Check if tracking server is running
    This is a session-scoped check - tests requiring server should use this
    """
    import httpx

    try:
        response = httpx.get(f"{API_URL}/health", timeout=2.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    pytest.skip("Tracking server not running at {API_URL}. Start server first.")


# Mark all integration tests to require server
def pytest_collection_modifyitems(config, items):
    """Automatically mark integration tests"""
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.usefixtures("tracking_server_running"))
