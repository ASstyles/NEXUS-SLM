"""Pytest configuration and environment fixtures for NEXUS-SLM test suite."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path
import pytest

_TEST_TELEMETRY_DIR = tempfile.mkdtemp(prefix="nexus_test_telemetry_")
atexit.register(lambda: shutil.rmtree(_TEST_TELEMETRY_DIR, ignore_errors=True))

# Ensure test environment and valid test credentials are set prior to module collection
os.environ.setdefault("NEXUS_ENV", "test")
os.environ.setdefault("NEXUS_ADMIN_KEY", "nexus-test-environment-key-min-32-chars-long")
os.environ.setdefault("NEXUS_TRUSTED_PROXIES", "127.0.0.1,::1,localhost")
os.environ.setdefault(
    "NEXUS_TELEMETRY_LOG_FILE",
    str(Path(_TEST_TELEMETRY_DIR) / "telemetry.jsonl"),
)

from src.utils.config import reload_config


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Initializes and reloads clean test configuration."""
    os.environ["NEXUS_ENV"] = "test"
    os.environ["NEXUS_ADMIN_KEY"] = "nexus-test-environment-key-min-32-chars-long"
    reload_config()
    yield
