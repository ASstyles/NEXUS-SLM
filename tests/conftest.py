"""Pytest configuration and environment fixtures for NEXUS-SLM test suite."""

import os
import pytest

# Ensure test environment and valid test credentials are set prior to module collection
os.environ.setdefault("NEXUS_ENV", "test")
os.environ.setdefault("NEXUS_ADMIN_KEY", "nexus-test-environment-key-min-32-chars-long")
os.environ.setdefault("NEXUS_TRUSTED_PROXIES", "127.0.0.1,::1,localhost")

from src.utils.config import reload_config


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Initializes and reloads clean test configuration."""
    os.environ["NEXUS_ENV"] = "test"
    os.environ["NEXUS_ADMIN_KEY"] = "nexus-test-environment-key-min-32-chars-long"
    reload_config()
    yield
