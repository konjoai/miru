"""Shared pytest fixtures for Miru tests."""
import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from miru.main import app


@pytest.fixture
def client() -> TestClient:
    """Synchronous FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_image_b64() -> str:
    """Base64-encoded raw bytes for a 4×4 black RGB image (48 bytes)."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode()
