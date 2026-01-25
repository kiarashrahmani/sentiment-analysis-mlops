"""Tests configuration and fixtures."""
import pytest
from pathlib import Path
import tempfile
import json

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"text": "This product is amazing! I love it.", "stars": 5, "sentiment": "positive"},
        {"text": "Excellent quality and fast shipping.", "stars": 5, "sentiment": "positive"},
        {"text": "Great service and quality!", "stars": 5, "sentiment": "positive"},
        {"text": "Terrible experience. Never again.", "stars": 1, "sentiment": "negative"},
        {"text": "Disappointing and overpriced.", "stars": 2, "sentiment": "negative"},
        {"text": "Very bad product, waste of money.", "stars": 1, "sentiment": "negative"},
        {"text": "It's okay, nothing special.", "stars": 3, "sentiment": "neutral"},
        {"text": "Average product, meets basic needs.", "stars": 3, "sentiment": "neutral"},
        {"text": "Neither good nor bad, just okay.", "stars": 3, "sentiment": "neutral"},
    ]


@pytest.fixture
def temp_data_file(sample_data):
    """Create a temporary data file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup is handled by tempfile
