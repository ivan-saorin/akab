"""
Test suite for AKAB
"""

import pytest
from pathlib import Path

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary data directory for testing"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "campaigns").mkdir()
    (data_dir / "experiments").mkdir()
    (data_dir / "knowledge_bases").mkdir()
    (data_dir / "templates").mkdir()
    (data_dir / "results").mkdir()
    
    return data_dir


@pytest.fixture
def sample_campaign():
    """Sample campaign configuration"""
    return {
        "id": "test-campaign",
        "name": "Test Campaign",
        "description": "Testing AKAB functionality",
        "providers": ["anthropic-local", "openai/gpt-3.5-turbo"],
        "total_experiments": 10,
        "completed_experiments": [],
        "created_at": "2025-01-01T00:00:00"
    }
