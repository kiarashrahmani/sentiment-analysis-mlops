"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from main import app
from config import settings


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    """Mock model service."""
    with patch('main.model_service') as mock:
        mock.is_loaded.return_value = True
        mock.get_model_type.return_value = "baseline"
        yield mock


class TestHealthEndpoint:
    """Test cases for health endpoint."""
    
    def test_health_check_healthy(self, client, mock_model_service):
        """Test health check when model is loaded."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    
    def test_health_check_unhealthy(self, client):
        """Test health check when model is not loaded."""
        with patch('main.model_service.is_loaded', return_value=False):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Test cases for predict endpoint."""
    
    def test_predict_success(self, client, mock_model_service):
        """Test successful prediction."""
        mock_model_service.predict_single.return_value = {
            "text": "This is great!",
            "sentiment": "positive",
            "confidence": 0.95,
            "probabilities": {
                "negative": 0.02,
                "neutral": 0.03,
                "positive": 0.95
            }
        }
        
        response = client.post(
            "/predict",
            json={"text": "This is great!"},
            headers={"X-API-Key": settings.api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert "confidence" in data
        assert "probabilities" in data
    
    def test_predict_missing_api_key(self, client):
        """Test prediction without API key."""
        response = client.post(
            "/predict",
            json={"text": "This is great!"}
        )
        
        assert response.status_code == 401
    
    def test_predict_invalid_api_key(self, client):
        """Test prediction with invalid API key."""
        response = client.post(
            "/predict",
            json={"text": "This is great!"},
            headers={"X-API-Key": "invalid-key"}
        )
        
        assert response.status_code == 401
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text."""
        response = client.post(
            "/predict",
            json={"text": ""},
            headers={"X-API-Key": settings.api_key}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_model_not_loaded(self, client):
        """Test prediction when model is not loaded."""
        with patch('main.model_service.is_loaded', return_value=False):
            response = client.post(
                "/predict",
                json={"text": "This is great!"},
                headers={"X-API-Key": settings.api_key}
            )
            
            assert response.status_code == 503


class TestBatchPredictEndpoint:
    """Test cases for batch predict endpoint."""
    
    def test_batch_predict_success(self, client, mock_model_service):
        """Test successful batch prediction."""
        mock_model_service.predict_batch.return_value = [
            {
                "text": "Great!",
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {
                    "negative": 0.02,
                    "neutral": 0.03,
                    "positive": 0.95
                }
            },
            {
                "text": "Bad!",
                "sentiment": "negative",
                "confidence": 0.90,
                "probabilities": {
                    "negative": 0.90,
                    "neutral": 0.05,
                    "positive": 0.05
                }
            }
        ]
        
        response = client.post(
            "/predict/batch",
            json={"texts": ["Great!", "Bad!"]},
            headers={"X-API-Key": settings.api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"texts": []},
            headers={"X-API-Key": settings.api_key}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many_items(self, client):
        """Test batch prediction with too many items."""
        texts = ["text"] * 101  # Over the limit of 100
        
        response = client.post(
            "/predict/batch",
            json={"texts": texts},
            headers={"X-API-Key": settings.api_key}
        )
        
        assert response.status_code == 422  # Validation error


class TestRootEndpoint:
    """Test cases for root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
