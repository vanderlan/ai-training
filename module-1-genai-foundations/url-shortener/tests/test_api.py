import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.database import Database

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_short_url():
    """Test creating a short URL"""
    response = client.post(
        "/api/shorten",
        json={"url": "https://www.example.com/test"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "short_code" in data
    assert "short_url" in data
    assert data["clicks"] == 0


def test_create_short_url_with_custom_alias():
    """Test creating a short URL with custom alias"""
    response = client.post(
        "/api/shorten",
        json={
            "url": "https://www.example.com/custom",
            "custom_alias": "mylink"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["short_code"] == "mylink"


def test_invalid_url():
    """Test creating short URL with invalid URL"""
    response = client.post(
        "/api/shorten",
        json={"url": "not-a-valid-url"}
    )
    assert response.status_code == 422


def test_get_stats():
    """Test getting URL statistics"""
    # First create a URL
    create_response = client.post(
        "/api/shorten",
        json={"url": "https://www.example.com/stats-test"}
    )
    short_code = create_response.json()["short_code"]
    
    # Get stats
    stats_response = client.get(f"/api/stats/{short_code}")
    assert stats_response.status_code == 200
    data = stats_response.json()
    assert data["short_code"] == short_code
    assert data["clicks"] >= 0


def test_redirect():
    """Test URL redirection"""
    # Create a URL
    create_response = client.post(
        "/api/shorten",
        json={"url": "https://www.example.com/redirect-test"}
    )
    short_code = create_response.json()["short_code"]
    
    # Test redirect
    redirect_response = client.get(f"/{short_code}", follow_redirects=False)
    assert redirect_response.status_code == 301
    assert redirect_response.headers["location"] == "https://www.example.com/redirect-test"


def test_list_urls():
    """Test listing all URLs"""
    response = client.get("/api/urls")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "urls" in data
    assert isinstance(data["urls"], list)
