from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_main():
    # Ping a health route or root route to verify startup
    response = client.get("/")
    assert response.status_code in [200, 404]  # Verifies app initializes without crashing