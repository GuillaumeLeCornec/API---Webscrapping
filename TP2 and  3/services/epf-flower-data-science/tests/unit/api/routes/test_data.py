import pytest
from fastapi.testclient import TestClient


class TestDataRoute:
    @pytest.fixture
    def client(self) -> TestClient:
        """
        Test client for integration tests
        """
        import sys
        import os

        # Chemin du répertoire que tu veux ajouter
        new_directory = os.path.join(os.getcwd(), '..', '..','..','..')
        sys.path.append(new_directory)

        from main import get_application

        app = get_application()

        client = TestClient(app, base_url="http://testserver")

        return client
    
    def test_data(self, client):
        data = "iris"
        url = f"data/{data}"
        response = client.get(url)

        
        # Assert the output
        assert response.status_code == 200
        assert response.json() == {
            "message": "Hello testuser, from fastapi test route !"
        }


