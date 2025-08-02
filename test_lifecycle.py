from fastapi.testclient import TestClient
from .API import app

client = TestClient(app)

def test_read_root():
    response = client.get("/root")
    assert response.status_code == 200
    assert response.json() == {"message": "Project Management Lifecycle System is live"}

def test_create_project():
    response = client.post(
        "/projects",
        json = {"name": "Test Project", 
                "description": "Testing Creation",
                "start_date": "2025-07-08T20:23:33.518Z",
                "end_date": "2025-07-09T20:23:33.518Z",
                "is_complete": False
                }
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Test Project"

def test_get_project():
    response = client.get("/projects")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_update_project():
    create_response = client.post(
        "/projects",
        json = {"name": "Old Title", "description": "Old description",
                "start_date": "2025-07-08T20:23:33.518Z",
            "end_date": "2025-07-09T20:23:33.518Z",
            "is_complete": False
                }
    )
    project_id = create_response.json()["id"]

    update_response = client.put(
        f"/projects/{project_id}",
        json = {"name": "Updated Title"}
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Updated Title"
 
def test_delete_project():
    create_response = client.post(
        "/projects",
        json = {"name": "Delete Me", "description": "This needs to go",
            "start_date": "2025-07-08T20:23:33.518Z",
            "end_date": "2025-07-09T20:23:33.518Z",
            "is_complete": False
                }
    )
    project_id = create_response.json()["id"]

    delete_response = client.delete(f"/projects/{project_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"message": "Project deleted successfully"}
