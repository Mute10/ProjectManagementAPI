from fastapi import FastAPI, Header, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from database import Project, Base 
from schematics import ProjectCreate, Project as ProjectPydantic
from tools import SessionLocal, engine
import crud
from crud import update_project
from logic import Monitoring
from pydantic import BaseModel
from typing import Union, List, Dict


app = FastAPI()
Base.metadata.create_all(bind=engine)


class ProjectLifecycleManager:
    def __init__(self):
        self.phases = ["initiation", "planning", "execution", "review", "done"]
    def advance_phase(self, current_phase: str) -> str:
        if current_phase not in self.phases:
            raise ValueError(f"Invaild phase: {current_phase}")
        index = self.phases.index(current_phase)
        if index + 1 < len(self.phases):
            return self.phases[index + 1]
        return self.phases[-1]
    
    def is_valid_transition(self, current_phase: str, next_phase: str) -> bool:
        try:
            return self.phases.index(next_phase) == self.phases.index(current_phase) + 1
        except ValueError:
            return False
        
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()



pm_phase = Monitoring("initiation", "planning", "execution", "control")
pm_phase.load_tasks([
    ["urgent meeting"],
    ["project charter on starlink"],
    ["business Scope for starlink HQ building"],
    ["important starlink deadline"]
])


class TaskInput(BaseModel):
    task: Union[str, List[Union[str, List[str]]], Dict[str, str]]


def verify_token(x_token: str = Header(...)):
    if x_token != "supersecrettoken":
        raise HTTPException(status_code = 401, detail ="Invalid token")
    

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/tasks/review")
def get_reviewed_tasks():
    finished, unfinished = pm_phase.review_tasks()
    return {
        "finished": finished,
        "unfinished": unfinished
    }

@app.get("/tasks/list")
def list_all_tasks():
    return {"all_tasks": pm_phase.tasks}

@app.delete("/tasks/clear")
def clear_tasks():
    pm_phase.tasks.clear()
    return {"message": "All tasks have been cleared."}

@app.get("/tasks/count")
def count_tasks():
    return {"total_tasks": len(pm_phase.tasks)}



@app.get("/", include_in_schema = False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/task/add")
def add_task(task_input: TaskInput):
    try:
        pm_phase.load_tasks(task_input.task)
        return {"message": "Task(s) added successfully."}
    except ValueError as e:
        return {"error": str(e)}


@app.get("/metrics")
def get_metrics(
    limit: int = 5,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    total = db.query(Project).count()

    recent_projects = (
        db.query(Project)
        .order_by(Project.start_date.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    recent_data = [
        {"id": p.id, "name": p.name, "updated_at": p.updated_at}
        for p in recent_projects
    ]
    return {
        "total_projects": total,
        "recent_updates": recent_data
    }



@app.post("/projects", summary = " ", operation_id="createProjects", response_model=ProjectPydantic)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    return crud.create_project(db=db, project=project)

@app.get("/projects", summary=" ", operation_id="getProjects",response_model=list[ProjectPydantic])
def read_projects(db: Session = Depends(get_db)):
    return crud.get_projects(db)

@app.get("/projects/{project_id}", summary=" ", operation_id="getProjects", response_model = ProjectPydantic)
def read_project(project_id: int, db: Session = Depends(get_db)):
    project = crud.get_project_by_id(db, project_id)
    if project is None:
        raise HTTPException(status_code = 404, detail = "Project Not Found")
    return project

@app.put("/projects/{project_id}", summary=" ", operation_id="updateProject", response_model = ProjectPydantic)
def update_project_by_id(project_id: int, project_data: ProjectCreate, db: Session = Depends(get_db)):
    updated = update_project(db, project_id, project_data)
    if updated is None:
        raise HTTPException(status_code = 404, detail = "Project not found")
    return updated

@app.delete("/projects/{project_id}", summary=" ", response_model=dict)
def delete_project_by_id(project_id: int, db: Session = Depends(get_db)):
    deleted = crud.delete_project(db, project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail = "Project not found")
    return { "Project deleted successfully"}

#root endpoint
@app.get("/", summary = "Root Health check", operation_id="getRoot")
def read_root():
 return {"Project Management Lifecycle System is live"}

"""
how to redirect people to different sites
@app.get("/github", summary="Redirect to my GitHub")
def redirect_to_github():
    return RedirectResponse(url="https://github.com/your-username")
    """
