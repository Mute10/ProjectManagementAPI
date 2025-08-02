from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_complete: Optional[bool] = False 


class ProjectCreate(ProjectBase):
    pass
    

class Project(ProjectBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    deadline: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TaskBase(BaseModel):
    name: str
    status: str = "todo"

class TaskCreate(TaskBase):
    project_id: int

class TaskUpdate(BaseModel):
    name: str | None = None
    status: str | None = None

class Task(TaskBase):
    id: int
    project_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)