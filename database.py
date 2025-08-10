# Showcases my PostGreSQL setup

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from tools import Base

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index = True)
    description = Column(String)
    start_date = Column(DateTime, default = datetime.utcnow)
    end_date = Column(DateTime, nullable = True)
    is_complete = Column(Boolean, default = False)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    
    tasks = relationship("Task", back_populates="project")

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', is_complete={self.is_complete})>"

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    name = Column(String, nullable=False)
    status = Column(String, default="todo")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())

    project = relationship("Project", back_populates="tasks")
    
PHASES = ["initiation", "planning", "execution", "review", "done"]
