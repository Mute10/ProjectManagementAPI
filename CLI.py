import argparse
from database import SessionLocal
import crud
from date import datetime

db = SessionLocal()

def create_project(name, description=None, start_date=None, end_date=None):
    project_data = {
        "name": name,
        "description": description,
        "start_date": datetime.strptime(start_date, "%Y-%m-%d") if start_date else None,
        "end_date": datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    }
    project = crud.create_project(db, project_data)
    print(f"{project.name} (ID: {project.id})")

def list_projects():
    projects = crud.get_all_projects(db)
    for p in projects:
        print(f"{p.id}: {p.name} | Complete: {p.is_complete}")

def create_task(project_id, name, status="todo"):
    task_data = {
        "project_id": project_id,
        "name": name,
        "status": status
    }
    task = crud.create_task(db, task_data)
    print(f"Task created: {task.name} (ID: {task.id})")


def list_tasks(project_id=None):
    tasks = crud.get_all_tasks(db, project_id=project_id)
    for t in tasks:
        print(f"{t.id}: {t.name} | Status: {t.status} | Project: {t.project_id}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for projects management API")
    subparsers = parser.add_subparsers(dest="command")

    # Create project
    project_parser = subparsers.add_parser("create_project")
    project_parser.add_argument("name")
    project_parser.add_argument("--description")
    project_parser.add_argument("--start_date")
    project_parser.add_argument("--end_date")

    #list projects
    subparsers.add_parser("list_projects")

    # Create task
    task_parser = subparsers.add_parser("create_task")
    task_parser.add_argument("project_id", type=int)
    task_parser.add_argument("name")
    task_parser.add_argument("--status", default="todo")


    #list tasks
    list_tasks_parser = subparsers.add_parser("list_tasks")
    list_tasks_parser.add_argument("--project_id", type=int)
    args = parser.parse_args()

    match args.command:
            case "create_project":
              create_project(args.name, args.description, args.start_date, args.end_date)                            
            case "list_projects":
                list_projects()
            case "create_task":
                create_task(args.project_id, args.name, args.status)
            case "list_taks":
                list_tasks(args.project_id)
            case _:
                parser.print_help()




