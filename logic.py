import math
import random
import string
import functools
from fastapi import FastAPI
import csv
from io import StringIO
from enum import Enum
import socket
from collections import deque
import subprocess
import os
from typing import Dict, List, Optional


app = FastAPI()

status_pipeline = ['backlog', 'in_progress', 'review', 'done' ]

# This class handles initiation, planning, and execution stages and manages a list of tasks related to this phase.
# Note: The format of input tasks must be consistent, otherwise unexpected behavior/errors may occur.
class BeginningPhase:
    def __init__(self, initiation, planning, execution):
        self.initiation = initiation
        self.planning = planning
        self.execution = execution
        self.tasks = []
    def load_tasks(self, task_input):
          """
        Loads tasks into the phase's task list.
        Acceptable input types:
        - str: a single task string added directly
        - list or tuple: can contain strings or nested lists/tuples; 
          nested items are joined into a single string
        - dict: keys of the dictionary are added as task descriptions
        Raises ValueError if the input type is unsupported.
        """
          if isinstance(task_input, str):
                self.tasks.append(task_input)
          elif isinstance(task_input, (list, tuple)):
                for item in task_input:
                      if isinstance(item, (list, tuple)):
                          # Join nested list/tuple elements into one string separated by spaces
                            self.tasks.append(" ".join(item))
                      elif isinstance(item, str):
                             # Append a random 3-character ID tag to each string task
                            random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 3))
                            tagged_task = f"{item.strip()} [ID:{random_id}]"
                            self.tasks.append(tagged_task)
                      else:
                            self.tasks.append(str(item))
          elif isinstance(task_input, dict):
                self.tasks.extend(task_input.keys())
          else: 
                raise ValueError("Unsupported task data type")
            
# Reviews tasks and categorizes them into finished and unfinished based on urgency keywords.
# Returns two lists:
# finishedTasks: a list of tuples containing task descriptions marked as done, along with an urgency level score.
# unfinishedTasks: a list of task descriptions still needing completion.
    def review_tasks(self):
          finishedTasks = []
          unfinishedTasks = []
          for task in self.tasks:
              # Check if task contains any urgency-related keywords (case-insensitive)
                if ("urgent" in task.lower()
                    or "important" in task.lower()
                    or "critical" in task.lower()
                    or "priority" in task.lower()
                    ):
                    # Mark task as done and calculate an urgency score based on task length
                     project_list = task + " +  is done"
                     level_of_urgency = math.ceil(len(task) * 1.5)
                     finishedTasks.append((project_list, level_of_urgency))
                else:
                    unfinishedTasks.append(f"{task} needs to get done")
          return finishedTasks, unfinishedTasks

# Attempts to establish a connection to a local server.
# If the server is down or unreachable, provides a clear explanation of the error.
def simulate_connection():
    try:
          s = socket.create_connection(("127.0.0.1", 9999), timeout = 2)
          return "Connection successful"
    except ConnectionRefusedError:
          return "ConnectionRefusedError: Could not connect to the server"
    except Exception as e:
          return f"Other error: {type(e).__name__} - {e}"
    

# A linked list to manage completed tasks.
# Allows appending new tasks without manual insertion one by one,
# and provides a method to retrieve all tasks as a list.
class TaskNode:
      def __init__(self, data):
            self.data = data
            self.next = None

class TaskLinkedList:
      def __init__(self):
            self.head = None
      def append(self, data):
          """
        Adds a new task node with the given data to the end of the list.
        """
            new_node = TaskNode(data)
            if not self.head:
                  self.head = new_node
                  return
            current = self.head
            while current.next:
                  current = current.next
            current.next = new_node
      def display(self):
          """
        Returns a list of all task data stored in the linked list nodes.
        """
            tasks = []
            current = self.head
            while current:
                  tasks.append(current.data)
                  current = current.next
            return tasks
     
# Calculates how many months a project can continue based on the remaining budget and monthly expenses.
# Raises an error if monthly cost is zero to avoid division by zero.
def calculate_burn_rate(total_budget: float, spent_so_far: float, monthly_cost: float) -> float:
      if monthly_cost == 0:
            raise ValueError("Monthly cost can't be zero")
      remaining = total_budget - spent_so_far
      return round(remaining / monthly_cost, 2)


# Provides different methods to sort a list of tasks by urgency, length, or alphabetically.
def sort_tasks_by_length(task_list):
     """
    Sorts tasks based on their length (number of characters).
    Shorter tasks will come first.
    """
          return sorted(task_list, key = len) 

def sort_tasks_by_keyword_priority(task_list):
    """
    Sorts tasks based on urgency keywords within the task description.
    Priority order (highest to lowest):
        1 - "critical"
        2 - "urgent"
        3 - "important"
        4 - "priority"
        5 - no keyword found
    Tasks with higher urgency appear earlier in the list.
    """
          def priority_value(task):
                task_lower = task.lower()
                if "critical" in task_lower:
                      return 1   
                elif "urgent" in task_lower:
                      return 2
                elif "important" in task_lower:
                      return 3
                elif "priority" in task_lower:
                      return 4
                else: 
                      return 5
          return sorted(task_list, key = priority_value)  
    
def sort_tasks_alphabetically(task_list, reverse=False):
    """
    Sorts tasks alphabetically.
    By default, sorts A to Z; set reverse=True to sort Z to A.
    """
          return sorted(task_list, reverse=reverse)
  


# Combines multiple groups of tags into a single immutable set of tags.
# Accepts any number of iterable tag groups (lists, sets, etc.)
def get_unique_tags(*tag_groups):
          all_tags = set()
          for group in tag_groups:
                all_tags.update(group)
          return frozenset(all_tags) # Returns frozenset for immutability 
    
tags1 = ["urgent", "planning", "execution"]
tags2 = ["execution", "control", "review"]
unique_tags = get_unique_tags(tags1, tags2)

# Finds common members between two teams by returning the intersection of their member sets.
def find_common_team_members(team_a, team_b):
          return set(team_a) & set(team_b)
    
team1 = ["alice", "bob", "charlie"]
team2 = ["charlie", "dana", "bob"]
common_members = find_common_team_members(team1, team2)


from typing import List

def binary_search_task_by_title(sorted_task_list: List[str], target_title: str) -> int:
    """
    Performs binary search on a sorted list of task titles.

    Args:
        sorted_task_list (List[str]): A list of task titles already sorted alphabetically.
        target_title (str): The task title to search for.

    Returns:
        The index of target_title in the sorted_task_list if found.

    Note:
        The list must be sorted before calling this function.
        The returned index corresponds to the sorted list, not the original.
    """
    low, high = 0, len(sorted_task_list) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_task = sorted_task_list[mid]

        if mid_task == target_title:
            return mid
        elif mid_task < target_title:
            low = mid + 1
        else:
            high = mid - 1
    return -1


      
# Performs a Depth-First Search traversal on a task dependency graph.
# Returns the list of tasks visited in DFS order starting from start_task.
def dfs_traverse(graph, start_task, visited=None):
      """
    Args:
        graph (Dict[str, List[str]]): A dictionary representing the graph where keys are tasks and
        values are lists of neighboring tasks.
        start_task (str): The task to start traversal from.
        visited (Optional[List[str]]): List of tasks already visited (used in recursion).

    Returns: List of visited tasks in the order they were traversed.
    """
       if visited is None:
             visited = []
       visited.append(start_task)
       for neighbor in graph.get(start_task, []):
             if neighbor not in visited:
                   dfs_traverse(graph, neighbor, visited)
       return visited


# Dynamic programming to check if any combination of numbers from a file sums to a target value.
# Handles various file read errors and warns about potential future deprecation.
import warnings
def safe_budget_match(file_path: str, target: int) -> bool:
        """
    Args:
        file_path (str): Path to the file containing numbers (one per line).
        target (int): The target sum to check for.

    Returns:
        bool: True if any subset of numbers sums to the target, False otherwise.
    
    Notes:
        - The function reads the file safely, handling Unicode and other errors.
        - Emits a PendingDeprecationWarning about future deprecation.
        - Uses a dynamic programming approach to find subset sums.
    """
      try:
            with open(file_path, encoding="utf-8") as f:
                  lines = f.readlines()
            warnings.warn("this function may be deprecated in future verisons", PendingDeprecationWarning)
      except UnicodeDecodeError:
            print("Failed to decoe file: make sure it's tuf-8 encoded.")
            return False
      except KeyboardInterrupt:
            print("Interrupted by user.")
            return False
      except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            return False
      
      try:
            numbers = []
            for line in lines:
                  value = eval(line.strip())
                  if isinstance(value, complex):
                        raise ValueError("Complex numbers are not supported.")
                  numbers.append(int(value))
      except ValueError as ve:
            print(f"Value error while parsing numbers: {ve}")
            return False
      except ProcessLookupError:
            print("Process related erorr occured.")
            return False
      dp = {0}
      for num in numbers:
            dp |= {x + num for x in dp}
      return target in dp



# More DP that implements a better task handling system
def select_tasks_by_value(tasks, max_capacity):
 
      n = len(tasks)
      dp = [[0] * (max_capacity +1) for _ in range(n +1)]
      for i in range(1, n + 1):
            name, value, cost = tasks[i - 1] 
            for c in range(max_capacity + 1):
                  if cost > c:
                        dp[i][c] = dp[i - 1][c]
                  else:
                        dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - cost] + value)

      selected = []  
      c = max_capacity
      for i in range(n, 0, -1):
            if dp[i][c] != dp[i-1][c]:
                  name, value, task = tasks[i-1]
                  selected.append(name)
                  c -= cost
      return {
            "max_value": dp[n][max_capacity],
            "selected_tasks": selected[::-1]
      }

tasks = [
    ("Prototype", 7, 3),
    ("Testing", 6, 2),
    ("Documentation", 4, 1),
    ("Deployment", 8, 4),
]
best = select_tasks_by_value(tasks, max_capacity=5)

# Knapsack DP: not sure what this does
def max_project_value(costs: list[int], values: list[int], budget: int) -> int:
      n = len(costs)
      dp = [[0] * (budget + 1) for _ in range(n + 1)]
      for i in range(1, n + 1):
            for w in range(budget + 1):
                  if costs[i - 1] <= w:
                        dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - costs[i - 1]])
                  else:
                        dp[i][w] = dp[i - 1][w]
      return dp[n][budget]

# this array limits the amount of tasks that can be placed into a container
# before having clear the contents
class DynamicArray:
      def __init__(self):
            self.capacity = 4
            self.length = 0
            self.data = [None] * self.capacity
      def append(self, value):
            if self.length == self.capacity:
                  self._resize()
            self.data[self.length] = value
            self.length += 1
      def _resize(self):
            self.capacity *= 2
            new_data = [None] * self.capacity
            for i in range(self.length):
                  new_data[i] = self.data[i]
            self.data = new_data
      def __getitem__(self, index):
            if 0 <= index < self.length:
                  return self.data[index]
            raise IndexError("Index out of bounds")
      def __len__(self):
            return self.length
                             

# queue time that prevents rapid adding of projects into the directories
class TaskQueue:
      def __init__(self):
            self.queue = deque()

      def enqueue(self, task):
            self.queue.append(task)

      def dequeue(self):
            if not self.queue:
                  return None
            return self.queue[0]
      
      def is_empty(self):
            return len(self.queue) == 0
      
      def size(self):
            return len(self.queue)
      

# not sure what this does
class TasksStack:
      def __init__(self):
            self.stack = []
      def push(self, task):
            self.stack.append(task)
      def pop(self):
            if not self.stack:
                  return None
            return self.stack.pop()
      
      def peek(self):
            if not self.stack:
                  return None
            return self.stack[-1]
      
      def is_empty(self):
            return len(self.stack) == 0
      def size(self):
            return len(self.stack)


#  makes use of import string. validating a strong urge for string format
def longest_common_prefix(strings):
      if not strings:
            return ""
      
      strings.sort()
      first = strings[0]
      last = strings[-1]
      prefix = ""

      for i in range(min(len(first), len(last))):
            if first[i] == last[i]:
                  prefix += first[i]
            else:
                  break
      return prefix



 # regex, this block is like a security camera and prevents faulty entries
import re
def match_task_code(text):
      pattern = r'\b(?:PRJ|TASK|BUG) - \d{1, 4}\b'
      return re.findall(pattern, text)
text = "Update PRJ-001, check TASK-42, and ignore BUG-999 for now"


def extract_emails_and_deadlines(task_list):
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    deadline_pattern = r'\d{4}-\d{2}-\d{2}'
    
    emails = []
    deadlines = []
    
    for text in task_list:
        emails.extend(re.findall(email_pattern, text))
        deadlines.extend(re.findall(deadline_pattern, text))
    
    return {"emails": emails, "deadlines": deadlines}

text = "Contact alice@acme.com or bob_dev@example.org. Deadline: 2025-09-01. Another task is due by 2025-10-15."
      


# makes use of csv and string IO imports, no idea what this does
def export_tasks_to_csv(task_list):
      output = StringIO()
      writer = csv.writer(output)
      writer.writerow(["Task"])

      for task in task_list:
            writer.writerow([task])
      return output.getvalue()


def export_task_dicts_to_csv(task_dicts):
      if not task_dicts:
            return ""

      output = StringIO()
      fieldnames = task_dicts[0].keys()
      writer = csv.DictWriter(output, fieldnames = fieldnames)
      writer.writeheader()

      for row in task_dicts:
            writer.writerow(row)
      return output.getvalue()

task_data = [
    {"title": "urgent fix", "status": "in_progress"},
    {"title": "plan Q4", "status": "backlog"},
]

csv_data = export_task_dicts_to_csv(task_data)



# allows for many files of data to be stored up to 1 GB per contianer
def group_tasks_by_priority(task_list):
      priority_map = {
            "high": [],
            "medium": [],
            "low": []
      }
      for task in task_list:
            lower = task.lower()
            if "critical" in lower or "urgent" in lower:
                  priority_map["high"].append(task)
            elif "important" in lower:
                  priority_map["medium"].append(task)
            else:
                  priority_map["low"].append(tasks)
      return priority_map

def group_tasks_by_priority(task_list):
      priority_map = {
            "high": [],
            "medium": [],
            "low": []
      }

      for task in task_list:
            lower = task.lower()
            if "critical" in lower or "urgent" in lower:
                  priority_map["high"].append(task)
            elif "important" in lower:
                  priority_map["medium"].append(task)
            else:
                  priority_map["low"].append(tasks)
      return priority_map


# enum logic that handles a list of options regarding a projects' status
class ProjectStatus(Enum):
      BACKLOG = "backlog"
      IN_PROGRESS = "in_progress"
      REVIEW =  "review"
      DONE = "done"

      def is_valid_status(status: str):
            try:
                  ProjectStatus(status)
                  return True
            except ValueError:
                  return False
            
      def convert_to_status_enum(status: str):
            return ProjectStatus(status)

# R vs V: not sure if this is even needed
def demonstrate_reference_vs_value():
      original = ["alpha", "beta", "gamma"]
      reference = original
      value_copy = original[:]
      reference.append("delta")
      value_copy.append("epsilon")

      return {
            "original": original,
            "reference": reference,
            "value_copy": value_copy
      }


#  a file system that gathers information on what type of machine you're using and
# uses import subprocess and os for further clarification
def export_tasks_to_csv(tasks_with_status, filepath="exported_tasks.csv"):
      try:
            with open(filepath, "w") as f:
                  for task, status in tasks_with_status:
                        f.write(f"{task}, {status}\n")
            return f"Successfully exported to {filepath}"
      except Exception as e:
            return f"Error exporting CSV: {e}"


def open_csv_file(filepath="exported_tasks.csv"):
      try:
            if not os.path.exists(filepath):
                  return f"FileNotFoundError: '{filepath}' does not exist."
            
            #MacOS
            if os.name == "posix":
                  subprocess.run(["open",filepath], check = True)

            #windows
            elif os.name == "nt":
                  os.startfile(filepath)

            #linux
            else:
                  subprocess.run(["xdg-open", filepath])
            return f"Opened file: {filepath}"
      
      except FileNotFoundError:
            return "System command not found. Could not open CSV."
      except subprocess.CalledProcessError:
            return "CalledProcessError: Failed to open CSV"
      except Exception as e:
            return f"Unhandled error: {type(e).__name__} - {e}"


# no idea what's going on here, functools?
def case_study(self, task1, task2):
          process = f"Analyzing {task1} and {task2}"
          infrastructure = "tools"
          return f"{process} using {infrastructure}"

class Monitoring(BeginningPhase):
            def __init__(self, initiation, planning, execution, control):
                super().__init__(initiation, planning, execution)
                self.control = control
            def case_study(self, task1, task2):
                  process = f"Analyzing {task1} and {task2}"
                  infrastructure = "tools"
                  return f"{process} using {infrastructure}"
            def control_applied(self, risk):
                  return f"If '{risk}' occurs, apply control method {self.control}"

# fast API route
@app.get("/tasks/review")
def get_reviewed_tasks():
      finished, unfinished = pm_phase.review_tasks()
      return {
            "finished" : finished,
            "unfinished" : unfinished
      }

# max file name is 64 characters?
@functools.lru_cache(maxsize=64)
def memoized_effort_estimate(n):
      if n <= 1:
            return 1
      return memoized_effort_estimate(n -1) + memoized_effort_estimate(n -2)

# variables that maintain the programs functionality
pm_phase = Monitoring("initiation", "planning", "execution", "control")
pm_phase.load_tasks(["urgent client request", "check status report"])
pm_phase.load_tasks([
    ["urgent meeting"],
    ["project charter on starlink"],
    ["business Scope for starlink HQ building"],
    ["important starlink deadline"]
])
pm_phase.load_tasks({
    "call starbucks CEO": "priority/tomorrow @11AM",
    "call Meijers district manager": "eventually",
    "monitor budget for starlink project": "critical",
    "agile or waterfall?": "critical"
})


task_graph = {
      "Plan": ["Design", "Budget"],
      "Design": ["Prototyping"],
      "Prototyping": ["Review"],
      "Budget": [],
      "Review": ["Approval"],
      "Approval": []
      }

#not sure about these three lines
task_list = pm_phase.tasks
target = "urgent client request"
csv_data = export_tasks_to_csv(task_list)

# allows users to search for tasks in the contianers by key words
index = binary_search_task_by_title(task_list, target)
if index != -1:
      print(f"Found '{target}' at index '{index}'")
else: 
      print(f"'{target}' not found")

visted_order = dfs_traverse(task_graph, "Plan")
result = pm_phase.case_study("A", "B")
bp = BeginningPhase(pm_phase.initiation, pm_phase.planning, pm_phase.execution)    

sample_tasks = [
    "john@boot.com - due by 2025-07-31",
    "daveMac@oswego.edu, deadline: 2026-12-01",
    "maldaven@nyker.org (due by 2023-01-23)"
]
print(extract_emails_and_deadlines(sample_tasks))


def __repr__(self):
          return (
                f"<Monitoring(initiation='{self.initiation}', "
                f"planning='{self.planning}', "
                f"control='{self.control}', "
                f"tasks={len(self.tasks)})>"
          )


# time is always a factor in project management: this is a Utility function to calculate a project's expected deadline.
# it simply returns a YYYY-MM-DD in string format. only returns an error if the deadline is changed without permission
from datetime import datetime, timedelta
def forecast_deadline(start_date: str, duration_days: int) -> str:
      if duration_days < 0:
            raise ValueError("Duration can't be negative")
      try:
            date = datetime.strptime(start_date, "%Y-%m-%d")
      except ValueError:
            raise ValueError("Start date must be in 'YYYY-MM-DD' format")
      deadline = date + timedelta(days=duration_days)
      return deadline.strftime("%Y-%m-%d")


# pulls one type of data into another incase the files is placed into the wrong contianer
class UnionFind:
      def __init__(self):
            self.parent = {}
            self.rank = {}
      
      def add(self, person):
            if person not in self.parent:
                  self.parent[person] = person
                  self.rank[person] = 0

      def find(self, person):
            if self.parent[person] != person:
                  self.parent[person]
            return self.parent[person]
      
      def union(self, person1, person2):
            self.add(person1)
            self.add(person2)
            root1 = self.find(person1)
            root2 = self.find(person2)

            if root1 == root2:
                  return
            
            if self.rank[root1] < self.rank[root2]:
                  self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                  self.parent[root2] = root1
            else:
                  self.parent[root2] = root1
                  self.rank[root1] += 1

      def connected(self, person1, person2):
            if person1 not in self.parent or person2 not in self.parent:
                  return False
            return self.find(person1) == self.find(person2)

# do i even need this?
class RollingAverageCalculator:
      def __init__(self, window_size=3):
            self.window_size = window_size
            self.values = []

      def add_value(self, val):
            self.value.append(val)
            if len(self.value) > self.window_size:
                  self.values.pop(0)

      def get_average(self):
            if not self.values:
                  return 0.0
            return sum(self.values) / len(self.values)


# Linear probing short but sweet
def linear_probe_insert(table: list, key: str, size: int = 10) -> int:
      index = sum(ord(c) for c in key) % size
      for i in range(size):
            probe_index = (index + i) % size
            if table[probe_index] is None:
                  table[probe_index] = key
                  return probe_index
            raise Exception("Hash table is full.")
      
# my main problem solver, this block finds the most optimal solution when unsure where a project should go
class SimplexSolver:
      def __init__(self, a, b ,c):
            self.a = [row[:] for row in a]
            self.b = b[:]
            self.c = c[:]
            self.n = len(c)
            self.m = len(b)
            self.table = []
            self._build_table()

      def _pivot(self, row, col):
            pivot_element = self.tablet[row][col]
            self.table[row] = [x / pivot_element for x in self.table[row]]
            for r in range(len(self.table)):
                  if r != row:
                        ratio = self.table[r][col]
                        self.table[r] = [
                              self.table[r][i] - ratio * self.table[row][i]
                              for i in range(len(self.table[0]))
                        ]

      def solve(self):
            while True:
                  last_row = self.table[-1]
                  pivot_col = min(
                        (i for i in range(len(last_row) - 1) if last_row[i] < 0),
                        default =- 1,
                        key=lambda i: last_row[i]
                  )
                  
                  if pivot_col == -1:
                        break

                  ratios = []
                  for i in range(self.m):
                        col_val = self.table[i][pivot_col]
                        if col_val > 0:
                              ratios.append((self.table[i][-1] / col_val, i))
                  if not ratios:
                        raise Exception("Unbound solution.")
                  _, pivot_row = min(ratios)
                  self._pivot_row(pivot_row, pivot_col) 

            solution = [0] * self.n
            for i in range(self.m):
                  for j in range(self.n):
                        if self.table[i][j] == 1 and all(self.table[k][j] == 0 for k in range(self.m) if k != i):
                              solution[j] = self.table[i][-1]
            return solution, self.table[-1][-1]  
