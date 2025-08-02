import unittest
from unittest.mock import patch
import os
from logic import (sort_tasks_by_keyword_priority, longest_common_prefix, select_tasks_by_value,
extract_emails_and_deadlines, binary_search_task_by_title, dfs_traverse,
simulate_connection, export_tasks_to_csv, open_csv_file, calculate_burn_rate,
linear_probe_insert, max_project_value, forecast_deadline, DynamicArray)


class TestLogic(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run")
    def test_open_csv_file_success(self, mock_run, mock_exists):
        result = open_csv_file("dummy.csv")
        self.assertIn("Opened file", result)
        mock_run.assert_called()

    @patch("os.path.exists", return_value=False)
    def test_open_csv_file_not_found(self, mock_exists):
        result = open_csv_file("nonexist.csv")
        self.assertIn("FileNotFoundError", result)

    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run", side_effect=FileNotFoundError("No such command")) 
    def test_open_csv_file_fails_with_missing_command(self, mock_run, mock_exists):
        result = open_csv_file("dummy.csv")
        self.assertIn("System command not found", result)
        
          
    def testSortTasksByKeywordPriority(self):
        tasks = ["fix bug", "critical issue", "important update", "urgent task"]
        result = sort_tasks_by_keyword_priority(tasks)
        self.assertEqual(result[0], "critical issue")
        self.assertEqual(result[1], "urgent task")

    def test_longest_common_prefix(self):
        words = ["interstellar", "internet", "internal"]
        result = longest_common_prefix(words)
        self.assertEqual(result, "inter")
    def test_longest_common_prefix_none(self):
        words = ["frog", "fish", "squid"]
        result = longest_common_prefix(words)
        self.assertEqual(result, "")

    def test_select_tasks_by_value(self):
        tasks = [
            ("TaskA", 7, 3),
            ("TaskB", 6, 2),
            ("TaskC", 4, 1)
        ]
        result = select_tasks_by_value(tasks, max_capacity=4)
        self.assertEqual(result["max_value"], 11)
        self.assertIn("TaskA", result["selected_tasks"])
        self.assertIn("TaskC", result["selected_tasks"])
        self.assertNotIn("TaskB", result["selected_tasks"])


    def test_extract_emails_and_deadlines(self):
        task_list = [
            "john@boot.com - due by 2025-07-31",
            "daveMac@oswego.edu, deadline: 2026-12-01",
            "maldaven@nyker.org (due by 2023-01-23)"
        ]
        expected = {
        "emails": [
            "john@boot.com",
            "daveMac@oswego.edu",
            "maldaven@nyker.org"
        ],
        "deadlines": [
            "2025-07-31",
            "2026-12-01",
            "2023-01-23"
        ]
    }
        result = extract_emails_and_deadlines(task_list)
        self.assertEqual(result, expected)

    def test_binary_search_task_by_tilte(self):
        tasks = ["deploy feature", "urgent meeting", "code review", "client call"]
        result_found = binary_search_task_by_title(tasks, "code review")
        result_not_found = binary_search_task_by_title(tasks, "nonexistent task")
        self.assertIsInstance(result_found, int)
        self.assertGreaterEqual(result_found, 0)
        self.assertEqual(result_not_found, -1)


    def test_dfs_traverse(self):
        graph = {
            'TaskA': ['TaskB', 'TaskC'],
            'TaskB': ['TaskD'],
            'TaskC': [],
            'TaskD': []
        }
        result = dfs_traverse(graph, 'TaskA')
        self.assertEqual(result, ['TaskA', 'TaskB', 'TaskD', 'TaskC'])


    def test_sort_tasks_does_not_muate_original(self):
        original_tasks = ["fix bug", "critical issue", "important update", "urgen task"]
        copy_tasks = original_tasks.copy()
        _ = sort_tasks_by_keyword_priority(original_tasks)
        self.assertEqual(original_tasks, copy_tasks)

    def test_simulate_connection_refused(self):
        result = simulate_connection()
        self.assertIn("connectionrefusederror", result.lower())

    
    def test_export_tasks_to_csv(self):
        tasks = [("TaskA", "In Progress"), ("TaskB", "Completed")]
        export_tasks_to_csv(tasks)
        
        with open("exported_tasks.csv", "r") as f:
                content = f.read()
        self.assertIn("TaskA, In Progress", content)
        self.assertIn("TaskB, Completed", content)
        os.remove("exported_tasks.csv")

    def test_calculate_burn_rate(self):
        assert calculate_burn_rate(10000, 3000, 1000) == 7

    def test_linear_probe_insert(self):
        table = [None] * 10
        index = linear_probe_insert(table, "projectX")
        assert table[index] == "projectX"

    def test_max_project_value(self):
        costs = [10, 20, 30]
        values = [60, 100, 120]
        budget = 50
        assert max_project_value(costs, values, budget) == 220

    def test_forecast_deadline(self):
        assert forecast_deadline("2025-08-01", 30) == "2025-08-31"
        assert forecast_deadline("2025-12-01", 60) == "2026-01-30"
    try:
        forecast_deadline("2025-13-01", 10)
        assert False, "Expected ValueError for invalid month"
    except ValueError:
        pass

    try:
        forecast_deadline("2025-01-01", -5)
        assert False, "Expected ValueError for negative duration"
    except ValueError:
        pass

    
    def test_dynamic_array(self):
        arr = DynamicArray()
        arr.append(10)
        arr.append(20)
        arr.append(30)
        arr.append(40)
        arr.append(50)
        self.assertEqual(len(arr), 5)
        self.assertEqual(arr[0], 10)
        self.assertEqual(arr[4], 50)
        with self.assertRaises(IndexError):
            _ = arr[5]


if __name__ == '__main__':
    unittest.main()