import os
import unittest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock the database path before importing server
os.environ["DB_PATH"] = "surtur_test.db"
import db
# Ensure the DB path is set for test
db.DB_PATH = "surtur_test.db"

from server import app

class TestServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize test database
        if os.path.exists("surtur_test.db"):
            try:
                os.remove("surtur_test.db")
            except OSError:
                pass
        if os.path.exists("surtur_test.db-wal"):
            try:
                os.remove("surtur_test.db-wal")
            except OSError:
                pass
        if os.path.exists("surtur_test.db-shm"):
            try:
                os.remove("surtur_test.db-shm")
            except OSError:
                pass
        db.init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        # Clean up test database
        if os.path.exists("surtur_test.db"):
            try:
                os.remove("surtur_test.db")
            except OSError:
                pass
        if os.path.exists("surtur_test.db-wal"):
            try:
                os.remove("surtur_test.db-wal")
            except OSError:
                pass
        if os.path.exists("surtur_test.db-shm"):
            try:
                os.remove("surtur_test.db-shm")
            except OSError:
                pass

    def setUp(self):
        # Clear tables before each test
        conn = db.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evals")
            cursor.execute("DELETE FROM runs")
            conn.commit()
        finally:
            conn.close()

    def test_read_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_get_runs_empty(self):
        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_launch_run_invalid_arm(self):
        payload = {
            "model_id": "facebook/opt-125m",
            "arm": "invalid_arm_name",
            "seed": 42,
            "method": "sft",
            "layer_spec": "last_4"
        }
        response = self.client.post("/api/runs/launch", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid experiment arm", response.json()["detail"])

    @patch("subprocess.Popen")
    def test_launch_run_success(self, mock_popen):
        # Mock Popen to simulate a successful subprocess run
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        payload = {
            "model_id": "facebook/opt-125m",
            "arm": "surtur",
            "seed": 42,
            "method": "sft",
            "layer_spec": "last_4",
            "max_steps": 5,
            "batch_size": 2,
            "grad_accum": 1,
            "lr": 1e-5
        }
        response = self.client.post("/api/runs/launch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["run_id"].startswith("surtur_seed_42_"))

        # Give background task a split second to execute
        time.sleep(0.1)

        # Verify it exists in DB with 'completed' status since subprocess finished with exit code 0
        conn = db.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (data["run_id"],))
            run = cursor.fetchone()
            self.assertIsNotNone(run)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["arm"], "surtur")
        finally:
            conn.close()

    @patch("subprocess.Popen")
    def test_launch_run_failure(self, mock_popen):
        # Mock Popen to simulate a failed subprocess run (exit code 1)
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        payload = {
            "model_id": "facebook/opt-125m",
            "arm": "surtur",
            "seed": 42,
            "method": "sft",
            "layer_spec": "last_4",
            "max_steps": 5,
            "batch_size": 2,
            "grad_accum": 1,
            "lr": 1e-5
        }
        response = self.client.post("/api/runs/launch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Give background task a split second to execute
        time.sleep(0.1)

        conn = db.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (data["run_id"],))
            run = cursor.fetchone()
            self.assertIsNotNone(run)
            self.assertEqual(run["status"], "failed")
        finally:
            conn.close()

    def test_get_run_detail_not_found(self):
        response = self.client.get("/api/runs/nonexistent_run_id")
        self.assertEqual(response.status_code, 404)

    def test_get_run_detail_path_traversal(self):
        traversal_ids = [
            "../secret",
            "..\\secret",
            "arm/seed/../../etc/passwd",
            "run_id/logs/../../../"
        ]
        for run_id in traversal_ids:
            response = self.client.get(f"/api/runs/{run_id}")
            self.assertTrue(response.status_code in [400, 404])

    def test_get_run_logs_path_traversal(self):
        traversal_ids = [
            "../secret",
            "..\\secret"
        ]
        for run_id in traversal_ids:
            response = self.client.get(f"/api/runs/{run_id}/logs")
            self.assertTrue(response.status_code in [400, 404])

    def test_launch_eval_validation(self):
        # Insert a running run
        conn = db.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO runs (run_id, arm, seed, model_id, method, layer_spec, config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("test_run_1", "surtur", 42, "facebook/opt-125m", "sft", "last_4", "hash", "path", 0.0, 0, 0, "running"))
            conn.commit()
        finally:
            conn.close()

        # Cannot evaluate running run
        response = self.client.post("/api/runs/test_run_1/eval")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Cannot evaluate a run that is not completed", response.json()["detail"])

        # Invalid max_examples bounds
        response = self.client.post("/api/runs/test_run_1/eval?max_examples=0")
        self.assertEqual(response.status_code, 400)
        
        response = self.client.post("/api/runs/test_run_1/eval?max_examples=20000")
        self.assertEqual(response.status_code, 400)

    def test_get_verdict_incomplete(self):
        response = self.client.get("/api/verdict")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "incomplete")

    def test_get_phase_schema(self):
        response = self.client.get("/api/phase")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active", data)
        self.assertIn("milestones", data)
        self.assertIn("per_arm", data)
        self.assertIn(data["active"], ["M0", "M1", "M2", "M3", "M4"])
        # Every milestone has done/label/phase
        for mid, m in data["milestones"].items():
            self.assertIn("done", m)
            self.assertIn("label", m)
            self.assertIn("phase", m)

    def test_get_roadmap_schema(self):
        response = self.client.get("/api/roadmap")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in ("vision", "north_star", "principles", "phases",
                    "not_building", "open_questions"):
            self.assertIn(key, data)
        self.assertEqual(len(data["principles"]), 5)
        self.assertEqual(len(data["phases"]), 5)
        # Not-building is explicitly the "won't" list
        self.assertGreaterEqual(len(data["not_building"]), 5)

    def test_phase_advances_with_seeds(self):
        # Insert 5+ completed surtur and frozen runs -> M0 should be done
        conn = db.get_db()
        try:
            cursor = conn.cursor()
            for i, seed in enumerate([42, 1337, 7, 8, 9]):
                cursor.execute("""
                INSERT OR IGNORE INTO runs
                  (run_id, arm, seed, model_id, method, layer_spec,
                   config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"sur_test_{seed}", "surtur", seed, "facebook/opt-125m",
                      "sft", "last_4", "h", "p", 1.0, 1, 100, "completed"))
                cursor.execute("""
                INSERT OR IGNORE INTO runs
                  (run_id, arm, seed, model_id, method, layer_spec,
                   config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"frz_test_{seed}", "frozen", seed, "facebook/opt-125m",
                      "sft", "last_4", "h", "p", 1.0, 0, 100, "completed"))
            conn.commit()
        finally:
            conn.close()

        response = self.client.get("/api/phase")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["milestones"]["M0"]["done"])
        self.assertIn(data["active"], ["M1", "M2", "M3", "M4"])

if __name__ == "__main__":
    unittest.main()
