import os
import sys
import unittest
from unittest.mock import patch, MagicMock

REAL_EXISTS = os.path.exists

# Add src/ to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

# Set up test database path
os.environ["DB_PATH"] = "surtur_test.db"
import db
db.DB_PATH = "surtur_test.db"

import cli_dashboard

class TestCLIDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize clean test db
        if os.path.exists("surtur_test.db"):
            try:
                os.remove("surtur_test.db")
            except OSError:
                pass
        db.init_db()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("surtur_test.db"):
            try:
                os.remove("surtur_test.db")
            except OSError:
                pass

    def setUp(self):
        # Clear tables before each test to guarantee empty states
        conn = db.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evals")
            cursor.execute("DELETE FROM runs")
            conn.commit()
        finally:
            conn.close()

    def test_get_runs_empty(self):
        runs = cli_dashboard.get_runs_data()
        self.assertEqual(runs, [])

    def test_get_evals_empty(self):
        evals = cli_dashboard.get_evals_data()
        self.assertEqual(evals, [])

    @patch("os.path.exists")
    def test_make_verdict_panel_empty(self, mock_exists):
        mock_exists.return_value = False
        panel = cli_dashboard.get_latest_verdict()
        self.assertIsNone(panel)

    def test_render_inferno_returns_layout(self):
        runs = [
            {
                'run_id': 'test_run_1',
                'arm': 'surtur',
                'seed': 42,
                'epoch': 1,
                'step': 10,
                'loss': 0.5,
                'status': 'completed',
                'trainable_params': 1000,
                'total_params': 10000
            }
        ]
        evals = []
        verdict = {'pass': True, 'verdict': 'PASS'}
        
        layout = cli_dashboard.render_inferno(runs, evals, verdict)
        from rich.layout import Layout
        self.assertIsInstance(layout, Layout)
        self.assertEqual(len(layout.children), 3)
        
    def test_render_inferno_empty_runs(self):
        layout = cli_dashboard.render_inferno([], [], {})
        from rich.layout import Layout
        self.assertIsInstance(layout, Layout)

    @patch("cli_dashboard.input")
    @patch("cli_dashboard.console")
    @patch("cli_dashboard.get_runs_data")
    @patch("cli_dashboard.get_evals_data")
    @patch("cli_dashboard.get_latest_verdict")
    @patch("cli_dashboard.show_status_live")
    def test_run_dashboard_commands(self, mock_show_status_live, mock_get_verdict, mock_get_evals, mock_get_runs, mock_console, mock_input):
        mock_input.side_effect = ["help", "status", "style", "style split", "style invalid_style", "invalid", "exit"]
        mock_get_runs.return_value = []
        mock_get_evals.return_value = []
        mock_get_verdict.return_value = None

        class Args:
            refresh = 0.01
            style = "dual"

        cli_dashboard.run_dashboard(Args())
        self.assertEqual(mock_input.call_count, 7)
        self.assertTrue(mock_console.print.called)
        self.assertTrue(mock_show_status_live.called)

    @patch("cli_dashboard.input")
    @patch("cli_dashboard.console")
    def test_run_dashboard_keyboard_interrupt(self, mock_console, mock_input):
        mock_input.side_effect = [KeyboardInterrupt, "exit"]
        class Args:
            refresh = 0.01
            style = "dual"
        cli_dashboard.run_dashboard(Args())
        self.assertTrue(mock_console.print.called)

    @patch("cli_dashboard.input")
    @patch("cli_dashboard.console")
    def test_run_dashboard_eof_error(self, mock_console, mock_input):
        mock_input.side_effect = EOFError
        class Args:
            refresh = 0.01
            style = "dual"
        cli_dashboard.run_dashboard(Args())
        self.assertTrue(mock_console.print.called)

    @patch("cli_dashboard.input")
    @patch("cli_dashboard.console")
    @patch("surtur.run_cmd")
    @patch("surtur.eval_cmd")
    @patch("surtur.reproduce_cmd")
    @patch("os.path.exists")
    def test_tui_command_execution(self, mock_exists, mock_reproduce, mock_eval, mock_run, mock_console, mock_input):
        def side_effect(path):
            if path in ("src/config.yaml", "config.yaml"):
                return True
            return REAL_EXISTS(path)
        mock_exists.side_effect = side_effect

        mock_input.side_effect = [
            "run src/config.yaml --dry-run",
            "1",
            "eval run_123 500",
            "reproduce run_456",
            "config",
            "exit"
        ]
        class Args:
            refresh = 0.01
            style = "status"
        
        cli_dashboard.run_dashboard(Args())
        
        mock_run.assert_called_once()
        run_args = mock_run.call_args[0][0]
        self.assertEqual(run_args.config, "src/config.yaml")
        self.assertTrue(run_args.dry_run)
        
        mock_eval.assert_called_once()
        eval_args = mock_eval.call_args[0][0]
        self.assertEqual(eval_args.run_id, "run_123")
        self.assertEqual(eval_args.max_examples, 500)
        
        mock_reproduce.assert_called_once()
        reproduce_args = mock_reproduce.call_args[0][0]
        self.assertEqual(reproduce_args.run_id, "run_456")

    @patch("cli_dashboard.input")
    @patch("cli_dashboard.console")
    @patch("surtur.run_cmd")
    @patch("os.path.exists")
    def test_tui_custom_run_command(self, mock_exists, mock_run, mock_console, mock_input):
        def side_effect(path):
            if path in ("src/config.yaml", "config.yaml"):
                return True
            return REAL_EXISTS(path)
        mock_exists.side_effect = side_effect
        mock_input.side_effect = [
            "run",
            "2", # Choose custom mode
            "custom_model", # Model ID
            "grpo", # Method
            "last_2", # Layer spec
            "data/custom_train.jsonl", # Dataset path
            "12,34", # Seeds
            "surtur_arm", # Arms
            "5", # Max steps
            "2", # Batch size
            "3e-5", # Learning rate
            "y", # Dry run
            "exit"
        ]
        class Args:
            refresh = 0.01
            style = "status"
        
        cli_dashboard.run_dashboard(Args())
        
        mock_run.assert_called_once()
        run_args = mock_run.call_args[0][0]
        self.assertTrue(run_args.dry_run)
        self.assertTrue(run_args.config.startswith("surtur_out/temp_configs/custom_"))
        
        # Verify YAML content
        import yaml
        with open(run_args.config, "r") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["model_id"], "custom_model")
        self.assertEqual(cfg["method"], "grpo")
        self.assertEqual(cfg["layer_spec"], "last_2")
        self.assertEqual(cfg["dataset_path"], "data/custom_train.jsonl")
        self.assertEqual(cfg["seeds"], [12, 34])
        self.assertEqual(cfg["arms"], ["surtur_arm"])
        self.assertEqual(cfg["max_steps"], 5)
        self.assertEqual(cfg["batch_size"], 2)
        self.assertEqual(cfg["lr"], 3e-5)

if __name__ == "__main__":
    unittest.main()
