#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import sqlite3

# Add src/ to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

_run_experiment = None
def run_experiment(*args, **kwargs):
    global _run_experiment
    if _run_experiment is None:
        from orchestrator import run_experiment as _impl
        _run_experiment = _impl
    return _run_experiment(*args, **kwargs)

_run_arm = None
def run_arm(*args, **kwargs):
    global _run_arm
    if _run_arm is None:
        from orchestrator import run_arm as _impl
        _run_arm = _impl
    return _run_arm(*args, **kwargs)

_evaluate_checkpoint = None
def evaluate_checkpoint(*args, **kwargs):
    global _evaluate_checkpoint
    if _evaluate_checkpoint is None:
        from eval_harness import evaluate_checkpoint as _impl
        _evaluate_checkpoint = _impl
    return _evaluate_checkpoint(*args, **kwargs)

def run_cmd(args):
    import db
    from orchestrator import ArmConfig
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    if not isinstance(raw, dict):
        print("Error: Config must be a YAML dictionary")
        sys.exit(1)

    # Validate against ArmConfig fields
    invalid_keys = [k for k in raw.keys() if k not in ArmConfig.__dataclass_fields__]
    if invalid_keys:
        print(f"Error: Invalid configuration fields: {invalid_keys}")
        sys.exit(1)

    try:
        ac = ArmConfig(**raw)
    except Exception as e:
        print(f"Error validating config fields: {e}")
        sys.exit(1)

    # Check that database path exists
    if not os.path.exists(db.DB_PATH):
        print(f"Error: Database path '{db.DB_PATH}' does not exist.")
        sys.exit(1)

    if getattr(args, "dry_run", False):
        print("Dry run validation successful.")
        sys.exit(0)

    print(f"Starting experiment run using config: {config_path}")
    run_experiment(ac)

def eval_cmd(args):
    import db
    db.init_db()
    conn = db.get_db()
    cursor = conn.cursor()
    
    if args.run_id:
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (args.run_id,))
        runs = cursor.fetchall()
        if not runs:
            print(f"Error: Run ID {args.run_id} not found in database.")
            sys.exit(1)
    else:
        # Find runs that don't have all evaluation sets done
        cursor.execute("SELECT * FROM runs WHERE status = 'completed'")
        all_runs = cursor.fetchall()
        runs = []
        for run in all_runs:
            cursor.execute("SELECT DISTINCT eval_set FROM evals WHERE run_id = ?", (run["run_id"],))
            existing_sets = {r["eval_set"] for r in cursor.fetchall()}
            required_sets = {"mmlu", "arc", "gsm8k", "truthfulqa", "harmlessness"}
            if len(required_sets - existing_sets) > 0:
                runs.append(run)
                
    conn.close()
    
    if not runs:
        print("No pending runs to evaluate.")
        return
        
    for run in runs:
        run_id = run["run_id"]
        ckpt_path = run["ckpt_path"]
        seed = run["seed"]
        print(f"Evaluating run {run_id} (checkpoint: {ckpt_path}, seed: {seed})")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint path {ckpt_path} does not exist. Skipping.")
            continue
            
        evaluate_checkpoint(
            checkpoint_dir=ckpt_path,
            max_examples=args.max_examples,
            seed=seed,
            run_id=run_id,
            evals_jsonl_path="surtur_out/evals.jsonl"
        )

def reproduce_cmd(args):
    import db
    from orchestrator import ArmConfig
    db.init_db()
    conn = db.get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (args.run_id,))
    run = cursor.fetchone()
    conn.close()
    
    if not run:
        print(f"Error: Run {args.run_id} not found in database.")
        sys.exit(1)
        
    print(f"Reproducing run {args.run_id}...")
    ac = ArmConfig(
        model_id=run["model_id"],
        method=run["method"],
        layer_spec=run["layer_spec"],
        seeds=[run["seed"]],
        arms=[run["arm"]]
    )
    run_arm(run["arm"], ac, run["seed"])
    print(f"Reproduction run completed successfully.")

def dashboard_cmd(args):
    # Runs the terminal/TUI dashboard using cli_dashboard
    from cli_dashboard import run_dashboard
    run_dashboard(args)

def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Surtur Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # run
    parser_run = subparsers.add_parser("run", help="Run experiments config")
    parser_run.add_argument("--config", default="src/config.yaml", help="Path to config.yaml")
    parser_run.add_argument("--dry-run", action="store_true", help="Dry run validation check without execution")
    
    # eval
    parser_eval = subparsers.add_parser("eval", help="Evaluate runs")
    parser_eval.add_argument("--run-id", help="Run ID to evaluate")
    parser_eval.add_argument("--max-examples", type=int, default=1000, help="Max examples to evaluate")
    
    # reproduce
    parser_reproduce = subparsers.add_parser("reproduce", help="Reproduce a run")
    parser_reproduce.add_argument("run_id", help="Run ID to reproduce")
    
    # dashboard
    parser_dashboard = subparsers.add_parser("dashboard", help="Start the interactive Surtur CLI TUI dashboard")
    parser_dashboard.add_argument("--refresh", type=float, default=1.0, help="Refresh interval in seconds")
    parser_dashboard.add_argument("--style", default="dual", choices=["status", "dual", "split"], help="TUI layout style")
    
    return parser.parse_args(args_list)

def main(args_list=None):
    try:
        args = parse_args(args_list)
        if args.command == "run":
            run_cmd(args)
        elif args.command == "eval":
            eval_cmd(args)
        elif args.command == "reproduce":
            reproduce_cmd(args)
        elif args.command == "dashboard":
            dashboard_cmd(args)
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
