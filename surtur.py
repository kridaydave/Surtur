#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import sqlite3

# Add src/ to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import db
from orchestrator import ArmConfig, run_experiment, run_arm
from eval_harness import evaluate_checkpoint

def run_cmd(args):
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    ac = ArmConfig(**{k: v for k, v in raw.items() if k in ArmConfig.__dataclass_fields__})
    print(f"Starting experiment run using config: {config_path}")
    run_experiment(ac)

def eval_cmd(args):
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

def main():
    parser = argparse.ArgumentParser(description="Surtur Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # run
    parser_run = subparsers.add_parser("run", help="Run experiments config")
    parser_run.add_argument("--config", default="src/config.yaml", help="Path to config.yaml")
    
    # eval
    parser_eval = subparsers.add_parser("eval", help="Evaluate runs")
    parser_eval.add_argument("--run-id", help="Run ID to evaluate")
    parser_eval.add_argument("--max-examples", type=int, default=1000, help="Max examples to evaluate")
    
    # reproduce
    parser_reproduce = subparsers.add_parser("reproduce", help="Reproduce a run")
    parser_reproduce.add_argument("run_id", help="Run ID to reproduce")
    
    args = parser.parse_args()
    if args.command == "run":
        run_cmd(args)
    elif args.command == "eval":
        eval_cmd(args)
    elif args.command == "reproduce":
        reproduce_cmd(args)

if __name__ == "__main__":
    main()
