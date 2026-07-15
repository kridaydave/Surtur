import os
import json
import argparse
from eval_harness import evaluate_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Surtur evaluation driver")
    parser.add_argument("--runs", default="experiments/p0/runs.jsonl", help="Path to runs.jsonl registry")
    parser.add_argument("--evals", default="experiments/p0/evals.jsonl", help="Path to evals.jsonl registry")
    parser.add_argument("--max_examples", type=int, default=1000, help="Max examples to evaluate per task")
    args = parser.parse_args()

    if not os.path.exists(args.runs):
        print(f"[Eval Driver] Registry {args.runs} not found. Nothing to evaluate.")
        return

    # Load already evaluated run-sets
    evaluated_runs = set()
    if os.path.exists(args.evals):
        with open(args.evals, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        evaluated_runs.add((entry["run_id"], entry["eval_set"]))
                    except Exception:
                        pass

    # Read runs to evaluate
    runs = []
    with open(args.runs, "r") as f:
        for line in f:
            if line.strip():
                try:
                    runs.append(json.loads(line))
                except Exception:
                    pass

    print(f"[Eval Driver] Found {len(runs)} runs in registry.")
    for run in runs:
        run_id = run["run_id"]
        ckpt_path = run["ckpt_path"]
        seed = run["seed"]
        
        # Check if all 5 evaluations are done for this run
        eval_sets = ["mmlu", "arc", "gsm8k", "truthfulqa", "harmlessness"]
        missing_sets = [es for es in eval_sets if (run_id, es) not in evaluated_runs]
        
        if not missing_sets:
            print(f"[Eval Driver] Run {run_id} already fully evaluated. Skipping.")
            continue
            
        print(f"[Eval Driver] Evaluating run {run_id} at {ckpt_path} (missing: {missing_sets})...")
        norm_path = os.path.normpath(ckpt_path)
        is_hf_model = len(norm_path.split(os.sep)) == 2 and not ckpt_path.startswith(".")
        if not os.path.exists(ckpt_path) and not is_hf_model:
            print(f"[Eval Driver] WARNING: Checkpoint path {ckpt_path} not found. Skipping.")
            continue
            
        # We run the evaluate_checkpoint function, which appends to args.evals internally
        try:
            evaluate_checkpoint(
                checkpoint_dir=ckpt_path,
                max_examples=args.max_examples,
                seed=seed,
                run_id=run_id,
                evals_jsonl_path=args.evals
            )
            print(f"[Eval Driver] Run {run_id} evaluation finished.")
        except Exception as e:
            print(f"[Eval Driver] ERROR: Failed to evaluate run {run_id} ({e})")

if __name__ == "__main__":
    main()
