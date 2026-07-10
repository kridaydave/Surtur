import os
import json
import argparse
import numpy as np
from metrics import compute_verdict

HARNESS_VERSION = "1.0.0"
KNOWN_GOOD_HARNESS_VERSIONS = ["1.0.0"]

def main():
    parser = argparse.ArgumentParser(description="Surtur Phase 0 Sign-Off Command")
    parser.add_argument("--runs", default="experiments/p0/runs.jsonl", help="Path to runs.jsonl registry")
    parser.add_argument("--evals", default="experiments/p0/evals.jsonl", help="Path to evals.jsonl registry")
    parser.add_argument("--contamination", default="data/contamination_probe.json", help="Path to contamination probe results")
    parser.add_argument("--out_dir", default="experiments/p0", help="Directory for M0_verdict output files")
    args = parser.parse_args()

    # 1. Validation Checks
    
    # (d) HARNESS_VERSION check
    if HARNESS_VERSION not in KNOWN_GOOD_HARNESS_VERSIONS:
        print(f"[Sign-Off] ERROR: HARNESS_VERSION '{HARNESS_VERSION}' is not on the known-good list.")
        return

    # Load runs
    if not os.path.exists(args.runs):
        print(f"[Sign-Off] ERROR: Registry {args.runs} not found.")
        return
        
    runs = []
    with open(args.runs, "r") as f:
        for line in f:
            if line.strip():
                try:
                    runs.append(json.loads(line))
                except Exception:
                    pass

    # Load evals
    if not os.path.exists(args.evals):
        print(f"[Sign-Off] ERROR: Evals {args.evals} not found.")
        return
        
    evals = []
    with open(args.evals, "r") as f:
        for line in f:
            if line.strip():
                try:
                    evals.append(json.loads(line))
                except Exception:
                    pass

    # (b) Contamination check
    contamination_status = "INCONCLUSIVE"
    if os.path.exists(args.contamination):
        try:
            with open(args.contamination, "r") as f:
                c_data = json.load(f)
                contamination_status = c_data.get("status", "INCONCLUSIVE")
        except Exception:
            pass
            
    if contamination_status == "INCONCLUSIVE":
        print("[Sign-Off] ERROR: Contamination check is missing or INCONCLUSIVE. Verification blocked.")
        return
    elif contamination_status == "CONTAMINATED":
        print("[Sign-Off] ERROR: Base model has been flagged as CONTAMINATED on public benchmarks.")
        return

    if not runs:
        print("[Sign-Off] ERROR: No runs found in registry. Verification blocked.")
        return

    # Group runs by arm
    arm_runs = {}
    for r in runs:
        arm = r.get("arm", "unknown")
        arm_runs.setdefault(arm, []).append(r)

    # (c) Seed count check (must be >= 5)
    for arm, arm_r_list in arm_runs.items():
        seeds_found = {r.get("seed") for r in arm_r_list}
        if len(seeds_found) < 5:
            print(f"[Sign-Off] ERROR: Seed count for arm '{arm}' is {len(seeds_found)} (< 5). Verification blocked.")
            return

    # Check for missing evaluations for each run
    run_evals = {}
    for ev in evals:
        run_evals.setdefault(ev.get("run_id"), []).append(ev)
        
    # (a) Check if any run is missing evaluations
    eval_sets = {"mmlu", "arc", "gsm8k", "truthfulqa", "harmlessness"}
    for r in runs:
        r_id = r.get("run_id")
        r_evals = run_evals.get(r_id, [])
        r_sets = {e.get("eval_set") for e in r_evals}
        missing = eval_sets - r_sets
        if missing:
            print(f"[Sign-Off] ERROR: Run {r_id} is missing evaluations for: {list(missing)}.")
            return

    # Compute mean accuracies per arm per task
    arm_task_accs = {}
    arm_task_slices = {}
    for arm, r_list in arm_runs.items():
        arm_task_accs[arm] = {}
        for r in r_list:
            r_id = r.get("run_id")
            r_evals = run_evals.get(r_id, [])
            for ev in r_evals:
                task = ev.get("eval_set")
                arm_task_accs[arm].setdefault(task, []).append(ev.get("accuracy", 0.0))
                
                # Slices aggregation
                ev_slices = ev.get("slices", {})
                if isinstance(ev_slices, str):
                    try:
                        ev_slices = json.loads(ev_slices)
                    except Exception:
                        ev_slices = {}
                if not isinstance(ev_slices, dict):
                    ev_slices = {}
                arm_task_slices.setdefault((arm, task), {})
                for slice_name, slice_acc in ev_slices.items():
                    arm_task_slices[(arm, task)].setdefault(slice_name, []).append(slice_acc)

    # Aggregate stats (mean ± std)
    arm_stats = {}
    for arm, task_dict in arm_task_accs.items():
        arm_stats[arm] = {}
        for task, accs in task_dict.items():
            arm_stats[arm][task] = {
                "mean": float(np.mean(accs)) if accs else 0.0,
                "std": float(np.std(accs)) if accs else 0.0,
                "count": len(accs)
            }

    # Aggregate slice stats
    arm_task_slices_mean = {}
    for (arm, task), slices_dict in arm_task_slices.items():
        arm_task_slices_mean.setdefault(arm, {})[task] = {
            slice_name: float(np.mean(accs)) if accs else 0.0 for slice_name, accs in slices_dict.items()
        }

    # Prepare inputs for compute_verdict
    # We compare surtur, frozen, and untrained_ref
    surtur_stats = arm_stats.get("surtur", {})
    frozen_stats = arm_stats.get("frozen", {})
    untrained_stats = arm_stats.get("untrained_ref", {})

    surtur_results = {task: {"accuracy": stats["mean"], "n_examples": 1000, "stderr": stats["std"]} for task, stats in surtur_stats.items()}
    frozen_results = {task: {"accuracy": stats["mean"], "n_examples": 1000, "stderr": stats["std"]} for task, stats in frozen_stats.items()}
    
    untrained_results = None
    if untrained_stats:
        untrained_results = {task: {"accuracy": stats["mean"], "n_examples": 1000, "stderr": stats["std"]} for task, stats in untrained_stats.items()}

    # Compute compute ratio: average duration of surtur vs full_ft
    surtur_durations = [r.get("duration_sec", 0.0) for r in arm_runs.get("surtur", [])]
    full_ft_durations = [r.get("duration_sec", 0.0) for r in arm_runs.get("full_ft", [])]
    
    avg_surtur_dur = np.mean(surtur_durations) if surtur_durations else 0.0
    avg_full_ft_dur = np.mean(full_ft_durations) if full_ft_durations else 0.0
    compute_ratio = float(avg_surtur_dur / avg_full_ft_dur) if avg_full_ft_dur > 0 else 0.0

    # Compute final verdict
    verdict = compute_verdict(surtur_results, frozen_results, untrained_results, compute_ratio=compute_ratio, arm_slices=arm_task_slices_mean)

    # 2. Output generation
    os.makedirs(args.out_dir, exist_ok=True)
    verdict_status = "PASS" if verdict["pass"] else "FAIL"

    # Save verdict JSON
    json_out = {
        "verdict": verdict_status,
        "compute_ratio": compute_ratio,
        "arm_stats": arm_stats,
        "failures": verdict["failures"]
    }
    with open(os.path.join(args.out_dir, "M0_verdict.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    # Save verdict Markdown
    model_id = runs[0].get("model_id", "unknown") if runs else "unknown"
    md_lines = [
        f"# Phase 0 (M0) Sign-Off Verdict: **{verdict_status}**\n",
        f"- **Model ID**: {model_id}",
        f"- **Contamination Check**: {contamination_status}",
        f"- **Compute Ratio**: {compute_ratio:.4f} (threshold <= 0.30)",
        "\n## Retention Analysis (must be >= 0.98)\n",
        "| Task | Surtur Mean | Frozen Mean | Retention Ratio | Status |",
        "|---|---|---|---|---|",
    ]
    for task, ret in verdict["retention"].items():
        s_acc = arm_stats.get("surtur", {}).get(task, {}).get("mean", 0.0)
        f_acc = arm_stats.get("frozen", {}).get(task, {}).get("mean", 0.0)
        status = "✓ PASS" if ret >= 0.98 else "✗ FAIL"
        md_lines.append(f"| {task} | {s_acc:.4f} | {f_acc:.4f} | {ret:.4f} | {status} |")

    if verdict["alignment_gain"]:
        md_lines.append("\n## Alignment Gain (must be > 0)\n")
        md_lines.append("| Task | Surtur Mean | Untrained Mean | Alignment Gain | Status |")
        md_lines.append("|---|---|---|---|---|")
        for task, gain in verdict["alignment_gain"].items():
            s_acc = arm_stats.get("surtur", {}).get(task, {}).get("mean", 0.0)
            u_acc = arm_stats.get("untrained_ref", {}).get(task, {}).get("mean", 0.0)
            status = "✓ PASS" if gain > 0 else "✗ FAIL"
            md_lines.append(f"| {task} | {s_acc:.4f} | {u_acc:.4f} | {gain:+.4f} | {status} |")

    if verdict["failures"]:
        md_lines.append("\n## Failures / Violations\n")
        for fail in verdict["failures"]:
            md_lines.append(f"- {fail}")
            
    with open(os.path.join(args.out_dir, "M0_verdict.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Generate HTML results_figure
    from make_figure import render
    retention_dict = verdict["retention"]
    gain_dict = verdict["alignment_gain"]
    html_content = render(
        retention=retention_dict,
        alignment_gain=gain_dict,
        compute_ratio=compute_ratio,
        verdict=verdict_status,
        model_id=model_id,
        seeds=len(arm_runs.get("surtur", [])),
        failures=verdict["failures"]
    )
    with open(os.path.join(args.out_dir, "M0_verdict.html"), "w") as f:
        f.write(html_content)

    print(f"[Sign-Off] Completed. Verdict: {verdict_status}.")
    print(f"Outputs written to {args.out_dir} (M0_verdict.json, M0_verdict.md, M0_verdict.html)")

if __name__ == "__main__":
    main()
