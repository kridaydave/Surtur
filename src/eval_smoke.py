from eval_harness import evaluate_checkpoint
from metrics import compute_verdict, print_results, print_verdict

CKPT = "src/smoke_out"

print("[Eval Smoke] Evaluating smoke_out checkpoint (max_examples=5)...")
results = evaluate_checkpoint(CKPT, max_examples=5)

print_results("Smoke Checkpoint Results", results)
print("\n[Eval Smoke] PASSED — harness runs without errors.")
