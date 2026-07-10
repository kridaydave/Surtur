#!/usr/bin/env bash
# Surtur — one-command reproduction script
# Usage:   bash reproduce.sh
# Result:  smoke tests pass + insulation verified + summary table printed
set -euo pipefail

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "ERROR: no python on PATH"; exit 1; fi
PIP="$PY -m pip"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== Surtur reproduction ==="
echo "Repo:  $REPO_ROOT"
echo "Python: $($PY --version)"
echo

echo "[1/5] Installing pinned environment..."
$PIP install -q -r requirements.lock.txt

echo "[2/5] Freezing core + guardrail test..."
cd src && $PY test_freeze.py

echo "[3/5] Training engine smoke test (3 SFT steps, CPU fallback)..."
$PY smoke_train.py

echo "[4/5] Orchestrator smoke test (arms surtur + frozen, 1 seed)..."
$PY test_orchestrator.py

echo "[5/5] Eval harness smoke test..."
$PY eval_smoke.py

echo
echo "=== Reproduction complete ==="
echo "Expected: ALL CHECKS PASSED, ORCHESTRATOR SMOKE TEST PASSED, [Eval Smoke] PASSED"
echo "Tolerance: bit-identical on CPU; +/- 1% on GPU (CUDA nondeterminism)"
echo "If any step failed, the exact line above is where reproducibility broke."
