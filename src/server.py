import os
import sys
import json
import sqlite3
import subprocess
import re
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# Add src/ to python path
sys.path.insert(0, os.path.dirname(__file__))

import db

db_lock = threading.Lock()

app = FastAPI(title="Surtur Web Dashboard Server")

# Paths anchored to this file, NOT cwd — so the server works no matter
# where it's launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_STATIC_DIR = os.path.join(_HERE, "static")
os.makedirs(_STATIC_DIR, exist_ok=True)

# Initialize database (anchored to project root, so the registry is shared
# regardless of cwd)
os.makedirs(_PROJECT_ROOT, exist_ok=True)
os.chdir(_PROJECT_ROOT)
db.init_db()

def validate_run_id(run_id: str):
    if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")

# Serve static files
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

class RunRequest(BaseModel):
    model_id: str
    arm: str
    seed: int
    method: str
    layer_spec: str
    dataset_path: Optional[str] = None
    max_steps: int = 100
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 5e-5

def execute_run(run_id: str, req: RunRequest):
    import yaml
    import time
    
    os.makedirs("surtur_out/temp_configs", exist_ok=True)
    config_path = f"surtur_out/temp_configs/{run_id}.yaml"
    
    config_data = {
        "run_id": run_id,
        "model_id": req.model_id,
        "method": req.method,
        "layer_spec": req.layer_spec,
        "dataset_path": req.dataset_path,
        "seeds": [req.seed],
        "arms": [req.arm],
        "max_steps": req.max_steps,
        "batch_size": req.batch_size,
        "grad_accum": req.grad_accum,
        "lr": req.lr
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    os.makedirs("surtur_out/logs", exist_ok=True)
    log_path = f"surtur_out/logs/{run_id}.log"
    
    py_executable = sys.executable
    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [py_executable, "surtur.py", "run", "--config", config_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait()
            
        with db_lock:
            conn = db.get_db()
            cursor = conn.cursor()
            if process.returncode != 0:
                cursor.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,))
            else:
                cursor.execute("UPDATE runs SET status = 'completed' WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()
    except Exception as e:
        with db_lock:
            conn = db.get_db()
            cursor = conn.cursor()
            cursor.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()
        with open(log_path, "a") as log_file:
            log_file.write(f"\n[Server Error] Run failed with exception: {e}\n")

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Surtur Dashboard index.html not found!</h1>"

@app.get("/api/runs")
def get_runs():
    conn = db.get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
    runs = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return runs

@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: str):
    validate_run_id(run_id)
    conn = db.get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail="Run not found")
        
    cursor.execute("SELECT * FROM evals WHERE run_id = ?", (run_id,))
    evals = [dict(e) for e in cursor.fetchall()]
    conn.close()
    
    log_content = ""
    log_path = f"surtur_out/logs/{run_id}.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
    else:
        log_content = f"Log file for run {run_id} is not available. This run may have been executed via the CLI or imported."
            
    return {
        "run": dict(run),
        "evals": evals,
        "logs": log_content
    }

@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str):
    validate_run_id(run_id)
    log_path = f"surtur_out/logs/{run_id}.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            return {"logs": f.read()}
    return {"logs": f"Log file for run {run_id} is not available. This run may have been executed via the CLI or imported."}

@app.post("/api/runs/launch")
def launch_run(req: RunRequest, background_tasks: BackgroundTasks):
    import time
    import hashlib
    
    if req.arm not in ["surtur", "full_ft", "frozen", "untrained_ref"]:
        raise HTTPException(status_code=400, detail="Invalid experiment arm")
        
    start_time = int(time.time())
    run_id = f"{req.arm}_seed_{req.seed}_{start_time}"
    
    config_str = f"{req.model_id}_{req.method}_{req.layer_spec}_{req.dataset_path}_{req.max_steps}_{req.batch_size}_{req.grad_accum}_{req.lr}_bf16"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    with db_lock:
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO runs (run_id, arm, seed, model_id, method, layer_spec, config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, req.arm, req.seed, req.model_id, req.method, req.layer_spec, config_hash,
            os.path.join("surtur_out", f"arm_{req.arm}", f"seed_{req.seed}"), 0.0, 0, 0, "running"
        ))
        conn.commit()
        conn.close()
    
    background_tasks.add_task(execute_run, run_id, req)
    
    return {"status": "success", "run_id": run_id}

@app.post("/api/runs/{run_id}/eval")
def launch_eval(run_id: str, background_tasks: BackgroundTasks, max_examples: int = 10):
    validate_run_id(run_id)
    if max_examples < 1 or max_examples > 10000:
        raise HTTPException(status_code=400, detail="max_examples must be between 1 and 10000")
    conn = db.get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    conn.close()
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Cannot evaluate a run that is not completed")
        
    def execute_eval():
        py_executable = sys.executable
        log_path = f"surtur_out/logs/{run_id}_eval.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [py_executable, "surtur.py", "eval", "--run-id", run_id, "--max-examples", str(max_examples)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True
            )
            process.wait()
            
    background_tasks.add_task(execute_eval)
    return {"status": "success", "message": f"Evaluation launched for run {run_id}."}

@app.get("/api/verdict")
def get_verdict():
    conn = db.get_db()
    cursor = conn.cursor()
    
    arms = ["surtur", "full_ft", "frozen", "untrained_ref"]
    results = {}
    for arm in arms:
        cursor.execute("""
        SELECT * FROM runs 
        WHERE arm = ? AND status = 'completed' 
        ORDER BY created_at DESC LIMIT 1
        """, (arm,))
        run = cursor.fetchone()
        if run:
            run_id = run["run_id"]
            cursor.execute("SELECT * FROM evals WHERE run_id = ?", (run_id,))
            evals = cursor.fetchall()
            results[arm] = {e["eval_set"]: {"accuracy": e["accuracy"], "stderr": e["stderr"]} for e in evals}
            results[arm]["duration_sec"] = run["duration_sec"]
            results[arm]["trainable_params"] = run["trainable_params"]
            results[arm]["total_params"] = run["total_params"]
            results[arm]["seed"] = run["seed"]
            results[arm]["layer_spec"] = run["layer_spec"]
            results[arm]["model_id"] = run["model_id"]
            results[arm]["method"] = run["method"]
            
    conn.close()
    
    if "surtur" not in results or "frozen" not in results:
        return {"status": "incomplete", "message": "Surtur and Frozen baseline runs are required to compute verdict."}
        
    from metrics import compute_verdict
    
    surtur_accs = {k: v for k, v in results["surtur"].items() if k not in ["duration_sec", "trainable_params", "total_params"]}
    frozen_accs = {k: v for k, v in results["frozen"].items() if k not in ["duration_sec", "trainable_params", "total_params"]}
    untrained_accs = {}
    if "untrained_ref" in results:
        untrained_accs = {k: v for k, v in results["untrained_ref"].items() if k not in ["duration_sec", "trainable_params", "total_params"]}
        
    compute_ratio = None
    if "full_ft" in results and results["full_ft"].get("trainable_params", 0) > 0:
        compute_ratio = results["surtur"].get("trainable_params", 0) / results["full_ft"].get("trainable_params", 1)
        
    verdict = compute_verdict(surtur_accs, frozen_accs, untrained_accs, compute_ratio)
    return {
        "status": "success",
        "results": results,
        "verdict": verdict
    }

def _phase_state():
    """Derive current milestone (M0..M5) and active phase (P0..P4) from the run registry.

    Heuristic (deterministic, no external state):
      M0 done  = 5+ completed seeds for *both* surtur and frozen arms (P0 head-to-head done)
      M1 done  = M0 done AND a verdict has been recorded (P1 runner reproducible)
      M2 done  = M1 done AND all surtur runs share one config_hash (P2 reproducibility)
      M3 done  = a M0_verdict artifact exists on disk (P3 dashboard ship)
      M4 done  = M3 done (P4 adoption — out of scope for this heuristic)
    Active phase = first phase whose gate is not yet passed.
    """
    conn = db.get_db()
    try:
        cursor = conn.cursor()

        # Per-arm seed counts and seed lists
        per_arm = {}
        cursor.execute("""
        SELECT arm, COUNT(*) AS n, GROUP_CONCAT(seed) AS seeds, COUNT(DISTINCT config_hash) AS distinct_hashes
        FROM runs WHERE status = 'completed' AND arm IN ('surtur','full_ft','frozen','untrained_ref')
        GROUP BY arm
        """)
        for r in cursor.fetchall():
            seeds = [int(s) for s in (r["seeds"] or "").split(",") if s]
            per_arm[r["arm"]] = {
                "n": r["n"],
                "seeds": seeds,
                "distinct_hashes": r["distinct_hashes"],
            }

        surtur = per_arm.get("surtur", {"n": 0, "seeds": [], "distinct_hashes": 0})
        frozen = per_arm.get("frozen", {"n": 0, "seeds": [], "distinct_hashes": 0})

        m0_done = surtur["n"] >= 5 and frozen["n"] >= 5
        m1_done = m0_done
        m2_done = m0_done and surtur["distinct_hashes"] == 1 and surtur["n"] >= 5

        import os
        signoff_path = "M0_verdict.json"
        m3_done = m2_done and os.path.exists(signoff_path)

        # Active phase
        if not m0_done:
            active = "M0"
        elif not m1_done:
            active = "M1"
        elif not m2_done:
            active = "M2"
        elif not m3_done:
            active = "M3"
        else:
            active = "M4"

        return {
            "active": active,
            "milestones": {
                "M0": {"done": m0_done, "label": "Eval suite + thresholds defined", "phase": "Pre-P0"},
                "M1": {"done": m1_done, "label": "Core claim validated", "phase": "P0"},
                "M2": {"done": m2_done, "label": "Reusable runner ships", "phase": "P1"},
                "M3": {"done": m3_done, "label": "Results reproducible 1-command", "phase": "P2"},
                "M4": {"done": m3_done, "label": "Dashboard launch", "phase": "P3"},
                "M5": {"done": False, "label": "Surtur is default", "phase": "P4"},
            },
            "per_arm": per_arm,
        }
    finally:
        conn.close()


@app.get("/api/phase")
def get_phase():
    """Return the derived current milestone + per-arm seed coverage.

    The dashboard's phase arc lights the current milestone from this endpoint.
    """
    return _phase_state()


@app.get("/api/milestones")
def get_milestones():
    return _phase_state()["milestones"]


@app.get("/api/roadmap")
def get_roadmap():
    """Serve the editorial roadmap content as JSON so the dashboard is the
    single source of truth (no duplicated copy in static/index.html)."""
    import json as _json
    md_path = os.path.join(os.path.dirname(__file__), "..", "ROADMAP.md")
    # We do not parse the markdown — we ship a curated summary that the dashboard
    # renders. Keeps the JSON endpoint stable, decoupled from prose edits.
    return {
        "vision": (
            "Make capability-preserving post-training a default, cheap, reproducible habit. "
            "Align models on a single GPU without silently burning the core they already earned."
        ),
        "north_star": {
            "retention": "≤2% drop vs frozen baseline",
            "alignment": "Measurable gain vs unaligned",
            "compute": "≤30% of full fine-tune cost",
            "hardware": "Single consumer / prosumer-class GPU",
        },
        "principles": [
            {"title": "Preservation is the product",
             "body": "If a Surtur run improves alignment but quietly degrades the core, it has failed. The core-eval suite is sacred."},
            {"title": "Single GPU is the constraint",
             "body": "Anything that requires a cluster is out of scope for v1."},
            {"title": "Reproducibility is non-negotiable",
             "body": "A number we can't rerun isn't evidence."},
            {"title": "Constraint strategy, not new architecture",
             "body": "Surtur is a where-gradients-flow decision. We do not invent optimizers or attention variants."},
            {"title": "Say no to good ideas",
             "body": "The cut lines are deliberate. A roadmap is what we chose not to do."},
        ],
        "phases": [
            {"id": "P0", "name": "Validate the core claim",
             "question": "Does layer-constrained training actually preserve the core AND align?",
             "gate": "Core claim holds within thresholds"},
            {"id": "P1", "name": "Core runner (engine)",
             "question": "Can we make this a reusable single-GPU runner?",
             "gate": "Runner completes, retention holds"},
            {"id": "P2", "name": "Reproducibility (Anchor)",
             "question": "Can anyone rerun any result, anywhere?",
             "gate": "Every P0/P1 number reruns from one command"},
            {"id": "P3", "name": "Internal dashboard (UI)",
             "question": "Can researchers use it day-to-day without CLI heroics?",
             "gate": "Launch → watch → see retention, all in UI"},
            {"id": "P4", "name": "Adoption & hardening",
             "question": "Is Surtur the default, not a curiosity?",
             "gate": "Adoption + median retention targets met"},
        ],
        "not_building": [
            "A hosted / productized platform with external customers",
            "An open-source / community release",
            "Multi-GPU / distributed training",
            "Multi-tenant auth, external sharing, billing/quotas",
            "Run scheduling / queueing",
            "Novel ML methods / optimizers / architectures",
        ],
        "open_questions": [
            "Who owns the eval suite?",
            "What are the retention / alignment thresholds?",
            "Which base model(s) does P0 target?",
            "Which alignment objective is the P0 proving ground?",
            "Compute budget reality — single-GPU class, full-FT baseline cost?",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
