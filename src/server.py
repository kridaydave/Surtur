import os
import sys
import json
import sqlite3
import subprocess
import re
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# Add src/ to python path
sys.path.insert(0, os.path.dirname(__file__))

import db

app = FastAPI(title="Surtur Web Dashboard Server")

# Initialize database
db.init_db()

def validate_run_id(run_id: str):
    if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")

# Serve static files
os.makedirs("src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

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
            
        conn = db.get_db()
        cursor = conn.cursor()
        if process.returncode != 0:
            cursor.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,))
        else:
            cursor.execute("UPDATE runs SET status = 'completed' WHERE run_id = ?", (run_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,))
        conn.commit()
        conn.close()
        with open(log_path, "a") as log_file:
            log_file.write(f"\n[Server Error] Run failed with exception: {e}\n")

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = "src/static/index.html"
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
def launch_eval(run_id: str, background_tasks: BackgroundTasks, max_examples: int = 1000):
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
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [py_executable, "surtur.py", "eval", "--run-id", run_id, "--max-examples", str(max_examples)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
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
            
    conn.close()
    
    if "surtur" not in results or "frozen" not in results:
        return {"status": "incomplete", "message": "Surtur and Frozen baseline runs are required to compute verdict."}
        
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
