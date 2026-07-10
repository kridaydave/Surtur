import sqlite3
import os
import json

DB_PATH = "surtur.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        conn.close()
        raise

def init_db():
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            arm TEXT,
            seed INTEGER,
            model_id TEXT,
            method TEXT,
            layer_spec TEXT,
            config_hash TEXT,
            ckpt_path TEXT,
            duration_sec REAL,
            trainable_params INTEGER,
            total_params INTEGER,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evals (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            eval_set TEXT,
            seed INTEGER,
            accuracy REAL,
            stderr REAL,
            n_items INTEGER,
            per_item_path TEXT,
            refusal_rate REAL,
            pass_through_rate REAL,
            duration_sec REAL,
            slices TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs (run_id)
        )
        """)
        conn.commit()
    finally:
        conn.close()
    
    # Import from JSONL files if they exist to migrate data
    import_from_jsonl()

def import_from_jsonl():
    paths_to_check_runs = ["surtur_out/runs.jsonl", "experiments/p0/runs.jsonl", "runs.jsonl"]
    paths_to_check_evals = ["surtur_out/evals.jsonl", "experiments/p0/evals.jsonl", "evals.jsonl"]
    
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        for path in paths_to_check_runs:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            cursor.execute("""
                            INSERT OR IGNORE INTO runs (run_id, arm, seed, model_id, method, layer_spec, config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                data.get("run_id"),
                                data.get("arm"),
                                data.get("seed"),
                                data.get("model_id"),
                                data.get("method", "sft"),
                                data.get("layer_spec"),
                                data.get("config_hash"),
                                data.get("ckpt_path"),
                                data.get("duration_sec"),
                                data.get("trainable_params"),
                                data.get("total_params"),
                                data.get("status", "completed")
                            ))
                        except Exception:
                            pass
                            
        for path in paths_to_check_evals:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            run_id = data.get("run_id")
                            cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
                            if cursor.fetchone() is None:
                                cursor.execute("""
                                INSERT OR IGNORE INTO runs (run_id, arm, seed, model_id, method, layer_spec, config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    run_id,
                                    "unknown",
                                    data.get("seed", 42),
                                    "unknown",
                                    "unknown",
                                    "unknown",
                                    "unknown",
                                    "unknown",
                                    0.0,
                                    0,
                                    0,
                                    "placeholder"
                                ))
                            cursor.execute("""
                            INSERT OR IGNORE INTO evals (run_id, eval_set, seed, accuracy, stderr, n_items, per_item_path, refusal_rate, pass_through_rate, duration_sec, slices)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                run_id,
                                data.get("eval_set"),
                                data.get("seed"),
                                data.get("accuracy"),
                                data.get("stderr"),
                                data.get("n_items"),
                                data.get("per_item_path"),
                                data.get("refusal_rate", 0.0),
                                data.get("pass_through_rate", 1.0),
                                data.get("duration_sec"),
                                json.dumps(data.get("slices", {}))
                            ))
                        except Exception:
                            pass
                            
        conn.commit()
    finally:
        conn.close()

def save_run(run_data):
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO runs (run_id, arm, seed, model_id, method, layer_spec, config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data["run_id"],
            run_data["arm"],
            run_data["seed"],
            run_data["model_id"],
            run_data["method"],
            run_data["layer_spec"],
            run_data["config_hash"],
            run_data["ckpt_path"],
            run_data["duration_sec"],
            run_data.get("trainable_params", 0),
            run_data.get("total_params", 0),
            run_data["status"]
        ))
        conn.commit()
    finally:
        conn.close()

def save_eval(eval_data):
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # Ensure run_id exists in runs to satisfy foreign key constraint
        cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (eval_data["run_id"],))
        if cursor.fetchone() is None:
            cursor.execute("""
            INSERT OR IGNORE INTO runs (run_id, arm, seed, model_id, method, layer_spec, config_hash, ckpt_path, duration_sec, trainable_params, total_params, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_data["run_id"],
                "unknown",
                eval_data["seed"],
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                0.0,
                0,
                0,
                "placeholder"
            ))
            
        cursor.execute("""
        INSERT INTO evals (run_id, eval_set, seed, accuracy, stderr, n_items, per_item_path, refusal_rate, pass_through_rate, duration_sec, slices)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_data["run_id"],
            eval_data["eval_set"],
            eval_data["seed"],
            eval_data["accuracy"],
            eval_data["stderr"],
            eval_data["n_items"],
            eval_data["per_item_path"],
            eval_data.get("refusal_rate", 0.0),
            eval_data.get("pass_through_rate", 1.0),
            eval_data["duration_sec"],
            json.dumps(eval_data.get("slices", {}))
        ))
        conn.commit()
    finally:
        conn.close()
