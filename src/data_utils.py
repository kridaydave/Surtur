"""Data loading with content-hash verification.

Every dataset used by Surtur has its SHA-256 recorded in data/MANIFEST.json.
load_verified(path) checks the file's hash against the manifest and refuses
to load if it changed. This catches silent data drift between runs.
"""
import hashlib
import json
import os
from typing import Optional


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(data_dir: str = "data") -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(project_root, data_dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def verify(path: str, data_dir: str = "data") -> str:
    manifest = load_manifest(data_dir)
    filename = os.path.basename(path)
    entry = manifest.get(filename)
    if entry is None:
        raise FileNotFoundError(
            f"{filename} not in {data_dir}/MANIFEST.json. "
            f"Add it with sha256 and a version label before use."
        )
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(project_root, path) if not os.path.isabs(path) else path
    
    actual = sha256_of_file(full_path)
    expected = entry["sha256"]
    if actual != expected:
        raise ValueError(
            f"DATA HASH MISMATCH for {filename}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"The data has changed since the last pinned run. "
            f"Update MANIFEST.json if this is intentional."
        )
    return actual


def load_verified(path: str, data_dir: str = "data") -> str:
    return verify(path, data_dir)
