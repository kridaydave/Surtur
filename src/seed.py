"""Surtur seed harness — pins every source of randomness.

Call set_seed(seed) at the START of every run, before model load.
For bit-reproducible GPU runs, also call enable_deterministic().
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_deterministic(warn_only: bool = True) -> None:
    """Force bit-reproducible CUDA ops. May slow training ~10-30%.
    Some ops (e.g. flash attention, scatter_add) have no deterministic
    kernel; with warn_only=True those will warn instead of crashing."""
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn — makes per-worker RNG deterministic."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
