# Surtur experiment orchestrator — runs arms A–D across seeds
import os
import time
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

import freeze
from train import TrainingConfig, run as train_run, resolve_band


ARM_NAMES = {"surtur", "full_ft", "frozen", "untrained_ref"}


@dataclass
class ArmConfig:
    model_id: str
    method: str = "sft"
    layer_spec: str = "last_4"
    dataset_path: Optional[str] = None
    output_dir_base: str = "./surtur_out"
    seeds: List[int] = field(default_factory=lambda: [42])
    arms: List[str] = field(default_factory=lambda: ["surtur"])
    max_steps: int = 100
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 5e-5
    dtype: str = "bf16"


def run_arm(arm_name: str, config: ArmConfig, seed: int) -> dict:
    if arm_name not in ARM_NAMES:
        raise ValueError(f"Unknown arm: {arm_name}. Must be one of {ARM_NAMES}")

    ckpt_dir = os.path.join(
        config.output_dir_base, f"arm_{arm_name}", f"seed_{seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    start = time.time()

    if arm_name in ("surtur", "full_ft"):
        spec = config.layer_spec if arm_name == "surtur" else "all"
        tc = TrainingConfig(
            model_id=config.model_id,
            method=config.method,
            layer_spec=spec,
            dataset_path=config.dataset_path,
            output_dir=ckpt_dir,
            seed=seed,
            max_steps=config.max_steps,
            batch_size=config.batch_size,
            grad_accum=config.grad_accum,
            lr=config.lr,
            dtype=config.dtype,
        )
        model = freeze.load_model(config.model_id)
        total_layers = len(freeze.get_layers(model))
        band = resolve_band(spec, total_layers)
        if spec != "all":
            freeze.apply_freeze(model, band)
        trainable, total = freeze.count_params(model)
        del model
        train_run(tc)

    elif arm_name in ("frozen", "untrained_ref"):
        model = freeze.load_model(config.model_id)
        trainable, total = freeze.count_params(model)
        model.save_pretrained(ckpt_dir)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(ckpt_dir)
        del model, tokenizer

    elapsed = time.time() - start

    return {
        "arm": arm_name,
        "seed": seed,
        "duration": elapsed,
        "trainable_params": trainable,
        "total_params": total,
        "checkpoint": ckpt_dir,
    }


def run_experiment(arm_config: ArmConfig) -> List[dict]:
    results = []
    total_arms = len(arm_config.arms) * len(arm_config.seeds)
    run_idx = 0

    for arm_name in arm_config.arms:
        for seed in arm_config.seeds:
            run_idx += 1
            print(
                f"\n{'='*60}\n"
                f"[Orchestrator] Run {run_idx}/{total_arms}: "
                f"arm={arm_name} seed={seed}\n"
                f"{'='*60}"
            )
            result = run_arm(arm_name, arm_config, seed)
            results.append(result)
            print(
                f"[Orchestrator] {arm_name} seed={seed} done in "
                f"{result['duration']:.1f}s — "
                f"trainable={result['trainable_params']:,}"
            )

    _print_summary(results)
    return results


def _print_summary(results: List[dict]) -> None:
    hdr = f"{'Arm':<16} {'Seed':<6} {'Duration':>10} {'Trainable':>14} {'Checkpoint'}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        print(
            f"{r['arm']:<16} {r['seed']:<6} {r['duration']:>9.1f}s "
            f"{r['trainable_params']:>14,} {r['checkpoint']}"
        )
    print(sep)


if __name__ == "__main__":
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    ac = ArmConfig(**{k: v for k, v in raw.items() if k in ArmConfig.__dataclass_fields__})
    run_experiment(ac)
