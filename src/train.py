# Unsloth MUST be imported before any HF/transformers import (research constraint:
# it patches torch internals). On the GPU training machine it is required; here it
# is optional so the code still imports on machines without it.
try:
    import unsloth  # noqa: F401
except ImportError:
    print(
        "[Surtur] WARNING: unsloth not installed. Required for real GPU runs; "
        "install it on the training machine before training."
    )

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

import freeze
import guard
from callbacks import GradientInsulationCallback
from seed import set_seed, enable_deterministic


@dataclass
class TrainingConfig:
    model_id: str
    method: str = "sft"
    layer_spec: str = "last_4"
    dataset_path: Optional[str] = None
    output_dir: str = "./surtur_out"
    seed: int = 42
    max_steps: int = 100
    num_epochs: float = -1.0
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 5e-5
    dtype: str = "bf16"
    resume: bool = False
    reward_funcs: Optional[List[Callable]] = field(default=None)


def resolve_band(spec: str, total_layers: int) -> List[int]:
    return freeze.layer_numbers(spec, total_layers)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(dataset_path: str):
    if not os.path.isabs(dataset_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(project_root, dataset_path)
    return load_dataset("json", data_files=dataset_path, split="train")


def run(config: TrainingConfig) -> None:
    set_seed(int(config.seed))
    if os.environ.get("SURTUR_DETERMINISTIC") == "1":
        enable_deterministic()
        
    if config.dataset_path:
        import data_utils
        data_utils.verify(config.dataset_path)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config.dtype == "bf16" and device == "cuda"
    use_fp16 = config.dtype == "fp16" and device == "cuda"
    
    if config.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif config.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"[Surtur] Loading {config.model_id} on {device}...")
    device_map = {"": device} if device == "cuda" else None
    model = freeze.load_model(config.model_id, dtype=torch_dtype, device_map=device_map)

    total_layers = len(freeze.get_layers(model))
    band = resolve_band(config.layer_spec, total_layers)
    print(f"[Surtur] Trainable band: {band} (of {total_layers} layers)")

    if config.layer_spec != "all":
        freeze.apply_freeze(model, band)
        guard.assert_only_expected_train(model, band)
    trainable, total = freeze.count_params(model)
    print(
        f"[Surtur] Trainable {trainable:,} / Total {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    tokenizer = load_tokenizer(config.model_id)
    dataset = build_dataset(config.dataset_path)

    num_epochs = config.num_epochs if config.num_epochs > 0 else 1.0
    common = dict(
        output_dir=config.output_dir,
        per_device_train_batch_size=int(config.batch_size),
        gradient_accumulation_steps=int(config.grad_accum),
        max_steps=int(config.max_steps),
        num_train_epochs=num_epochs,
        learning_rate=float(config.lr),
        seed=int(config.seed),
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    if config.method == "sft":
        args = SFTConfig(**common)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[GradientInsulationCallback(model=model)],
        )
    elif config.method == "grpo":
        if not config.reward_funcs:
            raise ValueError("GRPO requires reward_funcs in TrainingConfig")
        args = GRPOConfig(**common, num_generations=4)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=config.reward_funcs,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[GradientInsulationCallback(model=model)],
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")

    print("[Surtur] Starting training...")
    trainer.train(resume_from_checkpoint=config.resume)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[Surtur] Done. Checkpoint saved to {config.output_dir}")
