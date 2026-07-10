from train import TrainingConfig, run

config = TrainingConfig(
    model_id="facebook/opt-125m",
    method="sft",
    layer_spec="last_4",
    dataset_path="src/smoke_data.jsonl",
    output_dir="src/smoke_out",
    max_steps=3,
    batch_size=2,
    grad_accum=1,
    lr=5e-5,
    dtype="fp32",
)

run(config)
print("[Surtur] SMOKE TEST PASSED")
