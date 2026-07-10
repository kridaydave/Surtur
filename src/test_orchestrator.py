# Smoke test for orchestrator — runs arms A (surtur) and C (frozen)
# on opt-125m with 1 seed, max_steps=2.
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator import ArmConfig, run_experiment

OUT_BASE = os.path.join(os.path.dirname(__file__), "smoke_out", "orchestrator_test")


def test_orchestrator_smoke():
    config = ArmConfig(
        model_id="facebook/opt-125m",
        method="sft",
        layer_spec="last_4",
        dataset_path=os.path.join(os.path.dirname(__file__), "smoke_data.jsonl"),
        output_dir_base=OUT_BASE,
        seeds=[42],
        arms=["surtur", "frozen"],
        max_steps=2,
        batch_size=2,
        grad_accum=1,
        dtype="fp32",
    )
    results = run_experiment(config)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    arms_seen = {r["arm"] for r in results}
    assert arms_seen == {"surtur", "frozen"}, f"Unexpected arms: {arms_seen}"

    for r in results:
        assert os.path.isdir(r["checkpoint"]), f"Missing checkpoint: {r['checkpoint']}"
        assert r["trainable_params"] > 0
        assert r["duration"] >= 0

    surtur_r = next(r for r in results if r["arm"] == "surtur")
    frozen_r = next(r for r in results if r["arm"] == "frozen")
    assert surtur_r["trainable_params"] < frozen_r["trainable_params"], (
        f"Surtur should have fewer trainable params than frozen "
        f"({surtur_r['trainable_params']} vs {frozen_r['trainable_params']})"
    )

    shutil.rmtree(OUT_BASE, ignore_errors=True)
    print("\nORCHESTRATOR SMOKE TEST PASSED")


if __name__ == "__main__":
    test_orchestrator_smoke()
