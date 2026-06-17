from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def _find_checkpoint(run_root: Path, job_id: str) -> Path:
    candidates = [
        path
        for path in run_root.rglob(f"*j{job_id}")
        if path.is_dir() and (path / "config.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No HF checkpoint with config.json found under {run_root} for job {job_id}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safetensor_keys(checkpoint_dir: Path) -> list[str]:
    paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")

    keys: list[str] = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    return keys


def _validate_expert_layout(checkpoint_dir: Path, expected_num_experts: int) -> None:
    key_re = re.compile(
        r"^(model\.layers\.\d+)\.feed_forward\.experts\.(\d+)\.w1\.weight$"
    )
    layer_experts: dict[str, set[int]] = {}

    for key in _safetensor_keys(checkpoint_dir):
        match = key_re.match(key)
        if match is None:
            continue
        layer_experts.setdefault(match.group(1), set()).add(int(match.group(2)))

    if not layer_experts:
        raise AssertionError("No MoE expert tensors found in exported checkpoint")

    expected_ids = set(range(expected_num_experts))
    bad_layers = {
        layer: sorted(ids)
        for layer, ids in layer_experts.items()
        if ids != expected_ids
    }
    if bad_layers:
        preview = {
            layer: {"count": len(ids), "first": ids[:4], "last": ids[-4:]}
            for layer, ids in list(bad_layers.items())[:5]
        }
        raise AssertionError(
            "Exported checkpoint does not contain the full expert set per MoE "
            f"layer. Expected {expected_num_experts}; bad layer preview: {preview}"
        )

    print(
        f"Expert layout OK: {len(layer_experts)} MoE layers x "
        f"{expected_num_experts} experts"
    )


def _generate_text(checkpoint_dir: Path, prompt: str, max_new_tokens: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if output_ids.shape[-1] <= inputs["input_ids"].shape[-1]:
        raise AssertionError("Generation produced no new tokens")

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated text:")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an exported HF checkpoint has all MoE experts and loads."
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/lambdafs/alay/checkpoints")
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-num-experts", type=int, default=None)
    parser.add_argument(
        "--prompt", default="Explain expert parallelism in one sentence."
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    checkpoint_dir = _find_checkpoint(args.run_root, args.job_id)
    print(f"Validating checkpoint: {checkpoint_dir}")

    with open(checkpoint_dir / "config.json") as handle:
        config = json.load(handle)
    expected_num_experts = args.expected_num_experts or int(config["num_experts"])

    _validate_expert_layout(checkpoint_dir, expected_num_experts)
    _generate_text(checkpoint_dir, args.prompt, args.max_new_tokens)
    print("HF checkpoint validation passed")


if __name__ == "__main__":
    main()
