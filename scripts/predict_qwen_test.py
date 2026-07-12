"""Run one trained Qwen+LoRA model on Test and create submission.csv.

The script performs inference only.  It does not compute Test answer
distributions or use Test characteristics to alter training or model design.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, median

import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm

from scripts.evaluate_qwen_baseline import (
    load_model_and_processor,
    resolve_dtype,
    synchronize,
)
from src.data import load_samples, validate_submission_template
from src.qwen_baseline import build_qwen_messages, parse_qwen_output
from src.submission import (
    build_submission_rows,
    validate_submission_file,
    write_submission,
)
from src.training import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer Test with one Qwen LoRA adapter and write submission.csv."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/huggingface/hub")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_template_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["Id"] for row in csv.DictReader(handle)]


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive.")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    adapter_path = args.adapter_path.resolve()
    cache_dir = args.cache_dir.resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = validate_submission_template(data_dir)
    samples = load_samples(data_dir / "test.csv", labeled=False)
    template_ids = load_template_ids(data_dir / "sample_submission.csv")
    if len(samples) != sample_count or [s.sample_id for s in samples] != template_ids:
        raise ValueError("Test samples and submission template are not aligned.")

    device = select_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print("Scope: TEST INFERENCE ONLY; no Test distribution analysis is performed.")
    print(f"Samples: {len(samples):,}")
    print(f"Model: {args.model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Device/dtype: {device}/{dtype}")

    model, processor = load_model_and_processor(
        model_name=args.model_name,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
        adapter_path=adapter_path,
    )
    predictions: dict[str, tuple[int, int, int, int]] = {}
    diagnostic_rows: list[dict[str, object]] = []
    latencies: list[float] = []
    parse_failures = 0

    for sample in tqdm(samples, desc="Qwen Test inference"):
        messages = build_qwen_messages(
            sample,
            data_dir / "test",
            prompt_version="v2_direct_answer",
        )
        prompt_text = processor.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(  # type: ignore[operator]
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        synchronize(device)
        latency = time.perf_counter() - started
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs.input_ids, generated_ids, strict=True
            )
        ]
        raw_output = processor.batch_decode(  # type: ignore[attr-defined]
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_qwen_output(raw_output, output_mode="direct_answer")
        predictions[sample.sample_id] = parsed.answer
        latencies.append(latency)
        if not parsed.success:
            parse_failures += 1
        diagnostic_rows.append(
            {
                "Id": sample.sample_id,
                "predicted_answer": json.dumps(list(parsed.answer)),
                "parse_success": parsed.success,
                "parse_error": parsed.error or "",
                "latency_seconds": round(latency, 6),
                "raw_output": raw_output,
            }
        )

    diagnostics_path = output_dir / "test_inference_log.csv"
    with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Id",
                "predicted_answer",
                "parse_success",
                "parse_error",
                "latency_seconds",
                "raw_output",
            ),
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    summary = {
        "scope": "test_inference_only_no_distribution_analysis",
        "sample_count": len(samples),
        "model_name": args.model_name,
        "adapter_path": str(adapter_path),
        "device": str(device),
        "dtype": str(dtype),
        "parse_success_count": len(samples) - parse_failures,
        "parse_failure_count": parse_failures,
        "latency_seconds": {
            "total": round(sum(latencies), 6),
            "mean": round(mean(latencies), 6),
            "median": round(median(latencies), 6),
        },
    }
    with (output_dir / "inference_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    if parse_failures:
        raise RuntimeError(
            f"{parse_failures} Test outputs could not be parsed. "
            "submission.csv was not created; inspect test_inference_log.csv."
        )
    submission_rows = build_submission_rows(template_ids, predictions)
    submission_path = output_dir / "submission.csv"
    write_submission(submission_path, submission_rows)
    validated_count = validate_submission_file(submission_path, template_ids)
    print("Test inference completed without parsing failures.")
    print(f"Validated submission rows: {validated_count:,}")
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
