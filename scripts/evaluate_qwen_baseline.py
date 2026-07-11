"""Evaluate the official zero-shot Qwen baseline on fixed Train Validation IDs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.qwen_baseline import (
    QWEN_MODEL_NAME,
    build_qwen_messages,
    choose_evaluation_samples,
    load_validation_samples,
    parse_qwen_output,
    summarize_qwen_evaluation,
    write_qwen_predictions,
)
from src.training import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the official Qwen zero-shot prompt on Train Validation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--validation-split-csv",
        type=Path,
        default=Path("outputs/splits/validation_ids.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qwen_zero_shot")
    )
    parser.add_argument("--model-name", default=QWEN_MODEL_NAME)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/huggingface/hub")
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of random Validation samples; use 0 for all 1,907 samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model_and_processor(
    *, model_name: str, cache_dir: Path, device: torch.device, local_files_only: bool
) -> tuple[Qwen2VLForConditionalGeneration, object]:
    if device.type == "cuda":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return model, processor


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be 0 or a positive integer.")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    device = select_device(args.device)
    data_dir = args.data_dir.resolve()
    validation_split_csv = args.validation_split_csv.resolve()
    cache_dir = args.cache_dir.resolve()
    run_name = "validation_full" if args.limit == 0 else f"validation_{args.limit}"
    output_dir = args.output_dir.resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scope: TRAIN VALIDATION ONLY (Test data is not opened or analyzed).")
    print(f"device: {device}")
    print(f"model: {args.model_name}")
    all_validation_samples = load_validation_samples(
        data_dir=data_dir,
        validation_split_csv=validation_split_csv,
    )
    samples = choose_evaluation_samples(
        all_validation_samples,
        limit=None if args.limit == 0 else args.limit,
        seed=args.seed,
    )
    print(f"Validation samples selected: {len(samples):,}/{len(all_validation_samples):,}")

    model, processor = load_model_and_processor(
        model_name=args.model_name,
        cache_dir=cache_dir,
        device=device,
        local_files_only=args.local_files_only,
    )
    targets: list[tuple[int, int, int, int]] = []
    predictions: list[tuple[int, int, int, int]] = []
    parse_successes: list[bool] = []
    parse_errors: list[str | None] = []
    latencies_seconds: list[float] = []
    output_rows: list[dict[str, object]] = []

    progress = tqdm(samples, desc="Qwen Validation inference")
    for sample in progress:
        if sample.answer is None:
            raise ValueError(f"Validation sample {sample.sample_id!r} has no Answer.")
        synchronize(device)
        started_at = time.perf_counter()
        messages = build_qwen_messages(sample, data_dir / "train")
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
        )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        synchronize(device)
        latency_seconds = time.perf_counter() - started_at
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs.input_ids, generated_ids, strict=True
            )
        ]
        raw_output = processor.batch_decode(  # type: ignore[attr-defined]
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_qwen_output(raw_output)

        targets.append(sample.answer)
        predictions.append(parsed.answer)
        parse_successes.append(parsed.success)
        parse_errors.append(parsed.error)
        latencies_seconds.append(latency_seconds)
        output_rows.append(
            {
                "Id": sample.sample_id,
                "target_answer": json.dumps(list(sample.answer)),
                "predicted_answer": json.dumps(list(parsed.answer)),
                "chronological_order": (
                    json.dumps(list(parsed.chronological_order))
                    if parsed.chronological_order is not None
                    else ""
                ),
                "parse_success": parsed.success,
                "parse_error": parsed.error or "",
                "latency_seconds": round(latency_seconds, 6),
                "raw_output": raw_output,
            }
        )
        progress.set_postfix(
            parse_success=f"{sum(parse_successes)}/{len(parse_successes)}"
        )

    summary = summarize_qwen_evaluation(
        targets=targets,
        predictions=predictions,
        parse_successes=parse_successes,
        parse_errors=parse_errors,
        latencies_seconds=latencies_seconds,
    )
    summary["scope"] = "train_validation_only"
    summary["selection"] = {
        "available_validation_count": len(all_validation_samples),
        "evaluated_count": len(samples),
        "limit": args.limit,
        "seed": args.seed,
    }
    summary["model"] = {
        "name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "device": str(device),
    }
    write_qwen_predictions(output_dir / "predictions.csv", output_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    metrics = summary["metrics"]
    parsing = summary["parsing"]
    latency = summary["latency_seconds"]
    assert isinstance(metrics, dict)
    assert isinstance(parsing, dict)
    assert isinstance(latency, dict)
    print("\nQwen zero-shot Validation evaluation completed.")
    print(f"Exact Match: {metrics['exact_match_accuracy']:.6f}")
    print(f"Position Accuracy: {metrics['position_accuracy']:.6f}")
    print(f"Parse success: {parsing['success_count']}/{summary['sample_count']}")
    print(f"Mean latency: {latency['mean']:.3f} sec/sample")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
