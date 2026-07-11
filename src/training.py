"""Training and validation loops aligned with the competition metric."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dataset import decode_class
from .metrics import evaluate_predictions
from .model import CLIP_CONTEXT_LENGTH, TemporalOrderModel


@dataclass(frozen=True)
class EpochResult:
    loss: float
    sample_count: int
    exact_match_accuracy: float
    position_accuracy: float
    pairwise_ordering_accuracy: float
    mean_kendall_tau: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_sentences(
    tokenizer: Any, sentences: Sequence[str], device: torch.device
) -> dict[str, Tensor]:
    encoded = tokenizer(
        list(sentences),
        padding=True,
        truncation=True,
        max_length=CLIP_CONTEXT_LENGTH,
        return_tensors="pt",
    )
    inputs = {"input_ids": encoded["input_ids"].to(device)}
    if "attention_mask" in encoded:
        inputs["attention_mask"] = encoded["attention_mask"].to(device)
    return inputs


def _finish_epoch(
    *,
    total_loss: float,
    sample_count: int,
    target_answers: list[tuple[int, int, int, int]],
    predicted_answers: list[tuple[int, int, int, int]],
) -> EpochResult:
    if sample_count == 0:
        raise ValueError("An epoch must process at least one sample.")
    metrics = evaluate_predictions(target_answers, predicted_answers)
    return EpochResult(
        loss=total_loss / sample_count,
        sample_count=sample_count,
        exact_match_accuracy=float(metrics["exact_match_accuracy"]),
        position_accuracy=float(metrics["position_accuracy"]),
        pairwise_ordering_accuracy=float(metrics["pairwise_ordering_accuracy"]),
        mean_kendall_tau=float(metrics["mean_kendall_tau"]),
    )


def _record_predictions(
    logits: Tensor,
    answers: Tensor,
    target_answers: list[tuple[int, int, int, int]],
    predicted_answers: list[tuple[int, int, int, int]],
) -> None:
    predicted_classes = logits.argmax(dim=-1).detach().cpu().tolist()
    predicted_answers.extend(decode_class(int(value)) for value in predicted_classes)
    target_answers.extend(
        tuple(int(position) for position in row)
        for row in answers.detach().cpu().tolist()
    )


def train_one_epoch(
    model: TemporalOrderModel,
    batches: Iterable[Mapping[str, object]],
    *,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int | None = None,
    gradient_clip_norm: float = 1.0,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    sample_count = 0
    target_answers: list[tuple[int, int, int, int]] = []
    predicted_answers: list[tuple[int, int, int, int]] = []

    for step, batch in enumerate(batches, start=1):
        if max_steps is not None and step > max_steps:
            break
        images = batch["images"]
        targets = batch["targets"]
        answers = batch["answers"]
        sentences = batch["sentences"]
        assert isinstance(images, Tensor)
        assert isinstance(targets, Tensor)
        assert isinstance(answers, Tensor)
        assert isinstance(sentences, list)
        images = images.to(device)
        targets = targets.to(device)
        text_inputs = tokenize_sentences(tokenizer, sentences, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images=images, **text_inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        current_batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * current_batch_size
        sample_count += current_batch_size
        _record_predictions(logits, answers, target_answers, predicted_answers)

    return _finish_epoch(
        total_loss=total_loss,
        sample_count=sample_count,
        target_answers=target_answers,
        predicted_answers=predicted_answers,
    )


@torch.inference_mode()
def evaluate_model(
    model: TemporalOrderModel,
    batches: Iterable[Mapping[str, object]],
    *,
    tokenizer: Any,
    device: torch.device,
    max_steps: int | None = None,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    target_answers: list[tuple[int, int, int, int]] = []
    predicted_answers: list[tuple[int, int, int, int]] = []

    for step, batch in enumerate(batches, start=1):
        if max_steps is not None and step > max_steps:
            break
        images = batch["images"]
        targets = batch["targets"]
        answers = batch["answers"]
        sentences = batch["sentences"]
        assert isinstance(images, Tensor)
        assert isinstance(targets, Tensor)
        assert isinstance(answers, Tensor)
        assert isinstance(sentences, list)
        images = images.to(device)
        targets = targets.to(device)
        text_inputs = tokenize_sentences(tokenizer, sentences, device)

        logits = model(images=images, **text_inputs)
        loss = F.cross_entropy(logits, targets)
        current_batch_size = images.shape[0]
        total_loss += float(loss.item()) * current_batch_size
        sample_count += current_batch_size
        _record_predictions(logits, answers, target_answers, predicted_answers)

    return _finish_epoch(
        total_loss=total_loss,
        sample_count=sample_count,
        target_answers=target_answers,
        predicted_answers=predicted_answers,
    )
