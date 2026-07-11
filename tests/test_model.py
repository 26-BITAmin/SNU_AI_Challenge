from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from src.dataset import encode_answer
from src.model import FusionConfig, TemporalOrderModel
from src.training import evaluate_model, train_one_epoch


class FakeClipModel(nn.Module):
    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=embedding_dim)
        self.image_projection = nn.Linear(3, embedding_dim)
        self.text_embedding = nn.Embedding(32, embedding_dim)

    def get_image_features(self, *, pixel_values: Tensor) -> SimpleNamespace:
        pooled = pixel_values.mean(dim=(-1, -2))
        return SimpleNamespace(pooler_output=self.image_projection(pooled))

    def get_text_features(
        self, *, input_ids: Tensor, attention_mask: Tensor | None
    ) -> SimpleNamespace:
        embedded = self.text_embedding(input_ids)
        if attention_mask is None:
            pooled = embedded.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1)
            pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return SimpleNamespace(pooler_output=pooled)


class FakeTokenizer:
    def __call__(self, sentences: list[str], **_: object) -> dict[str, Tensor]:
        input_ids = torch.tensor(
            [[min(len(word), 31) for word in sentence.split()[:4]] for sentence in sentences],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def make_model() -> TemporalOrderModel:
    torch.manual_seed(7)
    return TemporalOrderModel(
        FakeClipModel(),
        FusionConfig(
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            dropout=0.0,
            freeze_clip=True,
        ),
    )


def make_batch() -> dict[str, object]:
    answers = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    return {
        "sample_ids": ["ABC123", "XYZ789"],
        "sentences": ["one short caption", "another short caption"],
        "input_names": [(), ()],
        "images": torch.rand(2, 4, 3, 8, 8),
        "answers": answers,
        "targets": torch.tensor(
            [encode_answer(row.tolist()) for row in answers], dtype=torch.long
        ),
        "no_ordering": torch.tensor([True, False]),
    }


class TemporalOrderModelTests(unittest.TestCase):
    def test_forward_shape_and_frozen_clip(self) -> None:
        model = make_model()
        logits = model(
            images=torch.rand(2, 4, 3, 8, 8),
            input_ids=torch.ones(2, 5, dtype=torch.long),
            attention_mask=torch.ones(2, 5, dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (2, 24))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.clip_model.parameters())
        )
        self.assertGreater(model.trainable_parameter_count(), 0)

    def test_backward_updates_head_not_clip(self) -> None:
        model = make_model()
        logits = model(
            images=torch.rand(2, 4, 3, 8, 8),
            input_ids=torch.ones(2, 5, dtype=torch.long),
        )
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.clip_model.parameters()))
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_rejects_wrong_image_contract(self) -> None:
        model = make_model()
        with self.assertRaises(ValueError):
            model(
                images=torch.rand(2, 3, 8, 8),
                input_ids=torch.ones(2, 5, dtype=torch.long),
            )


class TrainingLoopTests(unittest.TestCase):
    def test_one_train_and_validation_step(self) -> None:
        model = make_model()
        tokenizer = FakeTokenizer()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        batch = make_batch()
        train_result = train_one_epoch(
            model,
            [batch],
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_steps=1,
        )
        validation_result = evaluate_model(
            model,
            [batch],
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            max_steps=1,
        )
        self.assertEqual(train_result.sample_count, 2)
        self.assertEqual(validation_result.sample_count, 2)
        self.assertGreater(train_result.loss, 0.0)
        self.assertGreater(validation_result.loss, 0.0)
        self.assertTrue(0.0 <= validation_result.exact_match_accuracy <= 1.0)


if __name__ == "__main__":
    unittest.main()
