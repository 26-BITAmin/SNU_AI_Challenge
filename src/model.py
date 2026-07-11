"""CLIP-based multimodal baseline for four-frame temporal ordering."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import CLIPModel

from .dataset import ImagePreprocessConfig


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_CONTEXT_LENGTH = 77
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class FusionConfig:
    """Architecture choices for the lightweight temporal fusion head."""

    embedding_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    feedforward_dim: int = 2048
    dropout: float = 0.1
    num_classes: int = 24
    freeze_clip: bool = True

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.num_heads <= 0 or self.embedding_dim % self.num_heads != 0:
            raise ValueError("num_heads must evenly divide embedding_dim.")
        if self.num_layers <= 0 or self.feedforward_dim <= 0:
            raise ValueError("num_layers and feedforward_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.num_classes != 24:
            raise ValueError("This challenge requires exactly 24 permutation classes.")


def clip_image_preprocess_config() -> ImagePreprocessConfig:
    """Return preprocessing aligned with OpenAI CLIP ViT-B/32."""

    return ImagePreprocessConfig(
        image_size=224,
        resize_size=224,
        normalize=True,
        mean=CLIP_IMAGE_MEAN,
        std=CLIP_IMAGE_STD,
    )


def _pooled_feature(output: Any, *, source: str) -> Tensor:
    """Support both tensor and current Transformers pooled-output APIs."""

    if isinstance(output, Tensor):
        return output
    pooled_output = getattr(output, "pooler_output", None)
    if isinstance(pooled_output, Tensor):
        return pooled_output
    raise TypeError(f"{source} did not return a pooled feature tensor.")


class TemporalOrderModel(nn.Module):
    """Fuse one caption token and four frame tokens into 24 permutation logits."""

    sequence_length = 6  # [CLS], caption, Input_1, Input_2, Input_3, Input_4

    def __init__(self, clip_model: nn.Module, config: FusionConfig) -> None:
        super().__init__()
        config.validate()
        projection_dim = int(clip_model.config.projection_dim)  # type: ignore[attr-defined]
        if projection_dim != config.embedding_dim:
            raise ValueError(
                f"CLIP projection_dim={projection_dim} does not match "
                f"embedding_dim={config.embedding_dim}."
            )

        self.clip_model = clip_model
        self.config = config
        if config.freeze_clip:
            self.clip_model.requires_grad_(False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length, config.embedding_dim)
        )
        # 0=CLS, 1=caption, 2=image frame.
        self.modality_embedding = nn.Embedding(3, config.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        config: FusionConfig | None = None,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> "TemporalOrderModel":
        """Load a published CLIP encoder and attach a new temporal head."""

        clip_model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        projection_dim = int(clip_model.config.projection_dim)
        config = config or FusionConfig(embedding_dim=projection_dim)
        return cls(clip_model, config)

    def train(self, mode: bool = True) -> "TemporalOrderModel":
        """Keep a frozen CLIP deterministic while training the fusion head."""

        super().train(mode)
        if self.config.freeze_clip:
            self.clip_model.eval()
        return self

    def _encode_clip(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        context = torch.no_grad() if self.config.freeze_clip else nullcontext()
        batch_size = images.shape[0]
        with context:
            image_output = self.clip_model.get_image_features(  # type: ignore[attr-defined]
                pixel_values=images.reshape(-1, *images.shape[2:])
            )
            text_output = self.clip_model.get_text_features(  # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            image_features = _pooled_feature(image_output, source="CLIP image encoder")
            text_features = _pooled_feature(text_output, source="CLIP text encoder")

        image_features = F.normalize(image_features.float(), dim=-1).reshape(
            batch_size, 4, self.config.embedding_dim
        )
        text_features = F.normalize(text_features.float(), dim=-1)
        return image_features, text_features

    def forward(
        self,
        *,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one logit for each of the 24 possible answer permutations."""

        if images.ndim != 5 or images.shape[1] != 4 or images.shape[2] != 3:
            raise ValueError(
                "images must have shape [batch, 4, 3, height, width], "
                f"got {tuple(images.shape)}."
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != images.shape[0]:
            raise ValueError("input_ids must have shape [batch, text_length].")

        image_features, text_features = self._encode_clip(
            images, input_ids, attention_mask
        )
        batch_size = images.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat(
            (cls_tokens, text_features.unsqueeze(1), image_features), dim=1
        )
        modality_ids = torch.tensor(
            (0, 1, 2, 2, 2, 2), dtype=torch.long, device=tokens.device
        )
        tokens = (
            tokens
            + self.position_embedding
            + self.modality_embedding(modality_ids).unsqueeze(0)
        )
        fused_tokens = self.fusion_encoder(tokens)
        return self.classifier(self.output_norm(fused_tokens[:, 0]))

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def fusion_state_dict(self) -> dict[str, Tensor]:
        """Save only the learned head when the public CLIP backbone is frozen."""

        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("clip_model.")
        }

    def architecture_dict(self) -> dict[str, object]:
        return asdict(self.config)
