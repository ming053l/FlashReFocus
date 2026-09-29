"""Trainable RAM++ wrapper for image-conditioned tagging and prompting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import random

DEFAULT_CHECKPOINT = Path(
    "/ssd1/chingheng/Defocus/SOTA/recognize-anything/pretrained/ram_plus_swin_large_14m.pth"
)


try:
    from ram.models import ram_plus
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Failed to import AutoRefocus's vendored `ram` package. Make sure "
        "`AutoRefocus/ram` exists."
    ) from exc


@dataclass
class RamPlusConditionOutput:
    soft_text_embeddings: torch.Tensor
    image_embeds: torch.Tensor
    ip_adapter_image_embeds: torch.Tensor
    english_tags: List[List[str]]
    chinese_tags: List[List[str]]
    prompt_texts: List[str]
    probabilities: torch.Tensor
    logits: torch.Tensor


class RamPlusTagger(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        image_size: int = 384,
        vit: str = "swin_l",
        output_dim: int = 1024,
        fallback_prompt: str = "a sharp clear image",
        prompt_threshold: float | None = None,
        max_tags: int = 20,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parent / checkpoint_path
        self.checkpoint_path = checkpoint_path
        self.image_size = int(image_size)
        self.vit = vit
        self.output_dim = int(output_dim)
        self.fallback_prompt = fallback_prompt
        self.prompt_threshold = None if prompt_threshold is None else float(prompt_threshold)
        self.max_tags = int(max_tags)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"RAM++ checkpoint not found: {self.checkpoint_path}"
            )

        self.model = ram_plus(
            pretrained=str(self.checkpoint_path),
            image_size=self.image_size,
            vit=self.vit,
        )
        self.image_proj_to_sd = (
            nn.Identity() if self.output_dim == 512 else nn.Linear(512, self.output_dim)
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _load_pil_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _prepare_tensor_batch(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.dim() != 4 or image_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected image tensor shape [B, 3, H, W], got {tuple(image_tensor.shape)}"
            )
        image_tensor = image_tensor.float()
        img_min = float(image_tensor.min().item())
        img_max = float(image_tensor.max().item())
        if img_min >= -1.0 and img_max <= 1.0:
            image_tensor = (image_tensor + 1.0) / 2.0
        elif img_min >= 0.0 and img_max <= 1.0:
            pass
        else:
            image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.clamp(0.0, 1.0)
        image_tensor = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        image_tensor = image_tensor.to(
            device=self.pixel_mean.device,
            dtype=self.pixel_mean.dtype,
        )
        return (image_tensor - self.pixel_mean) / self.pixel_std

    def preprocess_batch(
        self,
        images: Sequence[str | Path | Image.Image],
    ) -> torch.Tensor:
        tensors = []
        for image in images:
            pil_image = self._load_pil_image(image)
            np_image = np.asarray(pil_image, dtype=np.float32).transpose(2, 0, 1)
            tensors.append(torch.from_numpy(np_image))
        batch = torch.stack(tensors, dim=0)
        return self._prepare_tensor_batch(batch)

    def _compute_tagging_outputs(self, image_embeds: torch.Tensor):
        device = image_embeds.device
        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=device,
        )
        image_cls_embeds = image_embeds[:, 0, :]
        batch_size = image_embeds.shape[0]
        descriptions_per_class = self.model.label_embed.shape[0] // self.model.num_class

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(
            dim=-1, keepdim=True
        )
        reweight_scale = self.model.reweight_scale.exp()
        logits_per_image = (
            reweight_scale * image_cls_embeds @ self.model.label_embed.t()
        )
        logits_per_image = logits_per_image.view(batch_size, -1, descriptions_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        reshaped_label_embed = self.model.label_embed.view(
            -1, descriptions_per_class, 512
        )
        label_embed_reweight = torch.einsum(
            "bcd,cdh->bch",
            weight_normalized,
            reshaped_label_embed,
        )
        label_embed = torch.relu(self.model.wordvec_proj(label_embed_reweight))

        tagging_embed = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )
        logits = self.model.fc(tagging_embed[0]).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        thresholds = self.model.class_threshold.to(device).unsqueeze(0)
        targets = probabilities > thresholds
        return logits, probabilities, targets

    def _targets_to_prompts(
        self,
        targets: torch.Tensor,
        probabilities: torch.Tensor | None = None,
    ):
        english_tags_batch: List[List[str]] = []
        chinese_tags_batch: List[List[str]] = []
        prompt_texts: List[str] = []
        text_prompt = "A sharp clear image, containing "

        tag_list_en = np.asarray(self.model.tag_list)
        tag_list_zh = np.asarray(self.model.tag_list_chinese)
        delete_indices = list(self.model.delete_tag_index)

        for idx in range(targets.shape[0]):
            if self.prompt_threshold is None:
                active_mask = targets[idx].detach().cpu().clone()
            else:
                if probabilities is None:
                    raise ValueError("probabilities must be provided when prompt_threshold is set.")
                active_mask = (probabilities[idx].detach().cpu() > self.prompt_threshold)
            if delete_indices:
                active_mask[delete_indices] = False
            tag_indices = torch.where(active_mask)[0].tolist()[: self.max_tags]
            english_tags = tag_list_en[tag_indices].tolist() if tag_indices else []
            chinese_tags = tag_list_zh[tag_indices].tolist() if tag_indices else []
            english_tags_batch.append(english_tags)
            chinese_tags_batch.append(chinese_tags)
            prompt_texts.append(
                text_prompt + ", ".join(english_tags)
                if english_tags
                else self.fallback_prompt
            )

        return english_tags_batch, chinese_tags_batch, prompt_texts

    def forward_batch(
        self,
        images: Sequence[str | Path | Image.Image] | torch.Tensor,
    ) -> RamPlusConditionOutput:
        if isinstance(images, torch.Tensor):
            image_tensor = self._prepare_tensor_batch(images)
        else:
            image_tensor = self.preprocess_batch(images)
        image_tensor = image_tensor.to(
            device=self.pixel_mean.device,
            dtype=self.pixel_mean.dtype,
        )

        image_embeds = self.model.image_proj(self.model.visual_encoder(image_tensor))
        logits, probabilities, targets = self._compute_tagging_outputs(image_embeds)
        english_tags, chinese_tags, prompt_texts = self._targets_to_prompts(
            targets,
            probabilities=probabilities,
        )
        ip_adapter_image_embeds = self.image_proj_to_sd(image_embeds[:, :1, :])

        return RamPlusConditionOutput(
            soft_text_embeddings=image_embeds,
            image_embeds=image_embeds,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            english_tags=english_tags,
            chinese_tags=chinese_tags,
            prompt_texts=prompt_texts,
            probabilities=probabilities,
            logits=logits,
        )

    def predict_batch(
        self,
        images: Sequence[str | Path | Image.Image] | torch.Tensor,
    ) -> RamPlusConditionOutput:
        self.eval()
        with torch.no_grad():
            return self.forward_batch(images)
