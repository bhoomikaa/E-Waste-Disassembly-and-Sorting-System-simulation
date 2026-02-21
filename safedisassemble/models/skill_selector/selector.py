"""Level 2: Skill Selector

VLA model that takes (image, subtask_instruction) and outputs:
- Skill ID (which primitive to execute)
- Skill parameters (target position, force limits, tool selection)

Built on top of OpenVLA / a frozen VLM with a learned action head.
Fine-tuned on domain-specific data from the simulation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SkillParameters:
    """Parameters for a specific skill execution."""

    def __init__(
        self,
        skill_id: str,
        target_position: np.ndarray,         # (3,) world-frame target
        approach_direction: np.ndarray,       # (3,) unit vector
        max_force: float = 10.0,             # Newtons
        gripper_width: float = 0.04,          # meters
        rotation_amount: float = 0.0,         # radians (for unscrewing)
        confidence: float = 1.0,
    ):
        self.skill_id = skill_id
        self.target_position = target_position
        self.approach_direction = approach_direction
        self.max_force = max_force
        self.gripper_width = gripper_width
        self.rotation_amount = rotation_amount
        self.confidence = confidence

    def to_tensor(self) -> torch.Tensor:
        """Flatten to a fixed-size tensor for training."""
        return torch.tensor([
            *self.target_position,
            *self.approach_direction,
            self.max_force,
            self.gripper_width,
            self.rotation_amount,
            self.confidence,
        ], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, skill_id: str) -> "SkillParameters":
        t = tensor.detach().cpu().numpy()
        return cls(
            skill_id=skill_id,
            target_position=t[0:3],
            approach_direction=t[3:6] / (np.linalg.norm(t[3:6]) + 1e-8),
            max_force=float(t[6]),
            gripper_width=float(t[7]),
            rotation_amount=float(t[8]),
            confidence=float(t[9]),
        )

    @classmethod
    def param_dim(cls) -> int:
        return 10


# --- Skill vocabulary ---
SKILL_VOCAB = [
    "unscrew",
    "pry_open",
    "pull_connector",
    "lift_component",
    "release_clip",
    "flip",
    "approach",
    "place",
]

SKILL_TO_IDX = {s: i for i, s in enumerate(SKILL_VOCAB)}
IDX_TO_SKILL = {i: s for s, i in SKILL_TO_IDX.items()}


class VisionEncoder(nn.Module):
    """Lightweight vision encoder (can be swapped with frozen ViT from VLM)."""

    def __init__(self, image_size: int = 224, embed_dim: int = 512):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim

        # Simple CNN backbone (replace with frozen DINOv2 / SigLIP for better results)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Linear(256, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float tensor, normalized to [0, 1]
        Returns:
            (B, embed_dim) image embeddings
        """
        features = self.backbone(images).squeeze(-1).squeeze(-1)
        return self.projector(features)


class LanguageEncoder(nn.Module):
    """Simple language encoder for subtask instructions.

    For production, replace with frozen text encoder from CLIP/SigLIP.
    Here we use a trainable embedding + Transformer for self-contained training.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_len: int = 64,
        embed_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=embed_dim * 4, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) long tensor of token indices

        Returns:
            (B, embed_dim) language embeddings (CLS token / mean pool)
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)

        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.transformer(x)

        # Mean pooling over sequence
        return x.mean(dim=1)


class SkillSelector(nn.Module):
    """Level 2: Vision-Language-Action model for skill selection.

    Architecture:
        image → VisionEncoder → visual_embed
        instruction → LanguageEncoder → lang_embed
        [visual_embed; lang_embed; proprioception] → MLP → (skill_logits, skill_params)
    """

    def __init__(
        self,
        image_size: int = 224,
        embed_dim: int = 512,
        proprioception_dim: int = 19,  # 7 joint pos + 7 joint vel + 2 gripper + 3 ee pos
        n_skills: int = len(SKILL_VOCAB),
        param_dim: int = SkillParameters.param_dim(),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_skills = n_skills
        self.param_dim = param_dim

        # Encoders
        self.vision_encoder = VisionEncoder(image_size, embed_dim)
        self.language_encoder = LanguageEncoder(embed_dim=embed_dim)

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprioception_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

        # Fusion + prediction heads
        fusion_dim = embed_dim * 3  # visual + language + proprioception
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Skill classification head
        self.skill_head = nn.Linear(256, n_skills)

        # Skill parameter regression head (per-skill parameters)
        self.param_head = nn.Linear(256, param_dim)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) normalized images
            token_ids: (B, L) token indices for instruction
            proprioception: (B, proprio_dim) robot state

        Returns:
            dict with 'skill_logits', 'skill_params', 'confidence'
        """
        visual_embed = self.vision_encoder(images)
        lang_embed = self.language_encoder(token_ids)
        proprio_embed = self.proprio_encoder(proprioception)

        fused = torch.cat([visual_embed, lang_embed, proprio_embed], dim=-1)
        features = self.fusion(fused)

        skill_logits = self.skill_head(features)
        skill_params = self.param_head(features)
        confidence = self.confidence_head(features)

        return {
            "skill_logits": skill_logits,
            "skill_params": skill_params,
            "confidence": confidence.squeeze(-1),
        }

    def predict(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> SkillParameters:
        """Inference: predict skill and parameters for a single observation."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                images.unsqueeze(0) if images.dim() == 3 else images,
                token_ids.unsqueeze(0) if token_ids.dim() == 1 else token_ids,
                proprioception.unsqueeze(0) if proprioception.dim() == 1 else proprioception,
            )

        skill_idx = outputs["skill_logits"][0].argmax().item()
        skill_id = IDX_TO_SKILL[skill_idx]
        params = SkillParameters.from_tensor(outputs["skill_params"][0], skill_id)
        params.confidence = outputs["confidence"][0].item()
        return params

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target_skills: torch.Tensor,
        target_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training losses.

        Args:
            outputs: forward() output dict
            target_skills: (B,) long tensor of skill indices
            target_params: (B, param_dim) float tensor of target parameters
        """
        skill_loss = F.cross_entropy(outputs["skill_logits"], target_skills)
        param_loss = F.mse_loss(outputs["skill_params"], target_params)

        total_loss = skill_loss + 0.5 * param_loss

        return {
            "total_loss": total_loss,
            "skill_loss": skill_loss,
            "param_loss": param_loss,
        }
