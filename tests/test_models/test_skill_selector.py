"""Tests for the Level 2 Skill Selector model."""

import torch
import pytest

from safedisassemble.models.skill_selector.selector import (
    SKILL_TO_IDX,
    SKILL_VOCAB,
    SkillParameters,
    SkillSelector,
)


class TestSkillSelector:
    def setup_method(self):
        self.model = SkillSelector(image_size=64, embed_dim=64)

    def test_forward_shape(self):
        B = 4
        images = torch.randn(B, 3, 64, 64)
        tokens = torch.randint(0, 1000, (B, 32))
        proprio = torch.randn(B, 19)

        outputs = self.model(images, tokens, proprio)

        assert outputs["skill_logits"].shape == (B, len(SKILL_VOCAB))
        assert outputs["skill_params"].shape == (B, SkillParameters.param_dim())
        assert outputs["confidence"].shape == (B,)

    def test_predict_returns_skill_params(self):
        image = torch.randn(3, 64, 64)
        tokens = torch.randint(0, 1000, (32,))
        proprio = torch.randn(19)

        params = self.model.predict(image, tokens, proprio)

        assert isinstance(params, SkillParameters)
        assert params.skill_id in SKILL_VOCAB
        assert 0 <= params.confidence <= 1

    def test_loss_computation(self):
        B = 4
        images = torch.randn(B, 3, 64, 64)
        tokens = torch.randint(0, 1000, (B, 32))
        proprio = torch.randn(B, 19)

        outputs = self.model(images, tokens, proprio)

        target_skills = torch.randint(0, len(SKILL_VOCAB), (B,))
        target_params = torch.randn(B, SkillParameters.param_dim())

        losses = self.model.compute_loss(outputs, target_skills, target_params)

        assert "total_loss" in losses
        assert "skill_loss" in losses
        assert "param_loss" in losses
        assert losses["total_loss"].requires_grad


class TestSkillParameters:
    def test_roundtrip_tensor(self):
        params = SkillParameters(
            skill_id="unscrew",
            target_position=torch.tensor([0.5, 0.1, 0.4]).numpy(),
            approach_direction=torch.tensor([0.0, 0.0, -1.0]).numpy(),
            max_force=5.0,
            gripper_width=0.02,
            rotation_amount=3.14,
            confidence=0.95,
        )

        tensor = params.to_tensor()
        assert tensor.shape == (SkillParameters.param_dim(),)

        recovered = SkillParameters.from_tensor(tensor, "unscrew")
        assert recovered.skill_id == "unscrew"
        assert abs(recovered.max_force - 5.0) < 0.01
