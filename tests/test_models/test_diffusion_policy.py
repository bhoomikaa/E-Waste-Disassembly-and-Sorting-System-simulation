"""Tests for the Level 3 Diffusion Motor Policy."""

import torch
import pytest

from safedisassemble.models.motor_policy.diffusion_policy import (
    ConditionalUNet1D,
    DiffusionMotorPolicy,
    DiffusionScheduler,
    ObservationEncoder,
)


class TestDiffusionScheduler:
    def test_add_noise(self):
        scheduler = DiffusionScheduler(num_train_steps=100)
        original = torch.randn(4, 7, 16)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([0, 25, 50, 99])

        noisy = scheduler.add_noise(original, noise, timesteps)
        assert noisy.shape == original.shape

        # At t=0, should be close to original
        assert torch.allclose(noisy[0], original[0], atol=0.1)

    def test_step(self):
        scheduler = DiffusionScheduler(num_train_steps=100)
        sample = torch.randn(4, 7, 16)
        model_output = torch.randn_like(sample)

        result = scheduler.step(model_output, 50, sample)
        assert result.shape == sample.shape


class TestConditionalUNet1D:
    def test_forward_shape(self):
        model = ConditionalUNet1D(
            action_dim=7, horizon=16, cond_dim=128,
            down_dims=(128, 256),
        )
        noisy = torch.randn(4, 7, 16)
        timesteps = torch.randint(0, 100, (4,))
        cond = torch.randn(4, 128)

        # Total cond = cond_dim + diffusion_step_embed_dim
        output = model(noisy, timesteps, cond)
        assert output.shape == (4, 7, 16)


class TestObservationEncoder:
    def test_forward_shape(self):
        encoder = ObservationEncoder(
            image_size=64, proprio_dim=19, output_dim=128,
        )
        images = torch.randn(4, 3, 64, 64)
        proprio = torch.randn(4, 19)
        skill_ids = torch.randint(0, 8, (4,))
        skill_params = torch.randn(4, 10)

        out = encoder(images, proprio, skill_ids, skill_params)
        assert out.shape == (4, 128)


class TestDiffusionMotorPolicy:
    def setup_method(self):
        self.policy = DiffusionMotorPolicy(
            action_dim=7,
            action_horizon=8,
            image_size=64,
            cond_dim=128,
            num_diffusion_steps=20,
            num_inference_steps=5,
        )

    def test_compute_loss(self):
        images = torch.randn(4, 3, 64, 64)
        proprio = torch.randn(4, 19)
        skill_ids = torch.randint(0, 8, (4,))
        skill_params = torch.randn(4, 10)
        actions = torch.randn(4, 8, 7)

        loss = self.policy.compute_loss(
            images, proprio, skill_ids, skill_params, actions
        )
        assert loss.requires_grad
        assert loss.item() > 0

    def test_predict_action_shape(self):
        images = torch.randn(2, 3, 64, 64)
        proprio = torch.randn(2, 19)
        skill_ids = torch.randint(0, 8, (2,))
        skill_params = torch.randn(2, 10)

        actions = self.policy.predict_action(
            images, proprio, skill_ids, skill_params
        )
        assert actions.shape == (2, 8, 7)

    def test_predict_single(self):
        image = torch.randn(3, 64, 64)
        proprio = torch.randn(19)
        skill_id = torch.tensor(0)
        skill_params = torch.randn(10)

        actions = self.policy.predict_action(
            image, proprio, skill_id, skill_params
        )
        assert actions.shape == (1, 8, 7)
