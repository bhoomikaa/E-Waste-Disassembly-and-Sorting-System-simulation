"""Level 3: Motor Policy using Diffusion Policy (Chi et al., 2023).

Generates continuous end-effector action trajectories conditioned on:
- Current observation (image + proprioception)
- Skill ID and parameters from Level 2
- Action horizon (predict H future actions at once)

Diffusion Policy is chosen because it handles multimodal action distributions
well — there are often multiple valid ways to perform a manipulation task
(e.g., prying from left or right edge).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings for the diffusion process."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class ConditionalResidualBlock(nn.Module):
    """1D residual block conditioned on diffusion timestep and context."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        # Conditional scaling and bias (FiLM conditioning)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) temporal feature map
            cond: (B, cond_dim) conditioning vector
        """
        h = self.blocks(x)

        # FiLM conditioning
        scale, bias = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)  # (B, C, 1)
        bias = bias.unsqueeze(-1)
        h = h * (1 + scale) + bias

        return h + self.residual_conv(x)


class ConditionalUNet1D(nn.Module):
    """1D U-Net for denoising action trajectories.

    Architecture follows Diffusion Policy (Chi et al., 2023):
    - Input: noisy action trajectory (B, action_dim, horizon)
    - Conditioning: observation embedding + skill embedding + diffusion timestep
    - Output: predicted noise (B, action_dim, horizon)
    """

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 16,
        cond_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Combined conditioning dimension
        total_cond_dim = cond_dim + diffusion_step_embed_dim

        # Input projection
        self.input_proj = nn.Conv1d(action_dim, down_dims[0], 1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i in range(len(down_dims) - 1):
            self.encoder_blocks.append(
                ConditionalResidualBlock(down_dims[i], down_dims[i], total_cond_dim)
            )
            self.downsample.append(
                nn.Conv1d(down_dims[i], down_dims[i + 1], 3, stride=2, padding=1)
            )

        # Bottleneck
        self.mid_block = ConditionalResidualBlock(
            down_dims[-1], down_dims[-1], total_cond_dim
        )

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(len(down_dims) - 1, 0, -1):
            self.upsample.append(
                nn.ConvTranspose1d(down_dims[i], down_dims[i - 1], 4, stride=2, padding=1)
            )
            self.decoder_blocks.append(
                ConditionalResidualBlock(
                    down_dims[i - 1] * 2, down_dims[i - 1], total_cond_dim
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], 3, padding=1),
            nn.GroupNorm(8, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, action_dim, horizon) noisy action trajectory
            timesteps: (B,) diffusion timestep indices
            condition: (B, cond_dim) observation + skill conditioning

        Returns:
            (B, action_dim, horizon) predicted noise
        """
        # Embed timesteps
        time_emb = self.time_embed(timesteps.float())

        # Combine conditioning
        cond = torch.cat([condition, time_emb], dim=-1)

        # Input projection
        x = self.input_proj(noisy_actions)

        # Encoder with skip connections
        skip_connections = []
        for enc_block, down in zip(self.encoder_blocks, self.downsample):
            x = enc_block(x, cond)
            skip_connections.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block(x, cond)

        # Decoder
        for dec_block, up, skip in zip(
            self.decoder_blocks, self.upsample, reversed(skip_connections)
        ):
            x = up(x)
            # Handle potential size mismatch from downsampling
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x, cond)

        return self.output_proj(x)


class DiffusionScheduler:
    """DDPM noise scheduler for training and inference."""

    def __init__(
        self,
        num_train_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        self.num_train_steps = num_train_steps

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_steps)
        elif beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule (improved DDPM)
            t = torch.linspace(0, num_train_steps, num_train_steps + 1) / num_train_steps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to actions according to the schedule."""
        device = original.device
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)

        # Reshape for broadcasting
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * original + sqrt_one_minus_alpha * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step (DDPM)."""
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]

        # Predicted x_0
        pred_original = (
            sample - torch.sqrt(1 - alpha_cumprod) * model_output
        ) / torch.sqrt(alpha_cumprod)

        # Clip
        pred_original = torch.clamp(pred_original, -1, 1)

        # Compute mean
        pred_mean = (
            torch.sqrt(alpha) * (1 - self.alphas_cumprod[max(timestep - 1, 0)])
            / (1 - alpha_cumprod) * sample
            + torch.sqrt(self.alphas_cumprod[max(timestep - 1, 0)]) * beta
            / (1 - alpha_cumprod) * pred_original
        )

        if timestep > 0:
            noise = torch.randn_like(sample)
            variance = beta * (1 - self.alphas_cumprod[timestep - 1]) / (1 - alpha_cumprod)
            pred_mean += torch.sqrt(variance) * noise

        return pred_mean


class ObservationEncoder(nn.Module):
    """Encode visual + proprioceptive observations into a conditioning vector."""

    def __init__(
        self,
        image_size: int = 224,
        proprio_dim: int = 19,
        skill_embed_dim: int = 32,
        n_skills: int = 8,
        output_dim: int = 256,
    ):
        super().__init__()

        # Image encoder (lightweight CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
        )

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Skill embedding
        self.skill_embed = nn.Embedding(n_skills, skill_embed_dim)

        # Skill parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(10, 32),  # SkillParameters.param_dim()
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Fusion
        fusion_in = 256 + 64 + skill_embed_dim + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        proprioception: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
            proprioception: (B, proprio_dim)
            skill_ids: (B,) long
            skill_params: (B, 10)

        Returns:
            (B, output_dim) conditioning vector
        """
        img_feat = self.image_encoder(images)
        proprio_feat = self.proprio_encoder(proprioception)
        skill_feat = self.skill_embed(skill_ids)
        param_feat = self.param_encoder(skill_params)

        combined = torch.cat([img_feat, proprio_feat, skill_feat, param_feat], dim=-1)
        return self.fusion(combined)


class DiffusionMotorPolicy(nn.Module):
    """Complete diffusion-based motor policy.

    Given observation + skill, generates an action trajectory of length H.
    Uses DDPM for training and DDIM for fast inference.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        observation_horizon: int = 2,
        image_size: int = 224,
        proprio_dim: int = 19,
        n_skills: int = 8,
        cond_dim: int = 256,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps

        # Observation encoder
        self.obs_encoder = ObservationEncoder(
            image_size=image_size,
            proprio_dim=proprio_dim,
            n_skills=n_skills,
            output_dim=cond_dim,
        )

        # Denoising network
        self.noise_predictor = ConditionalUNet1D(
            action_dim=action_dim,
            horizon=action_horizon,
            cond_dim=cond_dim,
        )

        # Noise scheduler
        self.scheduler = DiffusionScheduler(
            num_train_steps=num_diffusion_steps,
        )

    def compute_loss(
        self,
        images: torch.Tensor,
        proprioception: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_params: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Args:
            images: (B, 3, H, W)
            proprioception: (B, proprio_dim)
            skill_ids: (B,) long
            skill_params: (B, 10)
            actions: (B, action_horizon, action_dim) ground-truth action sequences
        """
        B = actions.shape[0]
        device = actions.device

        # Encode observations
        cond = self.obs_encoder(images, proprioception, skill_ids, skill_params)

        # Rearrange actions to (B, action_dim, horizon)
        actions_t = rearrange(actions, "b t a -> b a t")

        # Sample random timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(actions_t)

        # Add noise to actions
        noisy_actions = self.scheduler.add_noise(actions_t, noise, timesteps)

        # Predict noise
        noise_pred = self.noise_predictor(noisy_actions, timesteps, cond)

        # MSE loss
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        proprioception: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_params: torch.Tensor,
    ) -> torch.Tensor:
        """Generate action trajectory via iterative denoising.

        Args:
            images: (B, 3, H, W) or (3, H, W)
            proprioception: (B, proprio_dim) or (proprio_dim,)
            skill_ids: (B,) or scalar
            skill_params: (B, 10) or (10,)

        Returns:
            (B, action_horizon, action_dim) predicted action sequence
        """
        # Handle single inputs
        if images.dim() == 3:
            images = images.unsqueeze(0)
            proprioception = proprioception.unsqueeze(0)
            skill_ids = skill_ids.unsqueeze(0)
            skill_params = skill_params.unsqueeze(0)

        B = images.shape[0]
        device = images.device

        # Encode observations
        cond = self.obs_encoder(images, proprioception, skill_ids, skill_params)

        # Start from pure noise
        actions = torch.randn(B, self.action_dim, self.action_horizon, device=device)

        # DDIM-style inference (fewer steps)
        step_ratio = self.num_diffusion_steps // self.num_inference_steps
        timesteps = list(range(0, self.num_diffusion_steps, step_ratio))[::-1]

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.noise_predictor(actions, t_batch, cond)
            actions = self.scheduler.step(noise_pred, t, actions)

        # Clip to valid action range
        actions = torch.clamp(actions, -1, 1)

        # Rearrange to (B, horizon, action_dim)
        return rearrange(actions, "b a t -> b t a")
