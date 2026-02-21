"""Training pipeline for all three VLA levels.

Supports:
- Level 2 (Skill Selector): Behavioral cloning from demonstrations
- Level 3 (Motor Policy): Diffusion policy training from trajectories
- Combined fine-tuning of both levels

Level 1 (Task Planner) uses a prompted VLM and doesn't require training,
or can be fine-tuned separately via standard VLM fine-tuning recipes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from safedisassemble.data.trajectory import TrajectoryDataset
from safedisassemble.models.skill_selector.selector import (
    SKILL_TO_IDX,
    SkillParameters,
    SkillSelector,
)
from safedisassemble.models.motor_policy.diffusion_policy import DiffusionMotorPolicy


class SkillSelectorDataset(Dataset):
    """Dataset for training the Level 2 Skill Selector.

    Each sample: (image, instruction_tokens, proprioception) → (skill_id, skill_params)
    Samples from mid-level instruction boundaries in trajectories.
    """

    def __init__(
        self,
        trajectory_path: str | Path,
        image_size: int = 224,
        max_instruction_len: int = 64,
    ):
        self.image_size = image_size
        self.max_len = max_instruction_len

        # Load trajectory metadata to build index
        self._samples: list[dict] = []
        self._traj_path = Path(trajectory_path)

        self._build_index()

    def _build_index(self) -> None:
        """Build an index of (trajectory_idx, timestep_idx) for subtask boundaries."""
        with TrajectoryDataset(self._traj_path, mode="r") as ds:
            for traj_idx in range(ds.num_trajectories):
                traj = ds.get_trajectory(traj_idx)
                segments = traj.get_subtask_segments()

                for start, end, instruction in segments:
                    if start >= len(traj.timesteps):
                        continue
                    ts = traj.timesteps[start]
                    self._samples.append({
                        "traj_idx": traj_idx,
                        "ts_idx": start,
                        "instruction_mid": instruction,
                        "skill_id": ts.skill_id,
                    })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]

        with TrajectoryDataset(self._traj_path, mode="r") as ds:
            traj = ds.get_trajectory(sample["traj_idx"])

        ts = traj.timesteps[sample["ts_idx"]]

        # Prepare image
        image = torch.from_numpy(ts.image_wrist).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Tokenize instruction (simple hash tokenizer)
        tokens = self._tokenize(sample["instruction_mid"])

        # Proprioception
        proprio = np.concatenate([
            ts.joint_pos, ts.joint_vel, ts.gripper_pos, ts.ee_pos,
        ])

        # Targets
        skill_idx = SKILL_TO_IDX.get(sample["skill_id"], 0)

        # Skill parameters (from trajectory data)
        skill_params = np.zeros(SkillParameters.param_dim(), dtype=np.float32)
        skill_params[:3] = ts.ee_pos  # target position approximation
        skill_params[3:6] = np.array([0, 0, -1])  # default approach from above

        return {
            "image": image,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "proprioception": torch.tensor(proprio, dtype=torch.float32),
            "skill_target": torch.tensor(skill_idx, dtype=torch.long),
            "param_target": torch.tensor(skill_params, dtype=torch.float32),
        }

    def _tokenize(self, text: str) -> list[int]:
        words = text.lower().split()
        tokens = [hash(w) % 9999 + 1 for w in words[:self.max_len]]
        return tokens + [0] * (self.max_len - len(tokens))


class MotorPolicyDataset(Dataset):
    """Dataset for training the Level 3 Diffusion Motor Policy.

    Each sample: sliding window of (observations, actions) from trajectories.
    """

    def __init__(
        self,
        trajectory_path: str | Path,
        action_horizon: int = 16,
        observation_horizon: int = 2,
        image_size: int = 224,
    ):
        self.action_horizon = action_horizon
        self.obs_horizon = observation_horizon
        self.image_size = image_size

        self._traj_path = Path(trajectory_path)
        self._windows: list[tuple[int, int]] = []  # (traj_idx, start_idx)

        self._build_index()

    def _build_index(self) -> None:
        """Build sliding window index over all trajectories."""
        with TrajectoryDataset(self._traj_path, mode="r") as ds:
            for traj_idx in range(ds.num_trajectories):
                traj = ds.get_trajectory(traj_idx)
                max_start = len(traj.timesteps) - self.action_horizon
                for start in range(0, max(1, max_start), self.action_horizon // 2):
                    self._windows.append((traj_idx, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        traj_idx, start = self._windows[idx]

        with TrajectoryDataset(self._traj_path, mode="r") as ds:
            traj = ds.get_trajectory(traj_idx)

        end = min(start + self.action_horizon, len(traj.timesteps))

        # Observation (from start timestep)
        ts = traj.timesteps[start]
        image = torch.from_numpy(ts.image_wrist).float() / 255.0
        image = image.permute(2, 0, 1)

        proprio = np.concatenate([
            ts.joint_pos, ts.joint_vel, ts.gripper_pos, ts.ee_pos,
        ])

        # Action sequence
        actions = []
        for t in range(start, end):
            actions.append(traj.timesteps[t].action)

        # Pad if needed
        while len(actions) < self.action_horizon:
            actions.append(actions[-1])

        actions = np.stack(actions)

        # Skill info
        skill_id = SKILL_TO_IDX.get(ts.skill_id, 0)
        skill_params = np.zeros(SkillParameters.param_dim(), dtype=np.float32)
        skill_params[:3] = ts.ee_pos

        return {
            "image": image,
            "proprioception": torch.tensor(proprio, dtype=torch.float32),
            "skill_id": torch.tensor(skill_id, dtype=torch.long),
            "skill_params": torch.tensor(skill_params, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
        }


class Trainer:
    """Unified trainer for Level 2 and Level 3 models.

    Supports:
    - Individual training of each level
    - Joint fine-tuning
    - Logging to W&B
    - Checkpointing
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        num_workers: int = 4,
        use_wandb: bool = True,
        wandb_project: str = "safedisassemble",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self._wandb_run = None

    def _init_wandb(self, run_name: str, config: dict) -> None:
        if self.use_wandb:
            import wandb
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=config,
            )

    def _log(self, metrics: dict, step: int) -> None:
        if self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def train_skill_selector(
        self,
        model: SkillSelector,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        num_epochs: int = 100,
        run_name: str = "skill_selector",
    ) -> dict:
        """Train the Level 2 Skill Selector via behavioral cloning.

        Returns:
            Training metrics summary
        """
        config = {
            "model": "skill_selector",
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": num_epochs,
        }
        self._init_wandb(run_name, config)

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        train_dataset = SkillSelectorDataset(train_data_path)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )

        val_loader = None
        if val_data_path:
            val_dataset = SkillSelectorDataset(val_data_path)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers,
            )

        best_val_loss = float("inf")
        metrics_history = []
        global_step = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []

            for batch in train_loader:
                images = batch["image"].to(self.device)
                tokens = batch["tokens"].to(self.device)
                proprio = batch["proprioception"].to(self.device)
                skill_targets = batch["skill_target"].to(self.device)
                param_targets = batch["param_target"].to(self.device)

                outputs = model(images, tokens, proprio)
                losses = model.compute_loss(outputs, skill_targets, param_targets)

                optimizer.zero_grad()
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append({
                    k: v.item() for k, v in losses.items()
                })
                global_step += 1

                self._log({
                    "train/total_loss": losses["total_loss"].item(),
                    "train/skill_loss": losses["skill_loss"].item(),
                    "train/param_loss": losses["param_loss"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }, global_step)

            scheduler.step()

            # Compute epoch averages
            avg_loss = np.mean([l["total_loss"] for l in epoch_losses])

            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_skill_selector(model, val_loader)
                self._log({"val/total_loss": val_loss}, global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(model, "skill_selector_best.pt")

            metrics_history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
            })

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(model, f"skill_selector_epoch{epoch+1}.pt")

        # Final save
        self._save_checkpoint(model, "skill_selector_final.pt")

        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": avg_loss,
            "history": metrics_history,
        }

    @torch.no_grad()
    def _validate_skill_selector(
        self, model: SkillSelector, val_loader: DataLoader
    ) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            images = batch["image"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            proprio = batch["proprioception"].to(self.device)
            skill_targets = batch["skill_target"].to(self.device)
            param_targets = batch["param_target"].to(self.device)

            outputs = model(images, tokens, proprio)
            losses = model.compute_loss(outputs, skill_targets, param_targets)
            total_loss += losses["total_loss"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_motor_policy(
        self,
        model: DiffusionMotorPolicy,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        num_epochs: int = 200,
        run_name: str = "motor_policy",
    ) -> dict:
        """Train the Level 3 Diffusion Motor Policy.

        Returns:
            Training metrics summary
        """
        config = {
            "model": "diffusion_motor_policy",
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": num_epochs,
            "diffusion_steps": model.num_diffusion_steps,
        }
        self._init_wandb(run_name, config)

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs,
        )

        train_dataset = MotorPolicyDataset(train_data_path)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )

        best_loss = float("inf")
        metrics_history = []
        global_step = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []

            for batch in train_loader:
                images = batch["image"].to(self.device)
                proprio = batch["proprioception"].to(self.device)
                skill_ids = batch["skill_id"].to(self.device)
                skill_params = batch["skill_params"].to(self.device)
                actions = batch["actions"].to(self.device)

                loss = model.compute_loss(
                    images, proprio, skill_ids, skill_params, actions
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                global_step += 1

                self._log({
                    "train/diffusion_loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }, global_step)

            scheduler.step()

            avg_loss = np.mean(epoch_losses)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(model, "motor_policy_best.pt")

            metrics_history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
            })

            if (epoch + 1) % 20 == 0:
                self._save_checkpoint(model, f"motor_policy_epoch{epoch+1}.pt")

        self._save_checkpoint(model, "motor_policy_final.pt")

        return {
            "best_loss": best_loss,
            "final_loss": avg_loss,
            "history": metrics_history,
        }

    def _save_checkpoint(self, model: nn.Module, filename: str) -> None:
        path = self.output_dir / filename
        torch.save(model.state_dict(), path)

    def close(self) -> None:
        if self._wandb_run:
            import wandb
            wandb.finish()
