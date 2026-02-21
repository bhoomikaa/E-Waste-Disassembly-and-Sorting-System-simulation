"""Baseline methods for comparison in evaluation.

Baselines:
1. Scripted Policy (per-device) — upper bound on known devices
2. End-to-End VLA (no hierarchy) — shows hierarchy adds value
3. VLA without Safety Module — shows safety contribution
4. Random Exploration — sanity check
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ScriptedBaseline:
    """Hard-coded disassembly policy for a specific known device.

    This is the upper bound: perfect task knowledge, but zero generalization.
    Uses pre-programmed waypoints for each component of a specific device.
    """

    def __init__(self, device_name: str):
        self.device_name = device_name
        self._waypoints = self._load_waypoints(device_name)
        self._current_step = 0
        self._current_waypoint = 0

    def reset(self) -> None:
        self._current_step = 0
        self._current_waypoint = 0

    def predict(self, observation: dict) -> np.ndarray:
        """Return next action from scripted sequence."""
        if self._current_step >= len(self._waypoints):
            return np.zeros(7, dtype=np.float32)

        wp = self._waypoints[self._current_step]
        ee_pos = observation["ee_pos"]

        # Simple proportional control toward waypoint
        target = wp["target"]
        error = target - ee_pos
        dist = np.linalg.norm(error)

        if dist < 0.005:  # reached waypoint
            self._current_step += 1
            if self._current_step >= len(self._waypoints):
                return np.zeros(7, dtype=np.float32)
            wp = self._waypoints[self._current_step]
            target = wp["target"]
            error = target - ee_pos

        # Action: proportional position control + gripper
        action_pos = np.clip(error * 50, -1, 1)
        action_rot = np.zeros(3)
        gripper = wp.get("gripper", 0.0)

        return np.concatenate([action_pos, action_rot, [gripper]]).astype(np.float32)

    def _load_waypoints(self, device_name: str) -> list[dict]:
        """Load pre-programmed waypoints for known devices."""
        # These would be hand-tuned per device in the benchmark
        waypoints = {
            "laptop_v1": [
                {"target": np.array([0.36, -0.09, 0.50]), "gripper": 1.0},  # above screw 1
                {"target": np.array([0.36, -0.09, 0.46]), "gripper": 0.1},  # engage screw 1
                {"target": np.array([0.36, -0.09, 0.50]), "gripper": 0.1},  # extract
                # ... more waypoints per screw and component
                {"target": np.array([0.64, -0.09, 0.50]), "gripper": 1.0},  # above screw 2
                {"target": np.array([0.64, -0.09, 0.46]), "gripper": 0.1},
                {"target": np.array([0.64, -0.09, 0.50]), "gripper": 0.1},
            ],
            "router_v1": [
                {"target": np.array([0.50, -0.068, 0.47]), "gripper": 0.5},
                {"target": np.array([0.50, -0.068, 0.46]), "gripper": 0.2},
                {"target": np.array([0.50, 0.068, 0.47]), "gripper": 0.5},
                {"target": np.array([0.50, 0.068, 0.46]), "gripper": 0.2},
            ],
        }
        return waypoints.get(device_name, [])


class RandomBaseline:
    """Random action baseline (sanity check).

    Should perform very poorly — confirms metrics are working.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def predict(self, observation: dict) -> np.ndarray:
        return self.rng.uniform(-1, 1, 7).astype(np.float32)


class FlatVLABaseline:
    """End-to-end VLA without hierarchical decomposition.

    Same model capacity as the full system, but trained end-to-end
    without the 3-level hierarchy. Used to demonstrate that the
    hierarchical architecture adds value for long-horizon tasks.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self._model = None

    def reset(self) -> None:
        pass

    def load(self) -> None:
        """Load a trained flat VLA model."""
        if self.model_path is None:
            return

        import torch
        # The flat VLA would be a single model that directly maps
        # (image, instruction) → action, without skill decomposition
        # For the benchmark, this is trained separately
        from safedisassemble.models.motor_policy.diffusion_policy import DiffusionMotorPolicy

        self._model = DiffusionMotorPolicy(
            cond_dim=256,
            action_horizon=1,  # single-step prediction (no trajectory)
        ).to(self.device)

        if self.model_path:
            import torch
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, observation: dict) -> np.ndarray:
        """Predict action from flat VLA."""
        if self._model is None:
            return np.zeros(7, dtype=np.float32)

        import torch
        image = torch.from_numpy(observation["image_wrist"]).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

        proprio = np.concatenate([
            observation["joint_pos"],
            observation["joint_vel"],
            observation["gripper_pos"],
            observation["ee_pos"],
        ])
        proprio = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)

        skill_id = torch.zeros(1, dtype=torch.long, device=self.device)
        skill_params = torch.zeros(1, 10, device=self.device)

        with torch.no_grad():
            action_seq = self._model.predict_action(
                image, proprio, skill_id, skill_params
            )

        return action_seq[0, 0].cpu().numpy().astype(np.float32)


class NoSafetyBaseline:
    """Full hierarchical VLA but with safety module disabled.

    Demonstrates the value of the safety constraint module by showing
    higher violation rates without it.
    """

    def __init__(self, controller):
        self.controller = controller
        # Disable safety checking
        self.controller.safety_module.reset()
        self._safety_disabled = True

    def reset(self) -> None:
        self.controller.safety_module.reset()

    def step(self, observation: dict) -> np.ndarray:
        """Step without safety checks."""
        # Temporarily bypass safety
        original_check = self.controller.safety_module.check_runtime
        self.controller.safety_module.check_runtime = lambda *a, **k: []

        action = self.controller.step(observation)

        # Restore
        self.controller.safety_module.check_runtime = original_check
        return action
