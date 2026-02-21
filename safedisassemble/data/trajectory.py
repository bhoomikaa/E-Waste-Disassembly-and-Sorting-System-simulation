"""Trajectory data structures and storage for demonstration data.

Stores (image, language_instruction, action) tuples in HDF5 format
with support for hierarchical language annotations at three levels:
  - High-level: "Disassemble this laptop"
  - Mid-level: "Remove the back panel"
  - Low-level: "Unscrew the Phillips-head screw at bottom-left"
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


@dataclasses.dataclass
class Timestep:
    """Single timestep in a trajectory."""
    image_wrist: np.ndarray       # (H, W, 3) uint8
    image_overhead: np.ndarray    # (H, W, 3) uint8
    joint_pos: np.ndarray         # (7,) float64
    joint_vel: np.ndarray         # (7,) float64
    gripper_pos: np.ndarray       # (2,) float64
    ee_pos: np.ndarray            # (3,) float64
    ee_force: np.ndarray          # (3,) float64
    action: np.ndarray            # (7,) float32 - the action taken at this step
    # Hierarchical language annotations
    instruction_high: str         # e.g., "Disassemble this laptop"
    instruction_mid: str          # e.g., "Remove the back panel"
    instruction_low: str          # e.g., "Unscrew the screw at position (0.14, -0.09)"
    # Metadata
    timestamp: float = 0.0
    skill_id: str = ""            # which primitive skill is active
    is_terminal: bool = False
    safety_violation: bool = False


@dataclasses.dataclass
class Trajectory:
    """A complete disassembly trajectory (one episode)."""
    device_name: str
    timesteps: list[Timestep]
    success: bool = False
    total_reward: float = 0.0
    components_recovered: list[str] = dataclasses.field(default_factory=list)
    safety_violations: list[dict] = dataclasses.field(default_factory=list)
    metadata: dict = dataclasses.field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.timesteps)

    def get_subtask_segments(self) -> list[tuple[int, int, str]]:
        """Identify contiguous segments sharing the same mid-level instruction."""
        if not self.timesteps:
            return []

        segments = []
        start = 0
        current_mid = self.timesteps[0].instruction_mid

        for i, ts in enumerate(self.timesteps):
            if ts.instruction_mid != current_mid:
                segments.append((start, i, current_mid))
                start = i
                current_mid = ts.instruction_mid

        segments.append((start, len(self.timesteps), current_mid))
        return segments


class TrajectoryDataset:
    """HDF5-backed dataset of demonstration trajectories.

    File structure:
        /trajectories/
            /traj_000/
                images_wrist: (T, H, W, 3) uint8
                images_overhead: (T, H, W, 3) uint8
                joint_pos: (T, 7) float64
                joint_vel: (T, 7) float64
                gripper_pos: (T, 2) float64
                ee_pos: (T, 3) float64
                ee_force: (T, 3) float64
                actions: (T, 7) float32
                instructions_high: JSON string list
                instructions_mid: JSON string list
                instructions_low: JSON string list
                skill_ids: JSON string list
                is_terminal: (T,) bool
                attrs:
                    device_name, success, total_reward,
                    components_recovered, safety_violations, metadata
    """

    def __init__(self, path: str | Path, mode: str = "r"):
        self.path = Path(path)
        self.mode = mode
        self._file: Optional[h5py.File] = None

    def open(self) -> None:
        self._file = h5py.File(self.path, self.mode)
        if "trajectories" not in self._file:
            self._file.create_group("trajectories")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def num_trajectories(self) -> int:
        assert self._file is not None
        return len(self._file["trajectories"])

    @property
    def total_timesteps(self) -> int:
        assert self._file is not None
        total = 0
        for key in self._file["trajectories"]:
            total += self._file["trajectories"][key]["actions"].shape[0]
        return total

    def add_trajectory(self, traj: Trajectory) -> str:
        """Write a trajectory to the dataset. Returns the trajectory key."""
        assert self._file is not None
        assert self.mode in ("w", "a")

        traj_idx = self.num_trajectories
        key = f"traj_{traj_idx:06d}"
        grp = self._file["trajectories"].create_group(key)

        T = traj.length

        # Stack arrays
        images_wrist = np.stack([ts.image_wrist for ts in traj.timesteps])
        images_overhead = np.stack([ts.image_overhead for ts in traj.timesteps])
        joint_pos = np.stack([ts.joint_pos for ts in traj.timesteps])
        joint_vel = np.stack([ts.joint_vel for ts in traj.timesteps])
        gripper_pos = np.stack([ts.gripper_pos for ts in traj.timesteps])
        ee_pos = np.stack([ts.ee_pos for ts in traj.timesteps])
        ee_force = np.stack([ts.ee_force for ts in traj.timesteps])
        actions = np.stack([ts.action for ts in traj.timesteps])

        # Store arrays with compression
        grp.create_dataset("images_wrist", data=images_wrist,
                          chunks=(min(T, 32), *images_wrist.shape[1:]),
                          compression="gzip", compression_opts=4)
        grp.create_dataset("images_overhead", data=images_overhead,
                          chunks=(min(T, 32), *images_overhead.shape[1:]),
                          compression="gzip", compression_opts=4)
        grp.create_dataset("joint_pos", data=joint_pos)
        grp.create_dataset("joint_vel", data=joint_vel)
        grp.create_dataset("gripper_pos", data=gripper_pos)
        grp.create_dataset("ee_pos", data=ee_pos)
        grp.create_dataset("ee_force", data=ee_force)
        grp.create_dataset("actions", data=actions)

        # Store string data as JSON
        grp.attrs["instructions_high"] = json.dumps(
            [ts.instruction_high for ts in traj.timesteps]
        )
        grp.attrs["instructions_mid"] = json.dumps(
            [ts.instruction_mid for ts in traj.timesteps]
        )
        grp.attrs["instructions_low"] = json.dumps(
            [ts.instruction_low for ts in traj.timesteps]
        )
        grp.attrs["skill_ids"] = json.dumps(
            [ts.skill_id for ts in traj.timesteps]
        )

        # Terminal flags
        grp.create_dataset("is_terminal", data=np.array(
            [ts.is_terminal for ts in traj.timesteps]
        ))

        # Trajectory-level metadata
        grp.attrs["device_name"] = traj.device_name
        grp.attrs["success"] = traj.success
        grp.attrs["total_reward"] = traj.total_reward
        grp.attrs["components_recovered"] = json.dumps(traj.components_recovered)
        grp.attrs["safety_violations"] = json.dumps(traj.safety_violations)
        grp.attrs["metadata"] = json.dumps(traj.metadata)
        grp.attrs["timestamp"] = time.time()

        self._file.flush()
        return key

    def get_trajectory(self, idx: int) -> Trajectory:
        """Load a trajectory by index."""
        assert self._file is not None

        key = f"traj_{idx:06d}"
        grp = self._file["trajectories"][key]

        instructions_high = json.loads(grp.attrs["instructions_high"])
        instructions_mid = json.loads(grp.attrs["instructions_mid"])
        instructions_low = json.loads(grp.attrs["instructions_low"])
        skill_ids = json.loads(grp.attrs["skill_ids"])

        T = grp["actions"].shape[0]
        timesteps = []
        for t in range(T):
            ts = Timestep(
                image_wrist=grp["images_wrist"][t],
                image_overhead=grp["images_overhead"][t],
                joint_pos=grp["joint_pos"][t],
                joint_vel=grp["joint_vel"][t],
                gripper_pos=grp["gripper_pos"][t],
                ee_pos=grp["ee_pos"][t],
                ee_force=grp["ee_force"][t],
                action=grp["actions"][t],
                instruction_high=instructions_high[t],
                instruction_mid=instructions_mid[t],
                instruction_low=instructions_low[t],
                skill_id=skill_ids[t],
                is_terminal=bool(grp["is_terminal"][t]),
            )
            timesteps.append(ts)

        return Trajectory(
            device_name=grp.attrs["device_name"],
            timesteps=timesteps,
            success=bool(grp.attrs["success"]),
            total_reward=float(grp.attrs["total_reward"]),
            components_recovered=json.loads(grp.attrs["components_recovered"]),
            safety_violations=json.loads(grp.attrs["safety_violations"]),
            metadata=json.loads(grp.attrs["metadata"]),
        )

    def get_statistics(self) -> dict:
        """Compute dataset statistics."""
        assert self._file is not None
        stats = {
            "num_trajectories": self.num_trajectories,
            "total_timesteps": self.total_timesteps,
            "success_rate": 0.0,
            "devices": {},
        }

        success_count = 0
        for key in self._file["trajectories"]:
            grp = self._file["trajectories"][key]
            device = grp.attrs["device_name"]
            if device not in stats["devices"]:
                stats["devices"][device] = {"count": 0, "successes": 0}
            stats["devices"][device]["count"] += 1
            if grp.attrs["success"]:
                success_count += 1
                stats["devices"][device]["successes"] += 1

        if self.num_trajectories > 0:
            stats["success_rate"] = success_count / self.num_trajectories

        return stats
