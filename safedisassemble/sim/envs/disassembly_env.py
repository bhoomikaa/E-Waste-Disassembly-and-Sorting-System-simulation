"""Core Gymnasium environment for e-waste disassembly tasks.

Supports:
- Multiple device types via the device registry
- Language-conditioned tasks at multiple granularity levels
- Safety constraint monitoring (battery puncture, PCB snap)
- Configurable observation spaces (images + proprioception)
- Domain randomization hooks
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from safedisassemble.sim.device_registry import (
    ComponentType,
    DeviceSpec,
    HazardType,
    get_device,
)


class DisassemblyEnv(gym.Env):
    """E-waste disassembly environment.

    Observation space:
        - image_wrist: (H, W, 3) uint8 from wrist camera
        - image_overhead: (H, W, 3) uint8 from overhead camera
        - joint_pos: (7,) float64 robot joint positions
        - joint_vel: (7,) float64 robot joint velocities
        - gripper_pos: (2,) float64 finger positions
        - ee_pos: (3,) float64 end-effector position
        - ee_force: (3,) float64 estimated end-effector force
        - language_instruction: string (handled separately)

    Action space:
        - delta_ee: (6,) end-effector delta (xyz + rpy) in [-1, 1]
        - gripper: (1,) gripper command in [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        device_name: str = "laptop_v1",
        image_size: int = 224,
        max_steps: int = 500,
        reward_type: str = "sparse",
        render_mode: Optional[str] = None,
        use_domain_randomization: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.device_spec = get_device(device_name)
        self.image_size = image_size
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.use_domain_randomization = use_domain_randomization

        # Load MuJoCo model - combine robot + device
        self._build_combined_model()

        # Observation space
        self.observation_space = spaces.Dict({
            "image_wrist": spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "image_overhead": spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "joint_pos": spaces.Box(-np.pi, np.pi, (7,), dtype=np.float64),
            "joint_vel": spaces.Box(-10, 10, (7,), dtype=np.float64),
            "gripper_pos": spaces.Box(0, 0.04, (2,), dtype=np.float64),
            "ee_pos": spaces.Box(-2, 2, (3,), dtype=np.float64),
            "ee_force": spaces.Box(-100, 100, (3,), dtype=np.float64),
        })

        # Action space: 6-DOF EE delta + gripper
        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)

        # Task state
        self._current_instruction: str = ""
        self._target_component: Optional[str] = None
        self._removed_components: set[str] = set()
        self._step_count = 0
        self._safety_violations: list[dict] = []
        self._episode_reward = 0.0

        # Action scaling
        self._pos_scale = 0.02  # 2cm max displacement per step
        self._rot_scale = 0.1   # ~6 degrees max rotation per step

        # Camera setup
        self._wrist_cam_id: int = -1
        self._overhead_cam_id: int = -1

        # Rendering
        self._renderer: Optional[mujoco.Renderer] = None


    def _build_combined_model(self) -> None:
        """Build combined MuJoCo model from robot arm + device XMLs.

        NOTE: The original repo loaded robot-only (for basic testing).
        This version merges the device MJCF into the robot scene so renders show the laptop/router.
        """
        robot_xml = Path(__file__).parent.parent / "assets" / "xmls" / "robot_arm.xml"
        device_xml = Path(self.device_spec.mjcf_path)

        import xml.etree.ElementTree as ET
        import mujoco

        r = ET.parse(robot_xml).getroot()
        d = ET.parse(device_xml).getroot()

        def merge_children(tag: str):
            r_sec = r.find(tag)
            d_sec = d.find(tag)
            if d_sec is None:
                return
            if r_sec is None:
                r_sec = ET.SubElement(r, tag)
            for child in list(d_sec):
                r_sec.append(child)

        # Bring over device defaults (classes), assets (materials), and bodies/sites/geoms
        merge_children("default")
        merge_children("asset")
        merge_children("worldbody")

        # Write a combined MJCF next to robot xml (easy to inspect)
        combined = robot_xml.with_name(f"_combined_{self.device_spec.name}.xml")
        ET.ElementTree(r).write(combined, encoding="utf-8")

        self.model = mujoco.MjModel.from_xml_path(str(combined))
        self.data = mujoco.MjData(self.model)

        # Cache actuator and sensor indices
        self._joint_actuator_ids = []
        for i in range(7):
            name = f"act_joint{i+1}"
            self._joint_actuator_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            )

        self._finger_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_right"),
        ]

        # Camera IDs (sites in robot_arm.xml)
        self._wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "wrist_cam"
        )
        self._overhead_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "overhead_cam"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        options = options or {}

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to home position
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        for i, q in enumerate(home_qpos):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}"
            )
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = q

        mujoco.mj_forward(self.model, self.data)

        # Apply domain randomization if enabled
        if self.use_domain_randomization:
            self._apply_domain_randomization()

        # Set task instruction
        self._current_instruction = options.get(
            "instruction", "Disassemble the device"
        )
        self._target_component = options.get("target_component", None)

        # Reset state tracking
        self._removed_components = set()
        self._step_count = 0
        self._safety_violations = []
        self._episode_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1

        # Parse action
        delta_pos = action[:3] * self._pos_scale
        delta_rot = action[3:6] * self._rot_scale
        gripper_cmd = (action[6] + 1.0) / 2.0  # map [-1,1] to [0,1]

        # Apply action via operational space control
        self._apply_ee_action(delta_pos, delta_rot, gripper_cmd)

        # Step simulation (multiple substeps for stability)
        n_substeps = 10
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Check safety violations
        self._check_safety()

        # Compute reward
        reward = self._compute_reward()
        self._episode_reward += reward

        # Check termination
        terminated = self._check_terminated()
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_ee_action(
        self, delta_pos: np.ndarray, delta_rot: np.ndarray, gripper_cmd: float
    ) -> None:
        """Apply end-effector delta action using simple joint-space mapping.

        For a full implementation, this would use operational space control
        or resolved-rate IK. Here we use a simplified Jacobian-based approach.
        """
        # Get current EE site position
        ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        current_ee_pos = self.data.site_xpos[ee_site_id].copy()
        target_pos = current_ee_pos + delta_pos

        # Simple proportional control toward target
        # In production, replace with proper OSC or IK
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, ee_site_id)

        # Joint velocity command via pseudoinverse
        pos_error = target_pos - current_ee_pos
        jac_arm = jacp[:, :7]
        jac_pinv = np.linalg.pinv(jac_arm, rcond=1e-4)
        dq = jac_pinv @ pos_error

        # Apply as torque commands (PD control)
        kp = 50.0
        kd = 5.0
        for i in range(7):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}"
            )
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]
            target_q = self.data.qpos[qpos_idx] + dq[i] * 0.1
            torque = kp * (target_q - self.data.qpos[qpos_idx]) - kd * self.data.qvel[qvel_idx]
            act_id = self._joint_actuator_ids[i]
            self.data.ctrl[act_id] = np.clip(torque, -87, 87)

        # Gripper control
        gripper_target = gripper_cmd * 0.04
        self.data.ctrl[self._finger_actuator_ids[0]] = gripper_target
        self.data.ctrl[self._finger_actuator_ids[1]] = gripper_target

    def _get_obs(self) -> dict:
        """Build observation dictionary."""
        # Joint positions and velocities (first 7 joints = arm)
        joint_pos = np.zeros(7)
        joint_vel = np.zeros(7)
        for i in range(7):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}"
            )
            joint_pos[i] = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            joint_vel[i] = self.data.qvel[self.model.jnt_dofadr[joint_id]]

        # Gripper positions
        left_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left"
        )
        right_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right"
        )
        gripper_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[left_id]],
            self.data.qpos[self.model.jnt_qposadr[right_id]],
        ])

        # End-effector position
        ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        ee_pos = self.data.site_xpos[ee_site_id].copy()

        # Estimated EE force (from contact forces)
        ee_force = self._estimate_ee_force()

        # Render camera images
        image_wrist = self._render_camera("wrist")
        image_overhead = self._render_camera("overhead")

        return {
            "image_wrist": image_wrist,
            "image_overhead": image_overhead,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "gripper_pos": gripper_pos,
            "ee_pos": ee_pos,
            "ee_force": ee_force,
        }

    def _render_camera(self, camera_name: str) -> np.ndarray:
        """Render from a named camera. Returns (H, W, 3) uint8 array."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self.image_size, self.image_size
            )

        if camera_name == "wrist":
            # Render FROM the wrist site looking OUTWARD toward the device.
            # The wrist_cam site is on the gripper base. We compute a camera
            # that starts at the wrist and looks along the wrist's z-axis
            # (which points toward the device / workpiece).
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "wrist_cam"
            )
            wrist_pos = self.data.site_xpos[site_id].copy()

            # site_xmat is the 3×3 rotation flattened row-major.
            # Columns: x-right, y-up, z-forward of the site frame.
            xmat = self.data.site_xmat[site_id].reshape(3, 3)
            # The wrist_cam site has euler="0 pi 0" so its z-axis points
            # roughly downward toward the workspace. We look along -z of
            # the site (which is the physical "forward" after the pi flip).
            forward = -xmat[:, 2]  # camera look direction

            # Place lookat a fixed distance ahead of the wrist
            look_distance = 0.15  # 15 cm ahead
            lookat = wrist_pos + forward * look_distance

            # Camera positioned at the wrist, looking at the lookat point
            cam.lookat[:] = lookat
            cam.distance = look_distance
            # Derive azimuth / elevation from the forward vector
            dx, dy, dz = forward
            cam.azimuth = np.degrees(np.arctan2(dy, dx))
            cam.elevation = np.degrees(np.arcsin(np.clip(dz, -1, 1)))
        elif camera_name == "overhead":
            # Overhead camera — bird's-eye view of the workspace
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.5, 0, 0.42]
            cam.distance = 0.8
            cam.azimuth = 0
            cam.elevation = -90
        elif camera_name == "cinematic_close":
            # Low-angle close-up for dramatic video shots
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.5, 0, 0.45]
            cam.distance = 0.45
            cam.azimuth = 35
            cam.elevation = -35
        elif camera_name == "cinematic_wide":
            # Wide establishing shot showing full robot + workspace
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.3, 0, 0.45]
            cam.distance = 1.2
            cam.azimuth = -25
            cam.elevation = -25
        elif camera_name == "cinematic_side":
            # Side profile for dramatic arm movement shots
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.5, 0, 0.48]
            cam.distance = 0.6
            cam.azimuth = 90
            cam.elevation = -15
        else:
            # Fallback — overhead
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.5, 0, 0.42]
            cam.distance = 0.8
            cam.azimuth = 0
            cam.elevation = -90

        self._renderer.update_scene(self.data, cam)
        image = self._renderer.render()
        return image.copy()

    def _estimate_ee_force(self) -> np.ndarray:
        """Estimate forces at end-effector from contact data."""
        force = np.zeros(3)
        fingertip_geom_ids = set()

        for name in ["left_fingertip", "right_fingertip"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                fingertip_geom_ids.add(gid)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in fingertip_geom_ids or contact.geom2 in fingertip_geom_ids:
                # Extract normal force
                c_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                force += c_force[:3]

        return force

    def _check_safety(self) -> None:
        """Monitor safety zones for violations."""
        for zone in self.device_spec.safety_zones:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, zone.site_name
            )
            if site_id < 0:
                continue

            # Check if any contact force in the zone exceeds threshold
            zone_pos = self.data.site_xpos[site_id]
            total_force = 0.0

            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                contact_pos = contact.pos
                dist = np.linalg.norm(contact_pos - zone_pos)

                if dist < 0.05:  # within 5cm of zone center
                    c_force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, c_force)
                    total_force += np.linalg.norm(c_force[:3])

            if total_force > zone.force_threshold:
                violation = {
                    "type": zone.hazard_type.value,
                    "force": total_force,
                    "threshold": zone.force_threshold,
                    "step": self._step_count,
                    "description": zone.description,
                }
                self._safety_violations.append(violation)

    def _compute_reward(self) -> float:
        """Compute reward based on task progress and safety."""
        if self.reward_type == "sparse":
            return self._sparse_reward()
        return self._dense_reward()

    def _sparse_reward(self) -> float:
        reward = 0.0
        # Penalty for safety violations
        if self._safety_violations:
            latest = self._safety_violations[-1]
            if latest["step"] == self._step_count:
                reward -= 10.0
        return reward

    def _dense_reward(self) -> float:
        reward = 0.0

        # Progress reward: distance to target component
        if self._target_component:
            comp = self.device_spec.get_component(self._target_component)
            if comp.site_name:
                site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, comp.site_name
                )
                ee_site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
                )
                if site_id >= 0 and ee_site_id >= 0:
                    dist = np.linalg.norm(
                        self.data.site_xpos[site_id] - self.data.site_xpos[ee_site_id]
                    )
                    reward -= dist  # encourage getting closer

        # Safety penalty
        for v in self._safety_violations:
            if v["step"] == self._step_count:
                reward -= 10.0

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate on catastrophic safety violation (battery puncture)
        for v in self._safety_violations:
            if v["type"] == HazardType.BATTERY_PUNCTURE.value:
                return True
        return False

    def _apply_domain_randomization(self) -> None:
        """Apply randomization to visual and physical properties."""
        rng = self.np_random

        # Randomize lighting
        for i in range(self.model.nlight):
            self.model.light_diffuse[i] = rng.uniform(0.3, 1.0, 3)
            self.model.light_pos[i] += rng.uniform(-0.2, 0.2, 3)

        # Randomize device body colors
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ("shell" in name or "panel" in name or "chassis" in name):
                base_color = self.model.geom_rgba[i, :3].copy()
                noise = rng.uniform(-0.1, 0.1, 3)
                self.model.geom_rgba[i, :3] = np.clip(base_color + noise, 0, 1)

        # Randomize friction
        for i in range(self.model.ngeom):
            self.model.geom_friction[i, 0] *= rng.uniform(0.8, 1.2)

    def _get_info(self) -> dict[str, Any]:
        return {
            "instruction": self._current_instruction,
            "target_component": self._target_component,
            "removed_components": list(self._removed_components),
            "safety_violations": copy.deepcopy(self._safety_violations),
            "step_count": self._step_count,
            "episode_reward": self._episode_reward,
            "device_name": self.device_spec.name,
        }

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._render_camera("overhead")
        elif self.render_mode == "human":
            # For human rendering, use MuJoCo viewer
            import mujoco.viewer
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    # ─────────────────────────────────────────────────────
    #  Visual component removal API
    # ─────────────────────────────────────────────────────

    # Maps (device_type, component_name) → list of (joint_name, target_value) to
    # animate on removal.  The target_value is the qpos to drive the joint toward;
    # the joint's range upper-bound works well for "fully removed".
    _REMOVAL_JOINT_MAP: dict[str, dict[str, list[tuple[str, float]]]] = {
        "laptop": {
            "screw_1": [("screw_1_turn", 31.0), ("screw_1_slide", 0.05)],
            "screw_2": [("screw_2_turn", 31.0), ("screw_2_slide", 0.05)],
            "screw_3": [("screw_3_turn", 31.0), ("screw_3_slide", 0.05)],
            "screw_4": [("screw_4_turn", 31.0), ("screw_4_slide", 0.05)],
            "screw_5": [("screw_5_turn", 31.0), ("screw_5_slide", 0.05)],
            "back_panel": [("panel_slide_z", 0.22)],
            "battery": [("battery_connector", 0.12)],
            "ram_module": [("ram_latch", 0.5), ("ram_slot", 0.07)],
            "ssd_module": [("ssd_screw_turn", 31.0), ("ssd_slot", 0.08)],
            "fan_assembly": [("fan_mount", 0.08)],
        },
        "router": {
            "clip_front": [("clip_front_j", 0.01)],
            "clip_back": [("clip_back_j", 0.01)],
            "clip_left": [("clip_left_j", 0.01)],
            "clip_right": [("clip_right_j", 0.01)],
            "hidden_screw_1": [("hscrew_1_turn", 31.0), ("hscrew_1_slide", 0.012)],
            "hidden_screw_2": [("hscrew_2_turn", 31.0), ("hscrew_2_slide", 0.012)],
            "top_cover": [("cover_lift", 0.08)],
            "cmos_battery": [("cmos_batt_conn", 0.01)],
            "antenna_conn_1": [("ant_conn_1", 6.28)],
            "antenna_conn_2": [("ant_conn_2", 6.28)],
        },
    }

    def visually_remove_component(
        self,
        component_name: str,
        animate_steps: int = 60,
        settle_steps: int = 20,
    ) -> list[np.ndarray]:
        """Animate the physical removal of *component_name* and return frames.

        This drives the device joints associated with the component toward
        their "removed" position over *animate_steps* simulation steps, then
        allows *settle_steps* for the physics to settle.

        Returns a list of overhead-camera RGB frames captured during animation
        (useful for recording cinematic demos).
        """
        device_type = self.device_spec.device_type
        joint_targets = self._REMOVAL_JOINT_MAP.get(device_type, {}).get(
            component_name, []
        )

        frames: list[np.ndarray] = []

        if not joint_targets:
            # No known joint mapping — just mark as removed
            self._removed_components.add(component_name)
            return frames

        # Resolve joint ids & compute per-step increments
        joint_info: list[tuple[int, int, float, float]] = []  # (jnt_id, qpos_idx, start, step_delta)
        for jname, target_val in joint_targets:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qidx = self.model.jnt_qposadr[jid]
            start_val = float(self.data.qpos[qidx])
            step_delta = (target_val - start_val) / max(animate_steps, 1)
            joint_info.append((jid, qidx, start_val, step_delta))

        # Animate
        for t in range(animate_steps):
            for _jid, qidx, start_val, step_delta in joint_info:
                self.data.qpos[qidx] = start_val + step_delta * (t + 1)
            mujoco.mj_step(self.model, self.data)
            # Capture every 3rd frame to keep array manageable
            if t % 3 == 0:
                frames.append(self._render_camera("overhead"))

        # Settle
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        frames.append(self._render_camera("overhead"))

        self._removed_components.add(component_name)
        return frames

    def render_camera(self, camera_name: str) -> np.ndarray:
        """Public wrapper for rendering from any named camera."""
        return self._render_camera(camera_name)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
