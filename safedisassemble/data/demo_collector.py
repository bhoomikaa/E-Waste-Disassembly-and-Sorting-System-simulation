"""Demonstration data collection via scripted policies.

For a solo project, scripted demonstrations are more practical than teleop.
This module generates expert demonstrations by executing known disassembly
sequences in simulation using waypoint-based motion planning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from safedisassemble.data.trajectory import Timestep, Trajectory
from safedisassemble.sim.device_registry import DeviceSpec, get_device


class ScriptedDisassemblyPolicy:
    """Generates expert demonstrations via scripted waypoint sequences.

    Each skill (unscrew, pry, pull, etc.) is implemented as a sequence of
    end-effector waypoints with gripper commands. The policy chains skills
    according to the device's dependency graph.
    """

    # Skill primitives: each returns a list of (delta_pos, delta_rot, gripper) waypoints
    SKILLS = {
        "approach": "approach_target",
        "unscrew": "execute_unscrew",
        "pry_open": "execute_pry",
        "pull_connector": "execute_pull",
        "lift_component": "execute_lift",
        "release_clip": "execute_release_clip",
    }

    def __init__(
        self,
        device_spec: DeviceSpec,
        noise_std: float = 0.002,
        seed: Optional[int] = None,
    ):
        self.device_spec = device_spec
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def generate_plan(self) -> list[dict]:
        """Generate a valid disassembly plan respecting dependencies.

        Returns ordered list of actions with component, skill, and language annotations.
        """
        plan = []
        removed = set()
        removable = {c.name for c in self.device_spec.removable_components}

        # Topological sort based on dependencies
        while removable - removed:
            # Find components whose prerequisites are all met
            ready = []
            for comp_name in removable - removed:
                prereqs = self.device_spec.get_prerequisites(comp_name)
                if all(p in removed for p in prereqs):
                    ready.append(comp_name)

            if not ready:
                break  # Remaining components have unmet dependencies

            # Prioritize battery disconnection (safety constraint)
            battery_comps = [
                c for c in ready
                if self.device_spec.get_component(c).component_type.value in
                ("battery", "cmos_battery")
            ]
            if battery_comps:
                next_comp = battery_comps[0]
            else:
                next_comp = ready[0]

            comp = self.device_spec.get_component(next_comp)

            # Determine skill type
            skill = self._component_to_skill(comp)

            # Generate hierarchical language instructions
            instructions = self._generate_instructions(comp, skill)

            plan.append({
                "component": next_comp,
                "skill": skill,
                "instructions": instructions,
                "component_spec": comp,
            })
            removed.add(next_comp)

        return plan

    def _component_to_skill(self, comp) -> str:
        """Map component type to the appropriate disassembly skill."""
        from safedisassemble.sim.device_registry import ComponentType

        mapping = {
            ComponentType.SCREW: "unscrew",
            ComponentType.PANEL: "pry_open",
            ComponentType.BATTERY: "pull_connector",
            ComponentType.RAM: "lift_component",
            ComponentType.SSD: "lift_component",
            ComponentType.FAN: "lift_component",
            ComponentType.HEATSINK: "lift_component",
            ComponentType.CONNECTOR: "pull_connector",
            ComponentType.CLIP: "release_clip",
            ComponentType.ANTENNA: "pull_connector",
            ComponentType.CMOS_BATTERY: "lift_component",
        }
        return mapping.get(comp.component_type, "lift_component")

    def _generate_instructions(self, comp, skill: str) -> dict:
        """Generate hierarchical language instructions for a step."""
        device_type = self.device_spec.device_type
        comp_type = comp.component_type.value

        high = f"Disassemble this {device_type}"

        mid_templates = {
            "unscrew": f"Remove {comp.name.replace('_', ' ')}",
            "pry_open": f"Open the {comp.name.replace('_', ' ')}",
            "pull_connector": f"Disconnect the {comp.name.replace('_', ' ')}",
            "lift_component": f"Extract the {comp.name.replace('_', ' ')}",
            "release_clip": f"Release the {comp.name.replace('_', ' ')}",
        }
        mid = mid_templates.get(skill, f"Remove {comp.name}")

        low_templates = {
            "unscrew": f"Rotate {comp.name.replace('_', ' ')} counterclockwise until loose, then lift out",
            "pry_open": f"Insert tool at edge of {comp.name.replace('_', ' ')} and lever upward gently",
            "pull_connector": f"Grip the {comp.name.replace('_', ' ')} connector tab and pull straight out",
            "lift_component": f"Grip {comp.name.replace('_', ' ')} by edges and lift vertically",
            "release_clip": f"Push {comp.name.replace('_', ' ')} outward to disengage latch",
        }
        low = low_templates.get(skill, f"Remove {comp.name}")

        return {"high": high, "mid": mid, "low": low}

    def generate_waypoints(self, target_pos: np.ndarray, skill: str) -> list[dict]:
        """Generate end-effector waypoints for a skill execution.

        Args:
            target_pos: (3,) position of the target component
            skill: skill name

        Returns:
            List of waypoint dicts with 'delta_pos', 'delta_rot', 'gripper', 'steps'
        """
        noise = lambda: self.rng.normal(0, self.noise_std, 3)

        waypoints = []

        if skill == "approach":
            # Move above target, then descend
            above = target_pos.copy()
            above[2] += 0.05
            waypoints.append({
                "target_pos": above + noise(),
                "gripper": 1.0,  # open
                "steps": 30,
                "phase": "approach_above",
            })
            waypoints.append({
                "target_pos": target_pos + noise(),
                "gripper": 1.0,
                "steps": 20,
                "phase": "descend",
            })

        elif skill == "unscrew":
            # Approach, engage, rotate, lift
            above = target_pos.copy()
            above[2] += 0.03
            waypoints.extend([
                {"target_pos": above + noise(), "gripper": 1.0, "steps": 20,
                 "phase": "approach"},
                {"target_pos": target_pos + noise(), "gripper": 0.2, "steps": 15,
                 "phase": "engage"},
                # Simulate rotation via small circular motions
                {"target_pos": target_pos + np.array([0.002, 0, 0]) + noise(),
                 "gripper": 0.1, "steps": 10, "phase": "rotate_1"},
                {"target_pos": target_pos + np.array([0, 0.002, 0]) + noise(),
                 "gripper": 0.1, "steps": 10, "phase": "rotate_2"},
                {"target_pos": target_pos + np.array([-0.002, 0, 0]) + noise(),
                 "gripper": 0.1, "steps": 10, "phase": "rotate_3"},
                {"target_pos": target_pos + np.array([0, -0.002, 0]) + noise(),
                 "gripper": 0.1, "steps": 10, "phase": "rotate_4"},
                # Lift screw out
                {"target_pos": target_pos + np.array([0, 0, 0.03]) + noise(),
                 "gripper": 0.1, "steps": 15, "phase": "extract"},
                # Move to discard
                {"target_pos": np.array([0.3, 0.3, 0.6]) + noise(),
                 "gripper": 1.0, "steps": 20, "phase": "discard"},
            ])

        elif skill == "pry_open":
            edge = target_pos.copy()
            edge[1] -= 0.05  # approach from edge
            waypoints.extend([
                {"target_pos": edge + np.array([0, 0, 0.02]) + noise(),
                 "gripper": 0.3, "steps": 20, "phase": "position_tool"},
                {"target_pos": edge + noise(), "gripper": 0.1, "steps": 15,
                 "phase": "insert"},
                {"target_pos": edge + np.array([0, 0, 0.02]) + noise(),
                 "gripper": 0.1, "steps": 20, "phase": "lever_up"},
                {"target_pos": target_pos + np.array([0, 0, 0.08]) + noise(),
                 "gripper": 0.3, "steps": 25, "phase": "lift_panel"},
                {"target_pos": np.array([0.3, -0.3, 0.6]) + noise(),
                 "gripper": 1.0, "steps": 20, "phase": "set_aside"},
            ])

        elif skill == "pull_connector":
            waypoints.extend([
                {"target_pos": target_pos + np.array([0, 0, 0.02]) + noise(),
                 "gripper": 1.0, "steps": 20, "phase": "approach"},
                {"target_pos": target_pos + noise(), "gripper": 0.0, "steps": 15,
                 "phase": "grip"},
                {"target_pos": target_pos + np.array([0.02, 0, 0]) + noise(),
                 "gripper": 0.0, "steps": 20, "phase": "pull"},
                {"target_pos": target_pos + np.array([0.02, 0, 0.05]) + noise(),
                 "gripper": 0.0, "steps": 15, "phase": "extract"},
                {"target_pos": np.array([0.3, 0.2, 0.6]) + noise(),
                 "gripper": 1.0, "steps": 20, "phase": "place"},
            ])

        elif skill == "lift_component":
            waypoints.extend([
                {"target_pos": target_pos + np.array([0, 0, 0.02]) + noise(),
                 "gripper": 1.0, "steps": 20, "phase": "approach"},
                {"target_pos": target_pos + noise(), "gripper": 0.0, "steps": 15,
                 "phase": "grip"},
                {"target_pos": target_pos + np.array([0, 0, 0.08]) + noise(),
                 "gripper": 0.0, "steps": 25, "phase": "lift"},
                {"target_pos": np.array([0.3, 0.2, 0.6]) + noise(),
                 "gripper": 1.0, "steps": 20, "phase": "place"},
            ])

        elif skill == "release_clip":
            # Push clip outward
            waypoints.extend([
                {"target_pos": target_pos + np.array([0, 0, 0.01]) + noise(),
                 "gripper": 0.5, "steps": 15, "phase": "approach"},
                {"target_pos": target_pos + noise(), "gripper": 0.2, "steps": 10,
                 "phase": "contact"},
                # Push outward (direction depends on clip orientation - simplified)
                {"target_pos": target_pos + np.array([0, 0.01, 0]) + noise(),
                 "gripper": 0.2, "steps": 15, "phase": "push"},
            ])

        return waypoints

    def collect_trajectory(self, env) -> Trajectory:
        """Run a complete scripted disassembly and collect the trajectory.

        Args:
            env: DisassemblyEnv instance

        Returns:
            Trajectory with all timesteps and annotations
        """
        plan = self.generate_plan()

        obs, info = env.reset(options={
            "instruction": f"Disassemble this {self.device_spec.device_type}",
        })

        timesteps = []
        all_violations = []
        recovered = []

        for step_info in plan:
            comp = step_info["component_spec"]
            skill = step_info["skill"]
            instructions = step_info["instructions"]

            # Get target position (from site if available)
            if comp.site_name:
                import mujoco
                site_id = mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_SITE, comp.site_name
                )
                if site_id >= 0:
                    target_pos = env.data.site_xpos[site_id].copy()
                else:
                    target_pos = np.array([0.5, 0, 0.45])
            else:
                target_pos = np.array([0.5, 0, 0.45])

            waypoints = self.generate_waypoints(target_pos, skill)

            for wp in waypoints:
                # Convert waypoint to action
                ee_site_id = mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
                )
                current_ee = env.data.site_xpos[ee_site_id].copy()
                direction = wp["target_pos"] - current_ee
                dist = np.linalg.norm(direction)

                for s in range(wp["steps"]):
                    # Proportional action toward waypoint
                    ee_pos = env.data.site_xpos[ee_site_id].copy()
                    error = wp["target_pos"] - ee_pos
                    action_pos = np.clip(error / self._pos_scale_factor(), -1, 1)
                    action_rot = np.zeros(3)
                    gripper_action = wp["gripper"] * 2.0 - 1.0  # map [0,1] to [-1,1]
                    action = np.concatenate([action_pos, action_rot, [gripper_action]])
                    action = action.astype(np.float32)

                    obs, reward, terminated, truncated, info = env.step(action)

                    ts = Timestep(
                        image_wrist=obs["image_wrist"],
                        image_overhead=obs["image_overhead"],
                        joint_pos=obs["joint_pos"],
                        joint_vel=obs["joint_vel"],
                        gripper_pos=obs["gripper_pos"],
                        ee_pos=obs["ee_pos"],
                        ee_force=obs["ee_force"],
                        action=action,
                        instruction_high=instructions["high"],
                        instruction_mid=instructions["mid"],
                        instruction_low=instructions["low"],
                        skill_id=skill,
                        is_terminal=terminated or truncated,
                        safety_violation=len(info["safety_violations"]) > len(all_violations),
                    )
                    timesteps.append(ts)
                    all_violations = info["safety_violations"]

                    if terminated or truncated:
                        break

                if terminated or truncated:
                    break

            recovered.append(step_info["component"])

            if terminated or truncated:
                break

        return Trajectory(
            device_name=self.device_spec.name,
            timesteps=timesteps,
            success=not any(v for v in all_violations),
            total_reward=info.get("episode_reward", 0.0),
            components_recovered=recovered,
            safety_violations=all_violations,
        )

    def _pos_scale_factor(self) -> float:
        return 0.02
