"""Hierarchical VLA Controller

Orchestrates the three-level architecture:
  Level 1 (Task Planner) → Disassembly plan
  Safety Module → Plan validation + runtime monitoring
  Level 2 (Skill Selector) → Skill ID + parameters
  Level 3 (Motor Policy) → Continuous action trajectories

This is the main inference entry point for the full system.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import torch

from safedisassemble.models.task_planner.planner import (
    DisassemblyPlan,
    TaskPlanner,
)
from safedisassemble.models.skill_selector.selector import (
    SKILL_TO_IDX,
    SkillParameters,
    SkillSelector,
)
from safedisassemble.models.motor_policy.diffusion_policy import DiffusionMotorPolicy
from safedisassemble.models.safety.constraint_checker import (
    SafetyAction,
    SafetyAssessment,
    SafetyConstraintModule,
    SafetyLevel,
)


class ControllerState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING_SKILL = "executing_skill"
    SAFETY_PAUSE = "safety_pause"
    REPLANNING = "replanning"
    COMPLETE = "complete"
    FAILED = "failed"


class HierarchicalController:
    """Full hierarchical VLA controller for e-waste disassembly.

    Usage:
        controller = HierarchicalController(device="cuda")
        controller.load_models(planner_path, selector_path, policy_path)

        # Start a new disassembly task
        controller.begin_task(image, "Disassemble this laptop", device_spec)

        # Main control loop
        while not controller.is_done:
            action = controller.step(observation)
            # ... execute action in environment ...
    """

    def __init__(
        self,
        device: str = "cuda",
        action_horizon: int = 16,
        action_execution_horizon: int = 8,
        max_replan_attempts: int = 3,
    ):
        self.device = device
        self.action_horizon = action_horizon
        self.action_exec_horizon = action_execution_horizon
        self.max_replan_attempts = max_replan_attempts

        # Models
        self.planner = TaskPlanner(device=device)
        self.skill_selector = SkillSelector().to(device)
        self.motor_policy = DiffusionMotorPolicy().to(device)
        self.safety_module = SafetyConstraintModule()

        # State
        self._state = ControllerState.IDLE
        self._plan: Optional[DisassemblyPlan] = None
        self._current_skill: Optional[SkillParameters] = None
        self._action_buffer: list[np.ndarray] = []
        self._buffer_idx: int = 0
        self._replan_count: int = 0
        self._step_count: int = 0
        self._completed_components: list[str] = []

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def is_done(self) -> bool:
        return self._state in (ControllerState.COMPLETE, ControllerState.FAILED)

    def load_models(
        self,
        planner_path: Optional[str] = None,
        selector_path: Optional[str] = None,
        policy_path: Optional[str] = None,
    ) -> None:
        """Load pretrained model weights."""
        if planner_path:
            self.planner.load_model()

        if selector_path:
            state = torch.load(selector_path, map_location=self.device, weights_only=True)
            self.skill_selector.load_state_dict(state)

        if policy_path:
            state = torch.load(policy_path, map_location=self.device, weights_only=True)
            self.motor_policy.load_state_dict(state)

        self.skill_selector.eval()
        self.motor_policy.eval()

    def begin_task(
        self,
        image: np.ndarray,
        instruction: str,
        device_spec=None,
        device_type_hint: Optional[str] = None,
    ) -> DisassemblyPlan:
        """Initialize a new disassembly task.

        Args:
            image: (H, W, 3) overhead image of the device
            instruction: High-level instruction
            device_spec: Optional DeviceSpec for safety setup
            device_type_hint: Optional device type hint

        Returns:
            The generated disassembly plan
        """
        self._state = ControllerState.PLANNING
        self._step_count = 0
        self._completed_components = []
        self._replan_count = 0

        # Reset safety module
        self.safety_module.reset()
        if device_spec:
            self.safety_module.setup_from_device_spec(device_spec)

        # Generate plan (Level 1)
        self._plan = self.planner.plan(
            image=image,
            instruction=instruction,
            device_type_hint=device_type_hint,
        )

        # Validate plan safety
        plan_data = []
        for step in self._plan.steps:
            plan_data.append({
                "component": step.component,
                "component_type": step.component,
                "step_id": step.step_id,
            })

        safety_check = self.safety_module.validate_plan(plan_data)

        if safety_check.action == SafetyAction.REPLAN:
            # Re-order plan to fix safety violations
            self._plan = self._reorder_for_safety(self._plan)

        self._state = ControllerState.EXECUTING_SKILL
        return self._plan

    def step(self, observation: dict) -> np.ndarray:
        """Execute one control step.

        Args:
            observation: Dict with keys:
                'image_wrist': (H, W, 3) uint8
                'image_overhead': (H, W, 3) uint8
                'joint_pos': (7,) float
                'joint_vel': (7,) float
                'gripper_pos': (2,) float
                'ee_pos': (3,) float
                'ee_force': (3,) float

        Returns:
            (7,) action array
        """
        self._step_count += 1

        # Runtime safety check
        safety_assessments = self.safety_module.check_runtime(
            ee_position=observation["ee_pos"],
            ee_force=observation["ee_force"],
        )

        for assessment in safety_assessments:
            if assessment.action == SafetyAction.ABORT:
                self._state = ControllerState.SAFETY_PAUSE
                # Return zero action (stop moving)
                return np.zeros(7, dtype=np.float32)
            elif assessment.action == SafetyAction.REDUCE_FORCE:
                # Scale down action magnitude
                pass  # handled below

        # If we have buffered actions, execute them
        if self._action_buffer and self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1

            # Apply force reduction if warned
            force_scale = 1.0
            for a in safety_assessments:
                if a.action in (SafetyAction.REDUCE_FORCE, SafetyAction.REDUCE_SPEED):
                    force_scale = 0.5
            action = action * force_scale

            return action

        # Need new actions — run Level 2 and Level 3
        if self._plan is None or self._plan.is_complete:
            self._state = ControllerState.COMPLETE
            return np.zeros(7, dtype=np.float32)

        current_step = self._plan.current_step
        if current_step is None:
            self._state = ControllerState.COMPLETE
            return np.zeros(7, dtype=np.float32)

        # Pre-action safety check
        pre_check = self.safety_module.check_pre_action(current_step.component)
        if not pre_check.is_safe:
            if pre_check.action == SafetyAction.REPLAN:
                self._handle_replan(observation)
                return np.zeros(7, dtype=np.float32)
            elif pre_check.action == SafetyAction.ABORT:
                self._state = ControllerState.FAILED
                return np.zeros(7, dtype=np.float32)

        # Level 2: Select skill
        image_tensor = self._prep_image(observation["image_wrist"])
        proprio = self._prep_proprioception(observation)
        instruction_tokens = self._tokenize(current_step.action_description)

        skill_params = self.skill_selector.predict(
            image_tensor, instruction_tokens, proprio
        )
        self._current_skill = skill_params

        # Level 3: Generate action trajectory
        skill_id_tensor = torch.tensor(
            [SKILL_TO_IDX.get(skill_params.skill_id, 0)],
            device=self.device,
        )
        skill_param_tensor = skill_params.to_tensor().unsqueeze(0).to(self.device)

        action_seq = self.motor_policy.predict_action(
            image_tensor,
            proprio,
            skill_id_tensor,
            skill_param_tensor,
        )  # (1, H, 7)

        # Buffer the action sequence
        actions = action_seq[0].cpu().numpy()  # (H, 7)
        self._action_buffer = [actions[i] for i in range(self.action_exec_horizon)]
        self._buffer_idx = 0

        # Return first action
        action = self._action_buffer[0]
        self._buffer_idx = 1
        return action.astype(np.float32)

    def notify_subtask_complete(self, component_name: str, component_type: str) -> None:
        """Called when a subtask (component removal) is completed."""
        self._completed_components.append(component_name)
        self.safety_module.notify_removal(component_name, component_type)

        if self._plan:
            self._plan.advance()

        # Clear action buffer for next subtask
        self._action_buffer = []
        self._buffer_idx = 0

    def _handle_replan(self, observation: dict) -> None:
        """Handle a replanning request."""
        self._replan_count += 1
        if self._replan_count > self.max_replan_attempts:
            self._state = ControllerState.FAILED
            return

        self._state = ControllerState.REPLANNING
        # Re-plan from current state
        image = observation.get("image_overhead", observation.get("image_wrist"))
        if image is not None:
            new_plan = self.planner.plan(
                image=image,
                instruction="Continue disassembly from current state",
            )
            self._plan = new_plan
            self._state = ControllerState.EXECUTING_SKILL

    def _reorder_for_safety(self, plan: DisassemblyPlan) -> DisassemblyPlan:
        """Reorder plan steps to ensure battery-first safety constraint."""
        battery_steps = []
        other_steps = []
        panel_steps = []
        screw_steps = []

        for step in plan.steps:
            comp = step.component.lower()
            if "battery" in comp:
                battery_steps.append(step)
            elif "panel" in comp or "cover" in comp:
                panel_steps.append(step)
            elif "screw" in comp or "clip" in comp:
                screw_steps.append(step)
            else:
                other_steps.append(step)

        # Safe ordering: screws/clips → panels → battery → everything else
        reordered = screw_steps + panel_steps + battery_steps + other_steps

        # Re-number steps
        for i, step in enumerate(reordered):
            step.step_id = i + 1

        return DisassemblyPlan(steps=reordered, device_type=plan.device_type)

    def _prep_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert observation image to model input tensor."""
        img = torch.from_numpy(image).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC -> CHW
        return img.unsqueeze(0).to(self.device)

    def _prep_proprioception(self, obs: dict) -> torch.Tensor:
        """Stack proprioceptive observations into a single vector."""
        proprio = np.concatenate([
            obs["joint_pos"],    # 7
            obs["joint_vel"],    # 7
            obs["gripper_pos"],  # 2
            obs["ee_pos"],       # 3
        ])
        return torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)

    def _tokenize(self, text: str, max_len: int = 64) -> torch.Tensor:
        """Simple whitespace tokenizer (replace with proper tokenizer in production)."""
        words = text.lower().split()
        # Simple hash-based token assignment
        tokens = [hash(w) % 9999 + 1 for w in words[:max_len]]
        # Pad
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def get_status(self) -> dict:
        """Get current controller status for logging/visualization."""
        return {
            "state": self._state.value,
            "step_count": self._step_count,
            "plan_progress": (
                f"{self._plan._current_idx}/{self._plan.num_steps}"
                if self._plan else "no plan"
            ),
            "current_skill": (
                self._current_skill.skill_id if self._current_skill else None
            ),
            "completed_components": self._completed_components,
            "replan_count": self._replan_count,
            "safety_summary": self.safety_module.get_safety_summary(),
            "action_buffer_remaining": (
                len(self._action_buffer) - self._buffer_idx
                if self._action_buffer else 0
            ),
        }
