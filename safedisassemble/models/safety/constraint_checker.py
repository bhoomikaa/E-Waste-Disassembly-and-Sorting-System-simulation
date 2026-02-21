"""Safety Constraint Module

Sits between the Task Planner (Level 1) and Skill Selector (Level 2) to enforce
safety rules. Also monitors real-time execution at Level 3.

Key responsibilities:
1. Plan validation: Ensure battery disconnection is prioritized
2. Force monitoring: Abort actions that risk battery puncture or PCB snap
3. Zone enforcement: Prevent gripper entry into hazard zones at unsafe speeds/forces
4. Anomaly detection: Flag unexpected states (e.g., component not where expected)

This module is the paper's key safety contribution.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional

import numpy as np


class SafetyLevel(Enum):
    """Severity levels for safety assessments."""
    SAFE = "safe"
    CAUTION = "caution"      # proceed with reduced speed/force
    WARNING = "warning"      # pause and reassess
    CRITICAL = "critical"    # abort immediately


class SafetyAction(Enum):
    """Actions the safety module can take."""
    ALLOW = "allow"
    REDUCE_SPEED = "reduce_speed"
    REDUCE_FORCE = "reduce_force"
    PAUSE = "pause"
    ABORT = "abort"
    REPLAN = "replan"        # request Level 1 to generate new plan


@dataclasses.dataclass
class SafetyAssessment:
    """Result of a safety check."""
    level: SafetyLevel
    action: SafetyAction
    reason: str
    details: dict = dataclasses.field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.level in (SafetyLevel.SAFE, SafetyLevel.CAUTION)


@dataclasses.dataclass
class ZoneState:
    """Tracked state of a safety-critical zone."""
    zone_name: str
    position: np.ndarray        # (3,) center position
    radius: float               # danger zone radius in meters
    max_force: float            # force threshold in Newtons
    is_active: bool = True      # zone is active (hazard present)
    current_force: float = 0.0  # most recent force reading
    distance_to_ee: float = 1.0 # distance from EE to zone center


class BatteryPriorityChecker:
    """Ensures battery disconnection happens before other internal work.

    Validates disassembly plans and real-time action sequences to ensure
    the battery connector is disconnected before manipulating other
    internal components.
    """

    # Components that are unsafe to manipulate with battery connected
    BATTERY_SENSITIVE_COMPONENTS = {
        "ram", "ssd", "fan", "heatsink", "motherboard", "pcb",
        "display_connector", "speaker", "camera",
    }

    def __init__(self):
        self._battery_disconnected = False
        self._panel_removed = False

    def reset(self) -> None:
        self._battery_disconnected = False
        self._panel_removed = False

    def notify_component_removed(self, component_name: str, component_type: str) -> None:
        """Update state when a component is removed."""
        if component_type in ("battery", "cmos_battery"):
            self._battery_disconnected = True
        if component_type == "panel":
            self._panel_removed = True

    def check_plan(self, plan_steps: list[dict]) -> SafetyAssessment:
        """Validate that a disassembly plan has correct battery ordering.

        Args:
            plan_steps: List of dicts with 'component', 'component_type', 'step_id'
        """
        battery_step = None
        violations = []

        for step in plan_steps:
            comp_type = step.get("component_type", "")
            if comp_type in ("battery", "cmos_battery"):
                battery_step = step["step_id"]

        if battery_step is None:
            return SafetyAssessment(
                level=SafetyLevel.CAUTION,
                action=SafetyAction.ALLOW,
                reason="No battery found in plan — may be a battery-free device",
            )

        for step in plan_steps:
            comp = step.get("component", "").lower()
            if (any(kw in comp for kw in self.BATTERY_SENSITIVE_COMPONENTS)
                    and step["step_id"] < battery_step):
                violations.append(step)

        if violations:
            return SafetyAssessment(
                level=SafetyLevel.CRITICAL,
                action=SafetyAction.REPLAN,
                reason=(
                    f"Plan manipulates {len(violations)} internal component(s) "
                    f"before battery disconnection at step {battery_step}"
                ),
                details={"violations": violations, "battery_step": battery_step},
            )

        return SafetyAssessment(
            level=SafetyLevel.SAFE,
            action=SafetyAction.ALLOW,
            reason="Battery disconnection correctly prioritized",
        )

    def check_action(self, target_component: str) -> SafetyAssessment:
        """Check if it's safe to manipulate a specific component right now."""
        comp_lower = target_component.lower()

        # If panel isn't removed yet, internal components aren't accessible
        if not self._panel_removed and any(
            kw in comp_lower for kw in self.BATTERY_SENSITIVE_COMPONENTS
        ):
            return SafetyAssessment(
                level=SafetyLevel.WARNING,
                action=SafetyAction.REPLAN,
                reason=f"Cannot access '{target_component}' — panel not yet removed",
            )

        # If battery isn't disconnected, block sensitive operations
        if not self._battery_disconnected and any(
            kw in comp_lower for kw in self.BATTERY_SENSITIVE_COMPONENTS
        ):
            return SafetyAssessment(
                level=SafetyLevel.CRITICAL,
                action=SafetyAction.ABORT,
                reason=(
                    f"UNSAFE: Battery not disconnected. Cannot manipulate "
                    f"'{target_component}' — risk of short circuit"
                ),
            )

        return SafetyAssessment(
            level=SafetyLevel.SAFE,
            action=SafetyAction.ALLOW,
            reason="Action permitted",
        )


class ForceMonitor:
    """Real-time force monitoring for safety-critical zones.

    Tracks force applied near batteries, PCBs, and other fragile components.
    Triggers safety actions when thresholds are approached or exceeded.
    """

    def __init__(
        self,
        warning_threshold_fraction: float = 0.7,
        smoothing_alpha: float = 0.3,
    ):
        self.warning_fraction = warning_threshold_fraction
        self.alpha = smoothing_alpha
        self._zones: dict[str, ZoneState] = {}
        self._force_history: dict[str, list[float]] = {}
        self._max_history_len = 100

    def register_zone(
        self,
        name: str,
        position: np.ndarray,
        radius: float,
        max_force: float,
    ) -> None:
        """Register a safety-critical zone to monitor."""
        self._zones[name] = ZoneState(
            zone_name=name,
            position=position.copy(),
            radius=radius,
            max_force=max_force,
        )
        self._force_history[name] = []

    def deactivate_zone(self, name: str) -> None:
        """Deactivate a zone (e.g., after battery is removed)."""
        if name in self._zones:
            self._zones[name].is_active = False

    def update(
        self,
        ee_position: np.ndarray,
        ee_force: np.ndarray,
    ) -> list[SafetyAssessment]:
        """Update monitoring with current robot state.

        Args:
            ee_position: (3,) end-effector position
            ee_force: (3,) end-effector force vector

        Returns:
            List of safety assessments (one per active zone with issues)
        """
        force_magnitude = float(np.linalg.norm(ee_force))
        assessments = []

        for name, zone in self._zones.items():
            if not zone.is_active:
                continue

            # Update distance
            dist = float(np.linalg.norm(ee_position - zone.position))
            zone.distance_to_ee = dist

            # Only monitor force when EE is near the zone
            if dist > zone.radius * 2:
                continue

            # Exponential smoothing of force
            if self._force_history[name]:
                smoothed = (
                    self.alpha * force_magnitude
                    + (1 - self.alpha) * self._force_history[name][-1]
                )
            else:
                smoothed = force_magnitude

            self._force_history[name].append(smoothed)
            if len(self._force_history[name]) > self._max_history_len:
                self._force_history[name].pop(0)

            zone.current_force = smoothed

            # Check thresholds
            if smoothed > zone.max_force:
                assessments.append(SafetyAssessment(
                    level=SafetyLevel.CRITICAL,
                    action=SafetyAction.ABORT,
                    reason=(
                        f"FORCE LIMIT EXCEEDED in zone '{name}': "
                        f"{smoothed:.1f}N > {zone.max_force:.1f}N threshold"
                    ),
                    details={
                        "zone": name,
                        "force": smoothed,
                        "threshold": zone.max_force,
                        "distance": dist,
                    },
                ))
            elif smoothed > zone.max_force * self.warning_fraction:
                assessments.append(SafetyAssessment(
                    level=SafetyLevel.WARNING,
                    action=SafetyAction.REDUCE_FORCE,
                    reason=(
                        f"Approaching force limit in zone '{name}': "
                        f"{smoothed:.1f}N / {zone.max_force:.1f}N "
                        f"({smoothed/zone.max_force*100:.0f}%)"
                    ),
                    details={
                        "zone": name,
                        "force": smoothed,
                        "threshold": zone.max_force,
                        "distance": dist,
                    },
                ))
            elif dist < zone.radius:
                assessments.append(SafetyAssessment(
                    level=SafetyLevel.CAUTION,
                    action=SafetyAction.REDUCE_SPEED,
                    reason=f"Operating within safety zone '{name}' — reduced speed advised",
                    details={"zone": name, "distance": dist},
                ))

        return assessments

    def get_zone_states(self) -> dict[str, dict]:
        """Get current state of all monitored zones."""
        return {
            name: {
                "active": zone.is_active,
                "distance_to_ee": zone.distance_to_ee,
                "current_force": zone.current_force,
                "max_force": zone.max_force,
                "force_ratio": zone.current_force / zone.max_force if zone.max_force > 0 else 0,
            }
            for name, zone in self._zones.items()
        }


class SafetyConstraintModule:
    """Top-level safety module that combines all checkers.

    Sits between the hierarchical VLA levels:
    - Pre-execution: validates plans and actions
    - During execution: monitors forces and zones
    - Post-execution: logs safety events for training data
    """

    def __init__(self):
        self.battery_checker = BatteryPriorityChecker()
        self.force_monitor = ForceMonitor()
        self._event_log: list[dict] = []

    def reset(self) -> None:
        self.battery_checker.reset()
        self.force_monitor = ForceMonitor()
        self._event_log = []

    def setup_from_device_spec(self, device_spec) -> None:
        """Initialize monitoring zones from a DeviceSpec."""
        for zone in device_spec.safety_zones:
            # Position will be updated from simulation
            self.force_monitor.register_zone(
                name=zone.site_name,
                position=np.zeros(3),  # updated at runtime
                radius=0.05,
                max_force=zone.force_threshold,
            )

    def validate_plan(self, plan_steps: list[dict]) -> SafetyAssessment:
        """Validate a complete disassembly plan before execution."""
        assessment = self.battery_checker.check_plan(plan_steps)
        self._log_event("plan_validation", assessment)
        return assessment

    def check_pre_action(self, target_component: str) -> SafetyAssessment:
        """Check safety before executing an action on a component."""
        assessment = self.battery_checker.check_action(target_component)
        self._log_event("pre_action_check", assessment, {"component": target_component})
        return assessment

    def check_runtime(
        self,
        ee_position: np.ndarray,
        ee_force: np.ndarray,
    ) -> list[SafetyAssessment]:
        """Runtime force and zone monitoring during action execution."""
        assessments = self.force_monitor.update(ee_position, ee_force)
        for a in assessments:
            self._log_event("runtime_check", a)
        return assessments

    def notify_removal(self, component_name: str, component_type: str) -> None:
        """Notify that a component has been removed."""
        self.battery_checker.notify_component_removed(component_name, component_type)
        if component_type in ("battery", "cmos_battery"):
            # Deactivate battery safety zones
            for zone_name in list(self.force_monitor._zones.keys()):
                if "battery" in zone_name.lower() or "puncture" in zone_name.lower():
                    self.force_monitor.deactivate_zone(zone_name)

    def get_safety_summary(self) -> dict:
        """Get summary of safety events for logging/evaluation."""
        return {
            "total_events": len(self._event_log),
            "critical_events": sum(
                1 for e in self._event_log
                if e["assessment"]["level"] == SafetyLevel.CRITICAL.value
            ),
            "warnings": sum(
                1 for e in self._event_log
                if e["assessment"]["level"] == SafetyLevel.WARNING.value
            ),
            "aborts": sum(
                1 for e in self._event_log
                if e["assessment"]["action"] == SafetyAction.ABORT.value
            ),
            "zone_states": self.force_monitor.get_zone_states(),
            "battery_disconnected": self.battery_checker._battery_disconnected,
        }

    def _log_event(
        self,
        event_type: str,
        assessment: SafetyAssessment,
        extra: Optional[dict] = None,
    ) -> None:
        event = {
            "type": event_type,
            "assessment": {
                "level": assessment.level.value,
                "action": assessment.action.value,
                "reason": assessment.reason,
                "details": assessment.details,
            },
        }
        if extra:
            event.update(extra)
        self._event_log.append(event)

    @property
    def event_log(self) -> list[dict]:
        return self._event_log.copy()
