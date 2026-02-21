"""Tests for the safety constraint module."""

import numpy as np
import pytest

from safedisassemble.models.safety.constraint_checker import (
    BatteryPriorityChecker,
    ForceMonitor,
    SafetyAction,
    SafetyConstraintModule,
    SafetyLevel,
)


class TestBatteryPriorityChecker:
    def setup_method(self):
        self.checker = BatteryPriorityChecker()

    def test_valid_plan_passes(self):
        plan = [
            {"component": "screw_1", "component_type": "screw", "step_id": 1},
            {"component": "back_panel", "component_type": "panel", "step_id": 2},
            {"component": "battery", "component_type": "battery", "step_id": 3},
            {"component": "ram", "component_type": "ram", "step_id": 4},
        ]
        result = self.checker.check_plan(plan)
        assert result.level == SafetyLevel.SAFE

    def test_unsafe_plan_detected(self):
        plan = [
            {"component": "back_panel", "component_type": "panel", "step_id": 1},
            {"component": "ram", "component_type": "ram", "step_id": 2},  # before battery!
            {"component": "battery", "component_type": "battery", "step_id": 3},
        ]
        result = self.checker.check_plan(plan)
        assert result.level == SafetyLevel.CRITICAL
        assert result.action == SafetyAction.REPLAN

    def test_no_battery_is_caution(self):
        plan = [
            {"component": "screw_1", "component_type": "screw", "step_id": 1},
            {"component": "cover", "component_type": "panel", "step_id": 2},
        ]
        result = self.checker.check_plan(plan)
        assert result.level == SafetyLevel.CAUTION

    def test_action_blocked_without_panel_removed(self):
        # Panel not removed yet → WARNING (can't access internals)
        result = self.checker.check_action("ram")
        assert result.level == SafetyLevel.WARNING
        assert result.action == SafetyAction.REPLAN

    def test_action_blocked_without_battery_disconnect(self):
        # Panel removed but battery not disconnected → CRITICAL
        self.checker.notify_component_removed("back_panel", "panel")
        result = self.checker.check_action("ram")
        assert result.level == SafetyLevel.CRITICAL
        assert result.action == SafetyAction.ABORT

    def test_action_allowed_after_battery_disconnect(self):
        self.checker.notify_component_removed("back_panel", "panel")
        self.checker.notify_component_removed("battery", "battery")
        result = self.checker.check_action("ram")
        assert result.is_safe

    def test_screw_always_allowed(self):
        result = self.checker.check_action("screw_1")
        assert result.is_safe


class TestForceMonitor:
    def setup_method(self):
        self.monitor = ForceMonitor()
        self.monitor.register_zone(
            name="battery_zone",
            position=np.array([0.5, 0.0, 0.44]),
            radius=0.06,
            max_force=15.0,
        )

    def test_safe_when_far(self):
        assessments = self.monitor.update(
            ee_position=np.array([0.0, 0.0, 0.8]),
            ee_force=np.array([0.0, 0.0, 0.0]),
        )
        assert len(assessments) == 0

    def test_caution_when_near(self):
        assessments = self.monitor.update(
            ee_position=np.array([0.5, 0.0, 0.44]),  # inside zone
            ee_force=np.array([0.0, 0.0, 1.0]),       # low force
        )
        assert len(assessments) > 0
        assert assessments[0].level == SafetyLevel.CAUTION

    def test_critical_when_force_exceeded(self):
        # Build up force history
        for _ in range(5):
            self.monitor.update(
                ee_position=np.array([0.5, 0.0, 0.44]),
                ee_force=np.array([0.0, 0.0, 20.0]),  # over threshold
            )
        assessments = self.monitor.update(
            ee_position=np.array([0.5, 0.0, 0.44]),
            ee_force=np.array([0.0, 0.0, 20.0]),
        )
        critical = [a for a in assessments if a.level == SafetyLevel.CRITICAL]
        assert len(critical) > 0

    def test_zone_deactivation(self):
        self.monitor.deactivate_zone("battery_zone")
        assessments = self.monitor.update(
            ee_position=np.array([0.5, 0.0, 0.44]),
            ee_force=np.array([0.0, 0.0, 20.0]),
        )
        assert len(assessments) == 0


class TestSafetyConstraintModule:
    def test_full_workflow(self):
        module = SafetyConstraintModule()

        # Validate a good plan
        good_plan = [
            {"component": "screw", "component_type": "screw", "step_id": 1},
            {"component": "panel", "component_type": "panel", "step_id": 2},
            {"component": "battery", "component_type": "battery", "step_id": 3},
            {"component": "ssd", "component_type": "ssd", "step_id": 4},
        ]
        result = module.validate_plan(good_plan)
        assert result.level == SafetyLevel.SAFE

        # Notify removals in order
        module.notify_removal("panel", "panel")
        module.notify_removal("battery", "battery")

        # Now SSD is safe to remove
        result = module.check_pre_action("ssd")
        assert result.is_safe

        # Check summary
        summary = module.get_safety_summary()
        assert summary["battery_disconnected"] is True
