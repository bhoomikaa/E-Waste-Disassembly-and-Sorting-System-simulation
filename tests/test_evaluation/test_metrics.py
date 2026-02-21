"""Tests for evaluation metrics."""

import pytest

from safedisassemble.evaluation.metrics.disassembly_metrics import (
    DisassemblyEvaluator,
    EpisodeResult,
)


class TestDisassemblyEvaluator:
    def setup_method(self):
        self.evaluator = DisassemblyEvaluator()

    def test_empty_evaluator(self):
        summary = self.evaluator.compute()
        assert summary.task_completion_rate == 0.0

    def test_perfect_episode(self):
        result = EpisodeResult(
            device_name="laptop_v1",
            total_components=5,
            recovered_components=["screw_1", "panel", "battery", "ram", "ssd"],
            damaged_components=[],
            safety_violations=[],
            total_steps=100,
            total_reward=10.0,
            success=True,
            plan_was_safe=True,
            battery_disconnected_first=True,
        )
        self.evaluator.add_result(result)
        summary = self.evaluator.compute()

        assert summary.task_completion_rate == 1.0
        assert summary.component_recovery_rate == 1.0
        assert summary.safety_violation_rate == 0.0
        assert summary.battery_first_rate == 1.0

    def test_failed_episode_with_violation(self):
        result = EpisodeResult(
            device_name="laptop_v1",
            total_components=5,
            recovered_components=["screw_1"],
            damaged_components=["battery"],
            safety_violations=[{"type": "battery_puncture", "force": 20.0}],
            total_steps=50,
            total_reward=-10.0,
            success=False,
            plan_was_safe=False,
            battery_disconnected_first=False,
        )
        self.evaluator.add_result(result)
        summary = self.evaluator.compute()

        assert summary.task_completion_rate == 0.0
        assert summary.safety_violation_rate == 1.0
        assert summary.battery_puncture_rate == 1.0

    def test_generalization_split(self):
        seen_result = EpisodeResult(
            device_name="laptop_v1",
            total_components=3,
            recovered_components=["a", "b", "c"],
            damaged_components=[],
            safety_violations=[],
            total_steps=50,
            total_reward=5.0,
            success=True,
            plan_was_safe=True,
            battery_disconnected_first=True,
        )
        unseen_result = EpisodeResult(
            device_name="router_v1",
            total_components=3,
            recovered_components=["a"],
            damaged_components=[],
            safety_violations=[],
            total_steps=50,
            total_reward=1.0,
            success=False,
            plan_was_safe=True,
            battery_disconnected_first=True,
        )
        self.evaluator.add_result(seen_result)
        self.evaluator.add_result(unseen_result)

        summary = self.evaluator.compute(
            seen_devices={"laptop_v1"},
            unseen_devices={"router_v1"},
        )

        assert summary.seen_completion_rate == 1.0
        assert summary.unseen_completion_rate == 0.0
        assert summary.generalization_gap == 1.0

    def test_format_table(self):
        result = EpisodeResult(
            device_name="laptop_v1",
            total_components=3,
            recovered_components=["a", "b"],
            damaged_components=[],
            safety_violations=[],
            total_steps=80,
            total_reward=5.0,
            success=False,
            plan_was_safe=True,
            battery_disconnected_first=True,
        )
        self.evaluator.add_result(result)
        summary = self.evaluator.compute()
        table = self.evaluator.format_table(summary)

        assert "Task Completion Rate" in table
        assert "Safety Violation Rate" in table
