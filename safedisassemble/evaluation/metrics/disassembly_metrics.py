"""Evaluation metrics for e-waste disassembly performance.

Metrics designed to match what matters in real-world recycling:
1. Task completion rate
2. Component recovery rate (weighted by economic value)
3. Safety violation rate
4. Generalization score (performance on unseen devices)
5. Efficiency (steps per component)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""
    device_name: str
    total_components: int
    recovered_components: list[str]
    damaged_components: list[str]
    safety_violations: list[dict]
    total_steps: int
    total_reward: float
    success: bool
    plan_was_safe: bool                  # did the plan pass safety validation
    battery_disconnected_first: bool     # was battery handled before internals


@dataclasses.dataclass
class MetricsSummary:
    """Aggregated metrics across multiple evaluation episodes."""
    # Task completion
    task_completion_rate: float          # fraction of episodes fully completed
    subtask_completion_rate: float       # fraction of subtasks completed across all episodes

    # Component recovery
    component_recovery_rate: float       # fraction of components recovered intact
    weighted_recovery_rate: float        # recovery weighted by component value
    recovery_by_type: dict[str, float]   # per-component-type recovery rate

    # Safety
    safety_violation_rate: float         # fraction of episodes with violations
    battery_puncture_rate: float         # fraction with battery punctures specifically
    avg_violations_per_episode: float    # mean violations per episode
    safe_plan_rate: float                # fraction of plans passing safety check
    battery_first_rate: float            # fraction correctly prioritizing battery

    # Efficiency
    avg_steps_per_component: float       # mean steps to extract one component
    avg_total_steps: float               # mean total steps per episode

    # Generalization (if split is available)
    seen_completion_rate: Optional[float] = None
    unseen_completion_rate: Optional[float] = None
    generalization_gap: Optional[float] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class DisassemblyEvaluator:
    """Compute evaluation metrics from a set of episode results.

    Designed for the paper's evaluation section. Computes all metrics
    needed for Table 1 (main results) and ablation studies.
    """

    def __init__(self, component_values: Optional[dict[str, float]] = None):
        self.component_values = component_values or {}
        self._results: list[EpisodeResult] = []

    def add_result(self, result: EpisodeResult) -> None:
        self._results.append(result)

    def clear(self) -> None:
        self._results = []

    def compute(
        self,
        seen_devices: Optional[set[str]] = None,
        unseen_devices: Optional[set[str]] = None,
    ) -> MetricsSummary:
        """Compute all metrics from collected results.

        Args:
            seen_devices: Set of device names used in training
            unseen_devices: Set of device names held out for generalization test
        """
        if not self._results:
            return self._empty_summary()

        n = len(self._results)

        # Task completion
        task_completion_rate = sum(1 for r in self._results if r.success) / n

        total_subtasks = sum(r.total_components for r in self._results)
        completed_subtasks = sum(len(r.recovered_components) for r in self._results)
        subtask_completion_rate = (
            completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
        )

        # Component recovery
        total_possible = sum(r.total_components for r in self._results)
        total_recovered = sum(len(r.recovered_components) for r in self._results)
        total_damaged = sum(len(r.damaged_components) for r in self._results)

        component_recovery_rate = (
            total_recovered / total_possible if total_possible > 0 else 0.0
        )

        # Weighted recovery (by component economic value)
        weighted_recovered = 0.0
        weighted_total = 0.0
        for r in self._results:
            for comp in r.recovered_components:
                weighted_recovered += self.component_values.get(comp, 1.0)
            weighted_total += sum(
                self.component_values.get(comp, 1.0)
                for comp in (r.recovered_components + r.damaged_components)
            ) + r.total_components  # approximate

        weighted_recovery_rate = (
            weighted_recovered / weighted_total if weighted_total > 0 else 0.0
        )

        # Per-type recovery
        recovery_by_type = self._compute_per_type_recovery()

        # Safety metrics
        episodes_with_violations = sum(
            1 for r in self._results if r.safety_violations
        )
        safety_violation_rate = episodes_with_violations / n

        battery_punctures = sum(
            1 for r in self._results
            if any(v.get("type") == "battery_puncture" for v in r.safety_violations)
        )
        battery_puncture_rate = battery_punctures / n

        avg_violations = np.mean([
            len(r.safety_violations) for r in self._results
        ])

        safe_plan_rate = sum(1 for r in self._results if r.plan_was_safe) / n
        battery_first_rate = sum(
            1 for r in self._results if r.battery_disconnected_first
        ) / n

        # Efficiency
        steps_per_component = []
        for r in self._results:
            if r.recovered_components:
                steps_per_component.append(
                    r.total_steps / len(r.recovered_components)
                )
        avg_steps_per_comp = float(np.mean(steps_per_component)) if steps_per_component else 0.0
        avg_total_steps = float(np.mean([r.total_steps for r in self._results]))

        # Generalization
        seen_rate = None
        unseen_rate = None
        gen_gap = None

        if seen_devices and unseen_devices:
            seen_results = [r for r in self._results if r.device_name in seen_devices]
            unseen_results = [r for r in self._results if r.device_name in unseen_devices]

            if seen_results:
                seen_rate = sum(1 for r in seen_results if r.success) / len(seen_results)
            if unseen_results:
                unseen_rate = sum(1 for r in unseen_results if r.success) / len(unseen_results)
            if seen_rate is not None and unseen_rate is not None:
                gen_gap = seen_rate - unseen_rate

        return MetricsSummary(
            task_completion_rate=task_completion_rate,
            subtask_completion_rate=subtask_completion_rate,
            component_recovery_rate=component_recovery_rate,
            weighted_recovery_rate=weighted_recovery_rate,
            recovery_by_type=recovery_by_type,
            safety_violation_rate=safety_violation_rate,
            battery_puncture_rate=battery_puncture_rate,
            avg_violations_per_episode=float(avg_violations),
            safe_plan_rate=safe_plan_rate,
            battery_first_rate=battery_first_rate,
            avg_steps_per_component=avg_steps_per_comp,
            avg_total_steps=avg_total_steps,
            seen_completion_rate=seen_rate,
            unseen_completion_rate=unseen_rate,
            generalization_gap=gen_gap,
        )

    def _compute_per_type_recovery(self) -> dict[str, float]:
        """Compute recovery rates grouped by component type keyword."""
        type_recovered: dict[str, int] = {}
        type_total: dict[str, int] = {}

        type_keywords = [
            "screw", "panel", "battery", "ram", "ssd", "fan",
            "heatsink", "connector", "clip", "antenna",
        ]

        for r in self._results:
            for comp in r.recovered_components:
                comp_lower = comp.lower()
                for kw in type_keywords:
                    if kw in comp_lower:
                        type_recovered[kw] = type_recovered.get(kw, 0) + 1
                        type_total[kw] = type_total.get(kw, 0) + 1
                        break

            for comp in r.damaged_components:
                comp_lower = comp.lower()
                for kw in type_keywords:
                    if kw in comp_lower:
                        type_total[kw] = type_total.get(kw, 0) + 1
                        break

        return {
            kw: type_recovered.get(kw, 0) / type_total[kw]
            for kw in type_total
            if type_total[kw] > 0
        }

    def _empty_summary(self) -> MetricsSummary:
        return MetricsSummary(
            task_completion_rate=0.0,
            subtask_completion_rate=0.0,
            component_recovery_rate=0.0,
            weighted_recovery_rate=0.0,
            recovery_by_type={},
            safety_violation_rate=0.0,
            battery_puncture_rate=0.0,
            avg_violations_per_episode=0.0,
            safe_plan_rate=0.0,
            battery_first_rate=0.0,
            avg_steps_per_component=0.0,
            avg_total_steps=0.0,
        )

    def format_table(self, summary: MetricsSummary) -> str:
        """Format metrics as a markdown table (for papers/reports)."""
        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Task Completion Rate | {summary.task_completion_rate:.1%} |",
            f"| Component Recovery Rate | {summary.component_recovery_rate:.1%} |",
            f"| Weighted Recovery Rate | {summary.weighted_recovery_rate:.1%} |",
            f"| Safety Violation Rate | {summary.safety_violation_rate:.1%} |",
            f"| Battery Puncture Rate | {summary.battery_puncture_rate:.1%} |",
            f"| Battery-First Rate | {summary.battery_first_rate:.1%} |",
            f"| Avg Steps/Component | {summary.avg_steps_per_component:.1f} |",
        ]
        if summary.seen_completion_rate is not None:
            lines.append(
                f"| Seen Device Completion | {summary.seen_completion_rate:.1%} |"
            )
        if summary.unseen_completion_rate is not None:
            lines.append(
                f"| Unseen Device Completion | {summary.unseen_completion_rate:.1%} |"
            )
        if summary.generalization_gap is not None:
            lines.append(
                f"| Generalization Gap | {summary.generalization_gap:.1%} |"
            )

        return "\n".join(lines)
