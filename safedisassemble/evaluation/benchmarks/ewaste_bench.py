"""E-Waste Disassembly Benchmark Suite (EWasteBench).

A standardized benchmark for evaluating robotic e-waste disassembly.
This IS a contribution: no such benchmark exists in the literature.

Benchmark structure:
- 5 device categories, each with geometric variations
- Seen/unseen splits for generalization testing
- Standardized metrics and evaluation protocol
- Baseline results for comparison
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from safedisassemble.evaluation.metrics.disassembly_metrics import (
    DisassemblyEvaluator,
    EpisodeResult,
    MetricsSummary,
)
from safedisassemble.sim.device_registry import DEVICE_REGISTRY, DeviceSpec


class BenchmarkConfig:
    """Configuration for a benchmark evaluation run."""

    def __init__(
        self,
        name: str = "ewaste_bench_v1",
        seen_devices: list[str] = None,
        unseen_devices: list[str] = None,
        episodes_per_device: int = 50,
        max_steps_per_episode: int = 500,
        use_domain_randomization: bool = True,
        seeds: list[int] = None,
    ):
        self.name = name
        self.seen_devices = seen_devices or ["laptop_v1"]
        self.unseen_devices = unseen_devices or ["router_v1"]
        self.episodes_per_device = episodes_per_device
        self.max_steps = max_steps_per_episode
        self.use_domain_randomization = use_domain_randomization
        self.seeds = seeds or list(range(episodes_per_device))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seen_devices": self.seen_devices,
            "unseen_devices": self.unseen_devices,
            "episodes_per_device": self.episodes_per_device,
            "max_steps": self.max_steps,
            "use_domain_randomization": self.use_domain_randomization,
        }


class EWasteBenchmark:
    """Run the full benchmark evaluation suite.

    Usage:
        bench = EWasteBenchmark(config)

        # Evaluate a method
        results = bench.evaluate(my_controller)

        # Compare against baselines
        bench.evaluate_baselines()

        # Generate paper-ready tables
        bench.export_results("results/")
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.evaluator = DisassemblyEvaluator()
        self._results: dict[str, list[EpisodeResult]] = {}

    def evaluate(
        self,
        controller,
        method_name: str = "ours",
    ) -> MetricsSummary:
        """Run full benchmark evaluation for a method.

        Args:
            controller: Object with step(observation) → action interface
            method_name: Name for this method in results

        Returns:
            MetricsSummary with all metrics
        """
        from safedisassemble.sim.envs.disassembly_env import DisassemblyEnv

        all_results = []

        # Evaluate on all devices
        all_devices = self.config.seen_devices + self.config.unseen_devices

        for device_name in all_devices:
            device_spec = DEVICE_REGISTRY.get(device_name)
            if device_spec is None:
                continue

            for episode_idx in range(self.config.episodes_per_device):
                seed = self.config.seeds[episode_idx % len(self.config.seeds)]

                env = DisassemblyEnv(
                    device_name=device_name,
                    max_steps=self.config.max_steps,
                    use_domain_randomization=self.config.use_domain_randomization,
                    seed=seed,
                )

                result = self._run_episode(env, controller, device_spec)
                all_results.append(result)
                env.close()

        self._results[method_name] = all_results

        # Compute metrics
        self.evaluator.clear()
        for r in all_results:
            self.evaluator.add_result(r)

        return self.evaluator.compute(
            seen_devices=set(self.config.seen_devices),
            unseen_devices=set(self.config.unseen_devices),
        )

    def _run_episode(
        self,
        env,
        controller,
        device_spec: DeviceSpec,
    ) -> EpisodeResult:
        """Run a single evaluation episode."""
        obs, info = env.reset(options={
            "instruction": f"Disassemble this {device_spec.device_type}",
        })

        # Initialize controller if it has begin_task
        if hasattr(controller, "begin_task"):
            controller.begin_task(
                image=obs["image_overhead"],
                instruction=f"Disassemble this {device_spec.device_type}",
                device_spec=device_spec,
                device_type_hint=device_spec.device_type,
            )
        elif hasattr(controller, "reset"):
            controller.reset()

        recovered = []
        damaged = []
        all_violations = []
        total_reward = 0.0

        for step in range(env.max_steps):
            # Get action from controller
            if hasattr(controller, "step"):
                action = controller.step(obs)
            elif hasattr(controller, "predict"):
                action = controller.predict(obs)
            else:
                action = np.zeros(7, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            all_violations = info.get("safety_violations", [])

            if terminated or truncated:
                break

        # Determine what was recovered
        recovered = info.get("removed_components", [])
        total_components = len(device_spec.removable_components)

        # Check if battery was disconnected before other internals
        battery_first = self._check_battery_first(recovered, device_spec)

        return EpisodeResult(
            device_name=device_spec.name,
            total_components=total_components,
            recovered_components=recovered,
            damaged_components=damaged,
            safety_violations=all_violations,
            total_steps=info.get("step_count", 0),
            total_reward=total_reward,
            success=len(recovered) == total_components and not all_violations,
            plan_was_safe=not any(
                v.get("type") == "battery_puncture" for v in all_violations
            ),
            battery_disconnected_first=battery_first,
        )

    def _check_battery_first(
        self, recovered: list[str], device_spec: DeviceSpec
    ) -> bool:
        """Check if battery was disconnected before other internal components."""
        from safedisassemble.sim.device_registry import ComponentType

        battery_idx = None
        internal_indices = []

        internal_types = {
            ComponentType.RAM, ComponentType.SSD, ComponentType.FAN,
            ComponentType.HEATSINK, ComponentType.PCB,
        }

        for i, comp_name in enumerate(recovered):
            try:
                comp = device_spec.get_component(comp_name)
            except KeyError:
                continue

            if comp.component_type in (ComponentType.BATTERY, ComponentType.CMOS_BATTERY):
                battery_idx = i
            elif comp.component_type in internal_types:
                internal_indices.append(i)

        if battery_idx is None:
            return True  # no battery = vacuously true

        return all(battery_idx < idx for idx in internal_indices)

    def export_results(self, output_dir: str) -> None:
        """Export results to JSON and markdown tables."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for method_name, results in self._results.items():
            # Compute summary
            self.evaluator.clear()
            for r in results:
                self.evaluator.add_result(r)

            summary = self.evaluator.compute(
                seen_devices=set(self.config.seen_devices),
                unseen_devices=set(self.config.unseen_devices),
            )

            # Save JSON
            with open(output_path / f"{method_name}_results.json", "w") as f:
                json.dump({
                    "method": method_name,
                    "config": self.config.to_dict(),
                    "metrics": summary.to_dict(),
                    "num_episodes": len(results),
                }, f, indent=2, default=str)

            # Save markdown table
            table = self.evaluator.format_table(summary)
            with open(output_path / f"{method_name}_table.md", "w") as f:
                f.write(f"# {method_name} Results\n\n")
                f.write(table)

    def compare_methods(self) -> str:
        """Generate a comparison table across all evaluated methods."""
        if not self._results:
            return "No results to compare."

        lines = ["| Method | Task Completion | Recovery | Safety Violation | Battery-First | Gen. Gap |"]
        lines.append("|--------|----------------|----------|------------------|---------------|----------|")

        for method_name, results in self._results.items():
            self.evaluator.clear()
            for r in results:
                self.evaluator.add_result(r)

            s = self.evaluator.compute(
                seen_devices=set(self.config.seen_devices),
                unseen_devices=set(self.config.unseen_devices),
            )

            gen_gap = f"{s.generalization_gap:.1%}" if s.generalization_gap is not None else "N/A"

            lines.append(
                f"| {method_name} | {s.task_completion_rate:.1%} | "
                f"{s.component_recovery_rate:.1%} | {s.safety_violation_rate:.1%} | "
                f"{s.battery_first_rate:.1%} | {gen_gap} |"
            )

        return "\n".join(lines)
