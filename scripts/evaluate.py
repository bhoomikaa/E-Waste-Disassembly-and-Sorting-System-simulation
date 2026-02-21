"""Run the full EWasteBench evaluation suite.

Usage:
    python scripts/evaluate.py --method ours --selector-ckpt outputs/skill_selector_best.pt --policy-ckpt outputs/motor_policy_best.pt
    python scripts/evaluate.py --method scripted --device laptop_v1
    python scripts/evaluate.py --method random
    python scripts/evaluate.py --method all  # runs all methods and generates comparison
"""

from __future__ import annotations

import argparse
import json

from safedisassemble.evaluation.baselines.scripted_baseline import (
    FlatVLABaseline,
    RandomBaseline,
    ScriptedBaseline,
)
from safedisassemble.evaluation.benchmarks.ewaste_bench import (
    BenchmarkConfig,
    EWasteBenchmark,
)
from safedisassemble.models.hierarchical_controller import HierarchicalController


def main():
    parser = argparse.ArgumentParser(description="Run EWasteBench evaluation")
    parser.add_argument("--method", type=str,
                        choices=["ours", "scripted", "random", "flat_vla", "no_safety", "all"],
                        default="ours")
    parser.add_argument("--selector-ckpt", type=str, default=None)
    parser.add_argument("--policy-ckpt", type=str, default=None)
    parser.add_argument("--flat-vla-ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-randomization", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        seen_devices=["laptop_v1"],
        unseen_devices=["router_v1"],
        episodes_per_device=args.episodes,
        use_domain_randomization=not args.no_randomization,
    )

    bench = EWasteBenchmark(config)
    methods_to_run = []

    if args.method == "all":
        methods_to_run = ["ours", "scripted", "random"]
    else:
        methods_to_run = [args.method]

    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method}")
        print(f"{'='*60}")

        if method == "ours":
            controller = HierarchicalController(device=args.device)
            if args.selector_ckpt and args.policy_ckpt:
                controller.load_models(
                    selector_path=args.selector_ckpt,
                    policy_path=args.policy_ckpt,
                )
            summary = bench.evaluate(controller, method_name="ours")

        elif method == "scripted":
            # Scripted baseline — run per device
            for device_name in config.seen_devices + config.unseen_devices:
                baseline = ScriptedBaseline(device_name)
                summary = bench.evaluate(baseline, method_name=f"scripted_{device_name}")

        elif method == "random":
            baseline = RandomBaseline(seed=42)
            summary = bench.evaluate(baseline, method_name="random")

        elif method == "flat_vla":
            baseline = FlatVLABaseline(
                model_path=args.flat_vla_ckpt,
                device=args.device,
            )
            baseline.load()
            summary = bench.evaluate(baseline, method_name="flat_vla")

        elif method == "no_safety":
            controller = HierarchicalController(device=args.device)
            if args.selector_ckpt and args.policy_ckpt:
                controller.load_models(
                    selector_path=args.selector_ckpt,
                    policy_path=args.policy_ckpt,
                )
            from safedisassemble.evaluation.baselines.scripted_baseline import NoSafetyBaseline
            no_safety = NoSafetyBaseline(controller)
            summary = bench.evaluate(no_safety, method_name="no_safety")

    # Export results
    bench.export_results(args.output_dir)

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(bench.compare_methods())


if __name__ == "__main__":
    main()
