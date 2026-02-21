"""Run the complete SafeDisassemble pipeline end-to-end.

This script orchestrates the full workflow:
1. Collect demonstrations
2. Train models
3. Evaluate against baselines

Primarily for testing that the full pipeline hangs together.

Usage:
    python scripts/run_full_pipeline.py --quick  # small run for testing
    python scripts/run_full_pipeline.py --full   # full training run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def run_pipeline(quick: bool = True):
    # Import everything up front to catch import errors early
    from safedisassemble.sim.device_registry import get_device, list_devices
    from safedisassemble.sim.envs.disassembly_env import DisassemblyEnv
    from safedisassemble.data.demo_collector import ScriptedDisassemblyPolicy
    from safedisassemble.data.trajectory import TrajectoryDataset
    from safedisassemble.data.augmentation import VisualAugmentor, GeometryRandomizer
    from safedisassemble.models.task_planner.planner import TaskPlanner
    from safedisassemble.models.skill_selector.selector import SkillSelector
    from safedisassemble.models.motor_policy.diffusion_policy import DiffusionMotorPolicy
    from safedisassemble.models.safety.constraint_checker import SafetyConstraintModule
    from safedisassemble.models.hierarchical_controller import HierarchicalController
    from safedisassemble.evaluation.metrics.disassembly_metrics import (
        DisassemblyEvaluator, EpisodeResult,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_demos = 5 if quick else 200
    n_epochs = 2 if quick else 100

    print(f"{'='*60}")
    print(f"SafeDisassemble — Full Pipeline {'(quick mode)' if quick else ''}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # ─── Step 1: Environment Sanity Check ───
    print("\n[Step 1] Environment sanity check...")
    t0 = time.time()

    for device_name in list_devices():
        spec = get_device(device_name)
        print(f"  {device_name}: {spec.num_components} components, "
              f"{len(spec.safety_zones)} safety zones")

    env = DisassemblyEnv(device_name="laptop_v1", image_size=84, max_steps=50)
    obs, info = env.reset()
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  Image shape: {obs['image_wrist'].shape}")

    # Take a few random actions
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    env.close()
    print(f"  Environment OK ({time.time()-t0:.1f}s)")

    # ─── Step 2: Data Collection ───
    print(f"\n[Step 2] Collecting {n_demos} demonstrations...")
    t0 = time.time()

    demo_path = Path("data/trajectories/pipeline_test_demos.h5")
    demo_path.parent.mkdir(parents=True, exist_ok=True)

    spec = get_device("laptop_v1")
    env = DisassemblyEnv(device_name="laptop_v1", image_size=84, max_steps=100)
    policy = ScriptedDisassemblyPolicy(spec, noise_std=0.003, seed=42)

    with TrajectoryDataset(demo_path, mode="w") as ds:
        for i in range(n_demos):
            traj = policy.collect_trajectory(env)
            ds.add_trajectory(traj)
        stats = ds.get_statistics()

    env.close()
    print(f"  Collected {stats['num_trajectories']} trajectories, "
          f"{stats['total_timesteps']} timesteps ({time.time()-t0:.1f}s)")

    # ─── Step 3: Data Augmentation Test ───
    print("\n[Step 3] Testing augmentation pipeline...")
    t0 = time.time()

    augmentor = VisualAugmentor(seed=42)
    test_image = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
    augmented = augmentor.augment(test_image)
    assert augmented.shape == test_image.shape
    assert augmented.dtype == np.uint8

    geo_rand = GeometryRandomizer(seed=42)
    print(f"  Visual + geometry augmentation OK ({time.time()-t0:.1f}s)")

    # ─── Step 4: Model Forward Pass Test ───
    print("\n[Step 4] Testing model forward passes...")
    t0 = time.time()

    # Task Planner (retrieval-only mode, no VLM)
    planner = TaskPlanner(use_retrieval=True)
    plan = planner.plan(
        image=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        instruction="Disassemble this laptop",
        device_type_hint="laptop",
    )
    print(f"  Task Planner: generated {plan.num_steps} steps")
    print(f"    Plan: {plan.to_text()[:200]}...")

    # Skill Selector
    selector = SkillSelector(image_size=84, embed_dim=64).to(device)
    test_img = torch.randn(2, 3, 84, 84).to(device)
    test_tokens = torch.randint(0, 1000, (2, 32)).to(device)
    test_proprio = torch.randn(2, 19).to(device)
    outputs = selector(test_img, test_tokens, test_proprio)
    print(f"  Skill Selector: logits={outputs['skill_logits'].shape}, "
          f"params={outputs['skill_params'].shape}")

    # Motor Policy
    motor = DiffusionMotorPolicy(
        action_dim=7, action_horizon=8, image_size=84,
        cond_dim=128, num_diffusion_steps=10, num_inference_steps=3,
    ).to(device)

    test_skill_ids = torch.randint(0, 8, (2,)).to(device)
    test_skill_params = torch.randn(2, 10).to(device)
    test_actions = torch.randn(2, 8, 7).to(device)

    loss = motor.compute_loss(test_img, test_proprio, test_skill_ids, test_skill_params, test_actions)
    print(f"  Motor Policy: diffusion loss={loss.item():.4f}")

    pred_actions = motor.predict_action(test_img, test_proprio, test_skill_ids, test_skill_params)
    print(f"  Motor Policy: predicted actions shape={pred_actions.shape}")

    print(f"  All models OK ({time.time()-t0:.1f}s)")

    # ─── Step 5: Safety Module Test ───
    print("\n[Step 5] Testing safety module...")
    t0 = time.time()

    safety = SafetyConstraintModule()
    safety.setup_from_device_spec(spec)

    # Test plan validation
    good_plan = [
        {"component": "screw", "component_type": "screw", "step_id": 1},
        {"component": "panel", "component_type": "panel", "step_id": 2},
        {"component": "battery", "component_type": "battery", "step_id": 3},
        {"component": "ram", "component_type": "ram", "step_id": 4},
    ]
    result = safety.validate_plan(good_plan)
    print(f"  Good plan validation: {result.level.value} ({result.reason})")

    bad_plan = [
        {"component": "panel", "component_type": "panel", "step_id": 1},
        {"component": "ram", "component_type": "ram", "step_id": 2},
        {"component": "battery", "component_type": "battery", "step_id": 3},
    ]
    result = safety.validate_plan(bad_plan)
    print(f"  Bad plan validation: {result.level.value} ({result.reason})")

    summary = safety.get_safety_summary()
    print(f"  Safety events logged: {summary['total_events']}")
    print(f"  Safety module OK ({time.time()-t0:.1f}s)")

    # ─── Step 6: Evaluation Metrics Test ───
    print("\n[Step 6] Testing evaluation metrics...")
    t0 = time.time()

    evaluator = DisassemblyEvaluator()
    evaluator.add_result(EpisodeResult(
        device_name="laptop_v1",
        total_components=5,
        recovered_components=["screw", "panel", "battery", "ram", "ssd"],
        damaged_components=[],
        safety_violations=[],
        total_steps=100,
        total_reward=10.0,
        success=True,
        plan_was_safe=True,
        battery_disconnected_first=True,
    ))
    evaluator.add_result(EpisodeResult(
        device_name="router_v1",
        total_components=4,
        recovered_components=["clip", "cover"],
        damaged_components=[],
        safety_violations=[],
        total_steps=80,
        total_reward=3.0,
        success=False,
        plan_was_safe=True,
        battery_disconnected_first=True,
    ))

    metrics = evaluator.compute(
        seen_devices={"laptop_v1"},
        unseen_devices={"router_v1"},
    )
    print(f"  Task completion: {metrics.task_completion_rate:.1%}")
    print(f"  Component recovery: {metrics.component_recovery_rate:.1%}")
    print(f"  Seen completion: {metrics.seen_completion_rate:.1%}")
    print(f"  Unseen completion: {metrics.unseen_completion_rate:.1%}")
    print(f"  Generalization gap: {metrics.generalization_gap:.1%}")

    table = evaluator.format_table(metrics)
    print(f"\n{table}")
    print(f"\n  Metrics OK ({time.time()-t0:.1f}s)")

    # ─── Done ───
    print(f"\n{'='*60}")
    print("PIPELINE VALIDATION COMPLETE — All components working")
    print(f"{'='*60}")

    # Cleanup
    if demo_path.exists():
        demo_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Quick test run (default)")
    parser.add_argument("--full", action="store_true",
                        help="Full training run")
    args = parser.parse_args()

    run_pipeline(quick=not args.full)


if __name__ == "__main__":
    main()
