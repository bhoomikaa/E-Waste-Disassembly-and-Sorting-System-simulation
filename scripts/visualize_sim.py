"""Visualize the simulation environment and device models.

Usage:
    python scripts/visualize_sim.py --device laptop_v1
    python scripts/visualize_sim.py --device router_v1 --render-mode rgb_array --save-frames
    python scripts/visualize_sim.py --device laptop_v1 --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def render_static_views(device_name: str, output_dir: str = "renders"):
    """Render static camera views of the device for documentation/papers."""
    from safedisassemble.sim.envs.disassembly_env import DisassemblyEnv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = DisassemblyEnv(
        device_name=device_name,
        image_size=512,
        render_mode="rgb_array",
    )
    obs, info = env.reset()

    # Overhead view
    overhead = obs["image_overhead"]
    _save_image(overhead, out / f"{device_name}_overhead.png")
    print(f"Saved overhead view: {out / f'{device_name}_overhead.png'}")

    # Wrist cam view
    wrist = obs["image_wrist"]
    _save_image(wrist, out / f"{device_name}_wrist.png")
    print(f"Saved wrist view: {out / f'{device_name}_wrist.png'}")

    # Take a few random actions to show robot movement
    frames = [overhead]
    for i in range(10):
        action = np.zeros(7, dtype=np.float32)
        action[2] = -0.5  # move down
        action[6] = 0.5   # open gripper
        obs, _, _, _, _ = env.step(action)
        if i % 3 == 0:
            frames.append(obs["image_overhead"])

    # Save frame sequence
    for i, frame in enumerate(frames):
        _save_image(frame, out / f"{device_name}_frame_{i:03d}.png")

    env.close()
    print(f"Saved {len(frames)} frames to {out}/")


def run_scripted_demo(device_name: str, output_dir: str = "renders"):
    """Run a scripted disassembly demo and record frames."""
    from safedisassemble.data.demo_collector import ScriptedDisassemblyPolicy
    from safedisassemble.sim.device_registry import get_device
    from safedisassemble.sim.envs.disassembly_env import DisassemblyEnv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device_spec = get_device(device_name)

    env = DisassemblyEnv(
        device_name=device_name,
        image_size=512,
        render_mode="rgb_array",
    )

    policy = ScriptedDisassemblyPolicy(device_spec, noise_std=0.001, seed=42)
    plan = policy.generate_plan()

    print(f"\nDisassembly plan for {device_name}:")
    for step in plan:
        safety = " [SAFETY]" if "battery" in step["component"].lower() else ""
        print(f"  {step['skill']:20s} → {step['component']}{safety}")
        print(f"    High: {step['instructions']['high']}")
        print(f"    Mid:  {step['instructions']['mid']}")
        print(f"    Low:  {step['instructions']['low']}")

    env.close()
    print(f"\nPlan has {len(plan)} steps across {len(device_spec.removable_components)} components")


def show_device_info(device_name: str):
    """Print detailed info about a device model."""
    from safedisassemble.sim.device_registry import get_device

    spec = get_device(device_name)

    print(f"\n{'='*60}")
    print(f"Device: {spec.name} ({spec.device_type})")
    print(f"Description: {spec.description}")
    print(f"Difficulty: {spec.difficulty}/5.0")
    print(f"{'='*60}")

    print(f"\nComponents ({spec.num_components}):")
    for comp in spec.components:
        removable = "✓" if comp.joint_names else "✗"
        print(f"  [{removable}] {comp.name:25s} type={comp.component_type.value:15s} "
              f"value={comp.value_score:5.1f} force_limit={comp.removal_force_limit:.0f}N")

    print(f"\nSafety Zones ({len(spec.safety_zones)}):")
    for zone in spec.safety_zones:
        print(f"  ⚠ {zone.hazard_type.value:25s} threshold={zone.force_threshold:.0f}N  "
              f"({zone.description})")

    print(f"\nDependency Rules ({len(spec.dependencies)}):")
    for dep in spec.dependencies:
        print(f"  {dep.prerequisite:20s} → {dep.dependent:20s} ({dep.reason})")

    # Validate a topological ordering exists
    removable = {c.name for c in spec.removable_components}
    print(f"\nRemovable components: {len(removable)}")

    # Generate valid order
    removed = set()
    order = []
    max_iters = len(removable) * 2
    iters = 0
    while removable - removed and iters < max_iters:
        iters += 1
        for comp_name in removable - removed:
            prereqs = spec.get_prerequisites(comp_name)
            if all(p in removed for p in prereqs):
                order.append(comp_name)
                removed.add(comp_name)
                break

    if removed == removable:
        print(f"Valid disassembly order found ({len(order)} steps):")
        for i, name in enumerate(order):
            print(f"  {i+1}. {name}")
    else:
        remaining = removable - removed
        print(f"WARNING: Cannot find valid order! Stuck components: {remaining}")


def _save_image(image: np.ndarray, path: Path):
    """Save numpy image array as PNG."""
    try:
        from PIL import Image
        img = Image.fromarray(image)
        img.save(str(path))
    except ImportError:
        # Fallback: save as raw numpy
        np.save(str(path).replace(".png", ".npy"), image)
        print(f"  (PIL not available, saved as .npy)")


def main():
    parser = argparse.ArgumentParser(description="Visualize simulation environment")
    parser.add_argument("--device", type=str, default="laptop_v1",
                        help="Device name from registry")
    parser.add_argument("--mode", type=str, choices=["info", "render", "demo", "all"],
                        default="info",
                        help="What to visualize")
    parser.add_argument("--output-dir", type=str, default="renders")
    args = parser.parse_args()

    if args.mode in ("info", "all"):
        show_device_info(args.device)

    if args.mode in ("demo", "all"):
        run_scripted_demo(args.device, args.output_dir)

    if args.mode in ("render", "all"):
        render_static_views(args.device, args.output_dir)


if __name__ == "__main__":
    main()
