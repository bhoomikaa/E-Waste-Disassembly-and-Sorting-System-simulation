"""Collect scripted demonstration trajectories for training.

Usage:
    python scripts/collect_demos.py --device laptop_v1 --num-trajectories 200
    python scripts/collect_demos.py --device router_v1 --num-trajectories 200 --randomize
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safedisassemble.data.augmentation import GeometryRandomizer
from safedisassemble.data.demo_collector import ScriptedDisassemblyPolicy
from safedisassemble.data.trajectory import TrajectoryDataset
from safedisassemble.sim.device_registry import get_device
from safedisassemble.sim.envs.disassembly_env import DisassemblyEnv


def main():
    parser = argparse.ArgumentParser(description="Collect demonstration trajectories")
    parser.add_argument("--device", type=str, default="laptop_v1")
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="data/trajectories")
    parser.add_argument("--randomize", action="store_true",
                        help="Apply geometry randomization")
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device_spec = get_device(args.device)
    output_path = Path(args.output_dir) / f"{args.device}_demos.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = DisassemblyEnv(
        device_name=args.device,
        use_domain_randomization=args.randomize,
        seed=args.seed,
    )

    policy = ScriptedDisassemblyPolicy(
        device_spec=device_spec,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    with TrajectoryDataset(output_path, mode="w") as dataset:
        successes = 0
        for i in range(args.num_trajectories):
            traj = policy.collect_trajectory(env)
            key = dataset.add_trajectory(traj)

            if traj.success:
                successes += 1

            if (i + 1) % 10 == 0:
                print(
                    f"[{i+1}/{args.num_trajectories}] "
                    f"Success rate: {successes/(i+1):.1%} | "
                    f"Total timesteps: {dataset.total_timesteps}"
                )

        stats = dataset.get_statistics()
        print(f"\nCollection complete:")
        print(f"  Trajectories: {stats['num_trajectories']}")
        print(f"  Total timesteps: {stats['total_timesteps']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Saved to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
