"""Train VLA models for e-waste disassembly.

Usage:
    python scripts/train.py --model skill_selector --data data/trajectories/laptop_v1_demos.h5
    python scripts/train.py --model motor_policy --data data/trajectories/laptop_v1_demos.h5
    python scripts/train.py --model both --data data/trajectories/laptop_v1_demos.h5
"""

from __future__ import annotations

import argparse

from safedisassemble.models.motor_policy.diffusion_policy import DiffusionMotorPolicy
from safedisassemble.models.skill_selector.selector import SkillSelector
from safedisassemble.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train VLA models")
    parser.add_argument("--model", type=str, choices=["skill_selector", "motor_policy", "both"],
                        default="both")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectory dataset")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation dataset")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    trainer = Trainer(
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
    )

    if args.model in ("skill_selector", "both"):
        print("=" * 60)
        print("Training Level 2: Skill Selector")
        print("=" * 60)

        model = SkillSelector(image_size=224)
        epochs = args.epochs or 100

        results = trainer.train_skill_selector(
            model=model,
            train_data_path=args.data,
            val_data_path=args.val_data,
            num_epochs=epochs,
        )
        print(f"Skill Selector — Best val loss: {results['best_val_loss']:.4f}")

    if args.model in ("motor_policy", "both"):
        print("=" * 60)
        print("Training Level 3: Diffusion Motor Policy")
        print("=" * 60)

        model = DiffusionMotorPolicy(
            action_dim=7,
            action_horizon=16,
        )
        epochs = args.epochs or 200

        results = trainer.train_motor_policy(
            model=model,
            train_data_path=args.data,
            val_data_path=args.val_data,
            num_epochs=epochs,
        )
        print(f"Motor Policy — Best loss: {results['best_loss']:.4f}")

    trainer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
