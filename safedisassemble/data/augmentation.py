"""Domain randomization and data augmentation for training robustness.

Two types of augmentation:
1. Visual augmentation (applied to images at training time)
2. Geometry augmentation (applied in simulation to generate varied trajectories)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class VisualAugmentor:
    """Image-space augmentations for training-time diversity.

    Applied to camera observations before feeding to the VLA model.
    Designed to be fast and differentiable where possible.
    """

    def __init__(
        self,
        color_jitter: float = 0.3,
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.7, 1.3),
        noise_std: float = 5.0,
        blur_prob: float = 0.2,
        blur_kernel_range: tuple[int, int] = (3, 7),
        cutout_prob: float = 0.1,
        cutout_size_range: tuple[float, float] = (0.05, 0.15),
        seed: Optional[int] = None,
    ):
        self.color_jitter = color_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.cutout_prob = cutout_prob
        self.cutout_size_range = cutout_size_range
        self.rng = np.random.default_rng(seed)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random visual augmentations to a single image.

        Args:
            image: (H, W, 3) uint8 array

        Returns:
            Augmented (H, W, 3) uint8 array
        """
        img = image.astype(np.float32)

        # Color jitter
        if self.color_jitter > 0:
            img = self._color_jitter(img)

        # Brightness
        brightness = self.rng.uniform(*self.brightness_range)
        img = img * brightness

        # Contrast
        contrast = self.rng.uniform(*self.contrast_range)
        mean = img.mean()
        img = (img - mean) * contrast + mean

        # Gaussian noise
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = img + noise

        # Gaussian blur
        if self.rng.random() < self.blur_prob:
            img = self._gaussian_blur(img)

        # Random cutout (simulates occlusion)
        if self.rng.random() < self.cutout_prob:
            img = self._random_cutout(img)

        return np.clip(img, 0, 255).astype(np.uint8)

    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """Augment a batch of images. Shape: (B, H, W, 3)."""
        return np.stack([self.augment(img) for img in images])

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Random per-channel color shift."""
        jitter = self.rng.uniform(
            -self.color_jitter, self.color_jitter, (1, 1, 3)
        ).astype(np.float32) * 255
        return img + jitter

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Simple box blur approximation (avoids cv2 dependency in core)."""
        k = self.rng.integers(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1)
        if k % 2 == 0:
            k += 1
        # Separable box filter
        kernel = np.ones(k, dtype=np.float32) / k
        for c in range(3):
            # Horizontal pass
            img[:, :, c] = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=img[:, :, c]
            )
            # Vertical pass
            img[:, :, c] = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=img[:, :, c]
            )
        return img

    def _random_cutout(self, img: np.ndarray) -> np.ndarray:
        """Randomly zero out a rectangular patch."""
        h, w = img.shape[:2]
        size_frac = self.rng.uniform(*self.cutout_size_range)
        ch, cw = int(h * size_frac), int(w * size_frac)
        y = self.rng.integers(0, max(1, h - ch))
        x = self.rng.integers(0, max(1, w - cw))
        img[y:y+ch, x:x+cw] = 0
        return img


class GeometryRandomizer:
    """Simulation-level randomization for generating diverse device configurations.

    Modifies MuJoCo model properties to create variations:
    - Screw positions (within bounds)
    - Component offsets
    - Connector orientations
    - Material friction coefficients
    - Body masses (within physical bounds)
    """

    def __init__(
        self,
        position_noise: float = 0.005,    # 5mm max displacement
        rotation_noise: float = 0.1,       # ~6 degrees
        friction_range: tuple[float, float] = (0.5, 1.5),
        mass_range: tuple[float, float] = (0.8, 1.2),  # multiplicative
        stiffness_range: tuple[float, float] = (0.7, 1.3),
        seed: Optional[int] = None,
    ):
        self.position_noise = position_noise
        self.rotation_noise = rotation_noise
        self.friction_range = friction_range
        self.mass_range = mass_range
        self.stiffness_range = stiffness_range
        self.rng = np.random.default_rng(seed)

    def randomize_model(self, model) -> dict:
        """Apply randomizations to a MuJoCo model. Returns the applied perturbations.

        Args:
            model: mujoco.MjModel instance (modified in-place)

        Returns:
            Dictionary of applied randomization parameters for logging
        """
        import mujoco

        perturbations = {}

        # Randomize body positions (small offsets)
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and ("screw" in name or "connector" in name or "clip" in name):
                offset = self.rng.uniform(
                    -self.position_noise, self.position_noise, 3
                )
                model.body_pos[i, :3] += offset
                perturbations[f"body_{name}_offset"] = offset.tolist()

        # Randomize geom friction
        for i in range(model.ngeom):
            scale = self.rng.uniform(*self.friction_range)
            model.geom_friction[i, 0] *= scale
            perturbations[f"geom_{i}_friction_scale"] = scale

        # Randomize body masses
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name not in ("world", "worldbody"):
                scale = self.rng.uniform(*self.mass_range)
                # Mass is stored in geoms, not bodies directly
                # We scale inertia instead
                model.body_mass[i] *= scale

        # Randomize joint stiffness (affects screw tightness, clip force)
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and ("screw" in name or "clip" in name or "connector" in name):
                scale = self.rng.uniform(*self.stiffness_range)
                model.jnt_stiffness[i] *= scale
                perturbations[f"joint_{name}_stiffness_scale"] = scale

        # Randomize visual properties (colors, for sim-to-real transfer)
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and any(k in name for k in ["shell", "panel", "housing", "chassis"]):
                color_noise = self.rng.uniform(-0.08, 0.08, 3)
                model.geom_rgba[i, :3] = np.clip(
                    model.geom_rgba[i, :3] + color_noise, 0, 1
                )

        return perturbations

    def get_randomization_config(self) -> dict:
        """Return current randomization parameters as a dict for logging."""
        return {
            "position_noise": self.position_noise,
            "rotation_noise": self.rotation_noise,
            "friction_range": self.friction_range,
            "mass_range": self.mass_range,
            "stiffness_range": self.stiffness_range,
        }
