#!/usr/bin/env python3
"""Cinematic disassembly demo — LinkedIn-ready 16:9 video.

Produces a smooth, multi-angle, annotated 1280×720 video of a full
laptop disassembly.  Every single frame is rendered (no skipping).
Camera transitions use cubic-ease interpolation.  Text uses
documentary-style lower-third overlays.  Component-tracking close-ups
replace the old wrist camera for clearly visible detail shots.

Usage:
    PYTHONPATH=. python3 scripts/cinematic_demo.py
    PYTHONPATH=. python3 scripts/cinematic_demo.py --device laptop_v1 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  Easing & interpolation
# ═══════════════════════════════════════════════════════════════════════════

def ease_in_out_cubic(t: float) -> float:
    """Smooth S-curve: slow start, fast middle, slow end. t ∈ [0,1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def ease_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_array(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


# ═══════════════════════════════════════════════════════════════════════════
#  Camera state — holds lookat / distance / azimuth / elevation
# ═══════════════════════════════════════════════════════════════════════════

class CameraState:
    __slots__ = ("lookat", "distance", "azimuth", "elevation")

    def __init__(self, lookat=(0.5, 0.0, 0.44), distance=0.8,
                 azimuth=0.0, elevation=-45.0):
        self.lookat = np.array(lookat, dtype=float)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)

    def copy(self):
        c = CameraState()
        c.lookat = self.lookat.copy()
        c.distance = self.distance
        c.azimuth = self.azimuth
        c.elevation = self.elevation
        return c

    def interpolate(self, other: "CameraState", t: float) -> "CameraState":
        """Smooth interpolation using ease-in-out cubic."""
        s = ease_in_out_cubic(t)
        c = CameraState()
        c.lookat = lerp_array(self.lookat, other.lookat, s)
        c.distance = lerp(self.distance, other.distance, s)
        c.azimuth = lerp(self.azimuth, other.azimuth, s)
        c.elevation = lerp(self.elevation, other.elevation, s)
        return c


# Named presets — wide-angle for 16:9 (lookat, distance, azimuth, elevation)
CAM_WIDE       = CameraState([0.30, 0.0, 0.45], 1.20, -25, -25)
CAM_OVERHEAD   = CameraState([0.50, 0.0, 0.42], 0.80,   0, -85)
CAM_CLOSE      = CameraState([0.50, 0.0, 0.45], 0.45,  35, -35)
CAM_SIDE       = CameraState([0.50, 0.0, 0.48], 0.60,  90, -15)
CAM_HERO       = CameraState([0.50, 0.0, 0.44], 0.55,  20, -30)
CAM_LOW_ANGLE  = CameraState([0.50, 0.0, 0.44], 0.55, -40, -20)
CAM_FRONT      = CameraState([0.50, 0.0, 0.45], 0.50, 180, -30)
CAM_THREE_QTR  = CameraState([0.50, 0.0, 0.44], 0.65, -55, -28)


# ═══════════════════════════════════════════════════════════════════════════
#  Documentary-style lower-third text overlay
# ═══════════════════════════════════════════════════════════════════════════

def overlay_lower_third(image: np.ndarray, title: str,
                        subtitle: str = "", alpha: float = 1.0,
                        progress: float = -1.0) -> np.ndarray:
    """Render a documentary-style lower-third: left-aligned, bottom bar.

    - Narrow dark gradient bar at the bottom-left (not full-width)
    - Title in bold white, subtitle in lighter gray below
    - Optional thin progress bar at the very bottom
    """
    if alpha <= 0.01:
        return image
    try:
        import cv2
    except ImportError:
        return image

    img = image.copy()
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX

    # Scale relative to height (720p baseline) — LARGE readable text
    scale_factor = h / 720.0
    title_scale = 1.1 * scale_factor
    sub_scale = 0.7 * scale_factor
    title_thick = max(2, int(3 * scale_factor))
    sub_thick = max(1, int(2 * scale_factor))

    # Measure text
    (tw, th_), _ = cv2.getTextSize(title, font_bold, title_scale, title_thick)
    if subtitle:
        (sw, sh_), _ = cv2.getTextSize(subtitle, font, sub_scale, sub_thick)
    else:
        sw, sh_ = 0, 0

    # Bar dimensions — left-aligned, partial width
    pad_x = int(28 * scale_factor)
    pad_y = int(14 * scale_factor)
    bar_w = max(tw, sw) + pad_x * 3
    bar_w = min(bar_w, int(w * 0.65))  # never wider than 65% of frame

    total_text_h = th_ + (sh_ + pad_y if subtitle else 0)
    bar_h = total_text_h + pad_y * 3

    bar_y1 = h - bar_h - int(30 * scale_factor)
    bar_y2 = h - int(30 * scale_factor)
    bar_x1 = int(20 * scale_factor)
    bar_x2 = bar_x1 + bar_w

    # Accent stripe (left edge, teal)
    stripe_w = max(4, int(6 * scale_factor))

    # Draw dark bar with alpha
    overlay = img.copy()
    cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (15, 15, 20), -1)
    bar_alpha = 0.75 * alpha
    cv2.addWeighted(overlay, bar_alpha, img, 1 - bar_alpha, 0, img)

    # Accent stripe
    stripe_overlay = img.copy()
    cv2.rectangle(stripe_overlay, (bar_x1, bar_y1), (bar_x1 + stripe_w, bar_y2),
                  (0, 180, 220), -1)  # teal accent
    cv2.addWeighted(stripe_overlay, alpha, img, 1 - alpha, 0, img)

    # Title text
    title_x = bar_x1 + stripe_w + pad_x
    title_y = bar_y1 + pad_y + th_
    text_layer = img.copy()
    cv2.putText(text_layer, title, (title_x, title_y), font_bold,
                title_scale, (255, 255, 255), title_thick, cv2.LINE_AA)

    # Subtitle text
    if subtitle:
        sub_y = title_y + sh_ + pad_y
        cv2.putText(text_layer, subtitle, (title_x, sub_y), font,
                    sub_scale, (180, 190, 200), sub_thick, cv2.LINE_AA)

    cv2.addWeighted(text_layer, alpha, img, 1 - alpha, 0, img)

    # Progress bar at very bottom of frame
    if progress >= 0:
        pb_h = max(2, int(3 * scale_factor))
        pb_y = h - pb_h
        pb_filled = int(w * min(progress, 1.0))
        prog_overlay = img.copy()
        cv2.rectangle(prog_overlay, (0, pb_y), (w, h), (30, 30, 35), -1)
        cv2.rectangle(prog_overlay, (0, pb_y), (pb_filled, h), (0, 180, 220), -1)
        cv2.addWeighted(prog_overlay, 0.85 * alpha, img, 1 - 0.85 * alpha, 0, img)

    return img


def overlay_title_card(image: np.ndarray, title: str, subtitle: str = "",
                       alpha: float = 1.0) -> np.ndarray:
    """Centered large title card for opening/closing."""
    if alpha <= 0.01:
        return image
    try:
        import cv2
    except ImportError:
        return image

    img = image.copy()
    h, w = img.shape[:2]
    scale_factor = h / 720.0

    # Full-frame dark vignette
    vignette = img.copy()
    cv2.rectangle(vignette, (0, 0), (w, h), (0, 0, 0), -1)
    vig_alpha = 0.45 * alpha
    cv2.addWeighted(vignette, vig_alpha, img, 1 - vig_alpha, 0, img)

    font_bold = cv2.FONT_HERSHEY_DUPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = 1.3 * scale_factor
    title_thick = max(2, int(3 * scale_factor))
    (tw, th_), _ = cv2.getTextSize(title, font_bold, title_scale, title_thick)
    tx = (w - tw) // 2
    ty = h // 2 - int(10 * scale_factor)

    text_layer = img.copy()
    cv2.putText(text_layer, title, (tx, ty), font_bold,
                title_scale, (255, 255, 255), title_thick, cv2.LINE_AA)

    if subtitle:
        sub_scale = 0.55 * scale_factor
        sub_thick = max(1, int(1.5 * scale_factor))
        (sw, sh_), _ = cv2.getTextSize(subtitle, font, sub_scale, sub_thick)
        sx = (w - sw) // 2
        sy = ty + th_ + int(20 * scale_factor)
        cv2.putText(text_layer, subtitle, (sx, sy), font,
                    sub_scale, (0, 180, 220), sub_thick, cv2.LINE_AA)

    cv2.addWeighted(text_layer, alpha, img, 1 - alpha, 0, img)
    return img


def compute_text_alpha(frame_in_shot: int, total_shot_frames: int,
                       fade_in: int = 12, fade_out: int = 12) -> float:
    """Smooth fade-in / hold / fade-out alpha curve."""
    if frame_in_shot < fade_in:
        return frame_in_shot / fade_in
    if frame_in_shot > total_shot_frames - fade_out:
        return max(0.0, (total_shot_frames - frame_in_shot) / fade_out)
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Rendering helpers — 16:9 (width × height)
# ═══════════════════════════════════════════════════════════════════════════

_renderer_cache: dict[tuple[int, int], object] = {}


def get_renderer(model, width: int, height: int):
    """Get or create a MuJoCo renderer for given dimensions."""
    import mujoco
    key = (width, height)
    if key not in _renderer_cache:
        _renderer_cache[key] = mujoco.Renderer(model, height, width)
    return _renderer_cache[key]


def render_from_state(env, cam_state: CameraState,
                      width: int, height: int) -> np.ndarray:
    """Render a single frame from a CameraState at width×height."""
    import mujoco
    _enforce_pinned(env)  # ensure removed parts stay removed
    mujoco.mj_forward(env.model, env.data)  # recompute positions
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_state.lookat
    cam.distance = cam_state.distance
    cam.azimuth = cam_state.azimuth
    cam.elevation = cam_state.elevation
    renderer = get_renderer(env.model, width, height)
    renderer.update_scene(env.data, cam)
    return renderer.render().copy()


def camera_transition(env, start: CameraState, end: CameraState,
                      num_frames: int, width: int, height: int,
                      title: str = "", subtitle: str = "",
                      progress: float = -1.0) -> list[np.ndarray]:
    """Smooth eased camera move from *start* to *end*."""
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        state = start.interpolate(end, t)
        frame = render_from_state(env, state, width, height)
        if title:
            alpha = compute_text_alpha(i, num_frames, fade_in=15, fade_out=15)
            frame = overlay_lower_third(frame, title, subtitle, alpha, progress)
        frames.append(frame)
    return frames


def camera_hold(env, cam_state: CameraState, num_frames: int,
                width: int, height: int,
                title: str = "", subtitle: str = "",
                progress: float = -1.0) -> list[np.ndarray]:
    """Hold on a fixed camera, optionally with lower-third text."""
    frames = []
    base = render_from_state(env, cam_state, width, height)
    for i in range(num_frames):
        f = base.copy()
        if title:
            alpha = compute_text_alpha(i, num_frames)
            f = overlay_lower_third(f, title, subtitle, alpha, progress)
        frames.append(f)
    return frames


def camera_orbit(env, center_lookat, base_distance: float,
                 start_az: float, end_az: float, elevation: float,
                 num_frames: int, width: int, height: int,
                 title: str = "", subtitle: str = "",
                 progress: float = -1.0) -> list[np.ndarray]:
    """Smooth orbital pan."""
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        s = ease_in_out_cubic(t)
        az = lerp(start_az, end_az, s)
        # subtle distance breathing
        dist = base_distance + 0.04 * math.sin(t * math.pi)
        state = CameraState(center_lookat, dist, az, elevation)
        frame = render_from_state(env, state, width, height)
        if title:
            alpha = compute_text_alpha(i, num_frames, fade_in=20, fade_out=20)
            frame = overlay_lower_third(frame, title, subtitle, alpha, progress)
        frames.append(frame)
    return frames


# ═══════════════════════════════════════════════════════════════════════════
#  Component-tracking close-up camera
# ═══════════════════════════════════════════════════════════════════════════

def make_component_closeup(env, component_name: str,
                           fallback_pos=None) -> CameraState:
    """Create a camera that zooms into a specific component's location.

    This replaces the old wrist camera — it looks directly at the
    component from a close distance so you can see the detail.
    """
    import mujoco
    target = np.array([0.5, 0.0, 0.45])

    # Try to find the component's site
    comp_site_names = [
        f"{component_name}_site",
        f"{component_name.replace('_module', '')}_site",
        f"{component_name.replace('_assembly', '')}_site",
    ]
    for sname in comp_site_names:
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid >= 0:
            target = env.data.site_xpos[sid].copy()
            break
    else:
        if fallback_pos is not None:
            target = np.array(fallback_pos)

    # Camera offset — slightly above and to the side for depth
    return CameraState(
        lookat=target,
        distance=0.18,  # very close
        azimuth=25,
        elevation=-45,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Video I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_image(image: np.ndarray, path: Path):
    try:
        from PIL import Image
        Image.fromarray(image).save(str(path))
    except ImportError:
        np.save(str(path).replace(".png", ".npy"), image)


def frames_to_video(frames: list[np.ndarray], path: Path, fps: int = 30) -> bool:
    """Encode frames → MP4.  Prefers imageio+ffmpeg (H.264) for quality
    and broad compatibility; falls back to OpenCV mp4v."""
    # --- Try imageio + ffmpeg first (H.264, yuv420p) ---
    try:
        import imageio.v3 as iio
        from imageio_ffmpeg import get_ffmpeg_exe  # noqa: F401
        iio.imwrite(
            str(path),
            np.stack(frames),
            fps=fps,
            codec="libx264",
            plugin="pyav",
            output_params=["-crf", "18", "-preset", "slow",
                           "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart"],
        )
        return True
    except Exception:
        pass

    # --- Try imageio legacy writer ---
    try:
        import imageio
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p")
        for f in frames:
            writer.append_data(f)
        writer.close()
        return True
    except Exception:
        pass

    # --- Fallback: OpenCV mp4v ---
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return True
    except ImportError:
        print("  [!] No video encoder available")
        return False


def save_contact_sheet(frames: list[np.ndarray], path: Path):
    try:
        from PIL import Image
        n = 8
        idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
        h, w = frames[0].shape[:2]
        tw, th = w // 2, h // 2
        sheet = Image.new("RGB", (tw * 4, th * 2))
        for i, idx in enumerate(idxs):
            r, c = divmod(i, 4)
            sheet.paste(Image.fromarray(frames[idx]).resize((tw, th)),
                        (c * tw, r * th))
        sheet.save(str(path))
        print(f"  Contact sheet: {path}")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Smooth component removal (records EVERY frame)
# ═══════════════════════════════════════════════════════════════════════════

# Tracks all pinned joints so removed components STAY removed
_pinned_joints: dict[int, float] = {}  # qpos_idx → target_value


def _enforce_pinned(env):
    """Force all previously-removed joints to stay at their target values."""
    for qidx, val in _pinned_joints.items():
        env.data.qpos[qidx] = val


def animate_removal(env, component_name: str, cam_state: CameraState,
                    width: int, height: int,
                    animate_frames: int = 60,
                    settle_frames: int = 15,
                    title: str = "", subtitle: str = "",
                    progress: float = -1.0) -> list[np.ndarray]:
    """Drive removal joints with easing and capture every frame.
    After removal, joints are PINNED so physics can't pull them back."""
    import mujoco

    device_type = env.device_spec.device_type
    joint_targets = env._REMOVAL_JOINT_MAP.get(device_type, {}).get(
        component_name, [])

    frames: list[np.ndarray] = []

    if not joint_targets:
        env._removed_components.add(component_name)
        f = render_from_state(env, cam_state, width, height)
        for i in range(20):
            fr = f.copy()
            if title:
                alpha = compute_text_alpha(i, 20)
                fr = overlay_lower_third(fr, title, subtitle, alpha, progress)
            frames.append(fr)
        return frames

    # Resolve joint info
    joint_info = []
    for jname, target_val in joint_targets:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qidx = env.model.jnt_qposadr[jid]
        start_val = float(env.data.qpos[qidx])
        joint_info.append((jid, qidx, start_val, target_val))

    # Animate with eased interpolation — every frame
    for t in range(animate_frames):
        progress_t = ease_in_out_cubic(t / max(animate_frames - 1, 1))
        for _jid, qidx, start_val, target_val in joint_info:
            env.data.qpos[qidx] = lerp(start_val, target_val, progress_t)
        _enforce_pinned(env)  # keep previously removed parts in place
        mujoco.mj_step(env.model, env.data)

        frame = render_from_state(env, cam_state, width, height)
        if title:
            alpha = compute_text_alpha(t, animate_frames, fade_in=10, fade_out=8)
            frame = overlay_lower_third(frame, title, subtitle, alpha, progress)
        frames.append(frame)

    # PIN these joints permanently so they never snap back
    for _jid, qidx, _start_val, target_val in joint_info:
        _pinned_joints[qidx] = target_val
        env.data.qpos[qidx] = target_val
        # Zero out stiffness so physics doesn't fight the pin
        jid_model = _jid
        env.model.jnt_stiffness[jid_model] = 0.0

    # Settle physics (with pins enforced)
    for _ in range(settle_frames):
        _enforce_pinned(env)
        mujoco.mj_step(env.model, env.data)
    _enforce_pinned(env)
    frames.append(render_from_state(env, cam_state, width, height))

    env._removed_components.add(component_name)
    return frames


# ═══════════════════════════════════════════════════════════════════════════
#  Robot approach shot — short, smooth, every frame recorded
# ═══════════════════════════════════════════════════════════════════════════

def robot_approach_shot(env, target_pos: np.ndarray, skill: str,
                        cam_state: CameraState, width: int, height: int,
                        num_frames: int = 45,
                        title: str = "", subtitle: str = "",
                        progress: float = -1.0) -> list[np.ndarray]:
    """Brief robot motion toward the component. Records every frame."""
    import mujoco

    frames = []
    ee_site_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for i in range(num_frames):
        ee_pos = env.data.site_xpos[ee_site_id].copy()
        error = target_pos - ee_pos
        action_pos = np.clip(error / 0.02, -1, 1)
        gripper = 0.8 if i < num_frames // 2 else -0.5
        action = np.concatenate([
            action_pos, np.zeros(3), [gripper]]).astype(np.float32)

        _enforce_pinned(env)  # keep removed parts in place
        env.step(action)
        _enforce_pinned(env)  # re-enforce after physics step

        frame = render_from_state(env, cam_state, width, height)
        if title:
            alpha = compute_text_alpha(i, num_frames, fade_in=12, fade_out=10)
            frame = overlay_lower_third(frame, title, subtitle, alpha, progress)
        frames.append(frame)

    return frames


# ═══════════════════════════════════════════════════════════════════════════
#  Main cinematic pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_cinematic_demo(
    device_name: str = "laptop_v1",
    width: int = 1280,
    height: int = 720,
    output_dir: str = "renders/cinematic",
    make_video: bool = True,
    fps: int = 30,
):
    import mujoco
    from safedisassemble.data.demo_collector import ScriptedDisassemblyPolicy
    from safedisassemble.sim.device_registry import get_device
    from safedisassemble.sim.envs.disassembly_env import DisassemblyEnv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device_spec = get_device(device_name)

    res_str = f"{width}x{height}"
    print("+" + "=" * 56 + "+")
    print("|       SafeDisassemble — Cinematic Demo (16:9)        |")
    print(f"|  Device     : {device_name:<40s} |")
    print(f"|  Resolution : {res_str:<40s} |")
    print(f"|  FPS        : {fps:<40} |")
    print("+" + "=" * 56 + "+")

    env = DisassemblyEnv(
        device_name=device_name,
        image_size=224,  # obs size — cinematic frames use separate renderer
        max_steps=200_000,
        render_mode="rgb_array",
    )
    env.reset()
    F: list[np.ndarray] = []  # all frames

    # ╔═══════════════════════════════════════════════════════╗
    # ║  ACT 1 — ESTABLISHING (≈8 s)                        ║
    # ╚═══════════════════════════════════════════════════════╝
    print("\n[ACT 1] Establishing shots ...")

    # Opening title card — dark vignette with centered text
    title_frames = int(fps * 2.5)
    for i in range(title_frames):
        base = render_from_state(env, CAM_WIDE, width, height)
        a = compute_text_alpha(i, title_frames, fade_in=20, fade_out=15)
        base = overlay_title_card(base, "SafeDisassemble",
                                  "Hierarchical VLA for Automated E-Waste Disassembly", a)
        F.append(base)

    # Wide hold → overhead dolly
    F += camera_hold(env, CAM_WIDE, int(fps * 1.0), width, height,
                     "Automated E-Waste Disassembly",
                     "Vision-Language-Action Model")

    F += camera_transition(env, CAM_WIDE, CAM_OVERHEAD, int(fps * 1.2),
                           width, height,
                           "Safety-Aware Planning",
                           "Multi-level skill decomposition")

    # Overhead → three-quarter beauty shot
    F += camera_transition(env, CAM_OVERHEAD, CAM_THREE_QTR, int(fps * 1.0),
                           width, height,
                           "Target Device: Laptop", device_name)

    # Three-quarter → hero angle for disassembly
    F += camera_transition(env, CAM_THREE_QTR, CAM_HERO, int(fps * 0.8),
                           width, height)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  ACT 2 — DISASSEMBLY                                ║
    # ╚═══════════════════════════════════════════════════════╝
    policy = ScriptedDisassemblyPolicy(device_spec, noise_std=0.001, seed=42)
    plan = policy.generate_plan()

    print(f"\n[ACT 2] Disassembly — {len(plan)} steps")
    for i, s in enumerate(plan):
        tag = " [BATTERY]" if "battery" in s["component"].lower() else ""
        print(f"  {i+1:2d}. {s['skill']:18s} -> {s['component']}{tag}")

    # Stable camera setup — only 2 angles to avoid dizzying rotation:
    #   HERO  = primary wide-angle (shows robot + device, slight angle)
    #   Close-up = per-component tracking camera (zooms into the part)
    # The camera alternates: HERO (approach) → close-up (removal) → HERO
    # Occasional overhead for milestone steps (panel open, battery disconnect)

    current_cam = CAM_HERO
    total_steps = len(plan)

    # Clear pinned joints at start of disassembly
    _pinned_joints.clear()

    for step_idx, step_info in enumerate(plan):
        comp_name = step_info["component"]
        skill = step_info["skill"]
        instructions = step_info["instructions"]
        comp = step_info["component_spec"]

        step_progress = step_idx / total_steps
        step_num = f"Step {step_idx+1}/{total_steps}"
        step_title = instructions["mid"]
        comp_display = comp_name.replace("_", " ").title()
        print(f"  -> [{step_num}] {step_title}")

        # Resolve target position
        target_pos = np.array([0.5, 0.0, 0.45])
        if comp.site_name:
            sid = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_SITE, comp.site_name)
            if sid >= 0:
                target_pos = env.data.site_xpos[sid].copy()

        # Use HERO for most steps, OVERHEAD for milestone moments
        is_milestone = comp_name in (
            "back_panel", "battery", "fan_assembly",          # laptop milestones
            "top_cover", "cmos_battery",                       # router milestones
        )
        wide_cam = CAM_OVERHEAD if is_milestone else CAM_HERO

        # ── Smooth camera transition to wide (only if different) ──
        if current_cam is not wide_cam:
            F += camera_transition(env, current_cam, wide_cam,
                                   int(fps * 0.4), width, height,
                                   step_num, step_title, step_progress)

        # ── Robot approach from wide angle ──
        F += robot_approach_shot(
            env, target_pos, skill, wide_cam, width, height,
            num_frames=int(fps * 1.0),
            title=step_num, subtitle=f"Approaching: {comp_display}",
            progress=step_progress)

        # ── Zoom into component close-up ──
        closeup_cam = make_component_closeup(env, comp_name, target_pos)

        F += camera_transition(env, wide_cam, closeup_cam,
                               int(fps * 0.4), width, height,
                               step_num, f"Detail: {comp_display}",
                               step_progress)

        # ── Animated removal from close-up ──
        removal_subtitle = f"Removing: {comp_display}"
        F += animate_removal(
            env, comp_name, closeup_cam, width, height,
            animate_frames=int(fps * 2.0),
            settle_frames=int(fps * 0.3),
            title=step_num, subtitle=removal_subtitle,
            progress=step_progress)

        # ── Pull back to wide for brief "done" confirmation ──
        F += camera_transition(env, closeup_cam, wide_cam,
                               int(fps * 0.3), width, height,
                               "Completed", step_title,
                               step_progress)
        F += camera_hold(env, wide_cam, int(fps * 0.3), width, height,
                         "Completed", step_title,
                         step_progress)

        current_cam = wide_cam

    # ╔═══════════════════════════════════════════════════════╗
    # ║  ACT 3 — CLOSING (≈8 s)                             ║
    # ╚═══════════════════════════════════════════════════════╝
    print("\n[ACT 3] Closing sequence ...")

    # Transition to hero angle
    F += camera_transition(env, current_cam, CAM_HERO,
                           int(fps * 0.6), width, height)

    # Full 360° orbit around disassembled device
    F += camera_orbit(
        env, [0.5, 0, 0.44], 0.55,
        start_az=20, end_az=380,
        elevation=-30,
        num_frames=int(fps * 4.0),
        width=width, height=height,
        title="Disassembly Complete",
        subtitle=f"{total_steps} components safely removed",
        progress=1.0)

    # Final title card
    end_cam = CameraState([0.5, 0, 0.44], 0.55, 380, -25)
    title_frames = int(fps * 3.0)
    base_title = render_from_state(env, end_cam, width, height)
    for i in range(title_frames):
        frame = base_title.copy()
        a = compute_text_alpha(i, title_frames, fade_in=25, fade_out=20)
        frame = overlay_title_card(
            frame, "SafeDisassemble",
            "Hierarchical VLA  |  Safety-Aware  |  Vision-Language-Action", a)
        F.append(frame)

    # Close renderer
    for r in _renderer_cache.values():
        r.close()
    _renderer_cache.clear()
    env.close()

    # ═══════════════════════════════════════════════════════════
    #  Export
    # ═══════════════════════════════════════════════════════════
    duration = len(F) / fps
    print(f"\n{'='*50}")
    print(f"  Total frames : {len(F)}")
    print(f"  Duration     : {duration:.1f}s @ {fps} FPS")
    print(f"  Resolution   : {width}x{height}")
    print(f"{'='*50}")

    # Key frames
    key_idxs = np.linspace(0, len(F) - 1, 10, dtype=int)
    for ki in key_idxs:
        save_image(F[ki], out / f"keyframe_{ki:05d}.png")
    print(f"  Saved {len(key_idxs)} key frames -> {out}/")

    if make_video:
        video_path = out / f"{device_name}_cinematic.mp4"
        print(f"  Encoding -> {video_path} ...")
        ok = frames_to_video(F, video_path, fps=fps)
        if ok:
            mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  [OK] {video_path} ({mb:.1f} MB, {duration:.1f}s)")
        else:
            fd = out / "frames"
            fd.mkdir(exist_ok=True)
            for i, f in enumerate(F):
                save_image(f, fd / f"frame_{i:05d}.png")
            print(f"  Saved {len(F)} PNGs -> {fd}/")
            print(f"  ffmpeg -framerate {fps} -i {fd}/frame_%05d.png "
                  f"-c:v libx264 -pix_fmt yuv420p {video_path}")

    save_contact_sheet(F, out / f"{device_name}_contact_sheet.png")
    print(f"\n[DONE]  -> {out / f'{device_name}_cinematic.mp4'}")
    print(f"\nTo view: open {out / f'{device_name}_cinematic.mp4'}")


def main():
    p = argparse.ArgumentParser(description="Cinematic disassembly demo (16:9)")
    p.add_argument("--device", default="laptop_v1")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--output-dir", default="renders/cinematic")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    run_cinematic_demo(a.device, a.width, a.height, a.output_dir,
                       not a.no_video, a.fps)


if __name__ == "__main__":
    main()
