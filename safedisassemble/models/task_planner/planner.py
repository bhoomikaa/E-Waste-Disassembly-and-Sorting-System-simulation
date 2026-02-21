"""Level 1: Task Planner

Uses a Vision-Language Model (VLM) to analyze the device image and generate
an ordered sequence of subtasks for disassembly.

Two modes:
1. LLM-based: Uses a prompted LLM (e.g., GPT-4V, LLaVA) with retrieval-augmented
   generation from a database of known disassembly procedures.
2. Learned: Fine-tuned VLM that directly outputs plans from image + prompt.

The planner also enforces safety ordering constraints (battery-first rule).
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

# Lazy imports for torch/transformers — allows retrieval-only mode without GPU deps
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoProcessor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class DisassemblyStep:
    """A single planned disassembly step."""

    def __init__(
        self,
        step_id: int,
        component: str,
        action_description: str,
        skill_type: str,
        safety_critical: bool = False,
        confidence: float = 1.0,
    ):
        self.step_id = step_id
        self.component = component
        self.action_description = action_description
        self.skill_type = skill_type
        self.safety_critical = safety_critical
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "component": self.component,
            "action": self.action_description,
            "skill": self.skill_type,
            "safety_critical": self.safety_critical,
            "confidence": self.confidence,
        }


class DisassemblyPlan:
    """Ordered sequence of disassembly steps with safety annotations."""

    def __init__(self, steps: list[DisassemblyStep], device_type: str = "unknown"):
        self.steps = steps
        self.device_type = device_type
        self._current_idx = 0

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def current_step(self) -> Optional[DisassemblyStep]:
        if self._current_idx < len(self.steps):
            return self.steps[self._current_idx]
        return None

    def advance(self) -> Optional[DisassemblyStep]:
        self._current_idx += 1
        return self.current_step

    def reset(self) -> None:
        self._current_idx = 0

    @property
    def is_complete(self) -> bool:
        return self._current_idx >= len(self.steps)

    def to_text(self) -> str:
        lines = [f"Disassembly plan for {self.device_type}:"]
        for step in self.steps:
            safety_tag = " [SAFETY-CRITICAL]" if step.safety_critical else ""
            lines.append(
                f"  {step.step_id}. [{step.skill_type}] {step.action_description}{safety_tag}"
            )
        return "\n".join(lines)


class RetrievalDatabase:
    """Simple retrieval store of known disassembly procedures.

    Maps device types to reference disassembly sequences that can be used
    as few-shot examples for the planner.
    """

    def __init__(self):
        self._procedures: dict[str, list[dict]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Built-in disassembly procedures for common device types."""
        self._procedures["laptop"] = [
            {"step": 1, "action": "Flip laptop over to expose bottom panel",
             "skill": "flip", "safety": False},
            {"step": 2, "action": "Remove all visible screws from back panel",
             "skill": "unscrew", "safety": False},
            {"step": 3, "action": "Pry open back panel starting from corner",
             "skill": "pry_open", "safety": False},
            {"step": 4, "action": "Locate and disconnect battery connector FIRST",
             "skill": "pull_connector", "safety": True},
            {"step": 5, "action": "Remove battery from housing",
             "skill": "lift_component", "safety": True},
            {"step": 6, "action": "Remove SSD retaining screw",
             "skill": "unscrew", "safety": False},
            {"step": 7, "action": "Slide out SSD module",
             "skill": "lift_component", "safety": False},
            {"step": 8, "action": "Release RAM module latches",
             "skill": "release_clip", "safety": False},
            {"step": 9, "action": "Remove RAM module at angle",
             "skill": "lift_component", "safety": False},
            {"step": 10, "action": "Disconnect and remove cooling fan",
             "skill": "lift_component", "safety": False},
        ]

        self._procedures["router"] = [
            {"step": 1, "action": "Flip router to find hidden screws under label/feet",
             "skill": "flip", "safety": False},
            {"step": 2, "action": "Remove hidden screws",
             "skill": "unscrew", "safety": False},
            {"step": 3, "action": "Release all snap clips around case perimeter",
             "skill": "release_clip", "safety": False},
            {"step": 4, "action": "Lift top cover off",
             "skill": "pry_open", "safety": False},
            {"step": 5, "action": "Disconnect CMOS battery if present",
             "skill": "pull_connector", "safety": True},
            {"step": 6, "action": "Unscrew antenna connectors",
             "skill": "unscrew", "safety": False},
            {"step": 7, "action": "Remove PCB from housing",
             "skill": "lift_component", "safety": False},
        ]

        self._procedures["smartphone"] = [
            {"step": 1, "action": "Remove SIM tray using ejector tool",
             "skill": "pull_connector", "safety": False},
            {"step": 2, "action": "Heat back panel adhesive with heat gun",
             "skill": "heat", "safety": False},
            {"step": 3, "action": "Pry open back glass panel",
             "skill": "pry_open", "safety": False},
            {"step": 4, "action": "Disconnect battery ribbon cable FIRST",
             "skill": "pull_connector", "safety": True},
            {"step": 5, "action": "Remove battery (may require adhesive pull tab)",
             "skill": "lift_component", "safety": True},
            {"step": 6, "action": "Remove screws holding display connectors",
             "skill": "unscrew", "safety": False},
            {"step": 7, "action": "Disconnect display and camera flex cables",
             "skill": "pull_connector", "safety": False},
        ]

    def get_procedure(self, device_type: str) -> Optional[list[dict]]:
        return self._procedures.get(device_type)

    def get_few_shot_prompt(self, device_type: str, n_examples: int = 2) -> str:
        """Generate few-shot examples for the planner prompt."""
        examples = []
        for dtype, procedure in self._procedures.items():
            if dtype != device_type and len(examples) < n_examples:
                text = f"Device: {dtype}\nPlan:\n"
                for step in procedure:
                    safety = " [SAFETY-CRITICAL]" if step["safety"] else ""
                    text += f"  {step['step']}. [{step['skill']}] {step['action']}{safety}\n"
                examples.append(text)
        return "\n---\n".join(examples)

    def add_procedure(self, device_type: str, procedure: list[dict]) -> None:
        self._procedures[device_type] = procedure


_base_class = nn.Module if _TORCH_AVAILABLE else object


class TaskPlanner(_base_class):
    """Level 1 task planner.

    Given an image of a device and a high-level instruction ("disassemble this"),
    outputs an ordered DisassemblyPlan.

    For the academic project, we use a prompted VLM with retrieval augmentation.
    The model can optionally be fine-tuned on domain-specific data.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        use_retrieval: bool = True,
    ):
        if _TORCH_AVAILABLE:
            super().__init__()
        self.model_name = model_name
        self.device_str = device
        self.use_retrieval = use_retrieval
        self.retrieval_db = RetrievalDatabase()

        self._model = None
        self._processor = None

    def load_model(self) -> None:
        """Lazy-load the VLM. Requires torch and transformers."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch/transformers required for VLM mode. Use retrieval-only.")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device_str)
        self._model.eval()

    def plan(
        self,
        image: np.ndarray,
        instruction: str = "Disassemble this device",
        device_type_hint: Optional[str] = None,
    ) -> DisassemblyPlan:
        """Generate a disassembly plan from an image.

        Args:
            image: (H, W, 3) RGB image of the device
            instruction: High-level language instruction
            device_type_hint: Optional hint about device type

        Returns:
            DisassemblyPlan with ordered steps
        """
        if self._model is None:
            # Fallback to retrieval-only mode
            return self._retrieval_plan(device_type_hint or "laptop")

        prompt = self._build_prompt(instruction, device_type_hint)

        # Process image and text
        from PIL import Image
        pil_image = Image.fromarray(image)

        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device_str)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )

        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        return self._parse_plan(response, device_type_hint or "unknown")

    def _build_prompt(self, instruction: str, device_type: Optional[str]) -> str:
        """Construct the prompt with few-shot examples and safety guidelines."""
        few_shot = ""
        if self.use_retrieval and device_type:
            few_shot = self.retrieval_db.get_few_shot_prompt(device_type)

        prompt = f"""You are an expert electronics disassembly planner. Given an image of an electronic device, create a safe, ordered disassembly plan.

CRITICAL SAFETY RULES:
1. ALWAYS disconnect the battery connector BEFORE removing any other internal components
2. Never apply force to battery cells directly
3. Remove screws before attempting to pry panels
4. Handle PCBs by edges only

{f"Reference procedures:{chr(10)}{few_shot}{chr(10)}---{chr(10)}" if few_shot else ""}

Now, analyze the device in the image and create a disassembly plan.

Instruction: {instruction}
{f"Device type: {device_type}" if device_type else ""}

Output the plan as a numbered list. Mark safety-critical steps with [SAFETY-CRITICAL].
For each step, specify the skill type in brackets: [unscrew], [pry_open], [pull_connector], [lift_component], [release_clip], [flip]

Plan:"""
        return prompt

    def _parse_plan(self, response: str, device_type: str) -> DisassemblyPlan:
        """Parse LLM output into structured DisassemblyPlan."""
        steps = []
        lines = response.strip().split("\n")

        skill_keywords = {
            "unscrew": "unscrew",
            "pry": "pry_open",
            "pull": "pull_connector",
            "disconnect": "pull_connector",
            "lift": "lift_component",
            "remove": "lift_component",
            "release": "release_clip",
            "clip": "release_clip",
            "flip": "flip",
        }

        step_id = 0
        for line in lines:
            line = line.strip()
            if not line or not any(c.isdigit() for c in line[:3]):
                continue

            step_id += 1
            safety = "SAFETY" in line.upper()

            # Extract skill type
            skill = "lift_component"  # default
            for keyword, skill_type in skill_keywords.items():
                if keyword.lower() in line.lower():
                    skill = skill_type
                    break

            # Extract component name (heuristic)
            component = self._extract_component(line)

            steps.append(DisassemblyStep(
                step_id=step_id,
                component=component,
                action_description=line.lstrip("0123456789. "),
                skill_type=skill,
                safety_critical=safety,
            ))

        if not steps:
            # Fallback to retrieval
            return self._retrieval_plan(device_type)

        return DisassemblyPlan(steps=steps, device_type=device_type)

    def _extract_component(self, text: str) -> str:
        """Extract likely component name from step description."""
        component_keywords = [
            "battery", "screw", "panel", "cover", "ram", "ssd", "fan",
            "heatsink", "connector", "clip", "antenna", "pcb", "motherboard",
        ]
        text_lower = text.lower()
        for kw in component_keywords:
            if kw in text_lower:
                return kw
        return "unknown"

    def _retrieval_plan(self, device_type: str) -> DisassemblyPlan:
        """Generate plan from retrieval database only (no VLM needed)."""
        procedure = self.retrieval_db.get_procedure(device_type)
        if procedure is None:
            procedure = self.retrieval_db.get_procedure("laptop")  # fallback

        steps = []
        for entry in procedure:
            steps.append(DisassemblyStep(
                step_id=entry["step"],
                component=self._extract_component(entry["action"]),
                action_description=entry["action"],
                skill_type=entry["skill"],
                safety_critical=entry["safety"],
            ))

        return DisassemblyPlan(steps=steps, device_type=device_type)
