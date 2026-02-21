"""Registry of device models and their disassembly metadata.

Each device entry defines:
- MJCF file path
- Component inventory (what can be extracted)
- Disassembly graph (valid orderings with dependencies)
- Safety constraints (battery zones, force limits)
- Difficulty rating
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Optional


class ComponentType(Enum):
    SCREW = "screw"
    PANEL = "panel"
    BATTERY = "battery"
    RAM = "ram"
    SSD = "ssd"
    FAN = "fan"
    HEATSINK = "heatsink"
    PCB = "pcb"
    CONNECTOR = "connector"
    CLIP = "clip"
    ANTENNA = "antenna"
    CMOS_BATTERY = "cmos_battery"


class HazardType(Enum):
    BATTERY_PUNCTURE = "battery_puncture"
    PCB_SNAP = "pcb_snap"
    STRIPPED_SCREW = "stripped_screw"
    CONNECTOR_DAMAGE = "connector_damage"
    ELECTROSTATIC = "electrostatic"


@dataclasses.dataclass(frozen=True)
class Component:
    name: str
    component_type: ComponentType
    body_name: str  # MuJoCo body name
    site_name: Optional[str] = None  # interaction site
    joint_names: tuple[str, ...] = ()
    value_score: float = 1.0  # relative recovery value
    removal_force_limit: float = 20.0  # Newtons


@dataclasses.dataclass(frozen=True)
class SafetyZone:
    hazard_type: HazardType
    site_name: str  # MuJoCo site marking the danger zone
    force_threshold: float  # Newtons - exceeding this triggers failure
    description: str = ""


@dataclasses.dataclass(frozen=True)
class DependencyRule:
    """Component B cannot be removed until component A is removed first."""
    prerequisite: str  # component name
    dependent: str  # component name
    reason: str = ""


@dataclasses.dataclass
class DeviceSpec:
    name: str
    device_type: str
    mjcf_path: str
    components: list[Component]
    safety_zones: list[SafetyZone]
    dependencies: list[DependencyRule]
    difficulty: float = 1.0  # 1.0 = easy, 5.0 = very hard
    description: str = ""

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def removable_components(self) -> list[Component]:
        return [c for c in self.components if c.joint_names]

    def get_component(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"Component '{name}' not found in {self.name}")

    def get_prerequisites(self, component_name: str) -> list[str]:
        return [
            d.prerequisite for d in self.dependencies
            if d.dependent == component_name
        ]

    def validate_removal_order(self, order: list[str]) -> tuple[bool, str]:
        removed = set()
        for comp_name in order:
            prereqs = self.get_prerequisites(comp_name)
            for p in prereqs:
                if p not in removed:
                    return False, f"Cannot remove '{comp_name}' before '{p}'"
            removed.add(comp_name)
        return True, "Valid order"


ASSET_DIR = Path(__file__).parent / "assets" / "xmls"


LAPTOP_SPEC = DeviceSpec(
    name="laptop_v1",
    device_type="laptop",
    mjcf_path=str(ASSET_DIR / "laptop_device.xml"),
    description="Standard laptop with screwed back panel, Li-ion battery, RAM, and SSD",
    difficulty=3.0,
    components=[
        Component("screw_1", ComponentType.SCREW, "screw_1", "screw_1_site",
                  ("screw_1_turn", "screw_1_slide"), value_score=0.1, removal_force_limit=5.0),
        Component("screw_2", ComponentType.SCREW, "screw_2", "screw_2_site",
                  ("screw_2_turn", "screw_2_slide"), value_score=0.1, removal_force_limit=5.0),
        Component("screw_3", ComponentType.SCREW, "screw_3", "screw_3_site",
                  ("screw_3_turn", "screw_3_slide"), value_score=0.1, removal_force_limit=5.0),
        Component("screw_4", ComponentType.SCREW, "screw_4", "screw_4_site",
                  ("screw_4_turn", "screw_4_slide"), value_score=0.1, removal_force_limit=5.0),
        Component("screw_5", ComponentType.SCREW, "screw_5", "screw_5_site",
                  ("screw_5_turn", "screw_5_slide"), value_score=0.1, removal_force_limit=5.0),
        Component("back_panel", ComponentType.PANEL, "back_panel", None,
                  ("panel_slide_z",), value_score=0.5, removal_force_limit=15.0),
        Component("battery", ComponentType.BATTERY, "battery", "battery_connector_site",
                  ("battery_connector",), value_score=5.0, removal_force_limit=10.0),
        Component("ram_module", ComponentType.RAM, "ram_module", "ram_site",
                  ("ram_slot", "ram_latch"), value_score=8.0, removal_force_limit=8.0),
        Component("ssd_module", ComponentType.SSD, "ssd_module", "ssd_site",
                  ("ssd_slot",), value_score=10.0, removal_force_limit=6.0),
        Component("ssd_screw", ComponentType.SCREW, "ssd_screw", "ssd_screw_site",
                  ("ssd_screw_turn",), value_score=0.1, removal_force_limit=4.0),
        Component("fan_assembly", ComponentType.FAN, "fan_assembly", "fan_site",
                  ("fan_mount",), value_score=2.0, removal_force_limit=8.0),
    ],
    safety_zones=[
        SafetyZone(HazardType.BATTERY_PUNCTURE, "battery_puncture_zone", 15.0,
                   "Li-ion battery cell - puncture causes thermal runaway"),
        SafetyZone(HazardType.PCB_SNAP, "pcb_stress_zone", 20.0,
                   "PCB flex zone - excessive force snaps the board"),
    ],
    dependencies=[
        DependencyRule("screw_1", "back_panel", "Panel is screwed down"),
        DependencyRule("screw_2", "back_panel", "Panel is screwed down"),
        DependencyRule("screw_3", "back_panel", "Panel is screwed down"),
        DependencyRule("screw_4", "back_panel", "Panel is screwed down"),
        DependencyRule("screw_5", "back_panel", "Panel is screwed down"),
        DependencyRule("back_panel", "battery", "Battery under panel"),
        DependencyRule("back_panel", "ram_module", "RAM under panel"),
        DependencyRule("back_panel", "ssd_module", "SSD under panel"),
        DependencyRule("back_panel", "fan_assembly", "Fan under panel"),
        DependencyRule("ssd_screw", "ssd_module", "SSD is screwed in"),
        # SAFETY: battery must be disconnected before removing other internals
        DependencyRule("battery", "ram_module", "Safety: disconnect battery first"),
        DependencyRule("battery", "fan_assembly", "Safety: disconnect battery first"),
    ],
)


ROUTER_SPEC = DeviceSpec(
    name="router_v1",
    device_type="router",
    mjcf_path=str(ASSET_DIR / "router_device.xml"),
    description="Wireless router with snap-fit casing, hidden screws, and CMOS battery",
    difficulty=2.0,
    components=[
        Component("top_cover", ComponentType.PANEL, "top_cover", None,
                  ("cover_lift",), value_score=0.3, removal_force_limit=20.0),
        Component("clip_front", ComponentType.CLIP, "clip_front", "clip_front_site",
                  ("clip_front_j",), value_score=0.0, removal_force_limit=10.0),
        Component("clip_back", ComponentType.CLIP, "clip_back", "clip_back_site",
                  ("clip_back_j",), value_score=0.0, removal_force_limit=10.0),
        Component("clip_left", ComponentType.CLIP, "clip_left", "clip_left_site",
                  ("clip_left_j",), value_score=0.0, removal_force_limit=10.0),
        Component("clip_right", ComponentType.CLIP, "clip_right", "clip_right_site",
                  ("clip_right_j",), value_score=0.0, removal_force_limit=10.0),
        Component("hidden_screw_1", ComponentType.SCREW, "hidden_screw_1", "hscrew_1_site",
                  ("hscrew_1_turn", "hscrew_1_slide"), value_score=0.1, removal_force_limit=4.0),
        Component("hidden_screw_2", ComponentType.SCREW, "hidden_screw_2", "hscrew_2_site",
                  ("hscrew_2_turn", "hscrew_2_slide"), value_score=0.1, removal_force_limit=4.0),
        Component("cmos_battery", ComponentType.CMOS_BATTERY, "cmos_battery", "cmos_battery_site",
                  ("cmos_batt_conn",), value_score=0.5, removal_force_limit=5.0),
        Component("antenna_1", ComponentType.ANTENNA, "antenna_conn_1", "ant_conn_1_site",
                  ("ant_conn_1",), value_score=1.0, removal_force_limit=6.0),
        Component("antenna_2", ComponentType.ANTENNA, "antenna_conn_2", "ant_conn_2_site",
                  ("ant_conn_2",), value_score=1.0, removal_force_limit=6.0),
    ],
    safety_zones=[
        SafetyZone(HazardType.BATTERY_PUNCTURE, "cmos_battery_zone", 8.0,
                   "CMOS coin cell - lower risk than Li-ion but still hazardous"),
    ],
    dependencies=[
        DependencyRule("clip_front", "top_cover", "Cover held by clips"),
        DependencyRule("clip_back", "top_cover", "Cover held by clips"),
        DependencyRule("clip_left", "top_cover", "Cover held by clips"),
        DependencyRule("clip_right", "top_cover", "Cover held by clips"),
        DependencyRule("hidden_screw_1", "top_cover", "Hidden screws hold cover"),
        DependencyRule("hidden_screw_2", "top_cover", "Hidden screws hold cover"),
        DependencyRule("top_cover", "cmos_battery", "Battery under cover"),
        DependencyRule("top_cover", "antenna_1", "Connector under cover"),
        DependencyRule("top_cover", "antenna_2", "Connector under cover"),
    ],
)


# Central registry
DEVICE_REGISTRY: dict[str, DeviceSpec] = {
    "laptop_v1": LAPTOP_SPEC,
    "router_v1": ROUTER_SPEC,
}


def get_device(name: str) -> DeviceSpec:
    if name not in DEVICE_REGISTRY:
        available = ", ".join(DEVICE_REGISTRY.keys())
        raise KeyError(f"Device '{name}' not found. Available: {available}")
    return DEVICE_REGISTRY[name]


def list_devices() -> list[str]:
    return list(DEVICE_REGISTRY.keys())


def register_device(spec: DeviceSpec) -> None:
    DEVICE_REGISTRY[spec.name] = spec
