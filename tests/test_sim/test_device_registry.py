"""Tests for device registry and dependency graph validation."""

import pytest

from safedisassemble.sim.device_registry import (
    DEVICE_REGISTRY,
    LAPTOP_SPEC,
    ROUTER_SPEC,
    ComponentType,
    get_device,
    list_devices,
)


class TestDeviceRegistry:
    def test_list_devices(self):
        devices = list_devices()
        assert "laptop_v1" in devices
        assert "router_v1" in devices

    def test_get_device(self):
        spec = get_device("laptop_v1")
        assert spec.name == "laptop_v1"
        assert spec.device_type == "laptop"

    def test_get_unknown_device_raises(self):
        with pytest.raises(KeyError):
            get_device("nonexistent_device")


class TestLaptopSpec:
    def test_has_required_components(self):
        comp_types = {c.component_type for c in LAPTOP_SPEC.components}
        assert ComponentType.SCREW in comp_types
        assert ComponentType.PANEL in comp_types
        assert ComponentType.BATTERY in comp_types
        assert ComponentType.RAM in comp_types
        assert ComponentType.SSD in comp_types

    def test_battery_has_safety_zone(self):
        battery_zones = [
            z for z in LAPTOP_SPEC.safety_zones
            if "battery" in z.site_name.lower()
        ]
        assert len(battery_zones) > 0

    def test_dependency_chain_screws_to_panel(self):
        prereqs = LAPTOP_SPEC.get_prerequisites("back_panel")
        # All 5 screws should be prerequisites
        assert len(prereqs) == 5
        assert all("screw" in p for p in prereqs)

    def test_dependency_panel_to_internals(self):
        battery_prereqs = LAPTOP_SPEC.get_prerequisites("battery")
        assert "back_panel" in battery_prereqs

        ram_prereqs = LAPTOP_SPEC.get_prerequisites("ram_module")
        assert "back_panel" in ram_prereqs

    def test_safety_dependency_battery_before_ram(self):
        ram_prereqs = LAPTOP_SPEC.get_prerequisites("ram_module")
        assert "battery" in ram_prereqs

    def test_valid_removal_order(self):
        # Valid: screws → panel → battery → ram
        order = [
            "screw_1", "screw_2", "screw_3", "screw_4", "screw_5",
            "back_panel", "battery", "ssd_screw", "ssd_module",
            "ram_module", "fan_assembly",
        ]
        valid, msg = LAPTOP_SPEC.validate_removal_order(order)
        assert valid, msg

    def test_invalid_removal_order(self):
        # Invalid: trying to remove panel before screws
        order = ["back_panel", "screw_1"]
        valid, msg = LAPTOP_SPEC.validate_removal_order(order)
        assert not valid
        assert "screw" in msg.lower()

    def test_invalid_safety_order(self):
        # Invalid: removing RAM before battery
        order = [
            "screw_1", "screw_2", "screw_3", "screw_4", "screw_5",
            "back_panel", "ram_module",  # battery not yet removed!
        ]
        valid, msg = LAPTOP_SPEC.validate_removal_order(order)
        assert not valid


class TestRouterSpec:
    def test_has_snap_clips(self):
        clips = [c for c in ROUTER_SPEC.components if c.component_type == ComponentType.CLIP]
        assert len(clips) == 4

    def test_clips_prerequisite_for_cover(self):
        prereqs = ROUTER_SPEC.get_prerequisites("top_cover")
        clip_prereqs = [p for p in prereqs if "clip" in p]
        assert len(clip_prereqs) == 4
