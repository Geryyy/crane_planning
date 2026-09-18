"""An override for an undeclared name is dropped by rclpy in silence: happened to
`command_k`/`command_u_min`/`command_u_max` in crane_planner.yaml, unnoticed."""

from dataclasses import fields
from pathlib import Path

import yaml
from crane_planning import weights as crane_weights
from crane_planning.config import PlannerConfig
from crane_planning.node import (
    ARRAY_PARAMETERS,
    FLOAT_PARAMETERS,
    INT_PARAMETERS,
    OTHER_PARAMETERS,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "crane_planner.yaml"

TYPED = FLOAT_PARAMETERS + INT_PARAMETERS + ARRAY_PARAMETERS


def declared() -> set:
    """Every name `_declare` reaches, weights flattened as the yaml nests them."""
    return (
        set(TYPED)
        | set(OTHER_PARAMETERS)
        | {f"weights.{name}" for name in crane_weights.DEFAULTS}
    )


def written() -> set:
    """Every name the shipped yaml sets, weights flattened the same way."""
    parameters = yaml.safe_load(CONFIG.read_text())["crane_planner"]["ros__parameters"]
    names = set()
    for key, value in parameters.items():
        if key == "weights" and isinstance(value, dict):
            names |= {f"weights.{inner}" for inner in value}
        else:
            names.add(key)
    return names


def test_every_yaml_key_is_declared():
    assert written() - declared() == set()


def test_every_typed_parameter_is_a_config_field():
    """`_declare` reads its default off `PlannerConfig`, so a typo is fatal."""
    assert set(TYPED) - {field.name for field in fields(PlannerConfig)} == set()


# --- the pump is one number ---------------------------------------------------
# It lives in crane_model/config/hydraulics.yaml. `PlannerConfig` reads it there,
# the node declares that as the parameter default, and this yaml repeats neither.
def test_the_pump_is_the_one_in_crane_model():
    from crane_model import hydraulic_limits

    machine = hydraulic_limits()
    declared = PlannerConfig()
    assert declared.pump_flow_max == machine["pump_flow_max"]
    assert declared.pump_flow_planning_factor == machine["pump_flow_planning_factor"]

    deployed = yaml.safe_load(CONFIG.read_text())["crane_planner"]["ros__parameters"]
    assert "pump_flow_max" not in deployed
    assert "pump_flow_planning_factor" not in deployed


def test_the_actuator_and_its_domain_are_the_ones_in_crane_model():
    """The same discipline for C3: the planner refuses what the feedforward
    cannot deliver, so its `k` and its domain have to be the deliverer's."""
    import numpy as np
    from crane_model.symbolic import K_ACTUATOR_FIT
    from crane_model.velocity_loop import load_velocity_loop

    declared = PlannerConfig()
    assert np.array_equal(declared.command_k, np.asarray(K_ACTUATOR_FIT.k))
    assert declared.command_dead_time_s == K_ACTUATOR_FIT.dead_time_s

    # Psi's domain is the inner loop's clamp -- one number, not two that drift.
    from crane_model.conventions import ACTUATED_INDICES, canonical_joints

    gains, _ = load_velocity_loop()
    names = canonical_joints()
    joints = [names[index] for index in ACTUATED_INDICES[:5]]
    assert np.array_equal(
        declared.command_u_min, [gains[joint].u_clamp_min for joint in joints]
    )
    assert np.array_equal(
        declared.command_u_max, [gains[joint].u_clamp_max for joint in joints]
    )

    deployed = yaml.safe_load(CONFIG.read_text())["crane_planner"]["ros__parameters"]
    for name in ("command_k", "command_u_min", "command_u_max", "command_dead_time_s"):
        assert name not in deployed
