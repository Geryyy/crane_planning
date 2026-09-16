"""
An override for a name nobody declared is dropped in silence.

`rclpy` does not warn, raise or log: yaml key read, matched against nothing,
discarded; dataclass default runs. `command_k`, `command_u_min`, `command_u_max`
sat in `crane_planner.yaml` that way while the node read `PlannerConfig`'s copy.
They matched, so nothing moved and nothing said so.
"""

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
#
# `pump_flow_max` and `pump_flow_planning_factor` are also `crane_mpc`'s, and
# both packages turn them into an actuator limit. They had been typed into each
# separately -- 1.4e-3 here, 0.0014 there -- with nothing comparing them.
# `crane_model/config/hydraulics.yaml` is the source; this pins the copy.
def test_the_pump_is_the_one_in_crane_model():
    from crane_model.conventions import default_hydraulics_path

    with open(default_hydraulics_path(), encoding="utf-8") as handle:
        pump = yaml.safe_load(handle)["pump"]
    declared = PlannerConfig()
    deployed = yaml.safe_load(CONFIG.read_text())["crane_planner"]["ros__parameters"]

    for here, there in (
        ("pump_flow_max", "flow_max"),
        ("pump_flow_planning_factor", "planning_factor"),
    ):
        assert getattr(declared, here) == pump[there], here
        assert deployed[here] == pump[there], here
