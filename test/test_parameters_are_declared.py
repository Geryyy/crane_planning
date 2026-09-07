"""
An override for a name nobody declared is dropped in silence.

`rclpy` does not warn, does not raise and does not log: the yaml key is read
from the file, matched against nothing, and discarded, and the dataclass default
runs instead. `command_k`, `command_u_min` and `command_u_max` sat in
`crane_planner.yaml` that way until 2026-09-07 -- the page that documents them
as the compensator's identified domain, and the node reading `PlannerConfig`'s
copy. They matched, so nothing moved and nothing said so, which is exactly the
failure this file exists to make loud.
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
