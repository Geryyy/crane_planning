"""Cached .so is loaded, not compared -- name hashes description or two machines share one."""

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
from crane_planning.config import PlannerConfig, read_limits
from crane_planning.ocp import (
    SolverNotExported,
    TrajectoryOcp,
    baked_parameters,
    cache_tree,
)

CONFIG = PlannerConfig()
HYDRAULICS = {"pump_flow_max": CONFIG.pump_flow_max}


@lru_cache(maxsize=1)
def limits():
    """Real machine's `Limits`; the description string is only hashed, so any is fine here."""
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    return read_limits(
        path.read_text(), CONFIG.pump_flow_max, CONFIG.pump_flow_planning_factor
    )


def tree(description: str, lim=None):
    return cache_tree(
        baked_parameters(CONFIG, lim or limits()), HYDRAULICS, description
    )


def test_a_different_machine_is_a_different_tree():
    assert tree("<robot name='a'/>") != tree("<robot name='b'/>")


def test_the_same_machine_is_the_same_tree():
    assert tree("<robot name='a'/>") == tree("<robot name='a'/>")


def test_a_narrower_control_safe_box_is_a_different_tree():
    """
    `dq_max` divides the rate row and is generated into the C, so a
    `control_safe_limits.yaml` edit that left the tree where it was would load a
    solver scaled by the old ceiling and report it as the new one. The
    description hash does not cover this: the box narrows it after the
    description is read.
    """
    narrower = replace(limits(), dq_max=0.5 * limits().dq_max)
    assert tree("<robot name='a'/>", narrower) != tree("<robot name='a'/>")


def test_a_machine_nobody_exported_for_is_refused():
    """Refuses before paying for a model build."""
    with pytest.raises(SolverNotExported):
        TrajectoryOcp("<robot name='nobody'/>", limits(), CONFIG, {}, False)
