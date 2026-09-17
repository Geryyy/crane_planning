"""Cached .so is loaded, not compared -- name hashes description or two machines share one."""

import pytest
from crane_planning.config import PlannerConfig
from crane_planning.ocp import (
    SolverNotExported,
    TrajectoryOcp,
    baked_parameters,
    cache_tree,
)

HYDRAULICS = {"pump_flow_max": PlannerConfig().pump_flow_max}


def tree(description: str):
    return cache_tree(baked_parameters(PlannerConfig()), HYDRAULICS, description)


def test_a_different_machine_is_a_different_tree():
    assert tree("<robot name='a'/>") != tree("<robot name='b'/>")


def test_the_same_machine_is_the_same_tree():
    assert tree("<robot name='a'/>") == tree("<robot name='a'/>")


def test_a_machine_nobody_exported_for_is_refused():
    """Refuses before paying for a model build."""
    with pytest.raises(SolverNotExported):
        TrajectoryOcp("<robot name='nobody'/>", None, PlannerConfig(), {}, False)
