"""Two machines must not share one compiled solver.

The node builds its planner from whatever `/robot_description` carries; the
exporter bakes from a file. A cached `.so` is loaded, not compared, so while the
cache name said nothing about the description, a solver compiled for the fixture
answered every request planned for the sim's machine -- which is a different
xacro at `sim_hydraulics:=false`. Costs no solver build: the name is a hash.
"""

from crane_planning.config import PlannerConfig
from crane_planning.ocp import baked_parameters, cache_tree

HYDRAULICS = {"pump_flow_max": PlannerConfig().pump_flow_max}


def tree(description: str):
    return cache_tree(baked_parameters(PlannerConfig()), HYDRAULICS, description)


def test_a_different_machine_is_a_different_tree():
    assert tree("<robot name='a'/>") != tree("<robot name='b'/>")


def test_the_same_machine_is_the_same_tree():
    assert tree("<robot name='a'/>") == tree("<robot name='a'/>")
