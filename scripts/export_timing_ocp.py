#!/usr/bin/env python3
"""
Compile the trajectory OCP that `crane_planning.ocp.build_ocp` defines.

Definition lives in crane_planning/ocp.py; this is only the command line -- a
second copy here would drift.

    ./scripts/export_timing_ocp.py --fixture              # the test fixture
    ./scripts/export_timing_ocp.py --description live.urdf

Second compiles a solver for the machine on /robot_description; without it the
node refuses a description it wasn't baked for. Dump that description first:

    ./scripts/dump_robot_description.py live.urdf
    ./scripts/export_timing_ocp.py --description live.urdf

The solver lands in `ocp.CACHE` under `cache_key`, which is where the node loads
it from: `crane_ocp`'s persistent root, shared with `crane_mpc`'s export and not
swept with `/tmp`. `--output` moves it, and the node only follows if
`CRANE_PLANNING_OCP_CACHE` says the same. **There is no reviewable `generated/`
tree any more.**
`crane_ocp@1bc069d` retired the generate-into-git path for the whole stack --
it code-generated a tree nothing compiled, beside a hashed cache that generated
and compiled the same problem again, so the text a reviewer read was not the
artifact either node ran. The header that tree carried (`H_RATE` and friends,
the row divisors) had no consumer: this package is Python and nothing includes
it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent

# `crane_ocp_export` is installed (crane_ocp declares it); a from-scratch build
# of a sibling package runs this before that install exists, so fall back to the
# source tree -- same as crane_mpc/scripts/export_ocp.py.
try:
    import crane_ocp_export  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PACKAGE.parent / "crane_ocp"))

sys.path.insert(0, str(PACKAGE))

import crane_ocp_export as ox  # noqa: E402
from acados_template import AcadosOcpSolver  # noqa: E402
from crane_planning.config import PlannerConfig, read_limits  # noqa: E402
from crane_planning.ocp import (  # noqa: E402
    BAKED_LIMITS,
    DESCRIPTION,
    EXPORT_ENV,
    EXPORT_LEAF,
    build_ocp,
    cache_key,
    tree_of,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line, so a test can read it without running an export.

    `--descriptions`, `--output` and `--verbose` come from `crane_ocp_export`,
    the same call `crane_mpc/scripts/export_ocp.py` makes: its defaults are what
    the node reads, so a signature or convention change there has to fail here.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ox.add_arguments(parser, PACKAGE, EXPORT_ENV, EXPORT_LEAF)
    parser.add_argument(
        "--description",
        type=Path,
        help=(
            "the machine to bake, instead of the test fixture. What the node "
            "plans for is whatever `/robot_description` carries, which in sim is "
            "a different xacro than the fixture; dump it byte for byte with "
            "`scripts/dump_robot_description.py`"
        ),
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help=(
            "bake the test fixture instead. Only a fixture export is useful "
            "without a graph; it will not open in any node running the sim or "
            "the machine"
        ),
    )
    return parser


def compile_solver(
    description: str, parameters: dict, hydraulics: dict, base: Path, verbose: bool
) -> None:
    """
    Compile the solver into the cache the planner loads from.

    Without this the node refuses the description rather than pay code-gen +
    a C build (13 s warm, minutes cold) inside whatever asked for the plan.
    """
    ocp, _, _ = build_ocp(description, parameters, hydraulics)
    # Cache key covers path_segments, integrator, the row scales and the description
    # hash: cached .so is loaded not compared, so any change must land in a new
    # directory or the node silently reuses the old build. Node hashes
    # /robot_description and reports a mismatch instead of answering silently wrong.
    key = cache_key(parameters, hydraulics, description)
    # `--output` replaces the base only; the leaf is the name the node resolves.
    tree = base / tree_of(key).name
    tree.parent.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(tree)
    AcadosOcpSolver(ocp, json_file=str(tree.with_suffix(".json")), verbose=verbose)
    write_manifest(tree, key)
    print(f"compiled {tree} (description sha1 {key['description'][:10]})")


def main() -> int:
    arguments = build_parser().parse_args()

    # yaml over the declared defaults, which is what the node's declaration does.
    # Not the yaml alone: `command_k`, `command_u_*` and `pump_flow_max` were moved
    # out of it into crane_model and are only reachable through `PlannerConfig`.
    parameters = {
        **dataclasses.asdict(PlannerConfig()),
        **ox.read_ros_parameters(
            PACKAGE / "config" / "crane_planner.yaml", "crane_planner"
        ),
    }
    # Defaulting to the fixture is what made three runs of issue 155 record a
    # clean export for a robot no node runs: it compiles, prints success, and
    # the node then refuses the key. Which machine is baked has to be said.
    if arguments.description is None and not arguments.fixture:
        raise SystemExit(
            "refusing to guess which machine to bake. For a running graph:\n"
            "  scripts/dump_robot_description.py live.urdf\n"
            "  scripts/export_timing_ocp.py --description live.urdf\n"
            "or --fixture for the test fixture, which no node can open."
        )
    source = arguments.description or arguments.descriptions / DESCRIPTION
    print(f"baking {source}")
    description = source.read_text()
    # `dq_max`/`tau_max` are row scales the solver is generated with, and the node reads
    # them off its `Limits`; re-deriving them here from the description alone would export
    # a solver scaled by a different ceiling than the one that loads it.
    limits = read_limits(
        description,
        parameters["pump_flow_max"],
        parameters["pump_flow_planning_factor"],
    )
    parameters.update({name: getattr(limits, name) for name in BAKED_LIMITS})
    compile_solver(
        description, parameters, parameters, arguments.output, arguments.verbose
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
