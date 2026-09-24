"""
The exporter's options are `crane_ocp_export`'s, and its default output is the
directory the node opens. Both were broken at once by the extraction: the call
site kept an older `add_arguments` signature, and the node kept a `$TMPDIR`
cache no export wrote to. Neither shows up until an operator runs the command a
refusal told them to run, so it is pinned here instead.
"""

import importlib.util
from pathlib import Path

from crane_planning import ocp

PACKAGE = Path(__file__).resolve().parents[1]


def exporter():
    """`scripts/` is not a package; load the script the way `ros2 run` would."""
    spec = importlib.util.spec_from_file_location(
        "export_timing_ocp", PACKAGE / "scripts" / "export_timing_ocp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_exporter_writes_where_the_node_reads():
    assert exporter().build_parser().parse_args([]).output == ocp.CACHE
