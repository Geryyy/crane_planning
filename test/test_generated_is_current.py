"""`generated/` is still what `scripts/export_timing_ocp.py` writes."""

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent


def test_the_checked_in_solver_is_current():
    """
    The staleness gate: only three files of the tree are committed, the rest
    reaches the diff as `CRANE_PLANNING_OCP_GENERATED_DIGEST`, so this is the
    only reason a re-export ever happens.

    Subprocess, not import: the exporter puts `crane_ocp/scripts` on
    `sys.path` and regenerates a solver, none of which belongs in-process.
    `--check` writes into scratch and `main()` skips `compile_solver` under it,
    so no cache build here -- `--no-compile` would be redundant.
    """
    check = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "scripts" / "export_timing_ocp.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
