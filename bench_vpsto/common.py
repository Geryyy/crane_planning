"""Path setup and the one description every bench_vpsto script shares."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANNING = HERE.parent
sys.path[:0] = [
    str(PLANNING / "scripts"),
    str(PLANNING),
    str(PLANNING.parent / "crane_model"),
    str(PLANNING.parent / "cbs_figures" / "examples"),
]
GEN = HERE / "gen"
URDF = PLANNING.parent / "crane_model" / "test" / "description" / "pzs100.urdf"


def description() -> str:
    return URDF.read_text()
