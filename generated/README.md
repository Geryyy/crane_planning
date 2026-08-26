# `crane_planning/generated/` -- the shipped timing solver

Machine output. **Do not edit any file here**: change
`scripts/export_timing_ocp.py`, `crane_model/scripts/crane_symbolic.py` or one of
the config files below, re-run

    ./scripts/export_timing_ocp.py

and commit what changes. `./scripts/export_timing_ocp.py --check` regenerates into
a scratch tree and diffs, which is what says the tree still matches its inputs.

## What is in here

    crane_planning_timing_ocp_generated.h    the row offsets, the parameter
                                             packing and the conditioning
                                             divisors -- written by the export,
                                             read by `src/timing_ocp.cpp`
    crane_planning_timing_pzs100/            the PZS100 timing solver
    crane_planning_timing_epsilon7040/       the 7040 timing solver

Two solvers because the description is *baked in*: a generated solver is one
machine's, and `test_timing_ocp.cpp` solves on both.

Inside each solver directory, `crane_planning_timing_<tool>_output.{c,h}` is not
acados' -- it is `crane_symbolic`'s output map `z` code-generated beside the
solver, all six axes of `tau_a`, `F_cyl`, `v` and `Q` in physical units. acados
generates only what it solves, which is eleven rows already divided by their
conditioning constants, and three things need the physical quantity: `OcpNode`,
the static-force refusal and the bisected warm start.

It is the **same function `crane_mpc` ships**, up to the symbol prefix, because
both are `crane_symbolic`'s `z` over the module's own `(x, u, p)`.
`test_timing_ocp.cpp` compares the two trees on exactly that.

## There is no staleness guard, and that is a decision

The conditioning divisors come from `config/hydraulic_limits.yaml`, the smoothing
and the hydraulic constants from `crane_model/config/hydraulics.yaml`, and the
dynamics from the two descriptions under `crane_model/test/description/`.
**Nothing in the build or the test suite fails when one of those files and this
tree disagree.**

`docs/features/cbs-ocp-python/grill.md` §4 records the alternative that was
considered and rejected -- hashing the inputs into the generated code and
comparing in a test -- and records that this is a deliberate divergence from both
existing generator scripts in this repository.

What is **not** in the staleness surface is `config/crane_planner.yaml` and the
whole of `TimingOcpSettings`: kappa, the actuation limits, the boxes, the weights,
`intervals`, the tolerances and the wall-clock budget are every one of them set on
the generated solver before each solve. Only `max_iterations` has a ceiling, and
asking for more than it is refused rather than ignored.
