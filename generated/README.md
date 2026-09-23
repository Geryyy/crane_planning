# generated

`scripts/export_timing_ocp.py` writes this tree. Do not edit it; re-run the
script. `--check` regenerates into a scratch tree and diffs;
`test/test_generated_is_current.py` runs it.

Three files are committed -- this README, `crane_planning_ocp_generated.h` and
`crane_planning_ocp_pzs100/acados_solver_crane_planning_ocp_pzs100.h`. The rest
is gitignored and reaches `--check` through
`CRANE_PLANNING_OCP_GENERATED_DIGEST` in the header: sha256 over every
uncommitted generated file, so a description edit that only moves the CasADi
bodies still fails.

The horizon, the weights `W`, every box and every `L1` slack price are set on the
generated solver at run time, so moving any of them needs no re-export. What is
baked is the **structure**: the residual rows and their scaling, the row order of
`h`, and the description the dynamics were built from.
