"""An off-centre grasp is a payload, not a contradiction."""

import math
import types

import numpy as np
import pytest
from crane_planning.a2b import translate_payload
from crane_planning.planner import PlanningError


def request(*, s_log_8, p_cyl_8):
    """A `CalcMovement` request carrying one 0.4 x 0.2 x 0.2 m block."""
    shape = types.SimpleNamespace(length=0.4, radius_top=0.1, radius_bottom=0.1)
    return types.SimpleNamespace(
        carries_log=True,
        log_carrying=shape,
        coll_shape=shape,
        m_log=120.0,
        s_log_8=types.SimpleNamespace(**dict(zip("xyz", s_log_8))),
        p_cyl_8=types.SimpleNamespace(**dict(zip("xyz", p_cyl_8))),
    )


def test_the_measured_lateral_offset_reaches_the_payload():
    """
    `p_cyl_8` is x-only, so its zero y is not a claim that contradicts `s_log_8`.

    The behaviour tree measures the grasp offset after a pick and sends it in
    `s_log_8`; every caller leaves `p_cyl_8.y` at the .srv default. Reading that
    default as a second opinion refused every move that followed a real pick.
    """
    payload, shape = translate_payload(
        request(s_log_8=(0.0, -0.0712, math.nan), p_cyl_8=(0.0, 0.0, 0.0))
    )
    assert np.allclose(payload.center_of_mass_k8_m, [0.0, -0.0712, 0.0])
    assert np.allclose(shape[2], [0.0, -0.0712, 0.0])


def test_the_axis_both_callers_write_still_has_to_agree():
    with pytest.raises(PlanningError, match="two different points along the body"):
        translate_payload(
            request(s_log_8=(0.3, 0.0, math.nan), p_cyl_8=(0.0, 0.0, 0.0))
        )
