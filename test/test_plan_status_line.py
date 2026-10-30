"""`/cbs/status/plan` carries the numbers the logged result line carries, or the refusal."""

from crane_planning.node import plan_status
from diagnostic_msgs.msg import DiagnosticStatus


def test_an_answered_request_carries_duration_iterations_and_arrival_sway():
    stats = {
        "duration_s": 7.4231,
        "sqp_iterations": 12.0,
        "terminal_sway": 0.021456,
        "terminal_sway_rate": 0.01234,
        "acados_status": 0.0,
    }
    line = plan_status(DiagnosticStatus.OK, "curved_over; 7.42 s, 2 lifted", stats)
    assert line == (
        "ok=1 duration=7.423 iterations=12 sway_rad=0.02146 sway_rate=0.01234 "
        "msg=curved_over; 7.42 s, 2 lifted"
    )


def test_a_slack_bought_answer_is_still_an_answer():
    # WARN, not ERROR: the plan went out, so the panel shows the planner line.
    line = plan_status(DiagnosticStatus.WARN, "on 0.3 of slack", {"duration_s": 5.0})
    assert line == "ok=1 duration=5 msg=on 0.3 of slack"


def test_a_refusal_carries_the_reason_and_no_numbers():
    line = plan_status(
        DiagnosticStatus.ERROR, "the goal is inside the headboard", {"duration_s": 1.0}
    )
    assert line == "ok=0 msg=the goal is inside the headboard"
