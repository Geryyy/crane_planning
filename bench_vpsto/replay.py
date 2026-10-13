"""Both planners' trajectories through the same passive-dynamics sim (CBS model, RK4)."""

import casadi as ca
import numpy as np
from common import description
from crane_model.symbolic import CraneSymbolicModel
from crane_planning.config import passive_equilibrium

BOX_Q, BOX_DQ = 0.02, 0.04  # settled box, planner's terminal_q/dq_sway_max
DT, HOLD = 5e-3, 20.0


class Replay:
    def __init__(self):
        model = CraneSymbolicModel(description(), "pzs100")
        self.ddq_u = ca.Function("ddq_u", [model.x, model.u, model.p], [model.ddq_u])

    def run(self, sampler, T, q_u0, params, offset=None):
        """
        Replay a plan.

        sampler(t) -> (q_a, dq_a, ddq_a) on [0, T]; actuated axes hold still after T.
        Starts at rest at q_u0. offset(q_a) shifts the equilibrium (payload), params = p.
        """
        n = int(np.ceil(T / DT))
        h = T / n
        qa, dqa, ddqa = sampler(np.linspace(0.0, T, 2 * n + 1))
        hold_n = int(HOLD / DT)
        hold = (qa[-1], np.zeros(5), np.zeros(5))

        def rhs(a, v, acc, s):
            x = np.concatenate([a, s[:2], v, s[2:]])
            return np.concatenate(
                [s[2:], np.asarray(self.ddq_u(x, acc, params)).ravel()]
            )

        s = np.concatenate([q_u0, np.zeros(2)])
        history = [s]
        for k in range(n + hold_n):
            step = h if k < n else DT
            if k < n:
                i = 2 * k
                a0, a1, a2 = ((qa[j], dqa[j], ddqa[j]) for j in (i, i + 1, i + 2))
            else:
                a0 = a1 = a2 = hold
            k1 = rhs(*a0, s)
            k2 = rhs(*a1, s + 0.5 * step * k1)
            k3 = rhs(*a1, s + 0.5 * step * k2)
            k4 = rhs(*a2, s + step * k3)
            s = s + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            history.append(s)
        history = np.array(history)
        t = np.concatenate([np.arange(n + 1) * h, T + np.arange(1, hold_n + 1) * DT])
        q_a = np.vstack([qa[0::2], np.repeat(qa[-1:], hold_n, 0)])
        stride = max(
            1, n // 40
        )  # equilibrium offset is a numeric solve, so coarse in time
        eq = np.array([passive_equilibrium(a) for a in q_a])
        if offset is not None:
            keep = np.arange(0, n + 1, stride)
            shift = np.array([offset(q_a[i]) for i in keep] + [offset(q_a[-1])])
            at = np.concatenate([t[keep], t[-1:]])
            eq += np.stack([np.interp(t, at, shift[:, c]) for c in range(2)], axis=1)
        sway = history[:, :2] - eq
        rate = history[:, 2:]
        inside = (np.abs(sway) < BOX_Q).all(1) & (np.abs(rate) < BOX_DQ).all(1)
        outside = np.flatnonzero(~inside[n:])
        settle = (
            0.0
            if not outside.size
            else (np.inf if outside[-1] == hold_n else (outside[-1] + 1) * DT)
        )
        return dict(
            t=t[: n + 1],
            q_a=q_a[: n + 1],
            q_u=history[: n + 1, :2],
            terminal_sway=float(np.abs(sway[n]).max()),
            terminal_rate=float(np.abs(rate[n]).max()),
            peak_sway=float(np.abs(sway[: n + 1]).max()),
            settled_at_T=bool(inside[n]),
            settle_s=float(settle),
        )
