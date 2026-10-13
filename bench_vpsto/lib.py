"""ctypes face of libvpsto_crane.so: one VP-STO model bound to limits and a scene."""

import ctypes as C
import json

import numpy as np
from common import GEN, HERE

KIND = {"box": 0, "capsule": 1, "cylinder": 2, "sphere": 3}
NDOF = 5
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(str(HERE / "build" / "libvpsto_crane.so"))
        _lib.vc_create.restype = C.c_void_p
        _lib.vc_cost.restype = _lib.vc_collision.restype = C.c_double
        _lib.vc_pump.restype = _lib.vc_frames.restype = C.c_int
    return _lib


def _p(array):
    return np.ascontiguousarray(array, dtype=float).ctypes.data_as(
        C.POINTER(C.c_double)
    )


def casadi_call(name, *inputs, n_out):
    """Call the generated vc_pump / vc_frames, as the C++ cost does."""
    arrays = [np.ascontiguousarray(i, dtype=float) for i in inputs]
    out = np.zeros(n_out)
    arg = (C.c_void_p * len(arrays))(*[a.ctypes.data for a in arrays])
    res = (C.c_void_p * 1)(out.ctypes.data)
    getattr(lib(), name)(arg, res, None, None, 0)
    return out


class CraneVpsto:
    def __init__(
        self, dq_max, ddq_max, lower, upper, q_pump_max, q_tool, scene, payload_shape
    ):
        """scene: CollisionPrimitives in K0 (already truck-expanded); payload_shape as Planner's."""
        L = lib()
        self.h = C.c_void_p(
            L.vc_create(
                _p(dq_max),
                _p(ddq_max),
                _p(lower),
                _p(upper),
                C.c_double(q_pump_max),
                C.c_double(q_tool),
            )
        )
        for body in json.loads((GEN / "link_table.json").read_text())["bodies"]:
            L.vc_add_link(
                self.h,
                KIND[body["shape"]],
                _p(body["dims"]),
                C.c_double(body["weight"]),
            )
        for o in scene:
            pose = np.concatenate(
                [
                    np.asarray(o.pose_in_mounting_base.rotation).ravel(),
                    o.pose_in_mounting_base.translation,
                ]
            )
            L.vc_add_obstacle(self.h, KIND[o.shape], _p(o.dimensions_m), _p(pose))
        if payload_shape is not None:
            shape, dims, offset = payload_shape
            L.vc_set_payload(self.h, KIND[shape], _p(dims), _p(offset))

    def __del__(self):
        if getattr(self, "h", None) is not None:
            lib().vc_destroy(self.h)

    def solve(self, q0, qd, n_via=6, n_eval=80, n_samples=50, threads=8):
        via = np.zeros(NDOF * (n_via - 2))
        out = np.zeros(3)
        ev = np.zeros(3 * NDOF * n_eval)
        lib().vc_solve(
            self.h,
            _p(q0),
            _p(qd),
            n_via,
            n_eval,
            n_samples,
            threads,
            _p(via),
            _p(out),
            _p(ev),
        )
        ev = ev.reshape(3, n_eval, NDOF)
        return dict(
            via=via, T=out[0], cost=out[1], ms=out[2], q=ev[0], qp=ev[1], qpp=ev[2]
        )

    @staticmethod
    def sample(q0, qd, via, T, M):
        """M+1 uniform samples over [0, T] at exact T: (t, q, dq, ddq)."""
        n_via = via.size // NDOF + 2
        q, dq, ddq = (np.zeros(NDOF * (M + 1)) for _ in range(3))
        lib().vc_sample(
            _p(q0), _p(qd), _p(via), n_via, C.c_double(T), M, _p(q), _p(dq), _p(ddq)
        )
        return (
            np.linspace(0, T, M + 1),
            *(a.reshape(M + 1, NDOF) for a in (q, dq, ddq)),
        )

    def cost(self, q0, qd, via, n_eval=80):
        return lib().vc_cost(
            self.h, _p(q0), _p(qd), _p(via), via.size // NDOF + 2, n_eval
        )

    def collision(self, q5):
        return lib().vc_collision(self.h, _p(q5))
