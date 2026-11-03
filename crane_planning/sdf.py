"""Analytic signed distance with the machine as spheres: the CES cost only, never the certificate."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from crane_model import Frame

#: Largest cell half extent a covering sphere spans; over-covers a face by at most ~0.1 m.
CELL = 0.15


def _cover(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spheres covering a box of half extents `half`: cell centres, radius to the cell corner."""
    counts = np.maximum(1, np.ceil(half / CELL)).astype(int)
    cell = half / counts
    axes = [np.linspace(-h + c, h - c, n) for h, c, n in zip(half, cell, counts)]
    centres = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    return centres, np.full(len(centres), float(np.linalg.norm(cell)))


class MachineSpheres:
    """`crane_model`'s link primitives (and a carried body) as spheres, placed by FK."""

    def __init__(self, model, payload_shape=None):
        # crane_model exposes no per-link placement; its own collision query reads these
        self.model = model
        self.base = model._model.getFrameId(Frame.MOUNTING_BASE.value)
        links = model._link_geometry()
        #: per part: owner id, frame, placement in it, exact shape, half extents, spheres
        self.parts = []
        for index, (link, frame, geometry, placement) in enumerate(links.bodies):
            if type(geometry).__name__ == "Capsule":
                # spheres on the axis, spacing <= r/2, grown to close the gaps between them
                r, h = float(geometry.radius), float(geometry.halfLength)
                count = int(np.ceil(4.0 * h / r)) + 1
                local = np.zeros((count, 3))
                local[:, 2] = np.linspace(-h, h, count)
                step = 2.0 * h / max(count - 1, 1)
                shape, half = "capsule", np.array([r, r, h])
                radius = np.full(count, np.hypot(r, 0.5 * step))
            else:
                shape, half = "box", np.asarray(geometry.halfSide, dtype=float)
                local, radius = _cover(half)
            # base is constant in q: crane_model never queries it against static bodies
            self.parts.append(
                (
                    index,
                    frame,
                    placement,
                    shape,
                    half,
                    local,
                    radius,
                    index in links.swinging,
                    link != Frame.MOUNTING_BASE.value,
                )
            )
        if payload_shape is not None:
            kind, dimensions, offset = payload_shape
            half = 0.5 * np.asarray(dimensions, dtype=float)
            local, radius = _cover(
                half
            )  # cylinder by its box's spheres, corners over-cover
            placement = pin.SE3(np.eye(3), np.asarray(offset, dtype=float))
            frame = model._model.getFrameId(Frame.ROTATOR_LOWER_PART.value)
            self.parts.append(
                (-1, frame, placement, kind, half, local, radius, True, True)
            )
        sizes = [len(part[5]) for part in self.parts]
        owner = np.repeat([part[0] for part in self.parts], sizes)
        self.radii = np.concatenate([part[6] for part in self.parts])
        #: below the tool's upper hinge: these owe the sway envelope
        self.swinging = np.repeat([part[7] for part in self.parts], sizes)
        #: spheres checked against the scene
        self.scene_mask = np.repeat([part[8] for part in self.parts], sizes)
        # the pairs crane_model checks: its self pairs, and a carried body (-1) against the arm
        pairs = list(links.self_pairs)
        if payload_shape is not None:
            pairs += [(-1, k) for k in links.detached]
        slot = {part[0]: n for n, part in enumerate(self.parts)}
        self.spheres_of = {
            part[0]: np.flatnonzero(owner == part[0]) for part in self.parts
        }
        self.bound = {
            part[0]: float(np.max(np.linalg.norm(part[5], axis=1) + part[6]))
            for part in self.parts
        }
        # per pair: the part with fewer spheres against the other's exact shape
        self.pairs = []
        for i, j in pairs:
            if i not in slot or j not in slot:
                continue
            if len(self.spheres_of[i]) > len(self.spheres_of[j]):
                i, j = j, i
            self.pairs.append((slot[i], slot[j], self.spheres_of[i]))

    def place(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(centres, rotations, origins)` in the mounting base at generalised `q`, per part."""
        model, data = self.model._model, self.model._data
        pin.forwardKinematics(model, data, self.model._description.configuration(q))
        pin.updateFramePlacement(model, data, self.base)
        base = data.oMf[self.base]
        centres, rotations, origins = [], [], []
        for _owner, frame, placement, _shape, _half, local, *_rest in self.parts:
            pin.updateFramePlacement(model, data, frame)
            pose = base.actInv(data.oMf[frame]) * placement
            centres.append(local @ pose.rotation.T + pose.translation)
            rotations.append(pose.rotation)
            origins.append(pose.translation)
        return np.vstack(centres), np.array(rotations), np.array(origins)

    def self_distance(self, centres, rotations, origins) -> np.ndarray:
        """
        Per sample (leading axis): smallest machine-to-itself distance.

        One part's spheres against the other's exact shape. A pair whose bounding spheres
        stay apart over every sample is not refined: its bound is a positive lower bound,
        and only a touch is penalised.
        """
        nearest = np.full(centres.shape[0], np.inf)
        for small, big, spheres in self.pairs:
            apart = (
                np.linalg.norm(origins[:, small] - origins[:, big], axis=-1)
                - self.bound[self.parts[small][0]]
                - self.bound[self.parts[big][0]]
            )
            if np.all(apart > 0.0):
                nearest = np.minimum(nearest, apart)
                continue
            _owner, _frame, _placement, shape, half, *_rest = self.parts[big]
            local = np.einsum(
                "nkj,nji->nki",
                centres[:, spheres] - origins[:, big, None],
                rotations[:, big],
            )
            distance = _shape_distance(local, shape, half) - self.radii[spheres]
            nearest = np.minimum(nearest, np.min(distance, axis=-1))
        return nearest


def _shape_distance(local: np.ndarray, shape: str, half: np.ndarray) -> np.ndarray:
    """Signed distance of points in a primitive's own frame, negative inside."""
    if shape == "sphere":
        return np.linalg.norm(local, axis=-1) - half[0]
    if shape == "capsule":
        axis = np.clip(local[..., 2], -half[2], half[2])
        return (
            np.linalg.norm(local - np.stack([0 * axis, 0 * axis, axis], -1), axis=-1)
            - half[0]
        )
    if shape == "cylinder":
        outside = np.stack(
            [
                np.linalg.norm(local[..., :2], axis=-1) - half[0],
                np.abs(local[..., 2]) - half[2],
            ],
            axis=-1,
        )
    else:
        outside = np.abs(local) - half
    return np.linalg.norm(np.maximum(outside, 0.0), axis=-1) + np.minimum(
        np.max(outside, axis=-1), 0.0
    )


def signed_distance(centres: np.ndarray, radii: np.ndarray, scene) -> np.ndarray:
    """Per row of `centres` (n, k, 3): smallest sphere-to-scene signed distance, `inf` if none."""
    nearest = np.full(centres.shape[0], np.inf)
    for primitive in scene:
        pose = primitive.pose_in_mounting_base
        local = (centres - pose.translation) @ pose.rotation
        half = 0.5 * np.asarray(primitive.dimensions_m, dtype=float)
        distance = _shape_distance(local, primitive.shape, half)
        nearest = np.minimum(nearest, np.min(distance - radii, axis=-1))
    return nearest
