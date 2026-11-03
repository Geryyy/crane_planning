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
        self.frames, centres, radii, swinging = [], [], [], []
        for index, (link, frame, geometry, placement) in enumerate(links.bodies):
            if link == Frame.MOUNTING_BASE.value:
                continue  # constant in q, never queried against static bodies
            if type(geometry).__name__ == "Capsule":
                half = np.array([geometry.radius, geometry.radius, geometry.halfLength])
            else:
                half = np.asarray(geometry.halfSide, dtype=float)
            local, radius = _cover(half)
            self.frames.append(frame)
            centres.append(local @ placement.rotation.T + placement.translation)
            radii.append(radius)
            swinging.append(np.full(len(local), index in links.swinging))
        if payload_shape is not None:
            _shape, dimensions, offset = (
                payload_shape  # cylinder by its box, corners over-cover
            )
            local, radius = _cover(0.5 * np.asarray(dimensions, dtype=float))
            self.frames.append(model._model.getFrameId(Frame.ROTATOR_LOWER_PART.value))
            centres.append(local + np.asarray(offset, dtype=float))
            radii.append(radius)
            swinging.append(np.ones(len(local), dtype=bool))
        self.local = centres
        self.radii = np.concatenate(radii)
        #: below the tool's upper hinge: these owe the sway envelope
        self.swinging = np.concatenate(swinging)

    def centres(self, q: np.ndarray) -> np.ndarray:
        """Sphere centres in the mounting base at generalised `q`."""
        model, data = self.model._model, self.model._data
        pin.forwardKinematics(model, data, self.model._description.configuration(q))
        pin.updateFramePlacement(model, data, self.base)
        base = data.oMf[self.base]
        placed = []
        for frame, local in zip(self.frames, self.local):
            pin.updateFramePlacement(model, data, frame)
            pose = base.actInv(data.oMf[frame])
            placed.append(local @ pose.rotation.T + pose.translation)
        return np.vstack(placed)


def signed_distance(centres: np.ndarray, radii: np.ndarray, scene) -> np.ndarray:
    """Per row of `centres` (n, k, 3): smallest sphere-to-scene signed distance, `inf` if none."""
    nearest = np.full(centres.shape[0], np.inf)
    for primitive in scene:
        pose = primitive.pose_in_mounting_base
        local = (centres - pose.translation) @ pose.rotation
        half = 0.5 * np.asarray(primitive.dimensions_m, dtype=float)
        if primitive.shape == "sphere":
            distance = np.linalg.norm(local, axis=-1) - half[0]
        else:
            if primitive.shape == "cylinder":
                outside = np.stack(
                    [
                        np.linalg.norm(local[..., :2], axis=-1) - half[0],
                        np.abs(local[..., 2]) - half[2],
                    ],
                    axis=-1,
                )
            else:
                outside = np.abs(local) - half
            distance = np.linalg.norm(np.maximum(outside, 0.0), axis=-1) + np.minimum(
                np.max(outside, axis=-1), 0.0
            )
        nearest = np.minimum(nearest, np.min(distance - radii, axis=-1))
    return nearest
