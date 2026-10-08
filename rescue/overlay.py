"""Viewer overlays: translucent ghost robot, predicted tomato arc, contact marker.

Everything is drawn into viewer.user_scn, which is cleared and rebuilt every
frame by Overlay.draw().
"""

import mujoco
import numpy as np


class Overlay:
    def __init__(self, robot_xml: str, ghost_rgba=(0.2, 0.8, 1.0, 0.3)):
        # The ghost is a second, visual-only copy of the robot model.
        self.gm = mujoco.MjModel.from_xml_path(robot_xml)
        self.gd = mujoco.MjData(self.gm)
        self.ghost_rgba = np.array(ghost_rgba, np.float32)
        self.ghost_qpos = None
        self.path = None
        self.contact = None
        self.target_marks = []  # list of (pos, heading)
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 0
        self._opt.geomgroup[1] = 1  # G1 visual meshes live in group 1
        self._pert = mujoco.MjvPerturb()

    def draw(self, scn: mujoco.MjvScene, clear=True):
        """clear=True for a viewer's user_scn; False to append to a Renderer's scene."""
        if clear:
            scn.ngeom = 0
        if self.ghost_qpos is not None:
            self.gd.qpos[:] = self.ghost_qpos
            mujoco.mj_kinematics(self.gm, self.gd)
            start = scn.ngeom
            mujoco.mjv_addGeoms(self.gm, self.gd, self._opt, self._pert,
                                mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            for i in range(start, scn.ngeom):
                scn.geoms[i].rgba[:] = self.ghost_rgba
                scn.geoms[i].segid = -1
        if self.path is not None and len(self.path) > 1:
            step = max(1, len(self.path) // 60)
            pts = self.path[::step]
            for a, b in zip(pts[:-1], pts[1:]):
                self._line(scn, a, b, (1.0, 0.85, 0.1, 0.9), 0.006)
        if self.contact is not None:
            self._sphere(scn, self.contact, 0.03, (1.0, 0.1, 0.1, 0.7))
        for pos, heading in self.target_marks:
            self._sphere(scn, pos, 0.04, (0.2, 1.0, 0.3, 0.6))
            tip = np.asarray(pos) + 0.3 * np.array([np.cos(heading), np.sin(heading), 0.0])
            self._line(scn, pos, tip, (0.2, 1.0, 0.3, 0.9), 0.012)

    def _add(self, scn):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        return g

    def _sphere(self, scn, pos, r, rgba):
        g = self._add(scn)
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
                            np.asarray(pos, float), np.eye(3).ravel(), np.array(rgba, np.float32))

    def _line(self, scn, a, b, rgba, width):
        g = self._add(scn)
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(a, float), np.asarray(b, float))
