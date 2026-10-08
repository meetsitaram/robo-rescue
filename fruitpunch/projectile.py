"""Tomato launch + contact prediction.

The arena has no air drag (MuJoCo option density=0), so the tomato's free flight
is a parabola, discretized exactly the way MuJoCo's (semi-implicit) Euler integrator
does it: x(t) = p0 + v0 t + g t (t + dt) / 2. Using the continuous parabola instead
puts the prediction ~2.7 cm too high after 1.1 s at dt = 5 ms. Contact is predicted by sweeping the tomato along that
parabola in a scratch MjData and running MuJoCo's own collision detection
against the robot, either frozen at its current pose or following a planned
qpos trajectory (used to check whether a dodge clears the throw).
"""

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def launch_velocity(start, target, flight_time, g=GRAVITY):
    """Initial velocity that carries a point mass from start to target in flight_time."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    return (target - start - 0.5 * g * flight_time**2) / flight_time


SIM_DT = 0.005


def ballistic(p0, v0, t, g=GRAVITY, dt=SIM_DT):
    """Tomato position after t seconds of free flight under the sim's Euler integrator."""
    t = np.asarray(t, float)[..., None]
    return p0 + v0 * t + 0.5 * g * t * (t + dt)


@dataclass
class Contact:
    t: float               # seconds from now
    point: np.ndarray      # world-frame contact point
    body: str              # robot body name ("floor" if it lands first)
    tomato_pos: np.ndarray # tomato centre at contact


class ContactPredictor:
    def __init__(self, model: mujoco.MjModel, tomato_body="tomato", robot_root="pelvis", margin=0.05):
        # Private copy with the tomato inflated by `margin`, so near-misses count as hits.
        self.m = model = copy.deepcopy(model)
        tb = model.body(tomato_body)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == tb.id and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                model.geom_size[g, 0] += margin
                model.geom_rbound[g] += margin
        self.d = mujoco.MjData(model)
        self.tomato_qadr = model.jnt_qposadr[tb.jntadr[0]]
        self.tomato_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == tb.id}
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        root = model.body(robot_root).id
        self.robot_bodies = {b for b in range(model.nbody) if self._descends(b, root)}
        self.robot_nq = model.jnt_qposadr[tb.jntadr[0]]  # robot qpos comes before the tomato's

    def _descends(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def predict(self, robot_qpos, p0, v0, horizon=2.5, dt=0.005, robot_traj=None, traj_dt=None):
        """First contact of the tomato with the robot or floor.

        robot_qpos: current robot qpos (nq_robot,). If robot_traj (N, nq_robot) is
        given, the robot follows it (frame k at time k*traj_dt, held after the end).
        Returns (Contact or None, path (K,3)).
        """
        m, d = self.m, self.d
        ts = np.arange(0.0, horizon, dt)
        path = ballistic(np.asarray(p0, float), np.asarray(v0, float), ts)
        d.qpos[: self.robot_nq] = robot_qpos
        for i, t in enumerate(ts):
            if robot_traj is not None:
                k = min(int(t / traj_dt), len(robot_traj) - 1)
                d.qpos[: self.robot_nq] = robot_traj[k]
            d.qpos[self.tomato_qadr : self.tomato_qadr + 3] = path[i]
            d.qpos[self.tomato_qadr + 3 : self.tomato_qadr + 7] = (1, 0, 0, 0)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            for c in d.contact[: d.ncon]:
                g1, g2 = c.geom1, c.geom2
                if g1 in self.tomato_geoms:
                    other = g2
                elif g2 in self.tomato_geoms:
                    other = g1
                else:
                    continue
                if c.dist > 0:
                    continue
                if other == self.floor_geom:
                    return Contact(t, c.pos.copy(), "floor", path[i].copy()), path[: i + 1]
                b = m.geom_bodyid[other]
                if b in self.robot_bodies:
                    return Contact(t, c.pos.copy(), m.body(b).name, path[i].copy()), path[: i + 1]
        return None, path
