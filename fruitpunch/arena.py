"""MuJoCo arena: G1 + tomato, with the robot either kinematic or physically simulated.

kinematic: the robot's qpos/qvel are written from the reference motion every
           substep; only the tomato is simulated (it still collides with the
           moving robot because qvel is set consistently).
physics:   the robot is PD-controlled from joint targets produced by SONIC.
"""

from pathlib import Path

import mujoco
import numpy as np

from . import g1
from .rot import qconj, qmul, qrot

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "assets" / "scene_g1.xml"
ROBOT_XML = ROOT / "assets" / "g1" / "g1_29dof.xml"

STICKY_BODIES = ("torso_link", "pelvis")  # torso + head (head mesh lives on torso_link)
ROBOT_NQ, ROBOT_NV = 36, 35


BAT_LENGTH, BAT_RADIUS = 0.6, 0.035  # foam dodgeball bat, continuing the forearm
BAT_START = 0.10                      # m from the wrist_yaw_link origin (inside the fist)


def build_model(kinematic: bool, bat: str | None = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    # The included robot file's <compiler meshdir> is ignored under <include>; resolve meshes explicitly.
    for mesh in spec.meshes:
        mesh.file = str(ROBOT_XML.parent / "meshes" / Path(mesh.file).name)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.armature = 0.01
            j.damping = [0.05, 0, 0]  # mujoco>=3.15: damping is a 3-vector
            j.frictionloss = 0.1 if "wrist" in j.name else 0.2
    for side in ("left", "right"):
        hand = spec.body(f"{side}_wrist_yaw_link")
        hand.add_geom(name=f"{side}_hand_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[0.045, 0, 0], pos=[0.08, 0, 0], rgba=[0.8, 0.8, 0.8, 0], group=3,
                      mass=0)
    for side in {"right": ["right"], "left": ["left"], "both": ["left", "right"]}.get(bat or "", []):
        spec.body(f"{side}_wrist_yaw_link").add_geom(
            name=f"{side}_bat", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[BAT_RADIUS, 0, 0],
            fromto=[BAT_START, 0, 0, BAT_START + BAT_LENGTH, 0, 0], rgba=[1.0, 0.55, 0.1, 1], mass=0.15)
    spec.geom("floor").friction = [1.0, 0.005, 0.0001]
    # Non-zero at compile time so MuJoCo keeps gravity compensation active for the tomato;
    # set per throw with Arena.set_tomato_gravity().
    spec.body("tomato").gravcomp = 1e-6
    if kinematic:
        # Robot collides only with the tomato: floor (1), robot (2), tomato (3).
        robot_bodies = _robot_body_names(spec)
        for gm in spec.geoms:
            if gm.contype == 0 and gm.conaffinity == 0:
                continue
            body = gm.parent.name
            if body in robot_bodies:
                gm.contype, gm.conaffinity = 2, 2
            elif body == "tomato":
                gm.contype, gm.conaffinity = 3, 3
        # Near-infinite inertia: contacts must not push the robot, otherwise resetting it
        # to the reference every substep pumps energy into the tomato.
        for name in robot_bodies:
            b = spec.body(name)
            b.mass *= 1e4
            b.inertia = [x * 1e4 for x in b.inertia]
    return spec.compile()


def _robot_body_names(spec):
    names, stack = set(), [spec.body("pelvis")]
    while stack:
        b = stack.pop()
        names.add(b.name)
        stack.extend(b.bodies)
    return names


class Arena:
    def __init__(self, kinematic: bool = True, bat: str | None = None):
        self.kinematic = kinematic
        self.bat = bat
        self.m = build_model(kinematic, bat)
        self.m.opt.timestep = g1.SIM_DT
        self.d = mujoco.MjData(self.m)
        m = self.m
        tb = m.body("tomato")
        self.tomato_id = tb.id
        self.tq = m.jnt_qposadr[tb.jntadr[0]]
        self.tv = m.jnt_dofadr[tb.jntadr[0]]
        self.tomato_geom = m.geom("tomato_geom").id
        self.stick_eq = m.equality("tomato_stick").id
        self.sticky_bodies = {m.body(n).id for n in STICKY_BODIES}
        self.torso_id = m.body("torso_link").id
        self.stuck = False
        self.sticky = True
        self.first_hit = None  # (time, body name, world point)
        self.park_tomato()

    # --- robot --------------------------------------------------------------------
    def set_robot_state(self, qpos, qvel=None):
        self.d.qpos[:ROBOT_NQ] = qpos
        self.d.qvel[:ROBOT_NV] = 0 if qvel is None else qvel

    def robot_qpos(self):
        return self.d.qpos[:ROBOT_NQ].copy()

    def step_kinematic(self, q0, q1, frac0, frac1, n_sub):
        """Advance n_sub physics substeps while the robot moves from q0 to q1 (qpos)."""
        from .planner import interp_qpos
        pair = np.stack([q0, q1])
        qvel = qpos_velocity(q0, q1, n_sub * self.m.opt.timestep)
        for i in range(n_sub):
            a = frac0 + (frac1 - frac0) * (i / n_sub)
            self.set_robot_state(interp_qpos(pair, a), qvel)
            mujoco.mj_step(self.m, self.d)
            self._post_step()
        self.set_robot_state(q1, qvel)
        mujoco.mj_forward(self.m, self.d)

    def step_physics(self, q_target, n_sub, band=None):
        """PD control toward joint targets (MuJoCo order) for n_sub substeps."""
        for _ in range(n_sub):
            q = self.d.qpos[7:ROBOT_NQ]
            dq = self.d.qvel[6:ROBOT_NV]
            tau = g1.KP * (q_target - q) - g1.KD * dq
            self.d.ctrl[:] = np.clip(tau, -g1.TORQUE_LIMIT, g1.TORQUE_LIMIT)
            self.d.xfrc_applied[1] = 0
            if band is not None:
                band(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            self._post_step()

    # --- tomato -------------------------------------------------------------------
    def park_tomato(self):
        self.d.eq_active[self.stick_eq] = 0
        self.stuck = False
        self.first_hit = None
        self.launched = False
        self.d.qpos[self.tq:self.tq + 7] = (0, 6, 0.036, 1, 0, 0, 0)  # resting off to the side
        self.d.qvel[self.tv:self.tv + 6] = 0

    def launch(self, p0, v0):
        self.park_tomato()
        self.launched = True
        self.d.qpos[self.tq:self.tq + 3] = p0
        self.d.qvel[self.tv:self.tv + 3] = v0
        self.d.qvel[self.tv + 3:self.tv + 6] = np.random.uniform(-5, 5, 3)  # a bit of spin
        mujoco.mj_forward(self.m, self.d)

    def set_projectile(self, radius, mass, rgba, sticky):
        """Reshape the single projectile body into a ball from fruitpunch.balls."""
        m = self.m
        m.geom_size[self.tomato_geom, 0] = radius
        m.geom_rbound[self.tomato_geom] = radius
        m.geom_rgba[self.tomato_geom] = rgba
        stem = m.geom("tomato_stem").id
        m.geom_rgba[stem, 3] = 1.0 if sticky else 0.0  # the stem only on fruit
        m.geom_pos[stem, 2] = radius + 0.001
        m.body_mass[self.tomato_id] = mass
        # Floor on rotational inertia: a 58 g tennis ball has ~2.5e-5 kg m^2, which made the
        # ball's rotation numerically singular in contact (NaN QACC, then a MuJoCo auto-reset).
        m.body_inertia[self.tomato_id] = max(0.4 * mass * radius**2, 2e-4)
        # Recompute derived constants (dof_invweight0 etc. scale constraint softness); stale
        # values from the compiled 0.12 kg tomato made contacts with a 5 kg ball blow up.
        mujoco.mj_setConst(m, mujoco.MjData(m))
        self.sticky = sticky

    def set_tomato_gravity(self, scale: float):
        """Effective gravity on the tomato as a fraction of g (1 = real, <1 = floaty)."""
        self.m.body_gravcomp[self.tomato_id] = max(1e-6, 1.0 - scale)

    def tomato_state(self):
        return self.d.qpos[self.tq:self.tq + 3].copy(), self.d.qvel[self.tv:self.tv + 3].copy()

    def tomato_in_flight(self):
        return self.launched and self.first_hit is None

    def _post_step(self):
        if self.stuck:
            return
        m, d = self.m, self.d
        for c in d.contact[: d.ncon]:
            if self.tomato_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.tomato_geom else c.geom1
            body = m.geom_bodyid[other]
            if self.first_hit is None:
                self.first_hit = (d.time, m.body(body).name, c.pos.copy())
            if self.sticky and body in self.sticky_bodies:
                self._stick(body)
                return

    def _stick(self, body):
        """Activate the weld so the tomato stays where it touched."""
        m, d = self.m, self.d
        m.eq_obj1id[self.stick_eq] = body
        p1, q1 = d.xpos[body], d.xquat[body]
        p2, q2 = d.xpos[self.tomato_id], d.xquat[self.tomato_id]
        rel_p = qrot(qconj(q1), p2 - p1)
        rel_q = qmul(qconj(q1), q2)
        data = m.eq_data[self.stick_eq]
        data[0:3] = 0           # anchor at body2 origin
        data[3:6] = rel_p
        data[6:10] = rel_q
        data[10] = 1.0
        d.eq_active[self.stick_eq] = 1
        d.qvel[self.tv:self.tv + 6] = 0
        self.stuck = True


def qpos_velocity(q0, q1, dt):
    """Finite-difference free-joint qvel (world linear, body-frame angular) + joint vel."""
    v = np.zeros(ROBOT_NV)
    v[:3] = (q1[:3] - q0[:3]) / dt
    dq = qmul(qconj(q0[3:7]), q1[3:7])
    if dq[0] < 0:
        dq = -dq
    ang = 2 * np.arccos(np.clip(dq[0], -1, 1))
    s = np.sqrt(max(1 - dq[0] ** 2, 1e-12))
    v[3:6] = (dq[1:] / s) * ang / dt if ang > 1e-9 else 0
    v[6:] = (q1[7:] - q0[7:]) / dt
    return v
