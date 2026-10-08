"""Arm IK for swatting: put a hand's collision sphere on a point in space."""

import mujoco
import numpy as np

# MuJoCo joint indices (robot joints, 0-based) of shoulder pitch/roll/yaw + elbow.
ARM_JOINTS = {"left": [15, 16, 17, 18], "right": [22, 23, 24, 25]}
HAND_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
HAND_OFFSET = np.array([0.08, 0.0, 0.0])  # hand collision sphere centre in the wrist frame


class ArmIK:
    def __init__(self, model: mujoco.MjModel, effector=None):
        """effector: {side: offset in the wrist frame} of the point to place (hand or bat)."""
        self.m = model
        self.effector = {s: np.asarray((effector or {}).get(s, HAND_OFFSET), float) for s in HAND_BODY}
        self.d = mujoco.MjData(model)
        self.hand = {s: model.body(b).id for s, b in HAND_BODY.items()}
        self.dof = {s: [6 + j for j in js] for s, js in ARM_JOINTS.items()}
        self.qadr = {s: [7 + j for j in js] for s, js in ARM_JOINTS.items()}
        self.range = {s: model.jnt_range[[1 + j for j in js]] for s, js in ARM_JOINTS.items()}

    def hand_pos(self, side):
        b = self.hand[side]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.effector[side]

    def solve(self, robot_qpos, side, target, iters=40, damping=0.05):
        """Arm joint angles (4,) placing the hand at target with the rest of the body at
        robot_qpos. Returns (angles, residual distance in m)."""
        m, d = self.m, self.d
        d.qpos[: len(robot_qpos)] = robot_qpos
        qa, dofs, lo_hi = self.qadr[side], self.dof[side], self.range[side]
        jacp = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = self.hand_pos(side)
            err = np.asarray(target) - p
            if np.linalg.norm(err) < 0.005:
                break
            mujoco.mj_jac(m, d, jacp, None, p, self.hand[side])
            j = jacp[:, dofs]
            dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(3), err)
            d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo_hi[:, 0], lo_hi[:, 1])
        mujoco.mj_kinematics(m, d)
        return d.qpos[qa].copy(), float(np.linalg.norm(np.asarray(target) - self.hand_pos(side)))
