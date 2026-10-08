"""SONIC whole-body tracking policy (encoder + decoder ONNX), "g1" encoder mode.

Ported from gear_sonic_deploy g1_deploy_onnx_ref.cpp. Layouts:
  encoder obs (1762): [0:4] mode, [4:294] motion joint pos (10 frames, step 5, IsaacLab
      order, absolute), [294:584] motion joint vel, [601:661] motion anchor orientation
      (10 frames x 6). Everything else is zero in g1 mode.
  decoder obs (994): token(64) | ang vel hist(30) | joint pos hist(290, minus default) |
      joint vel hist(290) | last action hist(290) | gravity hist(30); histories oldest first.
"""

from collections import deque

import numpy as np
import onnxruntime as ort

from . import g1
from .planner import MotionTrack
from .rot import heading_quat, qconj, qmat, qmul, qrot

HIST = 10
ENC_STEP = 5


def to_il(x_mj):
    return x_mj[..., g1.MUJOCO_OF_IL]


class SonicPolicy:
    def __init__(self, encoder_path, decoder_path, providers=("CPUExecutionProvider",)):
        ort.set_default_logger_severity(3)
        self.enc = ort.InferenceSession(str(encoder_path), providers=list(providers))
        self.dec = ort.InferenceSession(str(decoder_path), providers=list(providers))
        self.hist = None
        self.last_action = np.zeros(g1.NUM_JOINTS)
        self.anchor_apply = None  # yaw alignment between motion and robot, set on first plan

    def reset(self, robot_qpos, robot_qvel):
        self.last_action = np.zeros(g1.NUM_JOINTS)
        self.anchor_apply = None
        # Upstream zero-pads missing history; we pre-fill with the initial state instead,
        # which avoids a zero gravity vector in the first ticks.
        frame = self._state_frame(robot_qpos, robot_qvel)
        self.hist = [deque([f.copy() for _ in range(HIST)], maxlen=HIST) for f in frame]

    def align_heading(self, robot_quat, ref_quat):
        """Called when the first plan is merged (upstream: reinitialize_heading_)."""
        self.anchor_apply = qmul(heading_quat(robot_quat), qconj(heading_quat(ref_quat)))

    def _state_frame(self, qpos, qvel):
        quat = qpos[3:7]
        q = to_il(qpos[7:36] - g1.DEFAULT_ANGLES)
        dq = to_il(qvel[6:35])
        gyro = qvel[3:6]  # free-joint angular velocity is in the body frame
        grav = qrot(qconj(quat), np.array([0.0, 0.0, -1.0]))
        return [gyro, q, dq, self.last_action.copy(), grav]

    def _encoder_obs(self, track: MotionTrack, base_quat):
        obs = np.zeros(1762, np.float32)
        last = len(track.frames) - 1
        apply = self.anchor_apply if self.anchor_apply is not None else np.array([1.0, 0, 0, 0])
        inv_base = qconj(base_quat)
        for k in range(HIST):
            f = min(track.cur + k * ENC_STEP, last)
            fr = track.frames[f]
            obs[4 + 29 * k: 4 + 29 * (k + 1)] = to_il(fr[7:])
            obs[294 + 29 * k: 294 + 29 * (k + 1)] = to_il(track.joint_vel(f))
            rel = qmul(inv_base, qmul(apply, fr[3:7]))
            r = qmat(rel)
            obs[601 + 6 * k: 601 + 6 * (k + 1)] = [r[0, 0], r[0, 1], r[1, 0], r[1, 1], r[2, 0], r[2, 1]]
        return obs

    def act(self, robot_qpos, robot_qvel, track: MotionTrack) -> np.ndarray:
        """One 50 Hz tick. Returns joint position targets in MuJoCo order."""
        for h, f in zip(self.hist, self._state_frame(robot_qpos, robot_qvel)):
            h.append(f)
        token = self.enc.run(None, {"obs_dict": self._encoder_obs(track, robot_qpos[3:7])[None]})[0][0]
        dec = np.concatenate([token] + [np.concatenate(list(h)) for h in self.hist]).astype(np.float32)
        action = self.dec.run(None, {"obs_dict": dec[None]})[0][0].astype(float)
        self.last_action = action
        return g1.DEFAULT_ANGLES + action[g1.IL_OF_MUJOCO] * g1.ACTION_SCALE
