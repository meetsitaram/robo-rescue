"""SONIC kinematic planner (planner_sonic.onnx) wrapper + the 50 Hz motion track.

Ported from gear_sonic_deploy localmotion_kplanner*.hpp and g1_deploy_onnx_ref.cpp.
Differences from upstream:
  * The planner works directly in the MuJoCo world frame (upstream uses the robot's
    frame at enable time). The network canonicalizes its inputs internally, so
    this is equivalent, and lets targets/ghosts be expressed in world coordinates.
  * Planning runs synchronously inside the sim loop (upstream: a 10 Hz thread).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import g1
from .rot import slerp, quat_yaw, yaw_of

DEFAULT_ALLOWED_TOKENS = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64)
BLEND_FRAMES = 8


@dataclass
class PlannerCommand:
    mode: int = 0
    target_vel: float = -1.0          # m/s; -1 = mode default
    move_dir: tuple = (0.0, 0.0)      # world xy direction (unnormalized ok); (0,0) = stand
    facing_yaw: float = 0.0           # world yaw (rad)
    height: float = -1.0              # only for squat/kneel modes, 0.2-0.8
    target_xy: tuple | None = None    # specific-target mode: world xy goal
    target_yaw: float | None = None   # specific-target mode: world heading at goal
    seed: int = 1234

    def key(self):
        return (self.mode, round(self.target_vel, 3), tuple(np.round(self.move_dir, 3)),
                round(self.facing_yaw, 3), round(self.height, 3),
                None if self.target_xy is None else tuple(np.round(self.target_xy, 3)),
                None if self.target_yaw is None else round(self.target_yaw, 3))


class KPlanner:
    def __init__(self, onnx_path: str | Path, providers=("CPUExecutionProvider",)):
        ort.set_default_logger_severity(3)
        self.sess = ort.InferenceSession(str(onnx_path), providers=list(providers))

    def plan(self, context: np.ndarray, cmd: PlannerCommand) -> np.ndarray:
        """context (4, 36) MuJoCo qpos at 30 Hz -> planned qpos (N, 36) at 30 Hz."""
        mv = np.array([*cmd.move_dir, 0.0], np.float32)
        n = np.linalg.norm(mv)
        mv = mv / n if n > 1e-6 else np.zeros(3, np.float32)
        face = np.array([np.cos(cmd.facing_yaw), np.sin(cmd.facing_yaw), 0.0], np.float32)
        has_target = cmd.target_xy is not None
        tp = np.zeros((1, 4, 3), np.float32)
        th = np.zeros((1, 4), np.float32)
        if has_target:
            tp[0, :, :2] = cmd.target_xy
            tp[0, :, 2] = g1.PLANNER_CONTEXT_Z
            th[0, :] = cmd.target_yaw if cmd.target_yaw is not None else cmd.facing_yaw
        height = cmd.height if cmd.mode in g1.HEIGHT_MODES else -1.0
        feed = {
            "context_mujoco_qpos": context[None].astype(np.float32),
            "target_vel": np.array([cmd.target_vel], np.float32),
            "mode": np.array([cmd.mode], np.int64),
            "movement_direction": mv[None],
            "facing_direction": face[None],
            "random_seed": np.array([cmd.seed], np.int64),
            "has_specific_target": np.array([[int(has_target)]], np.int64),
            "specific_target_positions": tp,
            "specific_target_headings": th,
            "allowed_pred_num_tokens": DEFAULT_ALLOWED_TOKENS,
            "height": np.array([height], np.float32),
        }
        qpos, num = self.sess.run(None, feed)
        out = qpos[0, : int(num[0])]
        if not np.isfinite(out).all() or len(out) < 4:
            raise RuntimeError("planner returned invalid motion")
        return out


def interp_qpos(frames: np.ndarray, f: float) -> np.ndarray:
    """Interpolate a (N, 36) qpos track at fractional index f (clamped)."""
    f = float(np.clip(f, 0, len(frames) - 1))
    i = int(np.floor(f))
    j = min(i + 1, len(frames) - 1)
    a = f - i
    out = (1 - a) * frames[i] + a * frames[j]
    out[3:7] = slerp(frames[i, 3:7], frames[j, 3:7], a)
    return out


def resample_30_to_50(frames30: np.ndarray) -> np.ndarray:
    n50 = int(np.floor(len(frames30) / g1.PLANNER_FPS * 50))
    return np.stack([interp_qpos(frames30, k * g1.PLANNER_FPS / 50) for k in range(n50)])


class MotionTrack:
    """Reference motion at 50 Hz (MuJoCo qpos layout). frame 0 = the current tick after a merge."""

    def __init__(self, qpos0: np.ndarray):
        self.frames = np.repeat(qpos0[None].astype(float), 4, axis=0)
        self.cur = 0
        self.fresh = True  # no plan merged yet

    # --- queries -----------------------------------------------------------------
    def at(self, f: float) -> np.ndarray:
        return interp_qpos(self.frames, f)

    def now(self, sub: float = 0.0) -> np.ndarray:
        return self.at(self.cur + sub)

    def joint_vel(self, f: int) -> np.ndarray:
        f = min(f, len(self.frames) - 1)
        if f + 1 < len(self.frames):
            return (self.frames[f + 1, 7:] - self.frames[f, 7:]) * 50.0
        if f > 0:
            return (self.frames[f, 7:] - self.frames[f - 1, 7:]) * 50.0
        return np.zeros(g1.NUM_JOINTS)

    def context(self, lookahead: int = 2) -> tuple[np.ndarray, int]:
        """4 context frames at 30 Hz starting at cur+lookahead (50 Hz frames)."""
        gen = self.cur + lookahead
        ctx = np.stack([self.at(gen + n * 50 / g1.PLANNER_FPS) for n in range(4)])
        return ctx, gen

    def remaining(self) -> int:
        return len(self.frames) - 1 - self.cur

    # --- updates -----------------------------------------------------------------
    def advance(self):
        self.cur = min(self.cur + 1, len(self.frames) - 1)

    def merge(self, plan30: np.ndarray, gen: int):
        new = resample_30_to_50(plan30)
        if self.fresh:
            self.frames, self.cur, self.fresh = new, 0, False
            return
        offset = max(0, gen - self.cur)
        n = offset + len(new)
        out = np.empty((n, new.shape[1]))
        for f in range(n):
            old = self.at(self.cur + f)
            k = f - offset
            if k < 0:
                out[f] = old
                continue
            w = np.clip(k / BLEND_FRAMES, 0.0, 1.0)
            o = (1 - w) * old + w * new[k]
            o[3:7] = slerp(old[3:7], new[k, 3:7], w)
            out[f] = o
        self.frames, self.cur = out, 0


def standing_qpos(xy=(0.0, 0.0), yaw=0.0, z=g1.PLANNER_CONTEXT_Z) -> np.ndarray:
    q = np.zeros(36)
    q[:2] = xy
    q[2] = z
    q[3:7] = quat_yaw(yaw)
    q[7:] = g1.DEFAULT_ANGLES
    return q


@dataclass
class ReplanPolicy:
    """When to call the planner (mirrors g1_deploy_onnx_ref.cpp:3618-3700)."""
    moving_interval: float = 1.0
    target_interval: float = 0.4
    min_remaining: int = 20  # also replan when the track is about to run out
    _last_key: tuple | None = field(default=None, repr=False)
    _since: float = field(default=1e9, repr=False)

    def mark(self, cmd: PlannerCommand):
        """Record that a plan for cmd was just merged."""
        self._last_key, self._since = cmd.key(), 0.0

    def should_replan(self, cmd: PlannerCommand, track: MotionTrack, dt: float) -> bool:
        self._since += dt
        key = cmd.key()
        changed = key != self._last_key
        moving = cmd.mode not in g1.STATIC_MODES or np.linalg.norm(cmd.move_dir) > 1e-6
        interval = self.target_interval if cmd.target_xy is not None else self.moving_interval
        due = (moving or cmd.target_xy is not None) and self._since >= interval
        starving = track.remaining() < self.min_remaining
        if changed or due or starving or track.fresh:
            self._last_key, self._since = key, 0.0
            return True
        return False


def root_yaw(qpos):
    return yaw_of(qpos[3:7])
