"""One 50 Hz control tick: replan if needed -> drive the robot (kinematic or SONIC physics)."""

import time
from dataclasses import replace

import mujoco
import numpy as np

from . import g1
from .arena import ROBOT_NQ, ROBOT_NV, Arena
from .config import DECODER_ONNX, ENCODER_ONNX, PLANNER_ONNX
from .planner import KPlanner, MotionTrack, PlannerCommand, ReplanPolicy, standing_qpos

N_SUB = int(round(g1.CONTROL_DT / g1.SIM_DT))


class Runner:
    def __init__(self, robot="kinematic", xy=(0.0, 0.0), yaw=0.0, band_seconds=1.0, bat=None):
        self.kinematic = robot == "kinematic"
        self.arena = Arena(kinematic=self.kinematic, bat=bat)
        self.planner = KPlanner(PLANNER_ONNX)
        self.replan = ReplanPolicy()
        self.cmd = PlannerCommand(facing_yaw=yaw)
        self.policy = None
        if not self.kinematic:
            from .sonic import SonicPolicy
            self.policy = SonicPolicy(ENCODER_ONNX, DECODER_ONNX)
        self.band_seconds = band_seconds
        self.reset(xy, yaw)

    def reset(self, xy=(0.0, 0.0), yaw=0.0):
        """Put the robot back standing at (xy, yaw) with a fresh reference track."""
        self.replan = ReplanPolicy()
        self.cmd = PlannerCommand(facing_yaw=yaw)
        q0 = standing_qpos(xy, yaw)
        if not self.kinematic:
            q0[2] = 0.80  # feet just above the floor
        self.band_until = self.arena.d.time + (0.0 if self.kinematic else self.band_seconds)
        self.arena.set_robot_state(q0)
        mujoco.mj_forward(self.arena.m, self.arena.d)
        self.track = MotionTrack(standing_qpos(xy, yaw))
        if self.policy:
            self.policy.reset(self.arena.d.qpos[:ROBOT_NQ].copy(), self.arena.d.qvel[:ROBOT_NV].copy())
        self.band_anchor = q0[:3].copy()
        self.last_plan_ms = 0.0
        self.ticks = 0
        self._arrived = False
        self.anchor_xy = True

    @property
    def time(self):
        return self.arena.d.time

    def drift(self):
        """Robot root xy minus reference root xy. SONIC tracks relative motion, so in physics
        mode the robot slowly drifts from the planner's world position."""
        return self.arena.d.qpos[:2] - self.track.now()[:2]

    CARROT = 0.6       # m: planner target distance used near the goal in physics mode
    ARRIVE = 0.06      # m: switch to idle inside this radius
    REENGAGE = 0.15    # m: leave idle again outside this radius

    def _corrected(self, cmd: PlannerCommand) -> PlannerCommand:
        """Physics-mode target following.

        Near a goal the planner closes the last ~0.3 m with a slow root creep (feet slide).
        A kinematic robot reproduces it; SONIC cannot slide, so it stops short. Instead,
        keep the planner's target at least CARROT ahead along the error so it plans real
        steps, and switch to idle (facing the goal heading) once within ARRIVE.
        """
        if self.kinematic or cmd.target_xy is None:
            self._arrived = False
            return cmd
        err = np.asarray(cmd.target_xy, float) - self.arena.d.qpos[:2]
        dist = np.linalg.norm(err)
        self._arrived = dist < (self.REENGAGE if self._arrived else self.ARRIVE)
        yaw = cmd.target_yaw if cmd.target_yaw is not None else cmd.facing_yaw
        if self._arrived:
            mode = cmd.mode if cmd.mode in g1.STATIC_MODES else 0
            return replace(cmd, mode=mode, target_xy=None, move_dir=(0.0, 0.0), facing_yaw=yaw)
        # The planner works around the reference root, which drifts from the robot:
        # express the goal relative to the reference.
        ref_xy = self.track.now()[:2]
        step = err if dist >= self.CARROT else err / dist * self.CARROT
        mode = 1 if cmd.mode in g1.STATIC_MODES else cmd.mode  # idle can't take real steps
        return replace(cmd, mode=mode, target_xy=tuple(ref_xy + step), target_yaw=yaw)

    def plan_now(self, cmd: PlannerCommand, lookahead=2):
        if not self.kinematic and self.anchor_xy:
            # Shift the reference onto the robot's real xy. The SONIC encoder never sees root
            # xy (only joints, joint velocities, relative heading), so this is invisible to
            # the policy but keeps the planner working from where the robot actually is.
            self.track.frames[:, :2] += self.drift()
        return self.plan_on(self.track, cmd, lookahead)

    def plan_on(self, track: MotionTrack, cmd: PlannerCommand, lookahead=2):
        """Plan from `track`'s context `lookahead` ticks ahead (used to schedule a move later)."""
        cmd = self._corrected(cmd)
        ctx, gen = track.context(lookahead)
        t = time.perf_counter()
        plan = self.planner.plan(ctx, cmd)
        self.last_plan_ms = 1000 * (time.perf_counter() - t)
        return plan, gen

    def commit(self, cmd: PlannerCommand, track: MotionTrack):
        """Adopt an already-evaluated plan without replanning (used by the dodge selector)."""
        self.track = track
        self.cmd = cmd
        self.replan.mark(cmd)

    def tick(self):
        if self.replan.should_replan(self.cmd, self.track, g1.CONTROL_DT):
            first = self.track.fresh
            plan, gen = self.plan_now(self.cmd)
            self.track.merge(plan, gen, speed=self.cmd.speed)
            if first and self.policy:
                self.policy.align_heading(self.arena.d.qpos[3:7].copy(), self.track.frames[0, 3:7])
        if self.kinematic:
            qa = self.track.now()
            self.track.advance()
            self.arena.step_kinematic(qa, self.track.now(), 0.0, 1.0, N_SUB)
        else:
            d = self.arena.d
            q_target = self.policy.act(d.qpos[:ROBOT_NQ].copy(), d.qvel[:ROBOT_NV].copy(), self.track)
            self.track.advance()
            band = self._band if self.time < self.band_until else None
            self.arena.step_physics(q_target, N_SUB, band)
        self.ticks += 1

    def _band(self, m, d):
        """Elastic band holding the pelvis up during the first second (upstream sim does the same)."""
        kp, kd = 2000.0, 200.0
        err = self.band_anchor - d.qpos[:3]
        f = kp * err - kd * d.qvel[:3]
        f[:2] *= 0.2
        d.xfrc_applied[1, :3] = f

    def robot_qpos(self):
        return self.arena.robot_qpos()

    def fallen(self):
        """Torso tipped more than ~60 deg (pelvis height alone misfires during squats/kneels)."""
        if self.kinematic:
            return False
        w, x, y, z = self.arena.d.qpos[3:7]
        up_z = 1 - 2 * (x * x + y * y)  # z component of the pelvis up axis
        return up_z < 0.5 or self.arena.d.qpos[2] < 0.25
