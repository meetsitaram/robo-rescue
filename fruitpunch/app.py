"""FruitPunch interactive MuJoCo viewer.

    python -m fruitpunch.app [--robot kinematic|physics] [--mode drive|target|game]
                             [--objective dodge|hit] [--bat left|right|both] [--balls tennis,tomato,...]

Keys (arrows / nav block, so they don't clash with MuJoCo viewer shortcuts):
  Delete        cycle mode: DRIVE -> TARGET -> GAME
  Insert        throw a ball now
  Home          cycle gait (slow walk, walk, run, crouch walk, boxing walk, ...)
  DRIVE   Up/Down walk forward/back   Left/Right turn 30 deg   PgUp/PgDn strafe   End squat on/off
  TARGET  arrows move the ghost 25 cm  PgUp/PgDn rotate ghost   End ghost squat on/off
          Enter = go to the ghost
  GAME    automatic throws every few seconds; the robot predicts and reacts
          End = switch objective DODGE <-> HIT (meet the ball with a hand or bat)
"""

import argparse
import queue
import time

import mujoco
import mujoco.viewer
import numpy as np

from . import g1
from .arena import ROBOT_XML
from .game import Game
from .overlay import Overlay
from .planner import PlannerCommand, root_yaw, standing_qpos
from .rot import wrap
from .runner import Runner

K_UP, K_DOWN, K_LEFT, K_RIGHT = 265, 264, 263, 262
K_PGUP, K_PGDN, K_HOME, K_END = 266, 267, 268, 269
K_INSERT, K_DELETE, K_ENTER, K_KP_ENTER = 260, 261, 257, 335

GAITS = [("walk", 2, -1.0), ("slow walk", 1, -1.0), ("run", 3, 2.0), ("crouch walk", 22, -1.0),
         ("boxing walk", 10, 1.0), ("happy walk", 23, -1.0), ("stealth walk", 18, -1.0)]
MODES = ["DRIVE", "TARGET", "GAME"]
MOMENTUM = 0.6  # s a movement key press keeps the robot moving


class App:
    def __init__(self, robot, mode, interval, objective="dodge", bat=None, balls=None):
        self.r = Runner(robot, bat=bat)
        extra = {"balls": tuple(balls)} if balls else {}
        self.game = Game(self.r, auto=False, interval=interval, objective=objective, **extra)
        self.overlay = Overlay(str(ROBOT_XML))
        self.keys = queue.Queue()
        self.mode = mode.upper()
        self.gait = 0
        q = self.r.robot_qpos()
        self.facing = root_yaw(q)
        self.move_until = -1.0
        self.move_local = np.zeros(2)  # (forward, left) in robot heading frame
        self.squat = False
        self.ghost_xy = q[:2] + np.array([1.0, 0.0])
        self.ghost_yaw = self.facing
        self.ghost_squat = False
        self._set_mode(self.mode)

    # --- input --------------------------------------------------------------------
    def on_key(self, key):
        self.keys.put(key)

    def _set_mode(self, mode):
        self.mode = mode
        self.game.auto = mode == "GAME"
        if mode == "GAME":
            self.game.home_xy = self.r.robot_qpos()[:2].copy()
            self.game.home_yaw = root_yaw(self.r.robot_qpos())
            self.game.next_throw = self.r.time + 1.5
        if mode != "GAME":
            self.r.cmd = PlannerCommand(mode=0, facing_yaw=root_yaw(self.r.robot_qpos()))
            self.facing = self.r.cmd.facing_yaw

    def _handle(self, key):
        gait_name, gait_mode, gait_vel = GAITS[self.gait]
        if key == K_DELETE:
            self._set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])
        elif key == K_INSERT:
            if self.game.state == "idle":
                self.game.throw()
        elif key == K_HOME:
            self.gait = (self.gait + 1) % len(GAITS)
        elif self.mode == "GAME" and key == K_END:
            self.game.objective = "hit" if self.game.objective == "dodge" else "dodge"
        elif self.mode == "DRIVE":
            if key in (K_UP, K_DOWN, K_PGUP, K_PGDN):
                self.move_local = {K_UP: (1, 0), K_DOWN: (-1, 0), K_PGUP: (0, 1), K_PGDN: (0, -1)}[key]
                self.move_local = np.array(self.move_local, float)
                self.move_until = self.r.time + MOMENTUM
                self.squat = False
            elif key in (K_LEFT, K_RIGHT):
                self.facing = wrap(self.facing + (np.pi / 6 if key == K_LEFT else -np.pi / 6))
            elif key == K_END:
                self.squat = not self.squat
        elif self.mode == "TARGET":
            step = {K_UP: (0.25, 0), K_DOWN: (-0.25, 0), K_LEFT: (0, 0.25), K_RIGHT: (0, -0.25)}
            if key in step:
                self.ghost_xy = self.ghost_xy + np.array(step[key])
            elif key in (K_PGUP, K_PGDN):
                self.ghost_yaw = wrap(self.ghost_yaw + (np.pi / 6 if key == K_PGUP else -np.pi / 6))
            elif key == K_END:
                self.ghost_squat = not self.ghost_squat
            elif key in (K_ENTER, K_KP_ENTER):
                self.r.cmd = PlannerCommand(mode=gait_mode, target_vel=gait_vel,
                                            target_xy=tuple(self.ghost_xy), target_yaw=self.ghost_yaw)
                self.target_sent = True

    def _drive_command(self):
        """Keyboard state -> planner command (DRIVE mode), like the upstream keyboard handler."""
        _, gait_mode, gait_vel = GAITS[self.gait]
        if self.squat:
            return PlannerCommand(mode=4, height=0.4, facing_yaw=self.facing)
        if self.r.time < self.move_until:
            c, s = np.cos(self.facing), np.sin(self.facing)
            f, l = self.move_local
            return PlannerCommand(mode=gait_mode, target_vel=gait_vel,
                                  move_dir=(f * c - l * s, f * s + l * c), facing_yaw=self.facing)
        return PlannerCommand(mode=0, facing_yaw=self.facing)

    def _target_command(self):
        """Once the robot reaches the ghost, settle into its posture (stand or squat)."""
        cmd = self.r.cmd
        if cmd.target_xy is None:
            return cmd
        q = self.r.robot_qpos()
        near = np.linalg.norm(q[:2] - self.ghost_xy) < 0.12 and abs(wrap(root_yaw(q) - self.ghost_yaw)) < 0.25
        if near and self.ghost_squat:
            return PlannerCommand(mode=4, height=0.4, facing_yaw=self.ghost_yaw)
        return cmd

    # --- loop ---------------------------------------------------------------------
    def run(self):
        r = self.r
        with mujoco.viewer.launch_passive(r.arena.m, r.arena.d, key_callback=self.on_key) as v:
            v.cam.lookat[:] = (0.5, 0, 0.8)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 150, -15
            wall0, sim0 = time.perf_counter(), r.time
            self._t_start = self._last_status = time.perf_counter()
            while v.is_running():
                while not self.keys.empty():
                    self._handle(self.keys.get())
                if self.mode == "DRIVE":
                    r.cmd = self._drive_command()
                elif self.mode == "TARGET":
                    r.cmd = self._target_command()
                with v.lock():
                    r.tick()
                    self.game.step()
                    if r.fallen():
                        print("robot fell - resetting")
                        r.reset()
                    texts = self._draw(v)
                # set_texts takes the viewer lock itself (not re-entrant): call it outside.
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, *texts))
                v.sync()
                if time.perf_counter() - self._last_status > 5.0:
                    self._last_status = time.perf_counter()
                    q = r.robot_qpos()
                    print(f"[{self.mode}] sim t={r.time:6.1f}s  wall={time.perf_counter() - self._t_start:6.1f}s  "
                          f"robot=({q[0]:+.2f},{q[1]:+.2f}) game={self.game.state} score={self.game.summary()}")
                ahead = (r.time - sim0) - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -0.25:  # fell behind (planning hiccup): don't try to catch up
                    wall0, sim0 = time.perf_counter(), r.time
            self._t_start = self._last_status = time.perf_counter()

    def _draw(self, v):
        r, g, o = self.r, self.game, self.overlay
        q = r.robot_qpos()
        o.target_marks, o.ghost_qpos = [], None
        if self.mode == "TARGET":
            ghost = standing_qpos(self.ghost_xy, self.ghost_yaw)
            if self.ghost_squat:
                ghost[2] -= 0.3
            o.ghost_qpos = ghost
            o.target_marks = [(np.array([*self.ghost_xy, 0.02]), self.ghost_yaw)]
        elif self.mode == "GAME" and g.state == "reacting":
            end = r.track.frames[-1].copy()  # where the committed reaction ends
            if not r.kinematic:
                end[:2] += r.drift()
            o.ghost_qpos = end
        o.path = g.path if g.state != "idle" else None
        o.contact = g.contact if g.state != "idle" else None
        o.draw(v.user_scn)
        gait = GAITS[self.gait][0]
        rec = g.records[-1] if g.records else None
        if rec is None:
            last = "-"
        elif g.state == "idle":
            last = f"{rec.action} -> {rec.outcome}"
        else:
            last = f"{rec.action} -> {rec.actual_body if rec.actual_t else 'in flight'}"
        counts = g.summary()
        left = "Mode\nGait\nRobot\nPlanner\nLast throw\nScore"
        mode = f"{self.mode} / {g.objective.upper()} (End: switch)" if self.mode == "GAME" else self.mode
        right = (f"{mode}  (Delete: next)\n{gait}  (Home: next)\n"
                 f"{'kinematic' if r.kinematic else 'SONIC physics'}  x={q[0]:+.2f} y={q[1]:+.2f} yaw={np.degrees(root_yaw(q)):+.0f}\n"
                 f"{r.last_plan_ms:.0f} ms/plan\n"
                 f"{last}\n"
                 f"{counts}  points {g.points()}")
        return left, right


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="kinematic", choices=["kinematic", "physics"])
    ap.add_argument("--mode", default="drive", choices=["drive", "target", "game"])
    ap.add_argument("--interval", type=float, default=4.0, help="seconds between throws in GAME mode")
    ap.add_argument("--objective", default="dodge", choices=["dodge", "hit"])
    ap.add_argument("--bat", default=None, choices=["left", "right", "both"], help="foam bat in hand(s)")
    ap.add_argument("--balls", default=None, help="comma-separated subset of fruitpunch.balls.BALLS")
    args = ap.parse_args()
    print(__doc__)
    App(args.robot, args.mode, args.interval, args.objective, args.bat,
        args.balls.split(",") if args.balls else None).run()


if __name__ == "__main__":
    main()
