"""Render review clips (MP4, H.264) with overlays and a caption bar, plus a JSON of outcomes.

    python scripts/render_clips.py [out_dir]
"""

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fruitpunch.arena import ROBOT_XML
from fruitpunch.game import Game
from fruitpunch.overlay import Overlay
from fruitpunch.planner import PlannerCommand, root_yaw, standing_qpos
from fruitpunch.runner import Runner

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "clips"
OUT.mkdir(parents=True, exist_ok=True)
W, H, FPS = 960, 540, 30
try:
    FONT = ImageFont.truetype("arial.ttf", 22)
    SMALL = ImageFont.truetype("arial.ttf", 17)
except OSError:
    FONT = SMALL = ImageFont.load_default()


class Clip:
    def __init__(self, name, robot, bat=None):
        self.name = name
        self.r = Runner(robot, bat=bat)
        self.ov = Overlay(str(ROBOT_XML))
        self.ren = mujoco.Renderer(self.r.arena.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 6.5, 115, -10
        self.writer = imageio.get_writer(OUT / f"{name}.mp4", fps=FPS, codec="libx264",
                                         quality=7, pixelformat="yuv420p", macro_block_size=16)
        self.next_frame = 0.0
        self.caption = ""
        self.sub = ""
        self.events = []

    def frame(self, look_offset=(0.8, 0, 0)):
        q = self.r.robot_qpos()
        self.cam.lookat[:] = (q[0] + look_offset[0], q[1] + look_offset[1], 1.05)
        self.ren.update_scene(self.r.arena.d, self.cam)
        self.ov.draw(self.ren.scene, clear=False)
        img = Image.fromarray(self.ren.render())
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, H - 64, W, H], fill=(0, 0, 0, 150))
        d.text((16, H - 58), self.caption, font=FONT, fill=(255, 255, 255))
        d.text((16, H - 28), self.sub, font=SMALL, fill=(200, 210, 220))
        d.text((W - 110, 12), f"t = {self.r.time:5.1f}s", font=SMALL, fill=(230, 230, 230))
        self.writer.append_data(np.asarray(img))

    def tick(self, game=None, **kw):
        self.r.tick()
        if game:
            game.step()
        while self.r.time >= self.next_frame:
            self.frame(**kw)
            self.next_frame += 1.0 / FPS

    def run(self, seconds, game=None, **kw):
        for _ in range(int(seconds / 0.02)):
            self.tick(game, **kw)

    def close(self):
        self.writer.close()
        print("wrote", OUT / f"{self.name}.mp4")


def clip_drive():
    c = Clip("drive_kinematic", "kinematic")
    c.cam.azimuth = 135
    steps = [("Idle", PlannerCommand(mode=0), 1.5),
             ("Walk forward (velocity command)", PlannerCommand(mode=2, move_dir=(1, 0)), 3.0),
             ("Turn in place to +90 deg", PlannerCommand(mode=0, facing_yaw=np.pi / 2), 2.0),
             ("Strafe right while facing +90 deg", PlannerCommand(mode=2, move_dir=(1, 0), facing_yaw=np.pi / 2), 2.5),
             ("Run toward +y", PlannerCommand(mode=3, target_vel=2.0, move_dir=(0, 1), facing_yaw=np.pi / 2), 2.5),
             ("Boxing walk", PlannerCommand(mode=10, target_vel=1.0, move_dir=(0, -1), facing_yaw=-np.pi / 2), 3.0),
             ("Squat (height 0.4)", PlannerCommand(mode=4, height=0.4, facing_yaw=-np.pi / 2), 2.5),
             ("Idle", PlannerCommand(mode=0, facing_yaw=-np.pi / 2), 1.5)]
    for label, cmd, secs in steps:
        c.caption, c.sub = label, "kplanner velocity/direction mode, kinematic robot"
        c.r.cmd = cmd
        c.run(secs, look_offset=(0, 0, 0))
    c.close()


def clip_target(robot):
    c = Clip(f"target_{robot}", robot)
    c.cam.azimuth, c.cam.distance = 140, 6.0
    c.run(1.0)
    targets = [("Go to (2.0, 1.0), face +90 deg", (2.0, 1.0), np.pi / 2, 2, False, 7.0),
               ("In-place: shift 0.4 m, turn to 180 deg", (2.0, 1.4), np.pi, 2, False, 6.0),
               ("Go to (0, 0), face 0 deg, then squat", (0.0, 0.0), 0.0, 2, True, 9.0)]
    for label, xy, yaw, mode, squat, secs in targets:
        ghost = standing_qpos(xy, yaw)
        if squat:
            ghost[2] -= 0.3
        c.ov.ghost_qpos = ghost
        c.ov.target_marks = [(np.array([*xy, 0.02]), yaw)]
        c.r.cmd = PlannerCommand(mode=mode, target_xy=xy, target_yaw=yaw)
        squatted = False
        for _ in range(int(secs / 0.02)):
            q = c.r.robot_qpos()
            err = np.linalg.norm(q[:2] - xy)
            dyaw = np.degrees(abs((root_yaw(q) - yaw + np.pi) % (2 * np.pi) - np.pi))
            if squat and not squatted and err < 0.12 and dyaw < 15:
                c.r.cmd, squatted = PlannerCommand(mode=4, height=0.4, facing_yaw=yaw), True
            c.caption = f"Target pose: {label}"
            c.sub = f"{robot} robot | position error {err * 100:4.0f} cm, heading error {dyaw:4.1f} deg"
            c.tick(look_offset=(0.2, 0.3, 0))
        c.events.append({"target": label, "pos_err_cm": round(err * 100, 1), "yaw_err_deg": round(dyaw, 1)})
    c.close()
    return c.events


def clip_game(robot, throws, seed, swat_only=False, objective="dodge", bat=None, balls=None):
    import fruitpunch.game as game_mod
    saved = game_mod.dodge_candidates
    if swat_only:
        game_mod.dodge_candidates = lambda *a, **k: []
    c = Clip(f"{objective}_{robot}{'_bat' if bat else ''}", robot, bat=bat)
    extra = {"balls": balls} if balls else {}
    g = Game(c.r, auto=True, interval=3.0, seed=seed, objective=objective, **extra,
             log_name=f"clip_{objective}_{robot}.csv")
    if g.log_path.exists():
        g.log_path.unlink()
    g.next_throw = 1.5
    done = 0
    while len(g.records) < throws or g.state != "idle":
        rec = g.records[-1] if g.records else None
        if rec is None:
            c.caption, c.sub = "Waiting for the first throw", f"{robot} robot"
        else:
            status = rec.outcome.upper() if g.state == "idle" else (
                "in flight" if rec.actual_t is None else f"hit {rec.actual_body}")
            pred = "-" if rec.pred_body is None else f"{rec.pred_body} @ {rec.pred_t:.2f}s"
            c.caption = (f"{objective.upper()} | throw {rec.n} ({rec.style}): "
                         f"{rec.action.split(' > ')[-1] if rec.action else '...'}  ->  {status}")
            c.sub = f"first prediction: {pred} | re-plans: {rec.replans} | {g.summary()} | points {g.points()}"
        c.ov.path = g.path if g.state != "idle" else None
        c.ov.contact = g.contact if g.state != "idle" else None
        if g.state == "reacting":
            end = c.r.track.frames[-1].copy()
            if not c.r.kinematic:
                end[:2] += c.r.drift()
            c.ov.ghost_qpos = end
        else:
            c.ov.ghost_qpos = None
        c.tick(g, look_offset=(1.2, 0, 0))
        if len(g.records) > done and g.state == "idle":
            done = len(g.records)
    c.run(1.0, look_offset=(1.2, 0, 0))
    c.close()
    game_mod.dodge_candidates = saved
    return [{"n": r.n, "ball": r.style, "first_prediction": r.pred_body, "pred_t": r.pred_t and round(r.pred_t, 2),
             "action": r.action, "replans": r.replans, "hit": r.actual_body, "outcome": r.outcome}
            for r in g.records if r.done], g.summary(), g.points()


if __name__ == "__main__":
    results = {}
    clip_drive()
    results["target_kinematic"] = clip_target("kinematic")
    results["target_physics"] = clip_target("physics")
    results["dodge_kinematic"] = clip_game("kinematic", 10, seed=1)
    results["dodge_physics"] = clip_game("physics", 10, seed=1)
    results["hit_kinematic_bat"] = clip_game("kinematic", 10, seed=2, objective="hit", bat="both")
    results["hit_physics_bat"] = clip_game("physics", 10, seed=2, objective="hit", bat="both")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, default=str)[:3000])
