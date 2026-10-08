"""Record README highlight shots: play games and keep the first throw that matches each shot's
criterion, saved as an MP4 and a small GIF in docs/media/.

    python scripts/render_highlights.py [shot ...]
"""

import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fruitpunch.arena import ROBOT_XML
from fruitpunch.game import Game
from fruitpunch.overlay import Overlay
from fruitpunch.planner import PlannerCommand, root_yaw, standing_qpos
from fruitpunch.runner import Runner

OUT = Path(__file__).resolve().parent.parent / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)
W, H, FPS = 960, 540, 30
try:
    FONT, SMALL = ImageFont.truetype("arial.ttf", 24), ImageFont.truetype("arial.ttf", 18)
except OSError:
    FONT = SMALL = ImageFont.load_default()


class Camera:
    def __init__(self, runner, distance=5.0, azimuth=115, elevation=-12):
        self.r = runner
        self.ov = Overlay(str(ROBOT_XML))
        self.ren = mujoco.Renderer(runner.arena.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.lookat = None

    def frame(self, title, sub, offset=(0.9, 0.0)):
        q = self.r.robot_qpos()
        want = np.array([q[0] + offset[0], q[1] + offset[1], 0.95])
        self.lookat = want if self.lookat is None else 0.9 * self.lookat + 0.1 * want  # smooth follow
        self.cam.lookat[:] = self.lookat
        self.ren.update_scene(self.r.arena.d, self.cam)
        self.ov.draw(self.ren.scene, clear=False)
        img = Image.fromarray(self.ren.render())
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, H - 70, W, H], fill=(0, 0, 0, 150))
        d.text((18, H - 64), title, font=FONT, fill=(255, 255, 255))
        d.text((18, H - 32), sub, font=SMALL, fill=(205, 215, 225))
        return np.asarray(img)


def save(name, frames):
    mp4 = OUT / f"{name}.mp4"
    imageio.mimsave(mp4, frames, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p",
                    macro_block_size=16)
    gif = OUT / f"{name}.gif"
    vf = ("fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(mp4),
                    "-vf", vf, str(gif)], check=True)
    print(f"saved {name}: {len(frames) / FPS:.1f}s, gif {gif.stat().st_size / 1e6:.1f} MB")


def game_shot(name, robot, objective, match, balls, seed=0, bat=None, react=True,
              max_throws=25, tail=1.2, label=""):
    """Play throws until one satisfies match(record); save that throw (launch to tail)."""
    r = Runner(robot, bat=bat)
    g = Game(r, auto=True, interval=tail + 1.0, seed=seed, objective=objective, balls=tuple(balls),
             log_name=f"highlight_{name}.csv")
    if g.log_path.exists():
        g.log_path.unlink()
    if not react:
        g._select = lambda p, v, q: ("none (taking it)", None)
    g.next_throw = r.time + 0.8
    cam = Camera(r)
    frames, rec, finish_t = [], None, None
    next_frame = r.time
    while len(g.records) <= max_throws:
        r.tick()
        g.step()
        if rec is None:
            if not g.records or g.records[-1].done:
                continue
            rec, frames, next_frame = g.records[-1], [], r.time  # follow this throw from launch
        if rec.done and finish_t is None:
            finish_t = r.time
        if r.time >= next_frame:
            next_frame += 1.0 / FPS
            state = rec.outcome.upper() if rec.done else (
                "in flight" if rec.actual_t is None else f"hit {rec.actual_body}")
            title = f"{label} | {rec.style}: {(rec.action or '...').split(' > ')[-1]} -> {state}"
            sub = (f"{'kinematic' if r.kinematic else 'SONIC physics'} robot | "
                   f"predicted {rec.pred_body or '-'}" + (f" @ {rec.pred_t:.2f}s" if rec.pred_t else ""))
            cam.ov.path = g.path if g.state != "idle" else None
            cam.ov.contact = g.contact if g.state != "idle" else None
            cam.ov.ghost_qpos = None
            if g.state == "reacting":
                end = r.track.frames[-1].copy()
                if not r.kinematic:
                    end[:2] += r.drift()
                cam.ov.ghost_qpos = end
            frames.append(cam.frame(title, sub))
        if finish_t is not None and r.time - finish_t >= tail:
            if match(rec, r):
                save(name, frames)
                return rec
            rec, finish_t = None, None
            if r.fallen():
                r.reset(tuple(g.home_xy), g.home_yaw)
            g.next_throw = min(g.next_throw, r.time + 0.3)
    print(f"{name}: no matching throw in {max_throws}")


def target_shot(name, robot, xy=(2.0, 1.0), yaw=np.pi / 2, seconds=8.0):
    """kplanner specific-target mode: walk to a ghost pose."""
    r = Runner(robot)
    cam = Camera(r, distance=5.5, azimuth=140, elevation=-16)
    for _ in range(50):
        r.tick()
    cam.ov.ghost_qpos = standing_qpos(xy, yaw)
    cam.ov.target_marks = [(np.array([*xy, 0.02]), yaw)]
    r.cmd = PlannerCommand(mode=2, target_xy=xy, target_yaw=yaw)
    frames, next_frame = [], r.time
    for _ in range(int(seconds / 0.02)):
        r.tick()
        if r.time >= next_frame:
            next_frame += 1.0 / FPS
            q = r.robot_qpos()
            err = np.linalg.norm(q[:2] - xy) * 100
            dyaw = np.degrees(abs((root_yaw(q) - yaw + np.pi) % (2 * np.pi) - np.pi))
            frames.append(cam.frame(
                f"kplanner target mode -> ({xy[0]:.1f}, {xy[1]:.1f}) facing {np.degrees(yaw):.0f} deg",
                f"{'kinematic' if r.kinematic else 'SONIC physics'} robot | error {err:4.0f} cm, {dyaw:4.1f} deg",
                offset=(0.5, 0.5)))
    save(name, frames)


def is_duck(rec, r):
    return (rec.action or "").split(" > ")[-1].startswith("duck") and rec.outcome == "dodged"


def is_punch_hit(rec, r):
    last = (rec.action or "").split(" > ")[-1]
    return rec.outcome == "swatted" and ("punch" in last or "hook" in last)


SHOTS = {
    "target_kinematic": lambda: target_shot("target_kinematic", "kinematic"),
    "target_sonic": lambda: target_shot("target_sonic", "physics"),
    "duck_kinematic": lambda: game_shot("duck_kinematic", "kinematic", "dodge", is_duck,
                                        ["basketball", "volleyball", "tomato"], label="DODGE"),
    "duck_sonic": lambda: game_shot("duck_sonic", "physics", "dodge", is_duck,
                                    ["volleyball", "basketball", "tomato"], seed=1, label="DODGE"),
    "cannonball_sonic": lambda: game_shot("cannonball_sonic", "physics", "dodge",
                                          lambda rec, r: rec.outcome == "struck" and r.fallen(), ["cannonball"],
                                          react=False, tail=2.5, max_throws=40, label="NO REACTION"),
    "punch_hand": lambda: game_shot("punch_hand", "kinematic", "hit", is_punch_hit,
                                    ["tomato", "volleyball", "dodgeball"], label="HIT"),
    "punch_bat": lambda: game_shot("punch_bat", "kinematic", "hit", is_punch_hit,
                                   ["tomato", "basketball", "volleyball"], seed=2, bat="both", label="HIT + BAT"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or SHOTS:
        SHOTS[name]()
