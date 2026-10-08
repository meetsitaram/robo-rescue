"""Render offscreen snapshots of each milestone (ghost target, trajectory, dodge) to docs/img."""

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rescue.arena import ROBOT_XML
from rescue.game import Game
from rescue.overlay import Overlay
from rescue.planner import PlannerCommand, standing_qpos
from rescue.runner import Runner

robot = sys.argv[1] if len(sys.argv) > 1 else "kinematic"
OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
r = Runner(robot)
ov = Overlay(str(ROBOT_XML))
ren = mujoco.Renderer(r.arena.m, 720, 1280)
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 4.5, 135, -18


def snap(name, look=None):
    cam.lookat[:] = look if look is not None else (r.robot_qpos()[0], r.robot_qpos()[1], 0.8)
    ren.update_scene(r.arena.d, cam)
    ov.draw(ren.scene, clear=False)
    Image.fromarray(ren.render()).save(OUT / f"{robot}_{name}.png")
    print("saved", OUT / f"{robot}_{name}.png")


def run(seconds):
    for _ in range(int(seconds / 0.02)):
        r.tick()


run(1.5)
# Target pose with ghost, before and after.
tgt, yaw = (1.2, 0.6), np.pi / 2
ov.ghost_qpos = standing_qpos(tgt, yaw)
ov.target_marks = [(np.array([*tgt, 0.02]), yaw)]
snap("target_before", look=(0.6, 0.3, 0.7))
r.cmd = PlannerCommand(mode=2, target_xy=tgt, target_yaw=yaw)
run(1.0)
snap("target_moving", look=(0.6, 0.3, 0.7))
run(6.0)
snap("target_arrived", look=(0.6, 0.3, 0.7))

# A throw with predicted path + contact, then the dodge.
ov.target_marks = []
g = Game(r, auto=False, seed=3, log_name="render.csv")
g.home_xy, g.home_yaw = r.robot_qpos()[:2].copy(), np.pi / 2
g.throw()
while g.state == "tracking" or g.path is None:
    r.tick(); g.step()
ov.path, ov.contact = g.path, g.contact
ov.ghost_qpos = r.track.frames[-1].copy()
snap("throw_predicted")
for _ in range(30):
    r.tick(); g.step()
ov.path, ov.contact = g.path, g.contact
snap("throw_dodging")
while g.state != "idle":
    r.tick(); g.step()
