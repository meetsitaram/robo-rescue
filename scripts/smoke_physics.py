"""Headless check of SONIC in physics mode: stand, walk, turn, go-to target, squat."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fruitpunch import g1
from fruitpunch.planner import PlannerCommand, root_yaw
from fruitpunch.runner import Runner

robot = sys.argv[1] if len(sys.argv) > 1 else "physics"
r = Runner(robot)


def run(cmd, seconds, label):
    r.cmd = cmd
    t0 = time.perf_counter()
    min_z = 9
    for _ in range(int(seconds / g1.CONTROL_DT)):
        r.tick()
        min_z = min(min_z, r.arena.d.qpos[2])
    q = r.robot_qpos()
    ref = r.track.now()
    wall = time.perf_counter() - t0
    print(f"{label:26s} root=({q[0]:+.2f},{q[1]:+.2f},{q[2]:.2f}) yaw={np.degrees(root_yaw(q)):+6.1f} "
          f"ref=({ref[0]:+.2f},{ref[1]:+.2f}) minz={min_z:.2f} rt={seconds / wall:.2f}x"
          + ("  FALLEN" if r.fallen() else ""))


run(PlannerCommand(mode=0), 4.0, "idle (band 1s)")
run(PlannerCommand(mode=2, move_dir=(1, 0)), 3.0, "walk +x")
run(PlannerCommand(mode=0, facing_yaw=np.pi / 2), 2.0, "turn to +90")
run(PlannerCommand(mode=1, target_xy=(0.0, 0.0), target_yaw=0.0), 6.0, "target (0,0) yaw 0")
run(PlannerCommand(mode=0, target_xy=(0.3, 0.4), target_yaw=-np.pi / 2), 4.0, "in-place (.3,.4) -90")
run(PlannerCommand(mode=4, height=0.4, facing_yaw=-np.pi / 2), 3.0, "squat h=0.4")
run(PlannerCommand(mode=0, facing_yaw=-np.pi / 2), 3.0, "idle")
