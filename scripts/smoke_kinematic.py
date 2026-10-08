"""Headless check of the kinematic pipeline: planner -> robot, target mode, tomato stick/bounce."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fruitpunch import g1
from fruitpunch.arena import Arena
from fruitpunch.config import PLANNER_ONNX
from fruitpunch.planner import KPlanner, MotionTrack, PlannerCommand, ReplanPolicy, standing_qpos, root_yaw
from fruitpunch.projectile import ContactPredictor, launch_velocity

N_SUB = int(round(g1.CONTROL_DT / g1.SIM_DT))

arena = Arena(kinematic=True)
planner = KPlanner(PLANNER_ONNX)
q0 = standing_qpos()
arena.set_robot_state(q0)
track = MotionTrack(q0)
rp = ReplanPolicy()


def run(cmd, seconds, label):
    t_plan = 0.0
    n_plan = 0
    for _ in range(int(seconds / g1.CONTROL_DT)):
        if rp.should_replan(cmd, track, g1.CONTROL_DT):
            ctx, gen = track.context()
            t = time.perf_counter()
            track.merge(planner.plan(ctx, cmd), gen)
            t_plan += time.perf_counter() - t
            n_plan += 1
        qa = track.now()
        track.advance()
        arena.step_kinematic(qa, track.now(), 0.0, 1.0, N_SUB)
    q = arena.robot_qpos()
    print(f"{label:28s} root=({q[0]:+.2f},{q[1]:+.2f},{q[2]:.2f}) yaw={np.degrees(root_yaw(q)):+6.1f}  "
          f"plans={n_plan} avg={1000 * t_plan / max(n_plan, 1):.0f}ms")
    return q


run(PlannerCommand(mode=0), 1.0, "idle")
run(PlannerCommand(mode=2, move_dir=(1, 0), facing_yaw=0), 3.0, "walk +x 3s")
run(PlannerCommand(mode=0, facing_yaw=np.pi / 2), 2.0, "turn in place to +90")
q = run(PlannerCommand(mode=1, target_xy=(0.0, 0.0), target_yaw=0.0), 6.0, "target (0,0) yaw 0")
run(PlannerCommand(mode=0, target_xy=(0.3, 0.4), target_yaw=-np.pi / 2), 4.0, "in-place target (.3,.4) -90")
run(PlannerCommand(mode=4, height=0.4), 2.0, "squat h=0.4")
run(PlannerCommand(mode=0), 2.0, "idle")

# Tomato thrown at the chest: predict, then simulate.
pred = ContactPredictor(arena.m)
q = arena.robot_qpos()
chest = q[:3] + np.array([0.05, 0, 0.35])
p0 = q[:3] + np.array([4.0, 0.3, 0.6])
v0 = launch_velocity(p0, chest, 1.2)
c, path = pred.predict(q, p0, v0)
print("predicted:", None if c is None else (round(c.t, 3), c.body, c.point.round(3)))
arena.launch(p0, v0)
t0 = arena.d.time
run(PlannerCommand(mode=0), 2.5, "idle during throw")
hit = arena.first_hit
print("actual   :", None if hit is None else (round(hit[0] - t0, 3), hit[1], hit[2].round(3)),
      "stuck=", arena.stuck, "tomato z=", arena.tomato_state()[0][2].round(3))

# Tomato thrown at the shin: should bounce, not stick.
q = arena.robot_qpos()
shin = q[:3] + np.array([0.05, 0.12, -0.55])
p0 = q[:3] + np.array([3.0, 0.0, 0.2])
v0 = launch_velocity(p0, shin, 1.0)
arena.launch(p0, v0)
t0 = arena.d.time
run(PlannerCommand(mode=0), 2.0, "idle during low throw")
hit = arena.first_hit
print("low hit  :", None if hit is None else (round(hit[0] - t0, 3), hit[1]), "stuck=", arena.stuck,
      "tomato pos=", arena.tomato_state()[0].round(2))
