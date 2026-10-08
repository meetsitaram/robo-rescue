"""Run N automatic throws without a viewer and print/log the outcome of each."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fruitpunch.game import Game
from fruitpunch.runner import Runner

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="kinematic", choices=["kinematic", "physics"])
ap.add_argument("--throws", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-dodge", action="store_true", help="baseline: never react")
args = ap.parse_args()

r = Runner(args.robot)
g = Game(r, seed=args.seed, log_name=f"throws_{args.robot}_s{args.seed}{'_nododge' if args.no_dodge else ''}.csv")
if g.log_path.exists():
    g.log_path.unlink()
if args.no_dodge:
    g._select = lambda p, v, q: ("none", None)
while len(g.records) < args.throws or g.state != "idle":
    r.tick()
    g.step()
    if r.fallen():
        print("ROBOT FELL at t=%.1f" % r.time)
        break
print(args.robot, "summary:", g.summary(), "log:", g.log_path)
