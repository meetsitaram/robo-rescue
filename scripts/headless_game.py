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
ap.add_argument("--objective", default="dodge", choices=["dodge", "hit"])
ap.add_argument("--bat", default=None, choices=["left", "right", "both"])
ap.add_argument("--balls", default=None, help="comma-separated subset of fruitpunch.balls.BALLS")
ap.add_argument("--no-dodge", action="store_true", help="baseline: never react")
ap.add_argument("--swat-only", action="store_true", help="no dodging moves: stay put or swat")
args = ap.parse_args()

r = Runner(args.robot, bat=args.bat)
extra = {"balls": tuple(args.balls.split(","))} if args.balls else {}
g = Game(r, seed=args.seed, objective=args.objective, **extra, log_name=f"throws_{args.robot}_{args.objective}_s{args.seed}{'_nododge' if args.no_dodge else ''}{'_swat' if args.swat_only else ''}{'_bat' if args.bat else ''}.csv")
if g.log_path.exists():
    g.log_path.unlink()
if args.swat_only:
    import fruitpunch.game as game_mod
    game_mod.dodge_candidates = lambda *a, **k: []
if args.no_dodge:
    g._select = lambda p, v, q: ("none", None)
falls = 0
while len(g.records) < args.throws or g.state != "idle":
    r.tick()
    g.step()
    if r.fallen() and g.state == "idle":
        falls += 1
        print("robot fell at t=%.1f - reset" % r.time)
        r.reset(tuple(g.home_xy), g.home_yaw)
print(args.robot, "summary:", g.summary(), "points:", g.points(), "falls:", falls, "log:", g.log_path)
