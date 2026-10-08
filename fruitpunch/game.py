"""FruitPunch game logic: throw, track, predict, dodge, score."""

import copy
import csv
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import g1
from .planner import PlannerCommand, root_yaw
from .projectile import GRAVITY, SIM_DT, ContactPredictor, launch_velocity

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
STICKY = {"torso_link", "pelvis"}


def random_throw(robot_qpos, rng, dist=(4.0, 6.0), azimuth_deg=40, flight=(1.0, 1.4)):
    """Launch position and velocity of a soft lob at the robot's torso/head."""
    yaw = root_yaw(robot_qpos)
    az = yaw + np.radians(rng.uniform(-azimuth_deg, azimuth_deg))
    d = rng.uniform(*dist)
    base = robot_qpos[:3].copy()
    base[2] = 0.0
    p0 = base + np.array([d * np.cos(az), d * np.sin(az), rng.uniform(1.2, 1.8)])
    aim = robot_qpos[:3] + np.array([0.0, 0.0, rng.uniform(0.25, 0.6)])
    aim[:2] += rng.uniform(-0.08, 0.08, 2)
    return p0, launch_velocity(p0, aim, rng.uniform(*flight))


class TrajectoryTracker:
    """Least-squares parabola (known gravity) fitted to observed tomato positions."""

    def __init__(self, noise_std=0.0, rng=None):
        self.noise = noise_std
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.t, self.p = [], []

    def observe(self, t, pos):
        self.t.append(t)
        self.p.append(np.asarray(pos) + self.rng.normal(0, self.noise, 3) * (self.noise > 0))

    def ready(self, n=4):
        return len(self.t) >= n

    def estimate(self):
        """(p, v) at the latest observation time."""
        t = np.asarray(self.t) - self.t[-1]
        y = np.asarray(self.p) - 0.5 * GRAVITY * (t * (t + SIM_DT))[:, None]  # sim's Euler parabola
        a = np.stack([np.ones_like(t), t], 1)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        return coef[0], coef[1]


@dataclass
class Candidate:
    name: str
    cmd: PlannerCommand


def dodge_candidates(q, threat_dir_xy, base_mode=2):
    """Candidate reactions. threat_dir_xy: unit xy direction the tomato travels."""
    x, y, yaw = q[0], q[1], root_yaw(q)
    side = np.array([-threat_dir_xy[1], threat_dir_xy[0]])  # left of the tomato's path
    c = []
    for s, label in ((1, "left"), (-1, "right")):
        for d in (0.5, 0.8):
            t = (x + s * side[0] * d, y + s * side[1] * d)
            c.append(Candidate(f"sidestep {label} {d:.1f}m", PlannerCommand(mode=base_mode, target_xy=t, target_yaw=yaw)))
    c.append(Candidate("squat", PlannerCommand(mode=4, height=0.3, facing_yaw=yaw)))
    c.append(Candidate("kneel", PlannerCommand(mode=5, height=0.3, facing_yaw=yaw)))
    back = (x + threat_dir_xy[0] * 0.8, y + threat_dir_xy[1] * 0.8)
    c.append(Candidate("back off", PlannerCommand(mode=base_mode, target_xy=back, target_yaw=yaw)))
    for s in (1, -1):
        c.append(Candidate(f"turn {'left' if s > 0 else 'right'}",
                           PlannerCommand(mode=0, facing_yaw=yaw + s * 1.4)))
    return c


@dataclass
class ThrowRecord:
    n: int
    t_launch: float
    pred_t: float | None = None
    pred_body: str | None = None
    action: str = "none"
    action_clears: bool | None = None
    decide_ms: float = 0.0
    replans: int = 0
    actual_body: str = "miss"
    actual_t: float | None = None
    stuck: bool = False

    @property
    def outcome(self):
        if self.actual_body in ("miss", "floor", "world"):
            return "dodged" if self.pred_body not in (None, "floor") else "miss"
        return "splat" if self.stuck else "bounced"


@dataclass
class Game:
    runner: object
    auto: bool = True
    interval: float = 3.5
    seed: int = 0
    log_name: str = "throws.csv"
    rng: np.random.Generator = field(init=False)
    state: str = field(default="idle", init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.predictor = ContactPredictor(self.runner.arena.m)
        self.tracker = TrajectoryTracker(noise_std=0.005, rng=self.rng)
        self.records: list[ThrowRecord] = []
        self.next_throw = self.runner.time + 2.0
        self.path = None
        self.contact = None
        self.rest_cmd = None
        self.last_eval = -1.0
        self.home_yaw = root_yaw(self.runner.robot_qpos())
        self.home_xy = self.runner.robot_qpos()[:2].copy()
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / self.log_name

    # --- throw lifecycle ----------------------------------------------------------
    def throw(self):
        r = self.runner
        p0, v0 = random_throw(r.robot_qpos(), self.rng)
        r.arena.launch(p0, v0)
        self.tracker.reset()
        self.path, self.contact = None, None
        self.records.append(ThrowRecord(len(self.records) + 1, r.time))
        self.rest_cmd = r.cmd
        self.state = "tracking"

    def step(self):
        """Call once per control tick (after runner.tick())."""
        r = self.runner
        a = r.arena
        if self.state == "idle":
            if self.auto and r.time >= self.next_throw:
                self.throw()
            return
        rec = self.records[-1]
        t_since = r.time - rec.t_launch
        if self.state in ("tracking", "reacting") and a.tomato_in_flight():
            p, _ = a.tomato_state()
            self.tracker.observe(r.time, p)
            first = self.state == "tracking"
            if self.tracker.ready(5) and (first or r.time - self.last_eval >= self.REEVAL):
                self._evaluate(rec, first)
        if a.first_hit is not None and rec.actual_t is None:
            rec.actual_body, rec.actual_t = a.first_hit[1], a.first_hit[0] - rec.t_launch
        if t_since > 2.5 or (a.first_hit is not None and t_since > (rec.actual_t or 0) + 1.0):
            self._finish(rec)

    REEVAL = 0.1        # s between re-predictions while the tomato flies
    MIN_REACT = 0.25    # s: too late to change the plan below this time-to-contact

    def _evaluate(self, rec, first):
        """Refit the trajectory, predict contact against the committed motion, re-plan if hit.

        The fit sharpens as observations accumulate (5 noisy samples give ~0.15 m/s velocity
        error, i.e. ~20 cm at impact), so the reaction is re-checked every REEVAL seconds.
        """
        r = self.runner
        t0 = time.perf_counter()
        self.last_eval = r.time
        p, v = self.tracker.estimate()
        q = r.robot_qpos()
        # Predict against where the robot is going (its reference track), not a frozen pose.
        future = r.track.frames[r.track.cur:].copy()
        if not r.kinematic:
            future[:, :2] += r.drift()
        contact, path = self.predictor.predict(q, p, v, robot_traj=future, traj_dt=g1.CONTROL_DT)
        self.path = path
        self.contact = contact.point if contact else None
        threat = contact is not None and contact.body != "floor"
        if first:
            rec.pred_body = contact.body if contact else None
            rec.pred_t = contact.t if contact else None
            self.state = "reacting"
            # A moving robot keeps going past the ~1 s reference horizon, so "no hit" is not
            # trustworthy for it: stop or dodge anyway.
            threat |= r.cmd.target_xy is not None or np.linalg.norm(r.cmd.move_dir) > 1e-6
            if not threat:
                rec.action = "stay"
        if threat and (contact is None or contact.t > self.MIN_REACT):
            name, clears = self._select(p, v, q)
            rec.action = name if first or rec.action == "stay" else f"{rec.action} > {name}"
            rec.action_clears = clears
            rec.replans += 0 if first else 1
        rec.decide_ms += 1000 * (time.perf_counter() - t0)

    def _select(self, p, v, q):
        """Try reactions (stop first, then dodges); commit the first whose blended plan clears."""
        r = self.runner
        d = v[:2] / (np.linalg.norm(v[:2]) + 1e-9)
        stop = Candidate("stop", PlannerCommand(mode=0, facing_yaw=root_yaw(q)))
        best = None
        for cand in [stop] + dodge_candidates(q, d):
            plan30, gen = r.plan_now(cand.cmd)
            # Evaluate exactly what would be executed: the plan blended into the live track.
            trial = copy.deepcopy(r.track)
            trial.merge(plan30, gen)
            traj = trial.frames.copy()
            if not r.kinematic:
                traj[:, :2] += r.drift()
            hit, _ = self.predictor.predict(q, p, v, robot_traj=traj, traj_dt=g1.CONTROL_DT)
            clears = hit is None or hit.body == "floor"
            score = (clears, hit.t if hit and not clears else 9.0)
            if best is None or score > best[0]:
                best = (score, cand, clears, trial)
            if clears:
                break
        _, cand, clears, trial = best
        r.commit(cand.cmd, trial)
        return cand.name, clears

    def _finish(self, rec):
        a = self.runner.arena
        rec.stuck = a.stuck
        print(f"throw {rec.n:3d}: predicted {rec.pred_body or 'miss':>18s}"
              f"{'' if rec.pred_t is None else f' @{rec.pred_t:.2f}s'}  action={rec.action:<20s}"
              f" clears={rec.action_clears} replans={rec.replans} cpu={rec.decide_ms:.0f}ms  ->  {rec.outcome}"
              f" ({rec.actual_body})")
        self._log(rec)
        # Walk back to the home spot, facing the throwing side.
        self.runner.cmd = PlannerCommand(mode=1, target_xy=tuple(self.home_xy), target_yaw=self.home_yaw)
        if not a.stuck:
            a.park_tomato()
        self.state = "idle"
        self.next_throw = self.runner.time + self.interval

    def _log(self, rec):
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["n", "t_launch", "pred_body", "pred_t", "action", "plan_clears",
                            "decide_ms", "replans", "actual_body", "actual_t", "stuck", "outcome"])
            w.writerow([rec.n, f"{rec.t_launch:.2f}", rec.pred_body, rec.pred_t and f"{rec.pred_t:.3f}",
                        rec.action, rec.action_clears, f"{rec.decide_ms:.0f}", rec.replans, rec.actual_body,
                        rec.actual_t and f"{rec.actual_t:.3f}", rec.stuck, rec.outcome])

    def summary(self):
        from collections import Counter
        c = Counter(r.outcome for r in self.records)
        return dict(c)
