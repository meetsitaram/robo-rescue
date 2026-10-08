"""Rescue game logic: throw, track, predict, dodge, score."""

import copy
import csv
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from . import g1
from .planner import PlannerCommand, apply_overlay, root_yaw
from .projectile import GRAVITY, SIM_DT, ContactPredictor, ballistic, launch_velocity
from .balls import BALLS
from .reach import ARM_JOINTS, ArmIK

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
STICKY = {"torso_link", "pelvis"}


def random_throw(robot_qpos, rng, ball="tomato", azimuth_deg=40):
    """Launch position, velocity and gravity vector of a throw of `ball` (see balls.BALLS) at
    the robot, aimed anywhere from the pelvis to the chest."""
    st = BALLS[ball]
    g = GRAVITY * st["gscale"]
    yaw = root_yaw(robot_qpos)
    az = yaw + np.radians(rng.uniform(-azimuth_deg, azimuth_deg))
    d = rng.uniform(*st["dist"])
    base = robot_qpos[:3].copy()
    base[2] = 0.0
    p0 = base + np.array([d * np.cos(az), d * np.sin(az), rng.uniform(*st["release"])])
    aim = robot_qpos[:3] + np.array([0.0, 0.0, rng.uniform(-0.05, 0.35)])  # ~0.74-1.14 m
    aim[:2] += rng.uniform(-0.08, 0.08, 2)
    return p0, launch_velocity(p0, aim, rng.uniform(*st["flight"]), g=g), g


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

    def estimate(self, g=GRAVITY):
        """(p, v) at the latest observation time, for a known gravity vector g."""
        t = np.asarray(self.t) - self.t[-1]
        y = np.asarray(self.p) - 0.5 * g * (t * (t + SIM_DT))[:, None]  # sim's Euler parabola
        a = np.stack([np.ones_like(t), t], 1)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        return coef[0], coef[1]


@dataclass
class Candidate:
    name: str
    cmd: PlannerCommand
    overlay: dict | None = None  # additive joint offsets {mujoco joint index: rad}
    overlay_timing: dict | None = None  # apply_overlay timing kwargs (start, ramp_in, hold, ramp_out)


# Whole-body forward lean: pitch the root forward and flex both hips by the same angle so
# the legs stay planted. (SONIC holds waist pitch fixed, so a waist-only lean never happens.)
LEAN = {"root_pitch": 0.45, g1.L_HIP_PITCH: -0.45, g1.R_HIP_PITCH: -0.45}
HAND_BODIES = {"left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
               "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"}
# Reaction moves use the planner's shortest horizon (6 tokens = 0.8 s), which reaches the
# target pose much sooner, played back 1.3x faster. SONIC squat duck: pelvis below 0.6 m in
# 0.40 s instead of 0.94 s; escape walk covers 0.27 m in 0.6 s instead of 0.09 m.
REACT_TOKENS, REACT_SPEED = 6, 1.3
SONIC_LAG = 0.12  # s the physics robot trails its reference; held at its current pose when predicting


def dodge_candidates(q, threat_dir_xy, contact_z=None, turn_walk_first=False, base_mode=2):
    """Candidate reactions, in the order they are tried.

    threat_dir_xy: unit xy direction the tomato travels; contact_z: predicted hit height.
    High throws (chest/head) try ducking first: it needs no travel. SONIC tracks forward
    walking far better than sidestepping, so in physics mode the "turn and walk out"
    escapes (heading along the escape direction) are tried before sidesteps.
    """
    x, y, yaw = q[0], q[1], root_yaw(q)
    side = np.array([-threat_dir_xy[1], threat_dir_xy[0]])  # left of the tomato's path
    still = PlannerCommand(mode=0, facing_yaw=yaw)
    squat = PlannerCommand(mode=4, height=0.3, facing_yaw=yaw)
    upper = [Candidate("twist torso left", still, {g1.WAIST_YAW: 1.2}),
             Candidate("twist torso right", still, {g1.WAIST_YAW: -1.2}),
             Candidate("lean forward", still, LEAN)]
    ducks = [Candidate("duck (squat 0.3)", squat),
             Candidate("duck + twist left", squat, {g1.WAIST_YAW: 1.0}),
             Candidate("duck + lean", squat, LEAN),
             Candidate("duck low (squat 0.2)", PlannerCommand(mode=4, height=0.2, facing_yaw=yaw)),
             Candidate("kneel on one leg", PlannerCommand(mode=6, height=0.3, facing_yaw=yaw)),
             Candidate("kneel", PlannerCommand(mode=5, height=0.3, facing_yaw=yaw))]
    sidesteps, walks, crouch = [], [], []
    for s, label in ((1, "left"), (-1, "right")):
        out = s * side
        out_yaw = float(np.arctan2(out[1], out[0]))
        for d in (0.5, 0.8):
            t = (x + out[0] * d, y + out[1] * d)
            sidesteps.append(Candidate(f"sidestep {label} {d:.1f}m",
                                       PlannerCommand(mode=base_mode, target_xy=t, target_yaw=yaw)))
        for d in (0.6, 0.9):
            t = (x + out[0] * d, y + out[1] * d)
            walks.append(Candidate(f"turn + walk {label} {d:.1f}m",
                                   PlannerCommand(mode=base_mode, target_xy=t, target_yaw=out_yaw)))
        t = (x + out[0] * 0.8, y + out[1] * 0.8)
        crouch.append(Candidate(f"crouch-walk {label} 0.8m",
                                PlannerCommand(mode=22, target_xy=t, target_yaw=out_yaw)))
    moves = walks + crouch + sidesteps if turn_walk_first else sidesteps + walks + crouch
    # Least effort first: upper-body moves need no steps, then ducks, then travel. For
    # throws below the chest, travelling escapes come before ducking.
    high = contact_z is not None and contact_z > 1.05
    c = upper + (ducks + moves if high else moves + ducks)
    back = (x + threat_dir_xy[0] * 0.8, y + threat_dir_xy[1] * 0.8)
    c.append(Candidate("back off", PlannerCommand(mode=base_mode, target_xy=back, target_yaw=yaw)))
    for s in (1, -1):
        c.append(Candidate(f"turn {'left' if s > 0 else 'right'}",
                           PlannerCommand(mode=0, facing_yaw=yaw + s * 1.4)))
    for cand in c:
        cand.cmd = fast(cand.cmd)
    return c


# RESCUE objective: run down a straight escape route to the exit while defenses fire.
ROUTE_LENGTH = 40.0   # m from the start to the exit
RUN_MODE, RUN_VEL, SPRINT_VEL = 3, 2.0, 2.5  # planner RUN mode accepts 1.5-3.0 m/s
MAX_BODY_HITS = 3     # a robot hit this many times is too damaged to make it out


def unit(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)])


def rescue_candidates(route_yaw):
    """Reactions while running: change speed or line, crouch, or stop; tried in this order."""
    run = lambda yaw, vel=RUN_VEL, mode=RUN_MODE: PlannerCommand(
        mode=mode, target_vel=vel, move_dir=tuple(unit(yaw)), facing_yaw=yaw)
    c = [Candidate("sprint", run(route_yaw, SPRINT_VEL)),
         Candidate("slow down", run(route_yaw, 1.5))]
    for deg in (35, 70):
        for s, label in ((1, "left"), (-1, "right")):
            c.append(Candidate(f"swerve {label} {deg}", run(route_yaw + s * np.radians(deg))))
    c += [Candidate("crouch-run", PlannerCommand(mode=g1.MODES["crouch_walk"], move_dir=tuple(unit(route_yaw)),
                                                 facing_yaw=route_yaw)),
          Candidate("stop", PlannerCommand(mode=0, facing_yaw=route_yaw)),
          Candidate("duck (squat 0.3)", PlannerCommand(mode=4, height=0.3, facing_yaw=route_yaw))]
    for cand in c:
        # Quick reactions (shortest horizon) but no faster playback: on SONIC, a 1.3x sprint
        # or swerve at running speed made the robot fall in every run.
        cand.cmd = replace(cand.cmd, horizon_tokens=REACT_TOKENS)
    return c


def fast(cmd: PlannerCommand) -> PlannerCommand:
    """Reaction timing: shortest planner horizon, faster playback."""
    return replace(cmd, horizon_tokens=REACT_TOKENS, speed=REACT_SPEED)


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
    style: str = "lob"
    done: bool = False
    actual_body: str = "miss"
    actual_t: float | None = None
    stuck: bool = False

    @property
    def outcome(self):
        if self.actual_body == "reset":
            return "reset"
        if self.actual_body in HAND_BODIES:
            return "swatted"
        if self.actual_body in ("miss", "floor", "world"):
            return "dodged" if self.pred_body not in (None, "floor") else "miss"
        if self.actual_body in STICKY:
            return "splat" if self.style.startswith("tomato") else "struck"
        return "bounced"


@dataclass
class Game:
    runner: object
    auto: bool = True
    interval: float = 3.5
    seed: int = 0
    log_name: str = "throws.csv"
    objective: str = "dodge"          # "dodge", "hit" (meet the ball with a hand) or "rescue" (run the escape route)
    balls: tuple = tuple(BALLS)
    rng: np.random.Generator = field(init=False)
    state: str = field(default="idle", init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.predictor = ContactPredictor(self.runner.arena.m)
        bat = self.runner.arena.bat
        bat_sides = {"right": ["right"], "left": ["left"], "both": ["left", "right"]}.get(bat or "", [])
        # With a bat, aim its middle at the tomato and start swinging from further out.
        from .arena import BAT_LENGTH, BAT_START
        self.ik = ArmIK(self.runner.arena.m,
                        {s: (BAT_START + 0.6 * BAT_LENGTH, 0.0, 0.0) for s in bat_sides})
        self.reach_radius = 0.55 + (0.5 if bat_sides else 0.0)
        self.tracker = TrajectoryTracker(noise_std=0.005, rng=self.rng)
        self.records: list[ThrowRecord] = []
        self.next_throw = self.runner.time + 2.0
        self.path = None
        self.contact = None
        self.rest_cmd = None
        self.last_eval = -1.0
        self.home_yaw = root_yaw(self.runner.robot_qpos())
        self.home_xy = self.runner.robot_qpos()[:2].copy()
        self.g = GRAVITY
        self.horizon = 2.5
        if self.objective == "hit":
            self.runner.cmd = PlannerCommand(mode=g1.MODES["idle_boxing"], facing_yaw=self.home_yaw)
        self.rescued = self.lost = False
        self.body_hits = 0
        self.t_start = self.runner.time
        self.exit_xy = self.home_xy + ROUTE_LENGTH * unit(self.home_yaw)
        self._last_steer = -1.0
        self.run_from = self.runner.time + 1.5  # stand for 1.5 s first: SONIC can fall going straight from a reset into a run
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / self.log_name

    # --- throw lifecycle ----------------------------------------------------------
    def throw(self):
        r = self.runner
        style = str(self.rng.choice(self.balls))
        ball = BALLS[style]
        if self.objective == "rescue":
            p0, v0, self.g = self._rescue_throw(style)
        else:
            p0, v0, self.g = random_throw(r.robot_qpos(), self.rng, style)
        r.arena.set_projectile(ball["radius"], ball["mass"], ball["rgba"], ball["sticky"])
        r.arena.launch(p0, v0)
        r.arena.set_tomato_gravity(float(ball["gscale"]))
        self.predictor.set_radius(ball["radius"])
        self.horizon = ball["flight"][1] + 1.0
        self.tracker.reset()
        self.path, self.contact = None, None
        self.records.append(ThrowRecord(len(self.records) + 1, r.time, style=style))
        self.rest_cmd = r.cmd
        self.state = "tracking"

    def step(self):
        """Call once per control tick (after runner.tick())."""
        r = self.runner
        a = r.arena
        if self.objective == "rescue":
            if self.rescued or self.lost:
                return
            q = r.robot_qpos()
            if np.dot(q[:2] - self.home_xy, unit(self.home_yaw)) >= ROUTE_LENGTH:
                self.rescued = True
                r.cmd = PlannerCommand(mode=0, facing_yaw=self.home_yaw)
                print(f"RESCUED in {r.time - self.t_start:.1f}s with {self.body_hits} body hits")
                return
            if r.time < self.run_from:
                return
            if self.state == "idle" and r.time - self._last_steer > 0.5:
                r.cmd, self._last_steer = self._run_cmd(), r.time  # steer back toward the exit
        if self.state == "idle":
            if self.auto and r.time >= self.next_throw:
                self.throw()
            return
        rec = self.records[-1]
        t_since = r.time - rec.t_launch
        if t_since < 0:  # MuJoCo auto-reset after an instability: abandon this throw
            rec.actual_body = "reset"
            self._finish(rec)
            return
        if self.state in ("tracking", "reacting") and a.tomato_in_flight():
            p, _ = a.tomato_state()
            self.tracker.observe(r.time, p)
            first = self.state == "tracking"
            if self.tracker.ready(5) and (first or r.time - self.last_eval >= self.REEVAL):
                self._evaluate(rec, first)
        if a.first_hit is not None and rec.actual_t is None:
            rec.actual_body, rec.actual_t = a.first_hit[1], a.first_hit[0] - rec.t_launch
        if t_since > self.horizon or (a.first_hit is not None and t_since > (rec.actual_t or 0) + 1.0):
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
        p, v = self.tracker.estimate(self.g)
        q = r.robot_qpos()
        # Predict against where the robot is going (its reference track), not a frozen pose.
        contact, path = self.predictor.predict(q, p, v, robot_traj=self._robot_traj(r.track),
                                               traj_dt=g1.CONTROL_DT, g=self.g, horizon=self.horizon)
        self.path = path
        self.contact = contact.point if contact else None
        # A planned swat (hand first) is not a threat.
        threat = contact is not None and contact.body != "floor" and contact.body not in HAND_BODIES
        if first:
            rec.pred_body = contact.body if contact else None
            rec.pred_t = contact.t if contact else None
            self.state = "reacting"
            # A moving robot keeps going past the ~1 s reference horizon, so "no hit" is not
            # trustworthy for it: stop or dodge anyway.
            if self.objective != "rescue":
                threat |= r.cmd.target_xy is not None or np.linalg.norm(r.cmd.move_dir) > 1e-6
            if not threat:
                rec.action = "stay"
        if self.objective == "hit":
            # Anything but a hand touching it first is a reason to (re)plan, up to 2 re-plans.
            threat = (contact is None or contact.body not in HAND_BODIES) and (first or rec.replans < 2)
        if threat and (contact is None or contact.t > self.MIN_REACT):
            if self.objective == "hit":
                name, clears = self._select_hit(p, v, q, path)
            else:
                name, clears = self._select(p, v, q)
            rec.action = name if first or rec.action == "stay" else f"{rec.action} > {name}"
            rec.action_clears = clears
            rec.replans += 0 if first else 1
        rec.decide_ms += 1000 * (time.perf_counter() - t0)

    def _robot_traj(self, track):
        """Predicted robot motion for a track: physics mode adds drift and SONIC's lag."""
        r = self.runner
        traj = track.frames[track.cur:].copy()
        if not r.kinematic:
            traj[:, :2] += r.drift()
            lag = np.repeat(r.robot_qpos()[None], int(SONIC_LAG / g1.CONTROL_DT), axis=0)
            traj = np.concatenate([lag, traj])
        if self.objective == "rescue" and len(traj) >= 6:
            # Keep running at the plan's final velocity until the ball can no longer arrive.
            vel = (traj[-1, :2] - traj[-6, :2]) / (5 * g1.CONTROL_DT)
            n = int((self.horizon + 0.5) / g1.CONTROL_DT) - len(traj)
            if n > 0:
                ext = np.repeat(traj[-1:], n, axis=0)
                ext[:, :2] += vel * (np.arange(1, n + 1) * g1.CONTROL_DT)[:, None]
                traj = np.concatenate([traj, ext])
        return traj

    def _try(self, cand, p, v, q):
        r = self.runner
        plan30, gen = r.plan_now(cand.cmd)
        # Evaluate exactly what would be executed: the plan blended into the live track.
        trial = copy.deepcopy(r.track)
        trial.merge(plan30, gen, speed=cand.cmd.speed)
        if cand.overlay:
            apply_overlay(trial, cand.overlay, **(cand.overlay_timing or {}))
        hit, _ = self.predictor.predict(q, p, v, robot_traj=self._robot_traj(trial),
                                        traj_dt=g1.CONTROL_DT, g=self.g, horizon=self.horizon)
        return hit, trial

    def _select(self, p, v, q):
        """Try reactions (stop, upper-body moves, ducks, escapes); commit the first whose
        blended plan clears. If none clears, try swatting the tomato with a hand."""
        r = self.runner
        if self.objective == "rescue":
            route_yaw = float(np.arctan2(*(self.exit_xy - q[:2])[::-1]))
            for cand in rescue_candidates(route_yaw):
                hit, trial = self._try(cand, p, v, q)
                if hit is None or hit.body == "floor":
                    r.commit(cand.cmd, trial)
                    return cand.name, True
            return "keep running", False
        d = v[:2] / (np.linalg.norm(v[:2]) + 1e-9)
        stop = Candidate("stop", fast(PlannerCommand(mode=0, facing_yaw=root_yaw(q))))
        z = None if self.contact is None else float(self.contact[2])
        best = None
        for cand in [stop] + dodge_candidates(q, d, contact_z=z, turn_walk_first=not r.kinematic):
            hit, trial = self._try(cand, p, v, q)
            clears = hit is None or hit.body == "floor"
            if clears:
                r.commit(cand.cmd, trial)
                return cand.name, True
            if best is None or hit.t > best[0]:
                best = (hit.t, cand, trial)
        swat = self._swat(p, v, q, stop)
        if swat is not None:
            return swat
        _, cand, trial = best
        r.commit(cand.cmd, trial)
        return cand.name, False

    PUNCHES = (("left punch", 11, 0.42, "left"), ("right punch", 12, 0.48, "right"),
               ("left hook", 15, 0.52, "left"), ("right hook", 16, 0.52, "right"))
    # name, planner mode, s from mode switch to full reach, punching hand

    def _select_hit(self, p, v, q, path):
        """HIT objective: face the tomato in a boxing guard, then throw a kplanner punch or hook
        timed (by delaying the mode switch) so a hand meets it first. Falls back to an arm-IK
        swat, then to just holding the guard."""
        r = self.runner
        face = float(np.arctan2(-v[1], -v[0]))
        guard = Candidate("guard", PlannerCommand(mode=g1.MODES["idle_boxing"], facing_yaw=face))
        plan30, gen = r.plan_now(guard.cmd)
        guard_trial = copy.deepcopy(r.track)
        guard_trial.merge(plan30, gen)
        # When does the tomato come within punching range (0.55 m) of the robot?
        d_xy = np.linalg.norm(path[:, :2] - q[:2], axis=1)
        near = np.nonzero(d_xy < self.reach_radius)[0]
        t_reach = near[0] * 0.005 if len(near) else 1.0
        lag = SONIC_LAG if not r.kinematic else 0.0
        for name, mode, peak, side in self.PUNCHES:
            base_delay = t_reach - peak - lag
            for dd in (0.0, -0.08, 0.08):
                delay = base_delay + dd
                if delay < 0:
                    continue
                cmd = PlannerCommand(mode=mode, facing_yaw=face)
                trial = copy.deepcopy(guard_trial)
                plan30, gen = r.plan_on(trial, cmd, lookahead=2 + int(round(delay / g1.CONTROL_DT)))
                trial.merge(plan30, gen)
                for aimed in (False, True):
                    if aimed and not self._aim(trial, side, p, v, delay + lag):
                        continue
                    hit, _ = self.predictor.predict(q, p, v, robot_traj=self._robot_traj(trial),
                                                    traj_dt=g1.CONTROL_DT, g=self.g, horizon=self.horizon)
                    if hit is not None and hit.body in HAND_BODIES:
                        r.commit(cmd, trial)
                        return name + (" (aimed)" if aimed else ""), True
        swat = self._swat(p, v, q, guard)
        if swat is not None:
            return swat
        r.commit(guard.cmd, guard_trial)
        return "guard", False

    def _aim(self, trial, side, p, v, t_switch):
        """Aim a planned punch: at the punching hand's full extension, layer an arm-IK
        correction that moves the fist onto the tomato's predicted position at that moment.
        Edits `trial` in place; returns False if the correction is out of reach."""
        r = self.runner
        frames = trial.frames
        k0 = trial.cur + int(t_switch / g1.CONTROL_DT)
        if k0 >= len(frames) - 2:
            return False
        # Full extension = hand farthest along the robot's heading after the mode switch.
        best, kbest = -9.0, None
        for k in range(k0, len(frames)):
            f = frames[k]
            self.ik.d.qpos[:36] = f
            mujoco.mj_kinematics(self.ik.m, self.ik.d)
            yaw = root_yaw(f)
            fwd = np.dot(self.ik.hand_pos(side)[:2] - f[:2], (np.cos(yaw), np.sin(yaw)))
            if fwd > best:
                best, kbest = fwd, k
        t_peak = (kbest - trial.cur) * g1.CONTROL_DT
        if t_peak < 0.2:
            return False
        pose = frames[kbest].copy()
        if not r.kinematic:
            pose[:2] += r.drift()
        target = ballistic(p, v, t_peak + (SONIC_LAG if not r.kinematic else 0.0), g=self.g)
        ang, resid = self.ik.solve(pose, side, target)
        if resid > 0.05:
            return False
        offsets = {j: float(a - pose[7 + j]) for j, a in zip(ARM_JOINTS[side], ang)}
        if max(abs(o) for o in offsets.values()) > 1.5:
            return False
        ramp = min(0.25, t_peak - 0.05)
        apply_overlay(trial, offsets, start=t_peak - ramp, ramp_in=ramp, hold=0.12, ramp_out=0.3)
        return True

    def _swat(self, p, v, q, base):
        """Arm IK: meet the tomato with a hand shortly before it would reach the body."""
        r = self.runner
        hit, base_trial = self._try(base, p, v, q)
        if hit is None or hit.body == "floor":
            return None
        traj = self._robot_traj(base_trial)
        # Walk back along the predicted arc (latest first) to the first point a hand can reach.
        tried = 0
        for t_int in np.arange(hit.t - 0.03, max(0.2, hit.t - 0.5), -0.03):
            target = ballistic(p, v, t_int, g=self.g)
            pose = traj[min(int(t_int / g1.CONTROL_DT), len(traj) - 1)]
            for side in ("left", "right"):
                ang, resid = self.ik.solve(pose, side, target)
                if resid > 0.05:
                    continue
                joints = ARM_JOINTS[side]
                offsets = {j: float(a - pose[7 + j]) for j, a in zip(joints, ang)}
                ramp = min(0.3, t_int - 0.05)
                cand = Candidate(f"swat {side} hand", base.cmd, offsets)
                cand.overlay_timing = dict(start=t_int - ramp - (SONIC_LAG if not r.kinematic else 0.0),
                                           ramp_in=ramp, hold=0.15, ramp_out=0.4)
                h, trial = self._try(cand, p, v, q)
                if h is not None and h.body in HAND_BODIES:
                    r.commit(cand.cmd, trial)
                    return cand.name, True
                tried += 1
                if tried >= 4:
                    return None
        return None

    def _finish(self, rec):
        a = self.runner.arena
        rec.stuck = a.stuck
        rec.done = True
        print(f"throw {rec.n:3d} [{rec.style:8s}]: predicted {rec.pred_body or 'miss':>18s}"
              f"{'' if rec.pred_t is None else f' @{rec.pred_t:.2f}s'}  action={rec.action:<20s}"
              f" clears={rec.action_clears} replans={rec.replans} cpu={rec.decide_ms:.0f}ms  ->  {rec.outcome}"
              f" ({rec.actual_body})")
        self._log(rec)
        # Walk back to the home spot, facing the throwing side.
        if self.objective == "rescue":
            if rec.outcome in ("splat", "struck"):
                self.body_hits += 1
            if self.body_hits >= MAX_BODY_HITS or self.runner.fallen():
                self.lost = True
                print(f"LOST after {self.runner.time - self.t_start:.1f}s ({self.body_hits} body hits)")
            else:
                self.runner.cmd, self._last_steer = self._run_cmd(), self.runner.time
        else:
            mode = g1.MODES["walk_boxing"] if self.objective == "hit" else 1
            self.runner.cmd = PlannerCommand(mode=mode, target_xy=tuple(self.home_xy), target_yaw=self.home_yaw)
        if not a.stuck:
            a.park_tomato()
        self.state = "idle"
        self.next_throw = self.runner.time + self.interval

    def _run_cmd(self):
        """Run toward the exit from wherever the robot is."""
        to_exit = self.exit_xy - self.runner.robot_qpos()[:2]
        yaw = float(np.arctan2(to_exit[1], to_exit[0]))
        return PlannerCommand(mode=RUN_MODE, target_vel=RUN_VEL, move_dir=tuple(unit(yaw)), facing_yaw=yaw)

    def _rescue_throw(self, style):
        """A defense beside or ahead of the route leads its shot at where the robot will be."""
        st = BALLS[style]
        g = GRAVITY * st["gscale"]
        r = self.runner
        flight = self.rng.uniform(*st["flight"])
        # Where the robot will be at impact: its reference track, extrapolated at run speed.
        k = r.track.cur + int(flight / g1.CONTROL_DT)
        future = r.track.at(min(k, len(r.track.frames) - 1)).copy()
        if k >= len(r.track.frames):
            future[:2] += RUN_VEL * unit(self.home_yaw) * (k - len(r.track.frames) + 1) * g1.CONTROL_DT
        if not r.kinematic:
            future[:2] += r.drift()
        aim = future[:3] + np.array([0.0, 0.0, self.rng.uniform(-0.05, 0.35)])
        aim[2] = 0.79 + aim[2] - future[2]  # pelvis-to-chest height of a standing robot
        az = self.home_yaw + np.radians(self.rng.uniform(-75, 75))  # from ahead of or beside the route
        d = self.rng.uniform(*st["dist"])
        p0 = np.array([aim[0] + d * np.cos(az), aim[1] + d * np.sin(az), self.rng.uniform(*st["release"])])
        return p0, launch_velocity(p0, aim, flight, g=g), g

    def _log(self, rec):
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["n", "t_launch", "pred_body", "pred_t", "action", "plan_clears",
                            "decide_ms", "replans", "style", "actual_body", "actual_t", "stuck", "outcome"])
            w.writerow([rec.n, f"{rec.t_launch:.2f}", rec.pred_body, rec.pred_t and f"{rec.pred_t:.3f}",
                        rec.action, rec.action_clears, f"{rec.decide_ms:.0f}", rec.replans, rec.style, rec.actual_body,
                        rec.actual_t and f"{rec.actual_t:.3f}", rec.stuck, rec.outcome])

    POINTS = {"dodge": {"dodged": 1, "miss": 1, "swatted": 2, "bounced": 0, "splat": -1, "struck": -1, "reset": 0},
              "hit": {"dodged": 0, "miss": 0, "swatted": 2, "bounced": 0, "splat": -1, "struck": -1, "reset": 0}}

    def points(self):
        return sum(self.POINTS[self.objective][r.outcome] for r in self.records if r.done)

    def summary(self):
        from collections import Counter
        c = Counter(r.outcome for r in self.records if r.done)
        return dict(c)
