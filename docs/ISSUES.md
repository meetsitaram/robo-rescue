# Issues log (phase 1)

Every problem hit while building phase 1, with its cause and fix, newest first.

| # | Area | Problem | Cause and fix |
| --- | --- | --- | --- |
| 29 | Renders | In close camera shots a ball appeared out of nowhere after each throw | It was the projectile parked between throws, resting on the floor 6 m to the side. It is now parked 50 m away and made invisible until the next launch |
| 28 | Physics | SONIC reacted too slowly: a duck took 0.94 s to get the pelvis below 0.6 m, an escape walk covered 0.09 m in 0.6 s | The planner picks a 0.8–1.47 s horizon and reaches the target pose at its end. Dodge reactions now force the shortest horizon (6 tokens) and play back 1.3× faster: duck 0.40 s, escape 0.27 m. Physics DODGE went from 19 to 20–21 avoided out of 28 |
| 27 | Physics | A headless run hung for an hour; MuJoCo logged a singular inertia at the ball's DOFs and NaN accelerations | Reshaping the one projectile body per throw (58 g tennis ball to 5 kg cannonball) left a tiny rotational inertia and stale derived constants (`dof_invweight0` etc.) from the compiled tomato. MuJoCo auto-reset the sim, which set the clock to 0, so the throw timeout never fired. Fix: inertia floor 2e-4 kg·m², `condim=3` instead of torsional friction, `mj_setConst` after every reshape, and a guard that abandons a throw if the clock jumps back |
| 26 | Game | "Straight" throws from far away are impossible with real gravity: a 1.5 s flight arcs about 2.8 m high | Game-style floater: the tomato's gravity is scaled per throw with MuJoCo `gravcomp` (8% g). The tracker and predictor use each throw's gravity |
| 25 | Game | HIT decisions took up to 13 s of CPU, freezing the interactive app | The predictor ran collision checks for the whole flight. It now skips steps while the ball is more than 1.8 m from the robot and above 0.3 m; decisions take 0.3–0.7 s |
| 24 | HIT | kplanner punches rarely connected | Punches reach full extension at about 1.1 m height while balls entered reach at 1.3–1.5 m. Fix: aim the planned punch with arm IK at full extension (an additive layer), timed by delaying the mode switch |
| 23 | HIT | Swats almost never found a reachable intercept | Intercepts were placed at fixed lead times; steep lobs are 1.6–1.8 m up at that point. Fix: scan back along the predicted arc for the first point a hand can reach |
| 22 | Review | Video captions counted the ball still in flight as "dodged" | Scores and points now count finished throws only |
| 21 | Physics | Robot reported as fallen right after a duck | The fall check used pelvis height only. It now uses torso tilt (more than 60°) or pelvis below 0.25 m |
| 20 | Physics | Forward-lean dodges were predicted to clear but splatted | SONIC holds waist pitch fixed (reference 0.67 rad, robot 0.01 rad). Lean is now whole-body: root pitched 0.45 rad with both hips flexed by the same angle |
| 19 | Physics | Sidestep dodges executed too slowly on SONIC | SONIC tracks forward walking far better than lateral steps. Added "turn and walk out" escapes (heading along the escape direction), tried before sidesteps on SONIC |
| 18 | Physics | Planned dodges cleared but the physical robot was still hit | SONIC trails its reference. The check now holds the robot's current pose for 0.12 s before following the plan |
| 17 | Eval | Physics scores varied run to run with the same seed | Multithreaded ONNX Runtime float sums differ slightly and contact physics amplifies them. Report ranges over several runs |
| 16 | App | Interactive viewer froze on the first frame | `viewer.set_texts()` takes the viewer lock itself (not re-entrant). Call it outside `with viewer.lock()` |
| 15 | Dodge | "Stop" predicted to clear but the tomato splatted | 5 noisy samples gave about 0.15 m/s velocity error, about 19 cm at impact. Re-fit and re-predict every 0.1 s in flight and re-plan if the committed motion is predicted to be hit |
| 14 | Dodge | Executed motion differed from the candidate that was checked | The runner replanned after the choice. Each candidate is evaluated on a copy of the track with the plan already blended in, and exactly that track is committed |
| 13 | Dodge | Throws predicted to land on the floor hit a robot walking home | A moving robot keeps going past the ~1 s reference horizon. Predict against the planned track, and check a "stop" candidate first when moving |
| 12 | Predict | Grazing head hits predicted as misses | MuJoCo's semi-implicit Euler drops the ball 0.5·g·dt·t below the true parabola (2.7 cm after 1.1 s). Use x(t) = p0 + v0·t + ½g·t(t+dt) and a 5 cm safety margin |
| 11 | Target | Physics robot stopped about 0.2 m short of targets | Near the goal the planner closes the last 0.3 m with a sliding root creep that SONIC cannot reproduce. Carrot target at least 0.6 m ahead, idle within 6 cm with hysteresis |
| 10 | Target | Physics robot drifted from the planner's world position | SONIC tracks relative motion. Before each replan the reference xy is shifted onto the robot; the encoder never sees root xy |
| 9 | Policy | Decoder input is 994 dims, not the 436 in `observation_config.yaml` | Stale yaml comment; layout taken from the C++ deploy code |
| 8 | Setup | Upstream G1 model files not on disk (sparse checkout) | Read with `git show`; vendored the 29-DoF MJCF and 35 meshes |
| 7 | Contact | A bouncing tomato left at 31 m/s and flew 14 m up | `solref` damping ratio 0.15 makes contact stiffness explode. `solref="0.02 0.5"`: restitution about 0.6 |
| 6 | Kinematic | Resetting a kinematic robot every substep can pump energy into the ball | Robot masses ×10⁴ in kinematic mode |
| 5 | Setup | `MjSpec` could not find meshes under `<include>` | Rewrite each `mesh.file` to an absolute path |
| 4 | Setup | `MjsJoint.damping = 0.05` raises on MuJoCo 3.15 | Damping is a 3-vector: `[0.05, 0, 0]` |
| 3 | Setup | G1 hands have no collision geoms | Add a 4.5 cm sphere per hand at load time |
| 2 | Setup | onnxruntime-gpu CUDA provider needs CUDA 13 cuBLAS | Run on CPU: planner about 28 ms per call |
| 1 | Setup | No `python` on PATH | `run.ps1` calls Python 3.12 by full path |
