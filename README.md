# FruitPunch

Robots dodge (and later punch and kick) fruit thrown at them. Phase 1 is a MuJoCo pipeline for the
Unitree G1. NVIDIA's GR00T SONIC kinematic planner generates the motion, and the SONIC whole-body
policy optionally tracks it in physics. A tomato is lobbed at the robot; its trajectory is tracked
and the contact predicted, and the robot picks a dodge that the planner verifies clears the path.

## Setup

```powershell
py -3.12 -m pip install -r requirements.txt
```

The SONIC ONNX models (`planner_sonic.onnx`, `model_encoder.onnx`, `model_decoder.onnx`) come from
the Hugging Face repo [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC). By default
they are read from `../gear-sonic-g1/`; override this with `FRUITPUNCH_SONIC_DIR`.

## Run

```powershell
.\run.ps1 -m fruitpunch.app --robot kinematic --mode drive   # keyboard driving
.\run.ps1 -m fruitpunch.app --robot physics --mode target    # ghost target pose
.\run.ps1 -m fruitpunch.app --robot physics --mode game      # throws + dodging
.\run.ps1 scripts\headless_game.py --robot kinematic --throws 20
```

`--robot kinematic` writes the planner's motion straight into the robot, and only the tomato is
simulated. `--robot physics` runs SONIC (encoder + decoder at 50 Hz, PD at 200 Hz).

| Key | DRIVE | TARGET | GAME |
| --- | --- | --- | --- |
| Delete | next mode | next mode | next mode |
| Insert | throw a tomato | throw a tomato | throw a tomato |
| Home | next gait | next gait | |
| Up / Down | walk forward / back | move ghost ±x 25 cm | |
| Left / Right | turn ±30° | move ghost ±y 25 cm | |
| PgUp / PgDn | strafe | rotate ghost ±30° | |
| End | squat on/off | ghost squat on/off | |
| Enter | | go to ghost | |

## Layout

| File | What |
| --- | --- |
| `fruitpunch/g1.py` | G1 constants: joint maps, default pose, gains, planner modes |
| `fruitpunch/planner.py` | `planner_sonic.onnx` wrapper, 30→50 Hz resampling, plan blending, replan policy |
| `fruitpunch/sonic.py` | SONIC encoder/decoder observation assembly (1762 / 994 dims) |
| `fruitpunch/arena.py` | MuJoCo scene build, kinematic vs physics robot, sticky tomato weld |
| `fruitpunch/runner.py` | 50 Hz tick, physics-mode target following |
| `fruitpunch/projectile.py` | launch solver, Euler-exact ballistic path, contact prediction |
| `fruitpunch/game.py` | thrower, trajectory tracker, dodge selection, scoring and CSV log |
| `fruitpunch/overlay.py` | ghost robot, predicted arc, contact marker |
| `fruitpunch/app.py` | interactive viewer |
| `scripts/` | headless smoke tests, headless game, offscreen renders |

The robot model in `assets/g1` comes from NVIDIA GR00T-WholeBodyControl (Apache-2.0); the meshes
are Unitree's.
