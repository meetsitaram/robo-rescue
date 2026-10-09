# Third-party notices

The code in this repository is licensed under the Apache License 2.0 (see `LICENSE`). It builds on
the third-party work below, which keeps its own license.

## Included in this repository

### Unitree G1 robot model: `assets/g1/`

`g1_29dof.xml` and `meshes/` are the Unitree G1 (29 DoF) robot description. They are copied from
NVIDIA's [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
(`gear_sonic/data/assets/robot_description`, Apache License 2.0). They originate from Unitree
Robotics' [unitree_ros](https://github.com/unitreerobotics/unitree_ros), BSD 3-Clause License:

> Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
> All rights reserved.

The full BSD 3-Clause text is in [`assets/g1/LICENSE`](assets/g1/LICENSE).

### Trailer: `docs/media/rescue_trailer.mp4`, `docs/media/trailer_finale.gif`

- X2 motions retargeted from the BONES-SEED dataset: Motion Data by Bones Studio,
  <https://bones.studio/>; subject to the
  [BONES-SEED Dataset License Agreement](https://bones.studio/info/seed-license).
- Music: "Transformers Uprising" by LumineWave (Pixabay Content License). It is included only as
  part of the trailer, not on its own.
- Backgrounds and some shots were generated with xAI Grok Imagine.

## Used but not included

### NVIDIA GEAR-SONIC models

The SONIC kinematic planner (`planner_sonic.onnx`) and the encoder/decoder policy
(`model_encoder.onnx`, `model_decoder.onnx`) are **not** distributed here. Download them from
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC) on Hugging Face. The model weights
are licensed by NVIDIA Corporation under the
[NVIDIA Open Model License](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/LICENSE);
the accompanying code is Apache License 2.0. The planner/policy interface in `rescue/planner.py` and
`rescue/sonic.py` follows the reference deployment in GR00T-WholeBodyControl.

### Python dependencies (installed with `pip`)

| Package | License |
|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache License 2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT |
| [NumPy](https://numpy.org) | BSD 3-Clause |
| [SciPy](https://scipy.org) | BSD 3-Clause |
| [Pillow](https://python-pillow.org) | MIT-CMU (HPND) |

## Motion data

This repository contains **no motion-capture data**. Every robot motion in the game is generated at
runtime by the SONIC kinematic planner and tracked by the SONIC policy (or played kinematically).

Related work by the same author uses motion clips from the **BONES-SEED** motion-capture dataset by
**Bones Studio** (for example the AgiBot X2 walk used in the Rescue trailer, via
[sonic-x2](https://github.com/meetsitaram/sonic-x2)). Where those appear:
Motion Data by Bones Studio, <https://bones.studio/>. Use of the underlying dataset is subject to the
[BONES-SEED Dataset License Agreement](https://bones.studio/info/seed-license); the raw dataset is
not redistributed.
