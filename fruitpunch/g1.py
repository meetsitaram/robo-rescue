"""Unitree G1 29-DoF constants for SONIC, ported from GR00T-WholeBodyControl
(gear_sonic_deploy/src/g1/.../policy_parameters.hpp). Arrays are in MuJoCo
joint order unless named *_il (IsaacLab order)."""

import numpy as np

NUM_JOINTS = 29

# MUJOCO_OF_IL[il] -> mujoco index;  IL_OF_MUJOCO[mj] -> isaaclab index
MUJOCO_OF_IL = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
                         24, 18, 25, 19, 26, 20, 27, 21, 28])
IL_OF_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23,
                         25, 27, 12, 16, 20, 22, 24, 26, 28])

DEFAULT_ANGLES = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,          # left leg
    -0.312, 0, 0, 0.669, -0.363, 0,          # right leg
    0, 0, 0,                                 # waist yaw/roll/pitch
    0.2, 0.2, 0, 0.6, 0, 0, 0,               # left arm
    0.2, -0.2, 0, 0.6, 0, 0, 0,              # right arm
])

# Motor models: armature, effort limit. stiffness = arm * w^2, damping = 4 * arm * w
_W = 2 * np.pi * 10.0
_MOTORS = {
    "5020": (0.003609725, 25.0),
    "7520_14": (0.010177520, 88.0),
    "7520_22": (0.025101925, 139.0),
    "4010": (0.00425, 5.0),
}
# (motor, kp/kd multiplier) per joint, MuJoCo order
_LEG = [("7520_22", 1), ("7520_22", 1), ("7520_14", 1), ("7520_22", 1), ("5020", 2), ("5020", 2)]
_ARM = [("5020", 1)] * 5 + [("4010", 1)] * 2
_JOINT_MOTORS = _LEG + _LEG + [("7520_14", 1), ("5020", 2), ("5020", 2)] + _ARM + _ARM

_arm = np.array([_MOTORS[m][0] for m, _ in _JOINT_MOTORS])
_mult = np.array([k for _, k in _JOINT_MOTORS], float)
EFFORT = np.array([_MOTORS[m][1] for m, _ in _JOINT_MOTORS])
KP = _arm * _W**2 * _mult
KD = 4 * _arm * _W * _mult
ACTION_SCALE = 0.25 * EFFORT / (_arm * _W**2)  # uses 1x stiffness even for doubled joints

# Simulated joint torque limits (actuatorfrcrange in the upstream sim model).
TORQUE_LIMIT = np.array([88, 88, 88, 139, 50, 50] * 2 + [88, 50, 50] + [25, 25, 25, 25, 25, 5, 5] * 2,
                        float)

CONTROL_DT = 0.02    # 50 Hz policy
SIM_DT = 0.005       # 200 Hz physics
PLANNER_FPS = 30.0
PLANNER_CONTEXT_Z = 0.788740

# Kinematic planner modes (planner_sonic.onnx V2)
MODES = {
    "idle": 0, "slow_walk": 1, "walk": 2, "run": 3, "squat": 4, "kneel_two": 5, "kneel": 6,
    "lying": 7, "crawl": 8, "idle_boxing": 9, "walk_boxing": 10, "left_punch": 11,
    "right_punch": 12, "random_punch": 13, "elbow_crawl": 14, "left_hook": 15, "right_hook": 16,
    "jump": 17, "stealth_walk": 18, "injured_walk": 19, "careful_walk": 20, "carry": 21,
    "crouch_walk": 22, "happy_walk": 23, "zombie_walk": 24, "gun_walk": 25, "scare_walk": 26,
}
STATIC_MODES = {0, 4, 5, 6, 7, 9}
HEIGHT_MODES = {4, 5, 6}  # use the `height` input (0.2-0.8)
