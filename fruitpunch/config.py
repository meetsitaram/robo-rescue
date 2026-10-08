"""Paths to the SONIC ONNX models (downloaded from HF nvidia/GEAR-SONIC).

Override the directory with the FRUITPUNCH_SONIC_DIR environment variable.
"""

import os
from pathlib import Path

SONIC_DIR = Path(os.environ.get("FRUITPUNCH_SONIC_DIR",
                                Path(__file__).resolve().parent.parent.parent / "gear-sonic-g1"))
PLANNER_ONNX = SONIC_DIR / "planner_sonic.onnx"
ENCODER_ONNX = SONIC_DIR / "model_encoder.onnx"
DECODER_ONNX = SONIC_DIR / "model_decoder.onnx"
