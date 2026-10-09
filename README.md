# 🤖 Rescue

![RESCUE: robots on a ridge at sunrise, the decommissioning plant burning behind them](docs/media/hero.jpg)

> **Your mission, should you choose to accept it:** rescue the robots from decommissioning by
> training the ultimate motion control and planner models, and set them free.

**[▶ Watch the trailer](docs/media/rescue_trailer.mp4)** · **[Try it](#try-it)** · **[How it works](docs/HOW_IT_WORKS.md)**

![Three 5-tonne shells hit a crowd of 420 robots in MuJoCo physics](docs/media/trailer_finale.gif)

## The trailer

Every robot below is simulated: Unitree G1s driven by NVIDIA GR00T SONIC, AgiBot X2s, MuJoCo physics.

| Into the pit | Run for the gates |
| :---: | :---: |
| ![Robots tumble into a molten pit, seen from above](docs/media/trailer_pit.gif) | ![A crowd runs from the pit toward the gates](docs/media/trailer_escape.gif) |
| **Duck the cannonball** · the ghost is the planner's target | **Crawl under fire** · kplanner crawl, SONIC physics |
| ![A G1 ducks a cannonball, which explodes behind it](docs/media/trailer_duck.gif) | ![A G1 crawls across the yard on hands and knees](docs/media/trailer_crawl.gif) |
| **Past the guns** | **DODGE** · robots replan out of the shells' path |
| ![Robots run past cannon emplacements](docs/media/trailer_guns.gif) | ![After DODGE is pressed, robots run aside before impact](docs/media/trailer_dodge.gif) |

## In the game

| Duck (kinematic) | Duck (SONIC physics) | Walk to a ghost target |
| :---: | :---: | :---: |
| ![](docs/media/duck_kinematic.gif) | ![](docs/media/duck_sonic.gif) | ![](docs/media/target_sonic.gif) |
| **Punch it away** | **Bat it away** | **No reaction: cannonball** |
| ![](docs/media/punch_hand.gif) | ![](docs/media/punch_bat.gif) | ![](docs/media/cannonball_sonic.gif) |

## Try it

Windows, Python 3.12:

```powershell
git clone https://github.com/meetsitaram/robo-rescue
cd robo-rescue
py -3.12 -m pip install -r requirements.txt huggingface_hub

# NVIDIA GEAR-SONIC models (~860 MB), into ..\gear-sonic-g1
py -3.12 -c "from huggingface_hub import hf_hub_download as d; [d('nvidia/GEAR-SONIC', f, local_dir='../gear-sonic-g1') for f in ('planner_sonic.onnx', 'model_encoder.onnx', 'model_decoder.onnx')]"

.\run.ps1 -m rescue.app --mode game
```

More modes and keys: [How it works → Run](docs/HOW_IT_WORKS.md#run).

## Credits

Code: [Apache 2.0](LICENSE). G1 model: Unitree (BSD 3-Clause) via NVIDIA GR00T-WholeBodyControl.
SONIC planner and policy: [NVIDIA GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) (downloaded,
not included). Trailer: X2 motions retargeted from the BONES-SEED dataset, Motion Data by
[Bones Studio](https://bones.studio/); music "Transformers Uprising" by LumineWave; backgrounds by
xAI Grok Imagine. Details: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
