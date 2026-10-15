<div align="center">

# VisionRL

**A robot that finds the exit using only its camera**

Camera-based navigation in Unity, trained with reinforcement learning (PPO)

<br/>

<img src="https://img.shields.io/badge/Unity-6000.6.5f1-000000?style=for-the-badge&logo=unity&logoColor=white" />
<img src="https://img.shields.io/badge/ML--Agents-4.0.0-00A67E?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />

<br/><br/>

[Watch the robot](#watch-the-trained-robot) &nbsp;|&nbsp;
[How it works](#how-it-works) &nbsp;|&nbsp;
[Results](#results) &nbsp;|&nbsp;
[Train it yourself](#train-it-yourself) &nbsp;|&nbsp;
[Project structure](#project-structure)

</div>

<br/>

---

## About

This project is a small room in Unity with a robot, twelve obstacles and a green exit. The robot has to reach the exit, but the only thing it can see is a single 84 x 84 camera image. It is not given any coordinates or sensor readings.

The robot is trained with PPO using Unity ML-Agents. The exit moves to a different spot every episode, so the robot can't memorize a path. It has to find the exit in the image each time.

The trained model is already included in the repo, so you can watch it run without training anything.

<br/>

## Watch the trained robot

You need Unity Hub with editor version **6000.6.5f1** (the same version the project was made in).

**1.** Clone the repo

```bash
git clone https://github.com/khrishith/VisionRL.git
```

**2.** In Unity Hub, click Add, then "Add project from disk", and pick the `VisionRL` folder.

**3.** Open the project with 6000.6.5f1. The first time takes a few minutes while Unity imports everything.

**4.** Open the scene `Assets/VisionRL/Scenes/EscapeRoomRandomExit_Demo.unity`

**5.** Press Play

The robot goes to the green exit on its own, then restarts with a new exit and a new start position. You don't need to press anything.

| Scene | What it is |
|:--|:--|
| `EscapeRoomRandomExit_Demo.unity` | **Open this one to watch.** It runs at a slower speed. |
| `EscapeRoomRandomExit.unity` | The same scene at full speed, used for testing and training. |
| `EscapeRoom.unity` | An earlier version where the exit never moves. |
| `GridWorld.unity` | The first 7 x 7 test grid. |

<br/>

## How it works

```
   camera image          neural network           one move              reward
  ┌────────────┐        ┌───────────────┐       ┌───────────┐       ┌─────────────┐
  │  84 x 84   │ ─────▶ │    decides    │ ────▶ │  forward  │ ────▶ │  PPO learns │
  │    RGB     │        │   the move    │       │  back     │       │  from it    │
  └────────────┘        └───────────────┘       │  left     │       └─────────────┘
                                                │  right    │
                                                └───────────┘
```

The robot gets one top-down camera image each step and chooses one of four moves: forward, backward, left or right. Each move takes it one cell. Walls and obstacles block it.

**Reward.** A small penalty for every move, a bigger penalty for walking into something, and +1 for reaching the exit. There is also a small bonus for getting closer to the exit, which helps it learn faster. An episode ends when the robot reaches the exit or after 48 moves.

**Exits.** There are 27 possible exit positions on the edge of the room. We trained on 21 of them and kept 6 back, to see if the robot could handle exits it had never seen.

<br/>

## Results

We ran each policy on the same 200 episodes.

| Policy | Reached the exit | Average steps |
|:--|--:|--:|
| Random moves | 5.5% | 47.1 |
| Best policy that ignores the image | 16.5% | 43.7 |
| **Our model** | **98.0%** (196 of 200) | **9.6** |
| Our model with the camera blanked | 11.0% | 46.7 |

With the camera on, the robot reaches the exit about 98% of the time. When we blank the camera it drops to 11%, which is close to what a policy that ignores the image gets. So the robot really is using what it sees.

We also ran the model inside Unity itself (not through Python). It reached the exit in 196 of 200 episodes, and in 88 of 100 episodes where the exit was one of the 6 positions it was never trained on. One of those 6 positions, at (0,3), was hard for it: it only got 3 of 15 there.

<br/>

## Train it yourself

> You only need this if you want to retrain or re-run the tests. It needs Python 3.10.12, and building the player needs a Mac.

<details>
<summary><b>1. Set up Python</b></summary>

<br/>

`Docs/SETUP.md` has the steps: a conda or micromamba environment with Python 3.10.12, `grpcio` from conda-forge, then

```bash
pip install -r Training/requirements.txt
```

</details>

<details>
<summary><b>2. Build the player</b></summary>

<br/>

In Unity, go to `VisionRL > Random Exit > Build macOS Player`. This makes `Builds/EscapeRoomRandomExit.app`.

</details>

<details>
<summary><b>3. Train</b></summary>

<br/>

```bash
mlagents-learn Training/ppo-random-exit.yaml \
  --env Builds/EscapeRoomRandomExit.app \
  --run-id my-run --seed 42 \
  --env-args -batchmode -screen-fullscreen 0 -screen-width 256 -screen-height 256
```

The results, including the exported `.onnx` model, go into `results/my-run/`.

The settings we used for the final run are in `Training/ppo-random-exit.yaml`: 200,000 steps, batch size 128, buffer size 2048, learning rate 3e-4 and a discount of 0.99.

</details>

<details>
<summary><b>4. Test a model</b></summary>

<br/>

```bash
python Training/evaluate.py --env Builds/EscapeRoomRandomExit.app \
  --model Assets/VisionRL/Models/VisionRL_random_exit_02.onnx \
  --actions sampled --exit-set 0 --episodes 200 --seed 9001
```

| Option | What it does |
|:--|:--|
| `--exit-set 2` | Tests only the 6 held-out exits |
| `--blank-camera` | Blanks the camera |
| `--blind-probs a,b,c,d` | Runs a policy that ignores the image |

</details>

<br/>

## Project structure

```
VisionRL/
├── Assets/VisionRL/
│   ├── Scripts/     the environment and the agent (RoomWorld.cs, RoomAgent.cs)
│   ├── Editor/      scene builders and automated checks
│   ├── Scenes/      the Unity scenes
│   ├── Models/      the trained ONNX models
│   └── Prefabs/     the robot
├── Packages/        Unity ML-Agents (included in the repo)
├── Training/        PPO settings, evaluate.py, requirements.txt
├── Reports/         test and evaluation results
└── Docs/            setup notes and the report files
```

There are also tests you can run from the Unity menu bar: `VisionRL > Random Exit > Run Checks`, and `Run Unity Runtime Validation` to run the trained model inside Unity.

<br/>

## Things to know

- **Keep the GPU Resident Drawer turned off.** In `Assets/Settings/PC_RPAsset.asset` it is set to `m_GPUResidentDrawerMode: 0`. When it was on, the camera image lagged behind the robot's real position and the training couldn't learn.
- The training logs and checkpoints are not in the repo because they are big. The final trained model is included.
- Building the player only works on macOS, but you can watch the model in the Unity Editor on any system that runs Unity 6.6.

<br/>

---

## Team Members: 
### Anush Bundel - 2023BCS0005
### K. Hrishith - 2023BCS0168
### Chaitanya Raste - 2023BCS0205

---

<div align="center">

Built with Unity 6.6, Unity ML-Agents 4.0.0, PyTorch 2.2.2, ONNX and Python 3.10

MIT License. See [LICENSE](LICENSE).

</div>
