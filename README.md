# VisionRL: Camera-Based Visual Navigation with Deep Reinforcement Learning in Unity

A Unity 6.6 / ML-Agents project in which a robot learns, with **PPO**, to reach a randomly placed exit in an 8 × 8 room using **only one 84 × 84 RGB camera frame**. There are no vector observations, no ray sensors and no coordinates: the policy has to find the exit in the image.

The trained policy is exported to **ONNX** and runs inside Unity's own inference engine (Burst CPU), with no Python needed at run time.

## Quick start: just watch the trained robot (no training needed)

The model is **already trained** and included in this repo, so you do not need Python or any training to see it work.

1. Install **Unity Hub** and the editor version **6000.6.5f1** (Unity 6.6) with macOS or Windows support.
2. Clone this repo (`git clone https://github.com/khrishith/VisionRL.git`) and add the folder in Unity Hub (**Add > Add project from disk**).
3. Open it with 6000.6.5f1. The first import takes a few minutes while Unity rebuilds its caches; some warnings are normal.
4. Open `Assets/VisionRL/Scenes/EscapeRoomRandomExit_Demo.unity` and press **Play**.
5. The robot drives itself to the green exit (no keyboard needed), then respawns with a new random exit and start. Watch the Game view; the step counter and reward are shown at the top left.

Use `EscapeRoomRandomExit_Demo.unity` to watch (it runs at about 5 moves per second). `EscapeRoomRandomExit.unity` is the same scene at full speed (50 moves per second), used for training and testing.

## Results

Evaluation on the same sequence of 200 episodes (reset seed 9001, all 27 exits, sampled actions). Intervals are 95% Wilson score intervals.

| Policy | Success | 95% CI | Timeouts | Mean length |
|---|---|---|---|---|
| Uniform random | 11/200 (5.5%) | 3.1–9.6% | 94.5% | 47.1 |
| Best blind policy (ignores the image) | 33/200 (16.5%) | 12.0–22.3% | 83.5% | 43.7 |
| 50k-step checkpoint, camera | 145/200 (72.5%) | 65.9–78.2% | 27.5% | 30.5 |
| 50k-step checkpoint, blank camera | 15/200 (7.5%) | 4.6–12.0% | 92.5% | 47.2 |
| **Final model, camera** | **196/200 (98.0%)** | **95.0–99.2%** | 2.0% | 9.6 |
| Final model, blank camera | 22/200 (11.0%) | 7.4–16.1% | 89.0% | 46.7 |

Run through **Unity's inference engine** (batch Play Mode, no Python trainer):

| Condition | Episodes | Success |
|---|---|---|
| All exits, camera | 200 | 196 (98.0%) |
| Held-out exits (never seen in training), camera | 100 | 88 (88.0%) |
| All exits, camera blanked | 200 | 24 (12.0%) |

Blanking the camera drops success from 98% to about 11–12%, the level of image-blind policies, which supports that the policy relies on the image. One held-out exit, `(0,3)`, was not solved reliably (3 of 15).

## How it works

- **Environment:** 8 × 8 grid, four solid walls, 12 static obstacles, a blue low-poly robot and a green exit on a border cell. Exit and start are randomized every episode (27 possible exits; 21 used for training, 6 held out).
- **Observation:** one 84 × 84 RGB frame from a fixed top-down orthographic camera.
- **Actions:** 4 discrete moves (forward, backward, left, right), one cell per action. Blocked moves cost a step.
- **Reward:** −0.01 per action, −0.02 extra for a blocked move, +1 for reaching the exit, −1 on timeout (48 actions), plus potential-based shaping on shortest-path distance (scale 0.05, γ = 0.99). The reward was audited so that wandering or back-and-forth loops never pay.
- **Training:** Unity ML-Agents PPO (`Training/ppo-random-exit.yaml`), 200,000 steps, seed 42, CPU.

## Repository layout

```
Assets/VisionRL/
  Scripts/    RoomWorld.cs, RoomAgent.cs (environment + agent), GridWorld.cs, GridAgent.cs (Stage 1 baseline)
  Editor/     scene builders, automated checks and the Unity runtime validator
  Scenes/     GridWorld.unity            Stage 1 baseline (7x7 grid)
              EscapeRoom.unity           fixed-exit room
              EscapeRoomRandomExit.unity final randomized-exit room, final model assigned (full speed)
              EscapeRoomRandomExit_Demo.unity same scene at a watchable speed: open this one to view
  Models/     VisionRL_random_exit_02.onnx   final trained model
              VisionRL_escape_train_01.onnx  earlier fixed-exit model
  Prefabs/    RobotVisual.prefab
Packages/     Unity ML-Agents 4.0.0 (embedded, with a one-line Unity 6.6 compatibility patch)
Training/     PPO configs, evaluate.py, requirements.txt
Reports/      test reports, evaluation JSON, figures
Docs/         SETUP.md, and Docs/Report/ (IEEE-style paper: LaTeX and Word sources)
```

Not included in the repo (kept local because of size): `Library/`, `Builds/`, `Backups/`, `results/` (training logs and checkpoints), and the Python environment.

## Requirements

- **Unity 6.6 (6000.6.5f1)** with the macOS build module. Developed and tested on a MacBook Air (Apple M3), macOS.
- **Python 3.10.12** for training and evaluation (see "Training" below).
- The ML-Agents Unity package is already embedded in `Packages/com.unity.ml-agents`.

## Run the trained robot at full speed (no Python needed)

1. Open the project folder in Unity Hub with editor version **6000.6.5f1**. The first import takes a few minutes.
2. Open `Assets/VisionRL/Scenes/EscapeRoomRandomExit.unity` (or the `_Demo` scene for a watchable speed).
3. Select `Blue Agent` and check its Behavior Parameters: model `VisionRL_random_exit_02`, Default behavior, Burst device, Deterministic Inference **off**.
4. Press **Play**. The robot navigates to the exit by itself and respawns with a new exit and start after each escape.
5. To slow it down for viewing, set **Decision Period** on the `Grid Decision Requester` to 10 (set it back to 1 for training).

## Automated checks

In the Unity menu bar:

- `VisionRL > Run Stage 1 Play Mode Checks` (7×7 baseline)
- `VisionRL > Escape Room > Run Checks`
- `VisionRL > Random Exit > Run Checks` (mechanics, reward audit, exit randomization, camera frames)
- `VisionRL > Random Exit > Run Unity Runtime Validation` (runs the assigned model inside Unity)

## Training and evaluation (Python)

Create the Python environment as described in `Docs/SETUP.md` (conda or micromamba, Python 3.10.12, `grpcio` from conda-forge, then `pip install -r Training/requirements.txt`). Then build the player with `VisionRL > Random Exit > Build macOS Player` and run:

```bash
mlagents-learn Training/ppo-random-exit.yaml \
  --env Builds/EscapeRoomRandomExit.app \
  --run-id my-run --seed 42 \
  --env-args -batchmode -screen-fullscreen 0 -screen-width 256 -screen-height 256
```

Evaluate an exported model (use `--actions sampled`; the model's deterministic output is not the trained policy):

```bash
python Training/evaluate.py --env Builds/EscapeRoomRandomExit.app \
  --model Assets/VisionRL/Models/VisionRL_random_exit_02.onnx \
  --actions sampled --exit-set 0 --episodes 200 --seed 9001
```

`--exit-set 1` uses the training exits and `--exit-set 2` the six held-out exits. `--blank-camera` runs the camera-blanking control, and `--blind-probs p0,p1,p2,p3` runs a policy that ignores the image.

## Important: GPU Resident Drawer

`Assets/Settings/PC_RPAsset.asset` has `m_GPUResidentDrawerMode: 0` on purpose. With the drawer enabled, camera frames in built players lagged behind the true robot position and training could not learn. Keep it disabled, and verify new players by checking that the robot in the camera frame matches the real state before training (see the report, Section V).

## Report

The full IEEE-style paper (method, experiments, rendering-fault analysis, limitations) is in `Docs/Report/`: the LaTeX source `overleaf/main.tex` and the Word file `VisionRL_Report.docx`.

## Limitations

One training seed, one room layout (only the exit and start change), a fixed top-down camera on a small grid, and one held-out exit not solved reliably. Unity validation was done in batch Play Mode, not on a rebuilt standalone player. The conclusions apply to this controlled simulation, not to real-world robot navigation.

## License

MIT. See [LICENSE](LICENSE).
