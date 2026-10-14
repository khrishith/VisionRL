# VisionRL prototype

## Requirements and inspection — 13 October 2026

Deadline: 14 October 2026 at night (Asia/Kolkata).

Original project: Unity 6000.6.5f1, URP 17.6.0, Input System 1.20.0,
SampleScene and template tutorial scripts only. No prior project requirement files
or AGENTS.md instructions found. Git 2.53.0 installed, project not initialized as a
repository. ARM64 host. Default Python 3.14.3 and Homebrew Python 3.11 found;
Conda not found on PATH or checked standard installation paths.

Use a native ARM64, project-local Python 3.10.12 environment. Do not install the
trainer into system Python. `.tools/bin/micromamba` manages `.venv-mlagents` without
changing shell startup files. Environment paths are local and not portable.

## Compatibility findings

Official ML-Agents Release 23 documents Unity >=6000.0 and Python 3.10.1–3.10.12.
The published pair is Unity 4.0.0 / Python 1.1.0. Direct testing found:

1. PyPI 1.1.0 requests grpcio <=1.48.2 and attempts a failing source build.
2. Official `release_23` source at commit
   `a2777719560e4676be99e2ea128c5eb1fbeb3dbb` accepts grpcio <=1.53.2. It internally
   labels both Python packages `1.2.0.dev0`. This is the installed, pinned source.
3. The pip grpcio 1.53.2 ARM64/universal wheel failed with missing `_CFRelease`.
   Conda-forge grpcio 1.51.1 imported successfully. Install it before pip.
   `pip check` reports its macOS `aarch64` wheel metadata as unsupported (pip
   expects `arm64`); retain this warning and use native import plus live connection
   tests to distinguish metadata from runtime failures.
4. Unity 6.6 rejects `Object.GetInstanceID()` in ML-Agents' Match3 helper. The
   official 4.0.0 package is embedded in `Packages/com.unity.ml-agents`, with one
   source change: `gameObject.GetEntityId().GetHashCode()` provides the integer
   random seed instead. Package licensing files are preserved. Do not edit Library
   caches; the embedded package takes precedence over the registry reference.

Sources:
- https://github.com/Unity-Technologies/ml-agents
- https://github.com/Unity-Technologies/ml-agents/blob/release_23/docs/Installation.md
- https://github.com/Unity-Technologies/ml-agents/blob/release_23/ml-agents/setup.py
- https://github.com/Unity-Technologies/ml-agents/blob/release_23/com.unity.ml-agents/package.json

These findings are specific to this machine and editor. CPU training is the tested
path. MPS acceleration is not assumed. Camera rendering requires graphics; do not
use `--no-graphics` or Unity `-nographics`. The Air's sustained throughput and visual
PPO convergence remain practical risks. A training run completing is not evidence
that a policy learned; use the evaluation reports.

## Environment contract

- 49 fixed cells; blue cube agent; green target; distinct uniformly random cells.
- One fixed overhead orthographic CameraSensor, 84×84 RGB, one frame, PNG transport.
  The installed release exposes Python tensors in CHW layout `(3,84,84)`.
- No vector observations, observable attributes, coordinate sensors, action masks,
  or coordinate-based policy inputs. Python evaluation asserts the live contract.
- One discrete branch with four actions: 0 +Z (forward), 1 -Z (backward),
  2 -X (left), 3 +X (right). One cell per action; wall attempts stay in place.
- Episodes end on target contact or after exactly 64 actions.
- Reward: -0.01 per action, -0.02 additional for a wall attempt, +1 success,
  -1 timeout. Potential shaping uses Phi(s)=-0.1*ManhattanDistance and
  0.99*Phi(next)-Phi(current); terminal Phi is zero. Gamma matches PPO config.
  Positions enter this private mechanics/reward calculation only.
- Success terminal reward is 1.09; timeout terminal reward is <=0.19. Evaluation
  uses this terminal distinction, not cumulative shaped return, to measure success.
- Manual fallback uses WASD/arrows; absent a direction it chooses forward because
  this action space has no no-op. Training supplies all actions through Python.

## Recreate Python environment

After downloading official ARM64 micromamba into `.tools/bin/micromamba`:

```sh
.tools/bin/micromamba create -y -r "$PWD/.tools/mamba" -p "$PWD/.venv-mlagents" -c conda-forge python=3.10.12 pip grpcio=1.51.1
.venv-mlagents/bin/python -m pip install -r Training/requirements.txt
.venv-mlagents/bin/python -m pip check
.venv-mlagents/bin/mlagents-learn --help
```

`Training/installed-lock.txt` records the installed Python packages. The explicit
source commit in requirements avoids silently following a changing branch.

## Staged workflow

1. In Unity, use **VisionRL > Create Grid Scene**. The builder creates a separate
   scene in `Assets/VisionRL/Scenes`, preserves SampleScene, and opens the existing
   grid scene on subsequent invocations. Save unsaved work when prompted.
2. Use **VisionRL > Validate Environment**, then **Validate and Build macOS**.
   The build uses only the grid scene without replacing existing build settings.
   Output: `Builds/VisionRL.app`. Mechanics tests cover every cell/action, resets,
   success, timeout, and scene sensor/action settings.
3. Verify runtime pixels and random baseline:

```sh
.venv-mlagents/bin/python Training/evaluate.py --env Builds/VisionRL.app --episodes 200 --seed 9001 --output Reports/random.json
```

4. Train the official PPO visual encoder on CPU:

```sh
OMP_NUM_THREADS=4 .venv-mlagents/bin/mlagents-learn Training/ppo.yaml --env Builds/VisionRL.app --run-id visionrl-ppo-01 --seed 42
```

Use a new run ID for a new experiment. To continue the same run, append `--resume`.
Do not use `--force` on existing results. To train through the Editor instead,
omit `--env`, open GridWorld, and press Play after Python is listening.

5. Evaluate exported ONNX with a fixed held-out seed and the same 200 episodes:

```sh
.venv-mlagents/bin/python Training/evaluate.py --env Builds/VisionRL.app --model results/visionrl-ppo-01/VisionRL.onnx --episodes 200 --seed 9001 --output Reports/trained.json
.venv-mlagents/bin/tensorboard --logdir results
```

Compare success rate and its 95% Wilson interval, mean episode length, and mean
successful-episode length. Proposed demo acceptance: >=90% success and materially
above random on the fixed evaluation seed; also run a second seed before claiming
generalization. All cells remain part of training; held-out seeds provide new
episode sequences, not a held-out spatial split.

Prioritize runtime validation, a short PPO export test, and evaluation before a
long run. Optional features are deferred until these pass. Current test outcomes
belong in `Reports`, not inferred from this procedure.

## Stage 2 audit — 13 October 2026

- Build was stale (older than scene/script edits); rebuilt. Previous build, scene and
  configs are in `Backups/`.
- `mlagents-learn` launched without `--env-args` opens a screen-sized player window and
  the player stalled mid-run twice (TimeoutError: Workers {0} stuck in waiting state).
  With `--env-args -batchmode -screen-fullscreen 0 -screen-width 256 -screen-height 256`
  the 4096-step smoke run completed. Always pass these args.
- Smoke runs: `results/stage2-smoke` and `stage2-smoke-2` (stalled, kept for reference),
  `stage2-smoke-3` (completed). They are connectivity tests, not trained policies.
