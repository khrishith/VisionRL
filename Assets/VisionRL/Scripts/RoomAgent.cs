using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

namespace VisionRL
{
    // ML-Agents agent for the EscapeRoom. Same interface as GridAgent: camera observation only,
    // one discrete branch of 4 actions, behavior name "VisionRL".
    public sealed class RoomAgent : Agent
    {
        public int seed = 1234;
        public int horizon = 128;                              // action limit per episode
        public RewardSettings rewards = new RewardSettings { distanceScale = 0.05f }; // editable in the Inspector
        // ---- Stage 8: random exit. Off by default so the Stage 4-7 EscapeRoom scene behaves exactly as before. ----
        public bool randomizeExit;
        public int minStartDistance = 6;   // start must be at least this many steps (around obstacles) from the exit
        public int exitSetDefault;         // 0 all exits, 1 training exits, 2 held-out exits; Python can override with "exit_set"
        public Transform goalPad;          // green floor marker, moved to the exit cell on every reset
        public Transform exitDoor;         // green door slab in the wall beside the exit cell
        RoomWorld world;
        System.Random random;
        bool manualControl;
        int manualAction;
        float lastReward;

        public RoomWorld World => world;
        public float LastReward => lastReward;
        public int LastAction { get; private set; } // diagnostics only; never an observation
        public int ManualAction { set { manualAction = value; } }
        public const float CellOffset = (RoomWorld.Size - 1) / 2f;

        public override void Initialize()
        {
            world = randomizeExit ? new RoomWorld(RoomWorld.RandomExitLayout, true) : new RoomWorld();
            world.Horizon = horizon; world.Rewards = rewards; world.MinStartDistance = minStartDistance;
            random = new System.Random(seed);
            MaxStep = 0; // RoomWorld owns the exact terminal boundary.
            var behavior = GetComponent<BehaviorParameters>();
            manualControl = behavior.BehaviorType == BehaviorType.HeuristicOnly ||
                (!Academy.Instance.IsCommunicatorOn && behavior.Model == null);
            GetComponent<DecisionRequester>().enabled = !manualControl;
        }

        public override void OnEpisodeBegin()
        {
            int exitSet = exitSetDefault;
            if (Academy.Instance.IsCommunicatorOn)
            {
                var parameters = Academy.Instance.EnvironmentParameters;
                int requestedSeed = (int)parameters.GetWithDefault("reset_seed", seed);
                if (requestedSeed != seed) { seed = requestedSeed; random = new System.Random(seed); }
                exitSet = (int)parameters.GetWithDefault("exit_set", exitSetDefault);
            }
            if (randomizeExit) world.ResetRandomExit(random, exitSet); else world.Reset(random);
            SyncTransforms();
            SyncExitVisual();
        }

        // Test/evaluation helper: restart the random sequence so different runs can see identical exit/start sequences.
        public void Reseed(int newSeed) { seed = newSeed; random = new System.Random(newSeed); }

        // Keeps the green pad and wall door on the logical exit cell. The visuals are what the camera (the only
        // observation) sees; the logical exit never reaches the policy any other way.
        public void SyncExitVisual()
        {
            if (!randomizeExit || goalPad == null || exitDoor == null) return;
            float ex = world.ExitX - CellOffset, ez = world.ExitZ - CellOffset;
            goalPad.position = new Vector3(ex, 0f, ez);
            // Door goes in the wall on the exit's side: north/south first (corners use north/south), else west/east.
            const float wallHeight = 1f, wallCentre = 4.25f;
            if (world.ExitZ == RoomWorld.Size - 1 || world.ExitZ == 0)
            {
                exitDoor.position = new Vector3(ex, wallHeight / 2, world.ExitZ == 0 ? -wallCentre : wallCentre);
                exitDoor.localScale = new Vector3(.9f, wallHeight + .08f, .52f);
            }
            else
            {
                exitDoor.position = new Vector3(world.ExitX == 0 ? -wallCentre : wallCentre, wallHeight / 2, ez);
                exitDoor.localScale = new Vector3(.52f, wallHeight + .08f, .9f);
            }
        }

        public void SyncTransforms()
        {
            transform.localPosition = new Vector3(world.X - CellOffset, 0.375f, world.Z - CellOffset);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            LastAction = actions.DiscreteActions[0];
            lastReward = world.Step(LastAction);
            AddReward(lastReward);
            SyncTransforms();
            if (!world.Done) return;
            Academy.Instance.StatsRecorder.Add("Room/Success", world.Success ? 1f : 0f);
            Academy.Instance.StatsRecorder.Add("Room/EpisodeLength", world.Steps);
            EndEpisode();
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var actions = actionsOut.DiscreteActions;
            actions[0] = manualAction;
        }

        void Update()
        {
            if (!manualControl) return;
            var k = Keyboard.current;
            if (k == null) return;
            if (k.wKey.wasPressedThisFrame || k.upArrowKey.wasPressedThisFrame) manualAction = 0;
            else if (k.sKey.wasPressedThisFrame || k.downArrowKey.wasPressedThisFrame) manualAction = 1;
            else if (k.aKey.wasPressedThisFrame || k.leftArrowKey.wasPressedThisFrame) manualAction = 2;
            else if (k.dKey.wasPressedThisFrame || k.rightArrowKey.wasPressedThisFrame) manualAction = 3;
            else return;
            RequestDecision();
        }

        void OnGUI()
        {
            if (world == null) return;
            GUI.Label(new Rect(12, 12, 600, 30), $"EscapeRoom | episode {CompletedEpisodes} | step {world.Steps}/{world.Horizon} | reward {lastReward:+0.000;-0.000;0.000}");
            if (manualControl) GUI.Label(new Rect(12, 36, 600, 30), "Click Game view. WASD / arrows: one cell per press. Reach the green exit.");
        }
    }
}
