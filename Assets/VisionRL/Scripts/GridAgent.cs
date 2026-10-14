using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

namespace VisionRL
{
    public sealed class GridAgent : Agent
    {
        public Transform target;
        public int seed = 1234;
        public RewardSettings rewards = new RewardSettings(); // editable in the Inspector
        readonly GridWorld world = new GridWorld();
        System.Random random;
        int episode;
        bool manualControl;
        int manualAction;
        float lastReward;
        public override void Initialize()
        {
            random = new System.Random(seed);
            world.Rewards = rewards;
            MaxStep = 0; // GridWorld owns the exact 64-action terminal boundary.
            var behavior = GetComponent<BehaviorParameters>();
            manualControl = behavior.BehaviorType == BehaviorType.HeuristicOnly ||
                (!Academy.Instance.IsCommunicatorOn && behavior.Model == null);
            GetComponent<DecisionRequester>().enabled = !manualControl;
        }
        public override void OnEpisodeBegin()
        {
            if (Academy.Instance.IsCommunicatorOn)
            {
                int requestedSeed = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("reset_seed", seed);
                if (requestedSeed != seed) { seed = requestedSeed; random = new System.Random(seed); }
            }
            world.Reset(random);
            SyncTransforms();
        }
        void SyncTransforms()
        {
            transform.localPosition = new Vector3(world.X - 3, 0.375f, world.Z - 3);
            target.localPosition = new Vector3(world.TargetX - 3, 0.375f, world.TargetZ - 3);
        }
        public override void OnActionReceived(ActionBuffers actions)
        {
            lastReward = world.Step(actions.DiscreteActions[0]);
            AddReward(lastReward);
            SyncTransforms();
            if (!world.Done) return;
            Academy.Instance.StatsRecorder.Add("Grid/Success", world.Success ? 1f : 0f);
            Academy.Instance.StatsRecorder.Add("Grid/EpisodeLength", world.Steps);
            episode++;
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
            else return; // No decision when idle; the policy still has exactly four actions.
            RequestDecision();
        }
        void OnGUI()
        {
            GUI.Label(new Rect(12, 12, 600, 30), $"VisionRL | episode {episode} | step {world.Steps}/64 | reward {lastReward:+0.000;-0.000;0.000}");
            if (manualControl) GUI.Label(new Rect(12, 36, 600, 30), "Click Game view. WASD / arrows: one cell per press. Reach the green cube.");
        }
    }
}
