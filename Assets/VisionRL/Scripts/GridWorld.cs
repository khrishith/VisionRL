using System;

namespace VisionRL
{
    // Tunable reward numbers. Defaults reproduce the originally tested behaviour.
    [Serializable]
    public sealed class RewardSettings
    {
        public float stepPenalty = -0.01f;      // paid on every action
        public float boundaryPenalty = -0.02f;  // extra, when the move would leave the grid
        public float targetReward = 1f;         // reaching the green cube
        public float timeoutPenalty = -1f;      // step limit hit without reaching it
        // Closer = positive, farther = negative, by this much per cell. It is
        // potential-based (gamma*Phi(next)-Phi(now)), so walking back and forth
        // sums to ~0 and cannot be farmed for reward.
        public float distanceScale = 0.1f;
        public float gamma = 0.99f;
    }

    // Pure environment mechanics. Never passed to a policy sensor.
    public sealed class GridWorld
    {
        public const int Size = 7;
        public const int Horizon = 64;
        public int X { get; private set; }
        public int Z { get; private set; }
        public int TargetX { get; private set; }
        public int TargetZ { get; private set; }
        public int Steps { get; private set; }
        public RewardSettings Rewards { get; set; } = new RewardSettings();
        public bool Success => X == TargetX && Z == TargetZ;
        public bool Done => Success || Steps >= Horizon;
        public int Distance => Math.Abs(X - TargetX) + Math.Abs(Z - TargetZ);

        public void Reset(Random random)
        {
            int start = random.Next(Size * Size);
            int goal = random.Next(Size * Size - 1);
            if (goal >= start) goal++;
            Set(start % Size, start / Size, goal % Size, goal / Size);
        }

        public void Set(int x, int z, int targetX, int targetZ)
        {
            if (x < 0 || x >= Size || z < 0 || z >= Size || targetX < 0 || targetX >= Size || targetZ < 0 || targetZ >= Size)
                throw new ArgumentOutOfRangeException();
            X = x; Z = z; TargetX = targetX; TargetZ = targetZ; Steps = 0;
        }

        public float Step(int action)
        {
            if (Done) throw new InvalidOperationException("Episode has ended.");
            int previous = Distance;
            int x = X, z = Z;
            switch (action)
            {
                case 0: z++; break; // forward = world +Z
                case 1: z--; break;
                case 2: x--; break;
                case 3: x++; break;
                default: throw new ArgumentOutOfRangeException(nameof(action));
            }
            bool wall = x < 0 || x >= Size || z < 0 || z >= Size;
            if (!wall) { X = x; Z = z; }
            Steps++;
            // Potential shaping with PPO gamma=0.99; terminal potential is zero.
            var r = Rewards;
            float potential = -r.distanceScale * previous;
            float nextPotential = Done ? 0f : -r.distanceScale * Distance;
            return r.stepPenalty + (wall ? r.boundaryPenalty : 0f)
                   + (Success ? r.targetReward : Done ? r.timeoutPenalty : 0f)
                   + r.gamma * nextPotential - potential;
        }
    }
}
