using System;
using System.Collections.Generic;

namespace VisionRL
{
    // Pure mechanics for the escape room. Never passed to a policy sensor.
    // The room is an 8x8 grid of cells. Obstacles are blocked cells; goal cells sit next to the exit.
    // Layout strings: top row of the array is the highest z. '#' obstacle, 'G' goal, '.' free.
    public sealed class RoomWorld
    {
        public const int Size = 8;
        public static readonly string[] DefaultLayout =
        {
            ".....GG.", // z=7 (exit is in the wall above the two G cells)
            "..##....", // z=6
            "......#.", // z=5
            "##.#..#.", // z=4
            "...#....", // z=3
            ".#...#..", // z=2
            ".#...#..", // z=1
            "........", // z=0
        };

        // Stage 8: same obstacles, no fixed exit. The exit is chosen on the border at every reset.
        public static readonly string[] RandomExitLayout =
        {
            "........", "..##....", "......#.", "##.#..#.", "...#....", ".#...#..", ".#...#..", "........",
        };

        readonly bool[,] blocked = new bool[Size, Size];
        readonly bool[,] goal = new bool[Size, Size];
        int[,] pathDistance = new int[Size, Size]; // steps to nearest goal, walking around obstacles
        readonly List<int> startCells = new List<int>();

        public int X { get; private set; }
        public int Z { get; private set; }
        public int Steps { get; private set; }
        public int Horizon { get; set; } = 128;
        // distanceScale 0.05 (Stage 1 used 0.1): with gamma 0.99 the shaping term leaks 0.0005*distance per
        // idle step, at most 0.006 here, which stays below the 0.01 step penalty. So no-progress loops never pay.
        public RewardSettings Rewards { get; set; } = new RewardSettings { distanceScale = 0.05f };
        public bool Success => goal[X, Z];
        public bool Done => Success || Steps >= Horizon;
        public int Distance => pathDistance[X, Z];
        public bool IsBlocked(int x, int z) => x < 0 || x >= Size || z < 0 || z >= Size || blocked[x, z];
        public bool IsObstacle(int x, int z) => blocked[x, z];
        public bool IsGoal(int x, int z) => goal[x, z];
        public int StartCellCount => startCells.Count;
        public int PathDistanceAt(int x, int z) => pathDistance[x, z];

        // ---- Random-exit mode (Stage 8) ----
        readonly List<int> exitCells = new List<int>();                 // border cells that are not obstacles, sorted
        readonly Dictionary<int, int[,]> distanceToExit = new Dictionary<int, int[,]>();
        public bool RandomExit { get; private set; }
        public int ExitX { get; private set; } = -1;
        public int ExitZ { get; private set; } = -1;
        public int ExitCandidateCount => exitCells.Count;
        // Every 4th candidate (index % 4 == 3) is held out of training, to test generalisation to unseen exit positions.
        public bool IsHeldOutExit(int index) => index % 4 == 3;
        public void ExitCandidate(int index, out int x, out int z) { x = exitCells[index] / Size; z = exitCells[index] % Size; }
        public int MinStartDistance { get; set; } = 6;

        public RoomWorld(string[] layout = null, bool randomExit = false)
        {
            layout = layout ?? DefaultLayout;
            if (layout.Length != Size) throw new ArgumentException("layout needs " + Size + " rows");
            for (int row = 0; row < Size; row++)
            {
                if (layout[row].Length != Size) throw new ArgumentException("layout row " + row + " needs " + Size + " chars");
                for (int x = 0; x < Size; x++)
                {
                    int z = Size - 1 - row;
                    char c = layout[row][x];
                    if (c == '#') blocked[x, z] = true;
                    else if (c == 'G') goal[x, z] = true;
                    else if (c != '.') throw new ArgumentException("bad layout char '" + c + "'");
                }
            }
            RandomExit = randomExit;
            if (randomExit) EnableRandomExit(); else ComputePathDistances();
        }

        void EnableRandomExit()
        {
            int free = 0;
            for (int x = 0; x < Size; x++) for (int z = 0; z < Size; z++) if (!blocked[x, z]) free++;
            for (int x = 0; x < Size; x++)
            for (int z = 0; z < Size; z++)
            {
                if (blocked[x, z] || goal[x, z]) { if (goal[x, z]) throw new ArgumentException("random-exit layouts must not contain G"); continue; }
                if (x != 0 && x != Size - 1 && z != 0 && z != Size - 1) continue; // border only
                var grid = BfsFrom(x, z);
                int reached = 0;
                foreach (int d in grid) if (d >= 0) reached++;
                if (reached != free) throw new ArgumentException($"exit {x},{z} cannot be reached from every free cell");
                exitCells.Add(x * Size + z);
                distanceToExit[x * Size + z] = grid;
            }
            if (exitCells.Count == 0) throw new ArgumentException("no valid exit cells");
        }

        int[,] BfsFrom(int gx, int gz)
        {
            var grid = new int[Size, Size];
            for (int x = 0; x < Size; x++) for (int z = 0; z < Size; z++) grid[x, z] = -1;
            var queue = new Queue<int>(); grid[gx, gz] = 0; queue.Enqueue(gx * Size + gz);
            int[] dx = { 0, 0, -1, 1 }, dz = { 1, -1, 0, 0 };
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue(); int cx = cell / Size, cz = cell % Size;
                for (int i = 0; i < 4; i++)
                {
                    int nx = cx + dx[i], nz = cz + dz[i];
                    if (IsBlocked(nx, nz) || grid[nx, nz] >= 0) continue;
                    grid[nx, nz] = grid[cx, cz] + 1; queue.Enqueue(nx * Size + nz);
                }
            }
            return grid;
        }

        // Test helper: switch to a specific exit candidate (no start chosen; call Set afterwards).
        public void ResetRandomExitTo(int exitIndex)
        {
            int key = exitCells[exitIndex];
            if (ExitX >= 0) goal[ExitX, ExitZ] = false;
            ExitX = key / Size; ExitZ = key % Size; goal[ExitX, ExitZ] = true; pathDistance = distanceToExit[key];
        }

        public int PathDistanceFromExit(int exitIndex, int x, int z) => distanceToExit[exitCells[exitIndex]][x, z];

        // exitSet: 0 = all exits, 1 = training exits, 2 = held-out exits. Picks a new exit, then a valid start that is
        // not the exit and at least MinStartDistance steps (walking around obstacles) away from it.
        public void ResetRandomExit(Random random, int exitSet = 0)
        {
            if (!RandomExit) throw new InvalidOperationException("world was not created in random-exit mode");
            var pool = new List<int>();
            for (int i = 0; i < exitCells.Count; i++)
            {
                bool held = IsHeldOutExit(i);
                if (exitSet == 0 || (exitSet == 1 && !held) || (exitSet == 2 && held)) pool.Add(i);
            }
            if (pool.Count == 0) throw new ArgumentException("empty exit set " + exitSet);
            int key = exitCells[pool[random.Next(pool.Count)]];
            if (ExitX >= 0) goal[ExitX, ExitZ] = false;
            ExitX = key / Size; ExitZ = key % Size;
            goal[ExitX, ExitZ] = true;
            pathDistance = distanceToExit[key];
            startCells.Clear();
            for (int x = 0; x < Size; x++)
            for (int z = 0; z < Size; z++)
                if (!blocked[x, z] && !goal[x, z] && pathDistance[x, z] >= MinStartDistance) startCells.Add(x * Size + z);
            if (startCells.Count == 0) throw new InvalidOperationException("no valid start for exit " + ExitX + "," + ExitZ);
            int cell = startCells[random.Next(startCells.Count)];
            Set(cell / Size, cell % Size);
        }

        // Breadth-first search outward from the goal cells. Used only for reward shaping.
        void ComputePathDistances()
        {
            var queue = new Queue<int>();
            for (int x = 0; x < Size; x++)
            for (int z = 0; z < Size; z++)
            {
                pathDistance[x, z] = -1;
                if (goal[x, z]) { pathDistance[x, z] = 0; queue.Enqueue(x * Size + z); }
            }
            if (queue.Count == 0) throw new ArgumentException("layout has no goal");
            int[] dx = { 0, 0, -1, 1 }, dz = { 1, -1, 0, 0 };
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue(); int cx = cell / Size, cz = cell % Size;
                for (int i = 0; i < 4; i++)
                {
                    int nx = cx + dx[i], nz = cz + dz[i];
                    if (IsBlocked(nx, nz) || pathDistance[nx, nz] >= 0) continue;
                    pathDistance[nx, nz] = pathDistance[cx, cz] + 1;
                    queue.Enqueue(nx * Size + nz);
                }
            }
            for (int x = 0; x < Size; x++)
            for (int z = 0; z < Size; z++)
            {
                if (blocked[x, z] || goal[x, z]) continue;
                if (pathDistance[x, z] < 0) throw new ArgumentException($"cell {x},{z} cannot reach the goal");
                startCells.Add(x * Size + z);
            }
        }

        public void Reset(Random random)
        {
            int cell = startCells[random.Next(startCells.Count)];
            Set(cell / Size, cell % Size);
        }

        public void Set(int x, int z)
        {
            if (IsBlocked(x, z)) throw new ArgumentOutOfRangeException($"cell {x},{z} is a wall or obstacle");
            X = x; Z = z; Steps = 0;
        }

        // Action 0 +Z (forward), 1 -Z (backward), 2 -X (left), 3 +X (right).
        public static void Delta(int action, out int dx, out int dz)
        {
            dx = 0; dz = 0;
            switch (action)
            {
                case 0: dz = 1; break;
                case 1: dz = -1; break;
                case 2: dx = -1; break;
                case 3: dx = 1; break;
                default: throw new ArgumentOutOfRangeException(nameof(action));
            }
        }

        public float Step(int action)
        {
            if (Done) throw new InvalidOperationException("Episode has ended.");
            int previous = Distance;
            Delta(action, out int dx, out int dz);
            bool bump = IsBlocked(X + dx, Z + dz); // outer wall or obstacle
            if (!bump) { X += dx; Z += dz; }
            Steps++;
            var r = Rewards;
            // Potential shaping on path distance: closer to the exit = positive. Bumping or walking
            // back and forth nets zero shaping, so repeated no-progress actions cannot be farmed.
            float potential = -r.distanceScale * previous;
            float nextPotential = Done ? 0f : -r.distanceScale * Distance;
            return r.stepPenalty + (bump ? r.boundaryPenalty : 0f)
                   + (Success ? r.targetReward : Done ? r.timeoutPenalty : 0f)
                   + r.gamma * nextPotential - potential;
        }

        // For tests and diagnostics only; never exposed to the agent.
        public int ShortestPathAction(int x, int z)
        {
            for (int a = 0; a < 4; a++)
            {
                Delta(a, out int dx, out int dz);
                if (!IsBlocked(x + dx, z + dz) && pathDistance[x + dx, z + dz] == pathDistance[x, z] - 1) return a;
            }
            throw new InvalidOperationException("no downhill action");
        }
    }
}
