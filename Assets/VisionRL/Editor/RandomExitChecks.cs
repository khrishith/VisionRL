using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace VisionRL.Editor
{
    // Stage 8 checks for the randomized-exit task. Edit mode: mechanics, reachability, reward audit, scene contents.
    // Play mode: resets update logical AND visual exit; camera frames show the exit where it really is.
    // Results: Reports/stage8-edit.txt, Reports/stage8-playmode.txt. No Python or trainer involved.
    [InitializeOnLoad]
    public static class RandomExitChecks
    {
        const string Pending = "VisionRL.RandomExitChecks";
        static readonly List<string> results = new List<string>();
        static double started;
        static RandomExitChecks() { EditorApplication.update += Tick; }

        static void Check(bool value, string name) { results.Add((value ? "PASS " : "FAIL ") + name); if (!value) throw new Exception(name); }
        static void Save(string file) { Directory.CreateDirectory("Reports"); File.WriteAllLines(file, results); Debug.Log("VISIONRL_CHECKS " + file + " " + string.Join("; ", results)); }

        [MenuItem("VisionRL/Random Exit/Run Checks")]
        public static void Run()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            RandomExitBuilder.CreateScene();
            results.Clear();
            try { EditMode(); }
            catch (Exception e) { results.Add("FAIL exception: " + e); Debug.LogException(e); Save("Reports/stage8-edit.txt"); return; }
            Save("Reports/stage8-edit.txt");
            SessionState.SetBool(Pending, true); started = 0;
            EditorApplication.EnterPlaymode();
        }

        static int[,] TestBfs(RoomWorld w, int gx, int gz)   // independent distance calculation, not using RoomWorld's
        {
            var d = new int[8, 8]; for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++) d[x, z] = -1;
            var q = new Queue<Vector2Int>(); d[gx, gz] = 0; q.Enqueue(new Vector2Int(gx, gz));
            var dirs = new[] { new Vector2Int(0, 1), new Vector2Int(0, -1), new Vector2Int(-1, 0), new Vector2Int(1, 0) };
            while (q.Count > 0) { var c = q.Dequeue(); foreach (var v in dirs) { int nx = c.x + v.x, nz = c.y + v.y; if (w.IsBlocked(nx, nz) || d[nx, nz] >= 0) continue; d[nx, nz] = d[c.x, c.y] + 1; q.Enqueue(new Vector2Int(nx, nz)); } }
            return d;
        }

        static void EditMode()
        {
            var w = new RoomWorld(RoomWorld.RandomExitLayout, true);
            int n = w.ExitCandidateCount;
            int obstacles = 0; for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++) if (w.IsObstacle(x, z)) obstacles++;
            Check(obstacles == 12 && n == 27, "layout: 12 obstacles, 27 candidate exits (all non-obstacle border cells)");
            var sides = new HashSet<string>(); int held = 0;
            for (int i = 0; i < n; i++)
            {
                w.ExitCandidate(i, out int x, out int z);
                if (!(x == 0 || x == 7 || z == 0 || z == 7) || w.IsObstacle(x, z)) throw new Exception($"candidate {x},{z} not a free border cell");
                if (z == 7) sides.Add("N"); if (z == 0) sides.Add("S"); if (x == 0) sides.Add("W"); if (x == 7) sides.Add("E");
                if (w.IsHeldOutExit(i)) held++;
            }
            Check(sides.Count == 4, "exit candidates lie on all four sides of the room");
            Check(held == 6, "6 of 27 exits are held out for generalisation tests (21 train)");

            // Reachability, checked with an independent BFS: from every exit, every free cell is reachable; distances agree.
            int mismatches = 0;
            for (int i = 0; i < n; i++)
            {
                w.ExitCandidate(i, out int gx, out int gz); var d = TestBfs(w, gx, gz);
                for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
                    if (!w.IsObstacle(x, z)) { if (d[x, z] < 0) throw new Exception($"unreachable {x},{z} from exit {gx},{gz}"); if (d[x, z] != w.PathDistanceFromExit(i, x, z)) mismatches++; }
            }
            Check(mismatches == 0, "every exit reaches every free cell, and RoomWorld path distances equal an independent BFS");

            // Resets
            var random = new System.Random(11);
            foreach (int set in new[] { 0, 1, 2 })
            {
                var exitsSeen = new HashSet<int>(); var sidesSeen = new HashSet<string>(); int minDist = 99;
                for (int r = 0; r < 4000; r++)
                {
                    w.ResetRandomExit(random, set);
                    if (w.IsBlocked(w.X, w.Z) || w.Success || w.Steps != 0) throw new Exception("invalid start");
                    if (!w.IsGoal(w.ExitX, w.ExitZ) || w.IsObstacle(w.ExitX, w.ExitZ)) throw new Exception("exit not a goal cell");
                    int goals = 0; for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++) if (w.IsGoal(x, z)) goals++;
                    if (goals != 1) throw new Exception("old exit not cleared: " + goals + " goal cells");
                    minDist = Math.Min(minDist, w.Distance);
                    exitsSeen.Add(w.ExitX * 8 + w.ExitZ);
                    int idx = -1; for (int i = 0; i < n; i++) { w.ExitCandidate(i, out int cx, out int cz); if (cx == w.ExitX && cz == w.ExitZ) idx = i; }
                    bool isHeld = w.IsHeldOutExit(idx);
                    if ((set == 1 && isHeld) || (set == 2 && !isHeld)) throw new Exception("exit from the wrong set");
                    sidesSeen.Add(w.ExitZ == 7 ? "N" : w.ExitZ == 0 ? "S" : w.ExitX == 0 ? "W" : "E");
                }
                int expected = set == 0 ? 27 : set == 1 ? 21 : 6;
                Check(exitsSeen.Count == expected && minDist >= 6, $"exit set {set}: 4000 resets use all {expected} exits, starts always valid, never the exit, >= 6 steps away (min {minDist})");
                if (set != 2) Check(sidesSeen.Count == 4, $"exit set {set}: exits appear on all four sides");
            }

            // Exhaustive mechanics + reward audit over every (exit, free cell, action).
            int steps = 0; float maxNonProgress = -9, minProgress = 9, minSuccess = 9; int[] dxs = { 0, 0, -1, 1 }, dzs = { 1, -1, 0, 0 };
            for (int i = 0; i < n; i++)
            {
                w.ExitCandidate(i, out int gx, out int gz); var d = TestBfs(w, gx, gz);
                for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
                {
                    if (w.IsObstacle(x, z) || (x == gx && z == gz)) continue;
                    for (int a = 0; a < 4; a++)
                    {
                        w.ResetRandomExitTo(i); w.Set(x, z);
                        int nx = x + dxs[a], nz = z + dzs[a]; bool bump = w.IsBlocked(nx, nz);
                        float r = w.Step(a); steps++;
                        if (bump ? (w.X != x || w.Z != z) : (w.X != nx || w.Z != nz)) throw new Exception($"move/bump wrong at {x},{z} a{a} exit {gx},{gz}");
                        int dNew = d[w.X, w.Z]; bool succ = w.X == gx && w.Z == gz;
                        float expected = -.01f + (bump ? -.02f : 0f) + (succ ? 1f : 0f) + .99f * (succ ? 0f : -.05f * dNew) - (-.05f * d[x, z]);
                        if (Mathf.Abs(r - expected) > 1e-4f) throw new Exception($"reward {r} != {expected} at {x},{z} a{a} exit {gx},{gz}");
                        if (succ) minSuccess = Mathf.Min(minSuccess, r);
                        else if (dNew < d[x, z]) minProgress = Mathf.Min(minProgress, r);   // moved closer
                        else maxNonProgress = Mathf.Max(maxNonProgress, r);                  // bump, same distance, or farther
                    }
                }
            }
            Check(true, $"all (exit, cell, action) combinations: moves, bumps and rewards match the formula ({steps} steps)");
            Check(maxNonProgress < 0f, $"REWARD AUDIT: every step that does not get closer (bump, same distance, farther) is negative in every exit/cell/action (max {maxNonProgress:F4})");
            Check(minProgress > 0f, $"REWARD AUDIT: only getting closer pays a positive step reward (min {minProgress:F4}); everything else costs");
            Check(minSuccess > .9f, $"reaching the exit always pays >= {minSuccess:F3}");

            // Two-step loops (forward/back etc.) are net negative everywhere, for every exit.
            float worstLoop = -9;
            for (int i = 0; i < n; i++)
            {
                w.ExitCandidate(i, out int gx, out int gz);
                for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
                {
                    if (w.IsObstacle(x, z) || (x == gx && z == gz)) continue;
                    for (int a = 0; a < 4; a++)
                    {
                        w.ResetRandomExitTo(i); w.Set(x, z);
                        if (w.IsBlocked(x + dxs[a], z + dzs[a]) || (x + dxs[a] == gx && z + dzs[a] == gz)) continue;
                        float loop = w.Step(a); loop += w.Step(a ^ 1);   // 0<->1 and 2<->3 are opposite actions
                        worstLoop = Mathf.Max(worstLoop, loop);
                    }
                }
            }
            Check(worstLoop < 0f, $"REWARD AUDIT: back-and-forth loops are net negative for every exit and cell (best {worstLoop:F4})");

            // Timeout and fixed-direction policies.
            w.Horizon = 48; w.ResetRandomExitTo(0); w.Set(4, 3);
            int t = 0; while (!w.Done) { w.Step(2); t++; }
            Check((t == 48 || w.Success) , "episodes end by the 48th action");
            int wins = 0, trials = 0;
            for (int i = 0; i < n; i++) for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
            {
                w.ExitCandidate(i, out int gx, out int gz);
                if (w.IsObstacle(x, z) || (x == gx && z == gz) || w.PathDistanceFromExit(i, x, z) < 6) continue;
                for (int a = 0; a < 4; a++) { w.ResetRandomExitTo(i); w.Set(x, z); while (!w.Done) w.Step(a); trials++; if (w.Success) wins++; }
            }
            Check(wins / (float)trials < .03f, $"FIXED-DIRECTION POLICIES FAIL: always-forward/back/left/right win {wins}/{trials} ({100f * wins / trials:F1}%) of valid (exit, start) pairs");
            // Blind random-walk policies, many episodes, Monte Carlo
            var rng = new System.Random(3);
            double[][] mixes = { new[] { .25, .25, .25, .25 }, new[] { .4838, .0195, .1214, .3753 }, new[] { .5, 0, 0, .5 } };
            string[] names = { "uniform random", "Stage 7 fixed probabilities", "forward/right 50/50" };
            for (int m = 0; m < mixes.Length; m++)
            {
                int ok = 0; const int E = 4000;
                for (int e = 0; e < E; e++)
                {
                    w.ResetRandomExit(rng, 0);
                    while (!w.Done) { double u = rng.NextDouble(), c = 0; int a = 3; for (int k = 0; k < 4; k++) { c += mixes[m][k]; if (u < c) { a = k; break; } } w.Step(a); }
                    if (w.Success) ok++;
                }
                Check(ok / (double)E < .25, $"blind policy '{names[m]}' wins only {100.0 * ok / E:F1}% of {E} randomized episodes (< 25%)");
            }
            w.Horizon = 128;

            // ---- Scene contents ----
            Check(EditorSceneManager.GetActiveScene().path == RandomExitBuilder.ScenePath, "EscapeRoomRandomExit.unity is open");
            Check(GameObject.Find("Floor").transform.childCount == 65, "floor: 64 tiles + slab");
            var walls = GameObject.Find("Walls").transform;
            foreach (var nme in new[] { "Wall_South", "Wall_North", "Wall_West", "Wall_East" }) Check(walls.Find(nme) != null, "solid wall present: " + nme);
            Check(walls.Find("Wall_North_Left") == null, "no fixed exit gap in the walls");
            Check(GameObject.Find("Obstacles").transform.childCount == 12, "12 obstacles");
            var agent = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            Check(agent != null && agent.randomizeExit && agent.goalPad != null && agent.exitDoor != null && agent.horizon == 48 && agent.minStartDistance == 6,
                  "RoomAgent: random exit on, pad/door linked, 48-step limit, start >= 6 from exit");
            var b = agent.GetComponent<BehaviorParameters>();
            Check(b.BehaviorName == "VisionRL" && b.Model == null && b.BehaviorType == BehaviorType.Default, "behavior VisionRL, Default, no model attached (training scene)");
            Check(b.BrainParameters.VectorObservationSize == 0 && b.BrainParameters.NumStackedVectorObservations == 1, "no vector observations");
            Check(b.BrainParameters.ActionSpec.NumContinuousActions == 0 && b.BrainParameters.ActionSpec.BranchSizes.Length == 1 && b.BrainParameters.ActionSpec.BranchSizes[0] == 4, "one discrete branch of 4 actions");
            var cs = agent.GetComponent<CameraSensorComponent>();
            Check(agent.GetComponents<SensorComponent>().Length == 1 && cs.Width == 84 && cs.Height == 84 && !cs.Grayscale, "single 84x84 RGB camera sensor (no ray sensors)");
            Check(agent.GetComponentsInChildren<Collider>().Length == 1, "only the agent root has a collider");
            Check(UnityEngine.Object.FindAnyObjectByType<GridAgent>() == null, "no GridAgent in the scene");
        }

        static void Tick()
        {
            if (!SessionState.GetBool(Pending, false) || !EditorApplication.isPlaying) return;
            if (started == 0) { started = EditorApplication.timeSinceStartup; return; }
            if (EditorApplication.timeSinceStartup - started < 1 || Time.frameCount < 2) return;
            SessionState.SetBool(Pending, false); results.Clear();
            try { PlayMode(); }
            catch (Exception e) { results.Add("FAIL exception: " + e); Debug.LogException(e); }
            finally { Save("Reports/stage8-playmode.txt"); EditorApplication.ExitPlaymode(); }
        }

        static void PlayMode()
        {
            var a = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            Check(a != null && !Academy.Instance.IsCommunicatorOn, "RoomAgent in Play mode, no trainer attached");
            Academy.Instance.EnvironmentStep();
            var w = a.World;
            Check(w.RandomExit && w.Horizon == 48, "world is in random-exit mode with a 48-step limit");
            void Act(int act) { a.ManualAction = act; a.RequestDecision(); Academy.Instance.EnvironmentStep(); }
            Vector3 Pos(int x, int z, float y) => new Vector3(x - 3.5f, y, z - 3.5f);

            // Resets: logical exit, visual pad and door, robot start all consistent.
            var exits = new HashSet<int>(); int sideN = 0, sideS = 0, sideW = 0, sideE = 0;
            for (int i = 0; i < 400; i++)
            {
                a.OnEpisodeBegin();
                if (w.IsBlocked(w.X, w.Z) || w.Success || w.Steps != 0 || w.Distance < 6) throw new Exception("invalid start after reset");
                if (!w.IsGoal(w.ExitX, w.ExitZ)) throw new Exception("exit is not a goal");
                if ((a.goalPad.position - Pos(w.ExitX, w.ExitZ, 0f)).magnitude > 1e-3f) throw new Exception($"pad {a.goalPad.position} not on exit {w.ExitX},{w.ExitZ}");
                var door = a.exitDoor.position; bool ns = w.ExitZ == 0 || w.ExitZ == 7;
                Vector3 want = ns ? new Vector3(w.ExitX - 3.5f, .5f, w.ExitZ == 0 ? -4.25f : 4.25f) : new Vector3(w.ExitX == 0 ? -4.25f : 4.25f, .5f, w.ExitZ - 3.5f);
                if ((door - want).magnitude > 1e-3f) throw new Exception($"door {door} should be {want} for exit {w.ExitX},{w.ExitZ}");
                if ((a.transform.localPosition - Pos(w.X, w.Z, .375f)).magnitude > 1e-3f) throw new Exception("robot not on start cell");
                exits.Add(w.ExitX * 8 + w.ExitZ);
                if (ns) { if (w.ExitZ == 7) sideN++; else sideS++; } else { if (w.ExitX == 0) sideW++; else sideE++; }
            }
            Check(exits.Count >= 24 && sideN > 0 && sideS > 0 && sideW > 0 && sideE > 0, $"400 resets: pad, door and robot always match the logical state; {exits.Count}/27 exits and all four sides used (N{sideN} S{sideS} W{sideW} E{sideE})");

            // Drive a full route with the real agent: follow the shortest path (diagnostic only) to the exit.
            a.OnEpisodeBegin(); int ep = a.CompletedEpisodes; int dist = w.Distance, moves = 0; int ex = w.ExitX, ez = w.ExitZ;
            while (a.CompletedEpisodes == ep && moves < 60) { Act(w.ShortestPathAction(w.X, w.Z)); moves++; }
            Check(moves == dist && a.CompletedEpisodes == ep + 1 && Mathf.Abs(a.LastReward - 1.04f) < 1e-3f && Mathf.Abs(a.LastReward) > 0, $"robot drove the {dist}-step route to the exit {ex},{ez}: episode ended, reward {a.LastReward:F3}");
            Check(!w.Success && w.Steps == 0 && !w.IsBlocked(w.X, w.Z), "after the exit, a new valid episode starts");

            // Walls / obstacles block; wandering is never positive.
            for (int tries = 0; tries < 50 && w.ExitX == 7 && w.ExitZ == 3; tries++) a.OnEpisodeBegin(); // keep (7,3) from being the exit
            w.Set(2, 5); a.SyncTransforms(); Act(0);
            Check(w.X == 2 && w.Z == 5 && a.LastReward < 0, $"obstacle at (2,6) blocks the robot, reward {a.LastReward:F3} < 0");
            w.Set(7, 3); a.SyncTransforms(); Act(3);
            Check(w.X == 7 && w.Z == 3 && a.LastReward < 0, $"east wall blocks the robot, reward {a.LastReward:F3} < 0");
            int badPositive = 0, steps2 = 0, timeouts = 0, positiveEpisodes = 0; var rnd = new System.Random(5);
            for (int e = 0; e < 60; e++)
            {
                a.OnEpisodeBegin(); int ep2 = a.CompletedEpisodes; float total = 0;
                while (a.CompletedEpisodes == ep2)
                {
                    int before = w.Distance; Act(rnd.Next(4)); total += a.LastReward; steps2++;
                    bool ended = a.CompletedEpisodes != ep2;
                    if (!ended && a.LastReward > 0 && a.World.Distance >= before) badPositive++;   // positive without getting closer
                }
                bool success = a.LastReward > .5f;
                if (!success) { timeouts++; if (total >= 0) positiveEpisodes++; }
            }
            Check(badPositive == 0 && positiveEpisodes == 0 && timeouts > 10, $"60 random-walk episodes in the real agent ({steps2} steps): {badPositive} positive rewards without progress; {timeouts} timeouts, all with negative total return");

            // Camera: the green exit in the frame sits where the logical exit is, for exits on every side.
            var cs = a.GetComponent<CameraSensorComponent>(); var tex = new Texture2D(84, 84, TextureFormat.RGB24, false);
            try
            {
                int checkedFrames = 0; float worstOffset = 0; int minGreen = int.MaxValue, minBlue = int.MaxValue; float scale = 84f / 9.2f;
                var targets = new[] { (0, 0), (7, 7), (3, 7), (7, 4), (0, 3), (3, 0), (6, 0), (0, 7) };
                foreach (var (tx, tz) in targets)
                {
                    int idx = -1; for (int i = 0; i < w.ExitCandidateCount; i++) { w.ExitCandidate(i, out int cx, out int cz); if (cx == tx && cz == tz) idx = i; }
                    if (idx < 0) continue;
                    w.ResetRandomExitTo(idx); a.SyncExitVisual();
                    // robot at a cell far from the exit
                    int rx = tx < 4 ? 6 : 1, rz = tz < 4 ? 6 : 1; if (w.IsObstacle(rx, rz)) rz = tz < 4 ? 7 : 0; w.Set(rx, rz); a.SyncTransforms();
                    CameraSensor.ObservationToTexture(cs.Camera, tex, 84, 84);
                    var px = tex.GetPixels(); double sumx = 0, sumy = 0; int green = 0, blue = 0;
                    for (int row = 0; row < 84; row++) for (int col = 0; col < 84; col++)
                    {
                        var c = px[(83 - row) * 84 + col];
                        if (c.g > .7f && c.r < .4f && c.b < .4f) { green++; sumx += col; sumy += row; }
                        if (c.b > .7f && c.r < .4f && c.g < .5f) blue++;
                    }
                    double wx = (sumx / green - 42) / scale, wz = (42 - sumy / green) / scale;
                    float offset = Mathf.Max(Mathf.Abs((float)wx - (tx - 3.5f)), Mathf.Abs((float)wz - (tz - 3.5f)));
                    worstOffset = Mathf.Max(worstOffset, offset); minGreen = Math.Min(minGreen, green); minBlue = Math.Min(minBlue, blue); checkedFrames++;
                    if (tx == 0 && tz == 3) { Directory.CreateDirectory("Reports"); File.WriteAllBytes("Reports/stage8-camera-exit-W.png", tex.EncodeToPNG()); }
                    if (tx == 7 && tz == 7) File.WriteAllBytes("Reports/stage8-camera-exit-NE.png", tex.EncodeToPNG());
                    if (tx == 3 && tz == 0) File.WriteAllBytes("Reports/stage8-camera-exit-S.png", tex.EncodeToPNG());
                    if (tx == 3 && tz == 7) File.WriteAllBytes("Reports/stage8-camera-exit-N.png", tex.EncodeToPNG());
                }
                Check(checkedFrames >= 6, $"{checkedFrames} exits rendered, covering all four sides");
                Check(minGreen >= 60, $"green exit visible in every frame (fewest green pixels {minGreen})");
                Check(minBlue >= 5, $"robot visible in every frame (fewest blue pixels {minBlue})");
                Check(worstOffset < 1.1f, $"the green blob in the 84x84 frame is centred within {worstOffset:F2} cells of the logical exit in every frame");
            }
            finally { UnityEngine.Object.Destroy(tex); }
            a.OnEpisodeBegin();
        }
    }
}
