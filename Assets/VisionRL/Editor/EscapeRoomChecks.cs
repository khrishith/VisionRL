using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace VisionRL.Editor
{
    // Stage 4 checks. Edit-mode part (mechanics + scene contents) then a Play-mode part.
    // No Python or trainer is launched. Results: Reports/stage4-edit.txt and stage4-playmode.txt.
    [InitializeOnLoad]
    public static class EscapeRoomChecks
    {
        const string Pending = "VisionRL.EscapeRoomChecks";
        const string Restore = "VisionRL.EscapeRoomChecks.Restore";
        static readonly List<string> results = new List<string>();
        static double started;
        static EscapeRoomChecks()
        {
            EditorApplication.update += Tick;
            EditorApplication.playModeStateChanged += change =>
            {
                if (change == PlayModeStateChange.EnteredEditMode && SessionState.GetBool(Restore, false))
                { SessionState.SetBool(Restore, false); EditorSceneManager.OpenScene(EscapeRoomBuilder.ScenePath); }
            };
        }

        static void Check(bool value, string name)
        {
            results.Add((value ? "PASS " : "FAIL ") + name);
            if (!value) throw new Exception(name);
        }

        static void Save(string file)
        {
            Directory.CreateDirectory("Reports");
            File.WriteAllLines(file, results);
            Debug.Log("VISIONRL_CHECKS " + file + " " + string.Join("; ", results));
        }

        [MenuItem("VisionRL/Escape Room/Run Checks")]
        public static void Run()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            EscapeRoomBuilder.CreateScene();
            results.Clear();
            try { EditMode(); }
            catch (Exception e) { results.Add("FAIL exception: " + e); Debug.LogException(e); Save("Reports/stage4-edit.txt"); return; }
            Save("Reports/stage4-edit.txt");
            // Stage 7: the scene now has the trained model, so the agent drives itself. The mechanics tests below drive it
            // by hand, so detach the model in memory only (never saved); the scene is reloaded from disk afterwards.
            var bpm = new SerializedObject(UnityEngine.Object.FindAnyObjectByType<RoomAgent>().GetComponent<BehaviorParameters>());
            bpm.FindProperty("m_Model").objectReferenceValue = null;
            bpm.ApplyModifiedPropertiesWithoutUndo();
            SessionState.SetBool(Restore, true);
            SessionState.SetBool(Pending, true);
            started = 0;
            EditorApplication.EnterPlaymode();
        }

        // ---------- Edit mode: pure mechanics and saved scene contents ----------
        static void EditMode()
        {
            var w = new RoomWorld();
            int free = 0, obstacles = 0, goals = 0;
            bool detour = false;
            for (int x = 0; x < 8; x++)
            for (int z = 0; z < 8; z++)
            {
                if (w.IsObstacle(x, z)) { obstacles++; continue; }
                if (w.IsGoal(x, z)) { goals++; continue; }
                free++;
                // Straight-line (Manhattan) distance to the nearest goal; a larger path distance proves a detour is needed.
                int best = int.MaxValue;
                for (int gx = 0; gx < 8; gx++) for (int gz = 0; gz < 8; gz++)
                    if (w.IsGoal(gx, gz)) best = Math.Min(best, Math.Abs(gx - x) + Math.Abs(gz - z));
                if (w.PathDistanceAt(x, z) > best) detour = true;
            }
            Check(obstacles == 12 && goals == 2 && free == 50 && w.StartCellCount == 50, "layout: 12 obstacles, 2 goal cells, 50 free start cells, all reach the goal");
            Check(detour, "obstacles force detours (path distance exceeds straight-line distance somewhere)");

            // Every free cell x every action: moves one cell, or bumps a wall/obstacle and stays.
            int assertions = 0;
            for (int x = 0; x < 8; x++)
            for (int z = 0; z < 8; z++)
            {
                if (w.IsObstacle(x, z) || w.IsGoal(x, z)) continue;
                for (int a = 0; a < 4; a++)
                {
                    w.Set(x, z);
                    RoomWorld.Delta(a, out int dx, out int dz);
                    bool shouldBump = x + dx < 0 || x + dx > 7 || z + dz < 0 || z + dz > 7 || w.IsObstacle(x + dx, z + dz);
                    int d0 = w.PathDistanceAt(x, z);
                    float reward = w.Step(a);
                    assertions += 3;
                    if (shouldBump)
                    {
                        if (w.X != x || w.Z != z) throw new Exception($"bump moved agent at {x},{z} action {a}");
                        // step -0.01 + bump -0.02 + shaping 0.0005*distance (gamma 0.99): still clearly negative
                        if (Mathf.Abs(reward - (-.03f + .0005f * d0)) > .0001f) throw new Exception($"bump reward wrong at {x},{z} action {a}: {reward}");
                    }
                    else if (w.X != x + dx || w.Z != z + dz) throw new Exception($"move failed at {x},{z} action {a}");
                    if (w.Steps != 1) throw new Exception("step count");
                }
            }
            Check(true, $"all free cells x 4 actions: moves, wall bumps and obstacle bumps correct ({assertions} assertions)");

            // Walking the shortest route reaches the goal in exactly Distance steps, and terminates with the goal reward.
            int farX = 0, farZ = 0, far = 0;
            for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
                if (!w.IsObstacle(x, z) && w.PathDistanceAt(x, z) > far) { far = w.PathDistanceAt(x, z); farX = x; farZ = z; }
            w.Set(farX, farZ);
            float total = 0; int taken = 0; float last = 0;
            while (!w.Done) { last = w.Step(w.ShortestPathAction(w.X, w.Z)); total += last; taken++; }
            Check(w.Success && taken == far && far > 10, $"shortest route from the farthest cell ({farX},{farZ}) takes {far} steps and reaches the goal");
            Check(Mathf.Abs(last - 1.04f) < .0001f, "final step into the goal pays 1.04 (goal 1.0 + step -0.01 + shaping 0.05)");
            // Shaping telescopes: total return along the shortest path is bounded, no reward farming.
            Check(total < 2.5f, "return along shortest path is bounded (" + total.ToString("F3") + ")");

            // Back-and-forth between two cells never earns net reward.
            w.Set(0, 0); float oscillation = 0;
            for (int i = 0; i < 40; i++) oscillation += w.Step(i % 2 == 0 ? 0 : 1);
            Check(oscillation < -.1f, "oscillating in place is net negative (" + oscillation.ToString("F3") + ")");

            // Timeout
            w.Set(0, 0); w.Horizon = 128;
            for (int i = 0; i < 127; i++) w.Step(2);
            Check(!w.Done && w.Steps == 127, "episode continues through action 127");
            w.Step(2);
            Check(w.Done && !w.Success, "action 128 ends the episode without success");

            // Resets
            var random = new System.Random(5);
            var seen = new HashSet<int>();
            for (int i = 0; i < 2000; i++)
            {
                w.Reset(random);
                if (w.IsBlocked(w.X, w.Z) || w.Success || w.Steps != 0) throw new Exception("invalid reset");
                seen.Add(w.X * 8 + w.Z);
            }
            Check(seen.Count == 50, "2000 resets: only valid cells, every one of the 50 start cells used");

            // Configurable rewards
            var custom = new RoomWorld { Rewards = new RewardSettings { stepPenalty = -.05f, boundaryPenalty = -.2f, targetReward = 5f, distanceScale = 0f } };
            custom.Set(0, 0);
            Check(Mathf.Abs(custom.Step(2) - -.25f) < .0001f, "reward settings are honoured (custom step + bump penalty)");

            // ---- Scene contents ----
            var scene = EditorSceneManager.GetActiveScene();
            Check(scene.path == EscapeRoomBuilder.ScenePath, "EscapeRoom.unity is open");
            Check(GameObject.Find("Floor").transform.childCount == 65, "floor: 64 tiles + slab");
            var wallRoot = GameObject.Find("Walls").transform;
            foreach (var n in new[] { "Wall_South", "Wall_West", "Wall_East", "Wall_North_Left", "Wall_North_Right" })
                Check(wallRoot.Find(n) != null, "wall present: " + n);
            // Exit gap: the two north wall segments leave world x in [1,3] open.
            var nl = wallRoot.Find("Wall_North_Left"); var nr = wallRoot.Find("Wall_North_Right");
            float gapLeft = nl.position.x + nl.localScale.x / 2, gapRight = nr.position.x - nr.localScale.x / 2;
            Check(Mathf.Abs(gapLeft - 1f) < .001f && Mathf.Abs(gapRight - 3f) < .001f, "exit gap spans world x 1..3, above goal cells x=5,6");
            var goalRoot = GameObject.Find("Goal").transform;
            Check(goalRoot.Find("GoalPad") != null && goalRoot.Find("ExitLintel") != null, "green goal pad and exit frame present");
            var obs = GameObject.Find("Obstacles").transform;
            int matched = 0;
            for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
            {
                var o = obs.Find($"Obstacle_{x}_{z}");
                if (w.IsObstacle(x, z)) { if (o != null && Mathf.Abs(o.position.x - (x - 3.5f)) < .001f && Mathf.Abs(o.position.z - (z - 3.5f)) < .001f) matched++; }
                else if (o != null) throw new Exception("stray obstacle");
            }
            Check(matched == 12 && obs.childCount == 12, "12 scene obstacles match the mechanics layout cell for cell");
            var agent = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            Check(agent != null && agent.transform.Find("Robot Visual") != null, "RoomAgent on 'Blue Agent' root, robot prefab as child");
            Check(UnityEngine.Object.FindAnyObjectByType<GridAgent>() == null, "no GridAgent in this scene");
            var b = agent.GetComponent<BehaviorParameters>();
            Check(b.BehaviorName == "VisionRL", "behavior name VisionRL matches Training/ppo.yaml");
            Check(b.Model != null && b.Model.name == "VisionRL_escape_train_01" && b.BehaviorType == BehaviorType.Default
                  && b.InferenceDevice == InferenceDevice.Burst && !b.DeterministicInference,
                  "trained ONNX assigned: Default behavior (Python -> trains, no Python -> inference), Burst CPU, sampled actions");
            Check(b.BrainParameters.VectorObservationSize == 0 && b.BrainParameters.NumStackedVectorObservations == 1, "no vector observations");
            Check(b.BrainParameters.ActionSpec.NumContinuousActions == 0 && b.BrainParameters.ActionSpec.BranchSizes.Length == 1 && b.BrainParameters.ActionSpec.BranchSizes[0] == 4, "one discrete branch of 4 actions");
            var sensors = agent.GetComponents<SensorComponent>();
            var cs = agent.GetComponent<CameraSensorComponent>();
            Check(sensors.Length == 1 && cs != null && cs.Width == 84 && cs.Height == 84 && !cs.Grayscale && cs.Camera != null, "single 84x84 RGB camera sensor");
            Check(agent.GetComponentsInChildren<Collider>().Length == 1, "only the agent root has a collider");
            Check(UnityEngine.Object.FindObjectsByType<Light>(FindObjectsInactive.Exclude).Length == 1, "one shadow-casting sun");
            Check(cs.Camera.orthographic && Mathf.Abs(cs.Camera.orthographicSize - 4.6f) < .001f, "camera: orthographic top-down covering the room and walls");
        }

        // ---------- Play mode ----------
        static void Tick()
        {
            if (!SessionState.GetBool(Pending, false) || !EditorApplication.isPlaying) return;
            if (started == 0) { started = EditorApplication.timeSinceStartup; return; }
            if (EditorApplication.timeSinceStartup - started < 1 || Time.frameCount < 2) return;
            SessionState.SetBool(Pending, false);
            results.Clear();
            try { PlayMode(); }
            catch (Exception e) { results.Add("FAIL exception: " + e); Debug.LogException(e); }
            finally { Save("Reports/stage4-playmode.txt"); EditorApplication.ExitPlaymode(); }
        }

        static void PlayMode()
        {
            var a = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            Check(a != null, "RoomAgent exists in Play mode");
            Check(!Academy.Instance.IsCommunicatorOn, "no trainer connection");
            Academy.Instance.EnvironmentStep(); // let the agent's own first reset happen
            var w = a.World;
            void Place(int x, int z) { w.Set(x, z); a.SyncTransforms(); }
            void Act(int action) { a.ManualAction = action; a.RequestDecision(); Academy.Instance.EnvironmentStep(); }
            Vector3 Expected() => new Vector3(w.X - 3.5f, .375f, w.Z - 3.5f);

            Place(0, 0);
            for (int i = 0; i < 10; i++) Academy.Instance.EnvironmentStep();
            Check(w.Steps == 0 && w.X == 0 && w.Z == 0, "idle simulation steps do nothing");

            Place(3, 0); Act(0);
            Check(w.X == 3 && w.Z == 1 && a.transform.localPosition == Expected(), "forward moves one cell (+Z) and the robot follows");
            Act(3); Check(w.X == 4 && w.Z == 1 && a.transform.localPosition == Expected(), "right moves +X");
            Act(1); Check(w.X == 4 && w.Z == 0 && a.transform.localPosition == Expected(), "backward moves -Z");
            Act(2); Check(w.X == 3 && w.Z == 0 && a.transform.localPosition == Expected(), "left moves -X");

            Place(0, 0); Act(1);
            Check(w.X == 0 && w.Z == 0 && w.Steps == 1 && a.transform.localPosition == Expected(), "south wall blocks the robot");
            Act(2); Check(w.X == 0 && w.Z == 0 && Mathf.Abs(a.LastReward - (-.03f + .0005f * w.PathDistanceAt(0, 0))) < .0001f, "west wall blocks; bump costs step + wall penalty (-0.03 plus tiny gamma shaping)");
            Place(7, 7); Act(0);
            Check(w.X == 7 && w.Z == 7, "north wall (beside the exit) blocks the robot");
            Place(2, 5); Act(0);
            Check(w.X == 2 && w.Z == 5 && a.transform.localPosition == Expected(), "obstacle at (2,6) blocks the robot");
            Place(4, 3); Act(2);
            Check(w.X == 4 && w.Z == 3, "obstacle at (3,3) blocks the robot");

            int episodes = a.CompletedEpisodes;
            Place(4, 7); Act(3);
            Check(a.CompletedEpisodes == episodes + 1, "entering the goal zone ends the episode");
            Check(Mathf.Abs(a.LastReward - 1.04f) < .0001f, "goal reward is 1.04");
            Check(!w.Success && !w.IsBlocked(w.X, w.Z) && w.Steps == 0, "after the goal the robot is reset to a valid non-goal cell");
            Check(a.transform.localPosition == Expected(), "robot transform matches the reset cell");
            episodes = a.CompletedEpisodes;
            Place(7, 7); Act(2); Act(2);
            Check(a.CompletedEpisodes == episodes + 1 || w.Steps == 0, "second goal cell (6,7) also ends the episode (via 7,7 -> 6,7)");

            // Full route from the farthest cell, following only the diagnostic shortest-path action.
            int fx = 0, fz = 0, far = 0;
            for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++)
                if (!w.IsObstacle(x, z) && w.PathDistanceAt(x, z) > far) { far = w.PathDistanceAt(x, z); fx = x; fz = z; }
            episodes = a.CompletedEpisodes;
            Place(fx, fz);
            int moves = 0;
            while (a.CompletedEpisodes == episodes && moves < 200)
            {
                int act = w.ShortestPathAction(w.X, w.Z);
                int px = w.X, pz = w.Z;
                Act(act); moves++;
                if (a.CompletedEpisodes == episodes && a.transform.localPosition != Expected()) throw new Exception("transform desync");
                if (moves > 200) break;
            }
            Check(moves == far && a.CompletedEpisodes == episodes + 1, $"robot drives the {far}-step detour route around obstacles to the exit");

            episodes = a.CompletedEpisodes;
            Place(0, 0);
            for (int i = 0; i < 127; i++) Act(2);
            Check(a.CompletedEpisodes == episodes && w.Steps == 127, "timeout: still running at action 127");
            Act(2);
            Check(a.CompletedEpisodes == episodes + 1 && w.Steps == 0, "timeout: action 128 ends and resets the episode");

            var starts = new HashSet<string>();
            for (int i = 0; i < 100; i++)
            {
                a.OnEpisodeBegin();
                if (w.IsBlocked(w.X, w.Z) || w.Success) throw new Exception("invalid reset");
                if (a.transform.localPosition != Expected()) throw new Exception("reset transform desync");
                starts.Add(w.X + "," + w.Z);
            }
            Check(starts.Count > 30, "100 episode resets: valid cells, varied (" + starts.Count + " distinct)");

            // Observation: camera frame at several robot positions.
            var cs = a.GetComponent<CameraSensorComponent>();
            var tex = new Texture2D(84, 84, TextureFormat.RGB24, false);
            try
            {
                int minBlue = int.MaxValue, minGreen = int.MaxValue;
                foreach (var cell in new[] { new Vector2Int(0, 0), new Vector2Int(7, 0), new Vector2Int(0, 7), new Vector2Int(7, 5), new Vector2Int(4, 3), new Vector2Int(2, 2), new Vector2Int(5, 6) })
                {
                    Place(cell.x, cell.y);
                    CameraSensor.ObservationToTexture(cs.Camera, tex, 84, 84);
                    int blue = 0, green = 0;
                    foreach (var c in tex.GetPixels())
                    {
                        if (c.b > .7f && c.r < .4f && c.g < .5f) blue++;
                        if (c.g > .7f && c.r < .4f && c.b < .4f) green++;
                    }
                    minBlue = Math.Min(minBlue, blue); minGreen = Math.Min(minGreen, green);
                    if (cell.x == 0 && cell.y == 0)
                    {
                        Directory.CreateDirectory("Reports"); File.WriteAllBytes("Reports/stage4-camera.png", tex.EncodeToPNG());
                        // Stage 6: obstacle tops must stand clearly apart from every open floor tile in the 84x84 frame.
                        var px = tex.GetPixels(); float Lum(int cx, int cz) {
                            int col = Mathf.RoundToInt(42 + (cx - 3.5f) * (84f / 9.2f)), row = Mathf.RoundToInt(42 - (cz - 3.5f) * (84f / 9.2f)); float t = 0;
                            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { var c = px[(83 - (row + dy)) * 84 + (col + dx)]; t += .2126f * c.r + .7152f * c.g + .0722f * c.b; }
                            return t / 9f; }
                        float minObs = 9, maxFloor = 0;
                        for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++) {
                            if (a.World.IsObstacle(x, z)) minObs = Mathf.Min(minObs, Lum(x, z));
                            else if (!a.World.IsGoal(x, z) && !(x == 0 && z == 0)) maxFloor = Mathf.Max(maxFloor, Lum(x, z));
                        }
                        Check(minObs > maxFloor * 1.6f, $"obstacle tops clearly brighter than every open floor tile at 84x84 (dimmest obstacle {minObs:F3} vs brightest floor {maxFloor:F3})");
                    }
                }
                Check(tex.width == 84 && tex.height == 84, "camera produces 84x84 RGB observations");
                Check(minBlue >= 5, "robot visible in every sampled frame (fewest blue pixels: " + minBlue + ")");
                Check(minGreen >= 20, "green goal visible in every sampled frame (fewest green pixels: " + minGreen + ")");
            }
            finally { UnityEngine.Object.Destroy(tex); }
            var lens = a.transform.Find("Robot Visual/Lens");
            Place(3, 0);
            var d = lens.position - a.transform.position;
            Check(d.z > .15f && Mathf.Abs(d.x) < .01f, "robot lens still faces +Z (Forward)");
            a.OnEpisodeBegin();
        }
    }
}
