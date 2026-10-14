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
    // Stage 8: validates trained models INSIDE Unity Play Mode (Unity's own Inference Engine policy, no Python) on the
    // randomized-exit scene. Sessions: A = new model, sampled; B = new model, deterministic; C = Stage 7 model, sampled.
    // Every phase reseeds the agent with 9001, so all phases see the same (exit, start) sequence as Python's seed 9001.
    [InitializeOnLoad]
    public static class RandomExitInference
    {
        public const string NewModel = "Assets/VisionRL/Models/VisionRL_random_exit_02.onnx";
        const string OldModel = "Assets/VisionRL/Models/VisionRL_escape_train_01.onnx";
        const string Queue = "VisionRL.S8.Queue", Armed = "VisionRL.S8.Armed", Active = "VisionRL.S8.Active";
        const string TmpTxt = "Reports/.stage8-unity.tmp", TmpCsv = "Reports/.stage8-unity-csv.tmp";
        static double started;
        static int unityErrors; static string firstError;

        static RandomExitInference()
        {
            EditorApplication.update += Tick;
            EditorApplication.playModeStateChanged += OnPlayState;
            Application.logMessageReceived += (msg, trace, type) =>
            {
                if (!EditorApplication.isPlaying || !SessionState.GetBool(Active, false)) return;
                if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert) { unityErrors++; if (firstError == null) firstError = msg; }
            };
        }

        static void SetModel(string path, bool deterministic)
        {
            var agent = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            var so = new SerializedObject(agent.GetComponent<BehaviorParameters>());
            var model = AssetDatabase.LoadMainAssetAtPath(path);
            if (model == null) throw new Exception("Model not imported: " + path);
            so.FindProperty("m_Model").objectReferenceValue = model;
            so.FindProperty("m_BehaviorType").enumValueIndex = (int)BehaviorType.Default;
            so.FindProperty("m_InferenceDevice").enumValueIndex = (int)InferenceDevice.Burst;
            so.FindProperty("m_DeterministicInference").boolValue = deterministic;
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.MarkSceneDirty(agent.gameObject.scene);
        }

        [MenuItem("VisionRL/Random Exit/Assign Trained Model")]
        public static void AssignModel()
        {
            RandomExitBuilder.CreateScene();
            SetModel(NewModel, false);
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
            Debug.Log("VISIONRL_RANDOMEXIT_MODEL_ASSIGNED " + NewModel);
        }

        [MenuItem("VisionRL/Random Exit/Run Unity Runtime Validation")]
        public static void Run()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            RandomExitBuilder.CreateScene();
            File.Delete(TmpTxt); File.Delete(TmpCsv);
            File.WriteAllText(TmpCsv, "phase,exit_x,exit_z,start_x,start_z,held_out_exit,success,length,return\n");
            SessionState.SetString(Queue, "A");   // final validation: new model, sampled inference only
            Next();
        }

        static void Next()
        {
            var q = SessionState.GetString(Queue, "").Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            if (q.Count == 0) { Finish(); return; }
            string s = q[0]; q.RemoveAt(0); SessionState.SetString(Queue, string.Join(",", q));
            SessionState.SetString("VisionRL.S8.Session", s);
            if (s == "A") SetModel(NewModel, false); else if (s == "B") SetModel(NewModel, true); else SetModel(OldModel, false);
            SessionState.SetBool(Armed, true); SessionState.SetBool(Active, true); started = 0; unityErrors = 0; firstError = null;
            EditorApplication.EnterPlaymode();
        }

        static void Finish()
        {
            SessionState.SetBool(Active, false);
            EditorSceneManager.OpenScene(RandomExitBuilder.ScenePath); // drop in-memory model/flag changes
            if (File.Exists(TmpTxt)) File.Copy(TmpTxt, "Reports/stage8-unity-runtime.txt", true);
            if (File.Exists(TmpCsv)) File.Copy(TmpCsv, "Reports/stage8-unity-episodes.csv", true);
            Debug.Log("VISIONRL_S8_DONE");
        }

        static void OnPlayState(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.EnteredEditMode && SessionState.GetBool(Active, false) && !SessionState.GetBool(Armed, false)) Next();
        }

        static void Tick()
        {
            if (!SessionState.GetBool(Armed, false) || !EditorApplication.isPlaying) return;
            if (started == 0) { started = EditorApplication.timeSinceStartup; return; }
            if (EditorApplication.timeSinceStartup - started < 1 || Time.frameCount < 2) return;
            SessionState.SetBool(Armed, false);
            var lines = new List<string>(); var csv = new List<string>();
            try { Execute(SessionState.GetString("VisionRL.S8.Session", "A"), lines, csv); }
            catch (Exception e) { lines.Add("FAIL exception: " + e); Debug.LogException(e); }
            finally
            {
                File.AppendAllLines(TmpTxt, lines); File.AppendAllLines(TmpCsv, csv);
                EditorApplication.ExitPlaymode(); // OnPlayState -> Next()
            }
        }

        class Stats { public int n, wins, timeouts; public long steps, winSteps; public double ret; public int[] actions = new int[4]; public int[] sideN = new int[4], sideWins = new int[4]; public int heldN, heldWins, trainN, trainWins; }

        static void Execute(string session, List<string> lines, List<string> csv)
        {
            var a = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            var bp = a.GetComponent<BehaviorParameters>(); var cs = a.GetComponent<CameraSensorComponent>();
            var policyObj = typeof(Agent).GetField("m_Brain", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).GetValue(a);
            string policyType = policyObj == null ? "null" : policyObj.GetType().Name;
            lines.Add($"=== session {session}: model={(bp.Model != null ? bp.Model.name : "NONE")} deterministic={bp.DeterministicInference} device={bp.InferenceDevice} activePolicy={policyType} behavior={bp.BehaviorName}/{bp.BehaviorType} ===");
            lines.Add($"communicatorOn={Academy.Instance.IsCommunicatorOn} sensors={a.GetComponents<SensorComponent>().Length} camera={cs.Width}x{cs.Height} rgb={!cs.Grayscale} vectorObsSize={bp.BrainParameters.VectorObservationSize} randomizeExit={a.randomizeExit} horizon={a.horizon}");
            if (bp.Model == null || policyType != "SentisPolicy" || bp.IsInHeuristicMode()) throw new Exception("not running the model: " + policyType);
            Academy.Instance.AutomaticSteppingEnabled = false;
            Academy.Instance.EnvironmentStep();
            var phases = session == "A"
                ? new[] { ("final model | all exits | camera", 0, false, 200), ("final model | HELD-OUT exits | camera", 2, false, 100), ("final model | all exits | BLANK camera", 0, true, 200) }
                : session == "B" ? new[] { ("new model | DETERMINISTIC | all exits", 0, false, 200) }
                : new[] { ("Stage 7 model | all exits | camera", 0, false, 300), ("Stage 7 model | all exits | BLANK camera", 0, true, 300) };
            foreach (var (label, exitSet, blank, episodes) in phases)
            {
                int savedMask = cs.Camera.cullingMask;
                if (blank) cs.Camera.cullingMask = 0;
                try { Report(label, RunEpisodes(a, label, exitSet, episodes, csv), lines); }
                finally { cs.Camera.cullingMask = savedMask; }
            }
            lines.Add($"unity errors/exceptions logged in session {session}: {unityErrors}" + (firstError != null ? " first: " + firstError : ""));
            Academy.Instance.AutomaticSteppingEnabled = true;
        }

        static Stats RunEpisodes(RoomAgent a, string label, int exitSet, int episodes, List<string> csv)
        {
            var w = a.World; var st = new Stats();
            a.exitSetDefault = exitSet; a.Reseed(9001); a.OnEpisodeBegin();   // identical (exit, start) sequence for every phase
            var heldKeys = new HashSet<int>(); for (int i = 0; i < w.ExitCandidateCount; i++) if (w.IsHeldOutExit(i)) { w.ExitCandidate(i, out int hx, out int hz); heldKeys.Add(hx * 8 + hz); }
            for (int e = 0; e < episodes; e++)
            {
                int ex = w.ExitX, ez = w.ExitZ, sx = w.X, sz = w.Z, ep0 = a.CompletedEpisodes, steps = 0; double ret = 0;
                while (a.CompletedEpisodes == ep0 && steps < 200) { Academy.Instance.EnvironmentStep(); steps++; ret += a.LastReward; st.actions[Mathf.Clamp(a.LastAction, 0, 3)]++; }
                bool win = a.CompletedEpisodes > ep0 && a.LastReward > .5f; bool held = heldKeys.Contains(ex * 8 + ez);
                int side = ez == 7 ? 0 : ez == 0 ? 1 : ex == 0 ? 2 : 3;
                st.n++; st.steps += steps; st.ret += ret; if (win) { st.wins++; st.winSteps += steps; } else st.timeouts++;
                st.sideN[side]++; if (win) st.sideWins[side]++; if (held) { st.heldN++; if (win) st.heldWins++; } else { st.trainN++; if (win) st.trainWins++; }
                csv.Add($"{label},{ex},{ez},{sx},{sz},{(held ? 1 : 0)},{(win ? 1 : 0)},{steps},{ret:F3}");
            }
            a.exitSetDefault = 0;
            return st;
        }

        static string Wilson(int k, int n) { if (n == 0) return "n/a"; double p = (double)k / n, z = 1.96, c = (p + z * z / (2 * n)) / (1 + z * z / n), r = z * Math.Sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / (1 + z * z / n); return $"{p:P1} (95% CI {c - r:P1}-{c + r:P1})"; }

        static void Report(string label, Stats st, List<string> lines)
        {
            lines.Add($"[{label}] episodes={st.n} success={st.wins} -> {Wilson(st.wins, st.n)} | timeouts={st.timeouts} ({(double)st.timeouts / st.n:P1}) | mean length={(double)st.steps / st.n:F1} | mean steps to success={(st.wins > 0 ? (double)st.winSteps / st.wins : 0):F1} | mean return={st.ret / st.n:F3}");
            lines.Add($"[{label}]   by exit side N/S/W/E: {st.sideWins[0]}/{st.sideN[0]}  {st.sideWins[1]}/{st.sideN[1]}  {st.sideWins[2]}/{st.sideN[2]}  {st.sideWins[3]}/{st.sideN[3]} | training exits {st.trainWins}/{st.trainN} | held-out exits {st.heldWins}/{st.heldN}");
            double tot = st.actions.Sum();
            lines.Add($"[{label}]   actions fwd/back/left/right = {st.actions[0] / tot:P0}/{st.actions[1] / tot:P0}/{st.actions[2] / tot:P0}/{st.actions[3] / tot:P0}");
        }
    }
}
