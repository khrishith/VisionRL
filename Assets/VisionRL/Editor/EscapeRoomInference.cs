using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace VisionRL.Editor
{
    // Stage 7: assigns the trained ONNX to the EscapeRoom agent and validates it INSIDE Unity (Play Mode),
    // using Unity's own Sentis/Inference Engine policy. No Python is involved.
    [InitializeOnLoad]
    public static class EscapeRoomInference
    {
        const string ModelPath = "Assets/VisionRL/Models/VisionRL_escape_train_01.onnx";
        const string Phase = "VisionRL.S7.Phase";      // 0 idle, 1 sampled(+blank camera), 2 deterministic
        const string Ready = "VisionRL.S7.Ready";
        static readonly List<string> lines = new List<string>();
        static readonly List<string> csv = new List<string>();
        static double started;
        static int unityErrors;
        static readonly List<string> errorSamples = new List<string>();

        static EscapeRoomInference()
        {
            EditorApplication.update += Tick;
            EditorApplication.playModeStateChanged += OnPlayState;
            Application.logMessageReceived += (msg, trace, type) =>
            {
                if (!EditorApplication.isPlaying || SessionState.GetInt(Phase, 0) == 0) return;
                if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert)
                { unityErrors++; if (errorSamples.Count < 5) errorSamples.Add(msg); }
            };
        }

        [MenuItem("VisionRL/Escape Room/Assign Trained Model")]
        public static void AssignModel()
        {
            EscapeRoomBuilder.CreateScene();
            var model = AssetDatabase.LoadMainAssetAtPath(ModelPath);
            if (model == null) throw new Exception("Model not imported: " + ModelPath);
            var agent = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            var so = new SerializedObject(agent.GetComponent<BehaviorParameters>());
            so.FindProperty("m_Model").objectReferenceValue = model;
            // Behavior stays Default: Python connected -> trains; no Python -> uses this model. (InferenceOnly would
            // silently stop mlagents-learn from ever controlling the agent.) CPU/Burst inference: reliable on Apple Silicon.
            so.FindProperty("m_BehaviorType").enumValueIndex = (int)BehaviorType.Default;
            so.FindProperty("m_InferenceDevice").enumValueIndex = (int)InferenceDevice.Burst;
            so.FindProperty("m_DeterministicInference").boolValue = false; // sampled action output, the training-time policy
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.MarkSceneDirty(agent.gameObject.scene);
            EditorSceneManager.SaveScene(agent.gameObject.scene);
            Debug.Log("VISIONRL_MODEL_ASSIGNED " + AssetDatabase.GetAssetPath(model));
        }

        [MenuItem("VisionRL/Escape Room/Run Unity Runtime Validation")]
        public static void Run()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            EscapeRoomBuilder.CreateScene();
            lines.Clear(); csv.Clear(); errorSamples.Clear(); unityErrors = 0;
            csv.Add("phase,start_x,start_z,success,length");
            SessionState.SetInt(Phase, 1);
            SessionState.SetBool(Ready, false);
            started = 0;
            EditorApplication.EnterPlaymode();
        }

        static void OnPlayState(PlayModeStateChange change)
        {
            if (change != PlayModeStateChange.EnteredEditMode) return;
            int phase = SessionState.GetInt(Phase, 0);
            if (phase == 2 && SessionState.GetBool(Ready, false))
            {
                // Edit mode again: tick "Deterministic Inference" on the in-memory scene object (never saved), play phase 2.
                SessionState.SetBool(Ready, false);
                var agent = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
                var so = new SerializedObject(agent.GetComponent<BehaviorParameters>());
                so.FindProperty("m_DeterministicInference").boolValue = true;
                so.ApplyModifiedPropertiesWithoutUndo();
                started = 0;
                EditorApplication.EnterPlaymode();
            }
            else if (phase == 3)
            {
                SessionState.SetInt(Phase, 0);
                EditorSceneManager.OpenScene(EscapeRoomBuilder.ScenePath); // discard the in-memory deterministic flag
                File.WriteAllLines("Reports/stage7-unity-runtime.txt", lines);
                File.WriteAllLines("Reports/stage7-unity-episodes.csv", csv);
                Debug.Log("VISIONRL_S7_DONE " + string.Join(" | ", lines));
            }
        }

        static void Tick()
        {
            int phase = SessionState.GetInt(Phase, 0);
            if (phase == 0 || phase == 3 || !EditorApplication.isPlaying) return;
            if (SessionState.GetBool(Ready, false)) return;
            if (started == 0) { started = EditorApplication.timeSinceStartup; return; }
            if (EditorApplication.timeSinceStartup - started < 1 || Time.frameCount < 2) return;
            SessionState.SetBool(Ready, true);
            try { Execute(phase); }
            catch (Exception e) { lines.Add("FAIL exception: " + e); Debug.LogException(e); }
            finally
            {
                if (phase == 1) { SessionState.SetInt(Phase, 2); SessionState.SetBool(Ready, true); }
                else SessionState.SetInt(Phase, 3);
                // phase 1 persists its lines across the play-mode transition via a file
                if (phase == 1) File.WriteAllLines("Reports/.stage7-phase1.tmp", lines.Concat(new[] { "#CSV" }).Concat(csv));
                else
                {
                    var prev = File.Exists("Reports/.stage7-phase1.tmp") ? File.ReadAllLines("Reports/.stage7-phase1.tmp") : new string[0];
                    int k = Array.IndexOf(prev, "#CSV");
                    var l = (k >= 0 ? prev.Take(k) : prev).Concat(lines).ToList(); var c = (k >= 0 ? prev.Skip(k + 1) : new string[0]).Concat(csv.Skip(1)).ToList();
                    lines.Clear(); lines.AddRange(l); csv.Clear(); csv.AddRange(c);
                }
                EditorApplication.ExitPlaymode();
            }
        }

        class Stats { public int episodes, successes, timeouts; public long steps, successSteps; public int[] actions = new int[4]; public List<string> failedStarts = new List<string>(); }

        static void Execute(int phase)
        {
            if (phase == 2) { lines.Clear(); csv.Clear(); csv.Add("phase,start_x,start_z,success,length"); } // phase 1 is kept in a temp file
            var a = UnityEngine.Object.FindAnyObjectByType<RoomAgent>();
            var bp = a.GetComponent<BehaviorParameters>();
            var cs = a.GetComponent<CameraSensorComponent>();
            bool deterministic = bp.DeterministicInference;
            lines.Add($"--- phase {phase}: Unity Play Mode, {(deterministic ? "DETERMINISTIC" : "SAMPLED")} inference ---");
            // The policy the agent is really running (private field), e.g. SentisPolicy = the ONNX model inside Unity.
            var policyObj = typeof(Agent).GetField("m_Brain", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).GetValue(a);
            string policyType = policyObj == null ? "null" : policyObj.GetType().Name;
            lines.Add($"behavior={bp.BehaviorName} requestedType={bp.BehaviorType} activePolicy={policyType} heuristicMode={bp.IsInHeuristicMode()} device={bp.InferenceDevice} deterministic={deterministic} model={(bp.Model != null ? bp.Model.name : "NONE")}");
            lines.Add($"communicatorOn={Academy.Instance.IsCommunicatorOn} sensors={a.GetComponents<SensorComponent>().Length} cameraSensor={cs.Width}x{cs.Height} grayscale={cs.Grayscale} vectorObsSize={bp.BrainParameters.VectorObservationSize}");
            if (bp.Model == null || policyType != "SentisPolicy" || bp.IsInHeuristicMode())
                throw new Exception("Agent is not running the trained model (active policy " + policyType + ")");
            Academy.Instance.AutomaticSteppingEnabled = false;
            var w = a.World;
            Academy.Instance.EnvironmentStep(); // agent's own first reset
            var starts = new List<Vector2Int>();
            for (int x = 0; x < 8; x++) for (int z = 0; z < 8; z++) if (!w.IsObstacle(x, z) && !w.IsGoal(x, z)) starts.Add(new Vector2Int(x, z));
            int repeats = phase == 1 ? 4 : 1;
            var order = new List<Vector2Int>(); for (int r = 0; r < repeats; r++) order.AddRange(starts);
            var rnd = new System.Random(7); order = order.OrderBy(_ => rnd.Next()).ToList();

            var normal = RunEpisodes(a, w, order, phase.ToString() + (deterministic ? "-deterministic" : "-sampled"));
            Report(normal, deterministic ? "deterministic" : "sampled", w);
            if (phase == 1)
            {
                // Control inside Unity: keep the model, but blank the camera (background colour only, no room visible).
                var mask = cs.Camera.cullingMask; cs.Camera.cullingMask = 0;
                try
                {
                    var tex = new Texture2D(84, 84, TextureFormat.RGB24, false);
                    CameraSensor.ObservationToTexture(cs.Camera, tex, 84, 84);
                    var px = tex.GetPixels(); float lo = 1, hi = 0; foreach (var c in px) { var l = c.grayscale; lo = Mathf.Min(lo, l); hi = Mathf.Max(hi, l); }
                    UnityEngine.Object.Destroy(tex);
                    lines.Add($"blank-camera control: frame luminance range {lo:F3}..{hi:F3} (constant = nothing visible)");
                    var blank = RunEpisodes(a, w, order, "1-sampled-BLANK-CAMERA");
                    Report(blank, "sampled, camera blanked", w);
                }
                finally { cs.Camera.cullingMask = mask; }
            }
            lines.Add($"unity errors/exceptions logged during phase {phase}: {unityErrors}" + (errorSamples.Count > 0 ? " e.g. " + errorSamples[0] : ""));
            Academy.Instance.AutomaticSteppingEnabled = true;
        }

        static Stats RunEpisodes(RoomAgent a, RoomWorld w, List<Vector2Int> order, string tag)
        {
            var st = new Stats();
            foreach (var s in order)
            {
                w.Set(s.x, s.y); a.SyncTransforms();
                int ep0 = a.CompletedEpisodes, steps = 0;
                while (a.CompletedEpisodes == ep0 && steps < 400)
                {
                    Academy.Instance.EnvironmentStep(); steps++;
                    st.actions[Mathf.Clamp(a.LastAction, 0, 3)]++;
                }
                bool success = a.CompletedEpisodes > ep0 && a.LastReward > .5f;
                st.episodes++; st.steps += steps;
                if (success) { st.successes++; st.successSteps += steps; } else { st.timeouts++; st.failedStarts.Add($"({s.x},{s.y})"); }
                csv.Add($"{tag},{s.x},{s.y},{(success ? 1 : 0)},{steps}");
            }
            return st;
        }

        static void Report(Stats st, string label, RoomWorld w)
        {
            double n = st.episodes, p = st.successes / n, z = 1.96;
            double ctr = (p + z * z / (2 * n)) / (1 + z * z / n), rad = z * Math.Sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n);
            lines.Add($"[{label}] episodes={st.episodes} successes={st.successes} ({p:P1}, 95% CI {ctr - rad:P1}-{ctr + rad:P1}) timeouts={st.timeouts} ({st.timeouts / n:P1}) mean length={st.steps / n:F1} mean steps to success={(st.successes > 0 ? (double)st.successSteps / st.successes : 0):F1}");
            double tot = st.actions.Sum();
            lines.Add($"[{label}] actions chosen [fwd,back,left,right] = {st.actions[0]},{st.actions[1]},{st.actions[2]},{st.actions[3]} = {st.actions[0] / tot:P0}/{st.actions[1] / tot:P0}/{st.actions[2] / tot:P0}/{st.actions[3] / tot:P0}");
            if (st.failedStarts.Count > 0) lines.Add($"[{label}] failed start cells ({st.failedStarts.Count}): " + string.Join(" ", st.failedStarts.Distinct().Take(20)));
        }
    }
}
