using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace VisionRL.Editor
{
    // Runs in the real Editor Play loop. No Python or trainer is launched.
    [InitializeOnLoad]
    public static class StageOnePlayModeChecks
    {
        const string Pending = "VisionRL.StageOneChecks";
        static readonly List<string> results = new List<string>();
        static double started;
        static StageOnePlayModeChecks() { EditorApplication.update += Tick; }

        [MenuItem("VisionRL/Run Stage 1 Play Mode Checks")]
        public static void Run()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            VisionRLBuilder.CreateScene();
            // Upgrade only our previously generated objects, preserving the original template.
            var agent = UnityEngine.Object.FindAnyObjectByType<GridAgent>();
            var oldRequester = agent.GetComponent<DecisionRequester>();
            if (!(oldRequester is GridDecisionRequester))
            {
                UnityEngine.Object.DestroyImmediate(oldRequester);
                var requester = agent.gameObject.AddComponent<GridDecisionRequester>();
                requester.DecisionPeriod = 1;
                requester.TakeActionsBetweenDecisions = false;
            }
            agent.transform.localScale = Vector3.one * .75f;
            agent.target.localScale = Vector3.one * .75f;
            var pos = agent.transform.position; pos.y = .375f; agent.transform.position = pos;
            pos = agent.target.position; pos.y = .375f; agent.target.position = pos;
            EditorSceneManager.MarkSceneDirty(agent.gameObject.scene);
            EditorSceneManager.SaveScene(agent.gameObject.scene);
            VisionRLBuilder.Validate();
            SessionState.SetBool(Pending, true);
            EditorApplication.EnterPlaymode();
        }

        static void Check(bool value, string name)
        {
            results.Add((value ? "PASS " : "FAIL ") + name);
            if (!value) throw new Exception(name);
        }

        static void Tick()
        {
            if (!SessionState.GetBool(Pending, false) || !EditorApplication.isPlaying) return;
            if (started == 0) { started = EditorApplication.timeSinceStartup; return; }
            if (EditorApplication.timeSinceStartup - started < 1 || Time.frameCount < 2) return;
            SessionState.SetBool(Pending, false);
            results.Clear();
            try { Test(); }
            catch (Exception e) { results.Add("FAIL exception: " + e); Debug.LogException(e); }
            finally
            {
                Directory.CreateDirectory("Reports");
                File.WriteAllLines("Reports/stage1-playmode.txt", results);
                Debug.Log("VISIONRL_PLAYMODE_CHECKS " + string.Join("; ", results));
                // Leave the saved scene open, returning to Edit mode after checks.
                EditorApplication.ExitPlaymode();
            }
        }

        static void Test()
        {
            var a = UnityEngine.Object.FindAnyObjectByType<GridAgent>();
            Check(a != null && a.target != null, "agent and target exist in Play mode");
            Check(GameObject.Find("7x7 Grid").transform.childCount == 49, "49 visible floor cells");
            Check(a.transform.localScale == Vector3.one*.75f && a.target.localScale == Vector3.one*.75f,
                "agent and target are cubes");
            Check(!Academy.Instance.IsCommunicatorOn, "no trainer connection");
            Check(!a.GetComponent<DecisionRequester>().enabled, "manual mode stays idle without key presses");
            var flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var w = (GridWorld)typeof(GridAgent).GetField("world", flags).GetValue(a);
            var sync = typeof(GridAgent).GetMethod("SyncTransforms", flags);
            void Place(int x, int z, int tx, int tz) { w.Set(x,z,tx,tz); sync.Invoke(a,null); }
            void Act(int action)
            {
                typeof(GridAgent).GetField("manualAction", flags).SetValue(a, action);
                a.RequestDecision();
                Academy.Instance.EnvironmentStep();
            }
            float LastReward() => (float)typeof(GridAgent).GetField("lastReward", flags).GetValue(a);
            Academy.Instance.EnvironmentStep(); // let the Agent's own first reset happen before we place cubes
            Place(3,3,6,6);
            for(int i=0;i<10;i++) Academy.Instance.EnvironmentStep();
            Check(w.Steps==0 && w.X==3 && w.Z==3, "ten idle simulation steps produce no movement or penalty");
            for (int i=0;i<4;i++)
            {
                Place(3,3,6,6); Act(i);
                Check(w.X == 3+(i==2?-1:i==3?1:0) && w.Z == 3+(i==1?-1:i==0?1:0), "movement action " + i);
                Check(a.transform.localPosition == new Vector3(w.X-3,.375f,w.Z-3), "transform follows action " + i);
                float expected = i==0 || i==3 ? .095f : -.103f;
                Check(Mathf.Abs(LastReward()-expected)<.0001f, "step penalty and distance reward action " + i);
            }
            Place(0,0,6,6); Act(2);
            Check(w.X==0 && w.Z==0 && w.Steps==1, "boundary blocks movement and consumes one step");
            Check(Mathf.Abs(LastReward() - -.018f)<.0001f, "boundary penalty plus step and distance shaping");
            int episode = a.CompletedEpisodes;
            Place(0,0,1,0); Act(3);
            Check(a.CompletedEpisodes==episode+1, "target reach terminates episode");
            Check(Mathf.Abs(LastReward()-1.09f)<.0001f, "target reached reward");
            Check(w.Steps==0 && !w.Success, "target reach resets to non-overlapping cells");
            episode = a.CompletedEpisodes;
            Place(0,0,6,6);
            for(int i=0;i<63;i++) Act(2);
            Check(w.Steps==63 && a.CompletedEpisodes==episode, "episode continues through action 63");
            Act(2);
            Check(a.CompletedEpisodes==episode+1 && w.Steps==0 && !w.Success, "action 64 times out and resets");
            Check(Mathf.Abs(LastReward()-.17f)<.0001f, "timeout reward includes terminal potential correction");
            var starts = new HashSet<string>();
            for(int i=0;i<100;i++)
            {
                a.OnEpisodeBegin();
                if(w.Success || w.X<0 || w.X>6 || w.Z<0 || w.Z>6 || w.TargetX<0 || w.TargetX>6 || w.TargetZ<0 || w.TargetZ>6)
                    throw new Exception("Invalid randomized reset");
                starts.Add($"{w.X},{w.Z},{w.TargetX},{w.TargetZ}");
            }
            Check(starts.Count>80, "100 valid non-overlapping randomized resets with varied starts");
            Place(1,1,5,5);
            var visual = a.transform.Find("Robot Visual");
            Check(visual != null, "robot visual is a child of the agent root");
            Check(a.GetComponentsInChildren<Collider>().Length == 1 && a.GetComponent<Collider>() != null,
                "only the agent root has a collider (robot parts have none)");
            Check(!a.GetComponent<MeshRenderer>().enabled, "original cube mesh hidden but still present");
            var lens = visual.Find("Lens");
            var toLens = lens.position - a.transform.position;
            Check(toLens.z > .15f && Mathf.Abs(toLens.x) < .01f, "robot lens faces +Z, the Forward direction");
            Check(a.GetComponentsInChildren<MonoBehaviour>().Length == 4, "no extra scripts on robot (agent, requester, behavior, sensor only)");
            var sensor = a.GetComponent<CameraSensorComponent>();
            var texture = new Texture2D(84,84,TextureFormat.RGB24,false);
            try
            {
                CameraSensor.ObservationToTexture(sensor.Camera,texture,84,84);
                // Strict colours so the robot's cyan details cannot count as the green target.
                var pixels=texture.GetPixels(); int blue=0,green=0;
                foreach(var c in pixels) { if(c.b>.7f && c.r<.4f && c.g<.5f) blue++; if(c.g>.7f && c.r<.4f && c.b<.4f) green++; }
                Check(texture.width==84 && texture.height==84 && blue>10 && green>10,
                    $"84x84 RGB camera frame contains blue agent ({blue}px) and green target ({green}px)");
                Directory.CreateDirectory("Reports");
                File.WriteAllBytes("Reports/stage1-camera.png",texture.EncodeToPNG());
            }
            finally { UnityEngine.Object.Destroy(texture); }
            a.OnEpisodeBegin();
        }
    }
}
