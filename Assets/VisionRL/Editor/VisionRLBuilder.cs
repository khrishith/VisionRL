using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace VisionRL.Editor
{
    public static class VisionRLBuilder
    {
        const string ScenePath = "Assets/VisionRL/Scenes/GridWorld.unity";
        [MenuItem("VisionRL/Create Grid Scene")]
        public static void CreateScene()
        {
            if (File.Exists(ScenePath))
            {
                EditorSceneManager.OpenScene(ScenePath);
                return;
            }
            if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Directory.CreateDirectory("Assets/VisionRL/Scenes");
            Directory.CreateDirectory("Assets/VisionRL/Materials");
            var floorA = Material("FloorA", new Color(.24f, .26f, .29f));
            var floorB = Material("FloorB", new Color(.32f, .34f, .37f));
            var blue = Material("AgentBlue", new Color(.02f, .15f, 1f));
            var green = Material("TargetGreen", new Color(.02f, 1f, .08f));
            var grid = new GameObject("7x7 Grid");
            for (int z = 0; z < 7; z++)
            for (int x = 0; x < 7; x++)
                Cube($"Tile_{x}_{z}", new Vector3(x-3, -.08f, z-3), new Vector3(.96f, .1f, .96f),
                    (x+z)%2 == 0 ? floorA : floorB).transform.SetParent(grid.transform);
            var target = Cube("Green Target", new Vector3(2, .375f, 2), Vector3.one * .75f, green);
            var agentObject = Cube("Blue Agent", new Vector3(-2,.375f,-2), Vector3.one * .75f, blue);
            // Configure all sensors before Agent is enabled.
            agentObject.SetActive(false);
            var cameraObject = new GameObject("Observation Camera");
            var camera = cameraObject.AddComponent<Camera>();
            camera.transform.position = new Vector3(0, 10, 0);
            camera.transform.rotation = Quaternion.Euler(90, 0, 0);
            camera.orthographic = true;
            camera.orthographicSize = 3.8f;
            camera.aspect = 1f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(.09f,.09f,.11f);
            camera.nearClipPlane = .1f;
            camera.farClipPlane = 20f;
            camera.allowHDR = false;
            camera.allowMSAA = false;
            cameraObject.tag = "MainCamera";
            var behavior = agentObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "VisionRL";
            behavior.BrainParameters.VectorObservationSize = 0;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(4);
            behavior.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            behavior.UseChildSensors = false;
            var sensor = agentObject.AddComponent<CameraSensorComponent>();
            sensor.Camera = camera;
            sensor.SensorName = "RGB84";
            sensor.Width = 84;
            sensor.Height = 84;
            sensor.Grayscale = false;
            sensor.ObservationStacks = 1;
            sensor.CompressionType = SensorCompressionType.PNG;
            sensor.RuntimeCameraEnable = true;
            var agent = agentObject.AddComponent<GridAgent>();
            agent.target = target.transform;
            var decision = agentObject.AddComponent<GridDecisionRequester>();
            decision.DecisionPeriod = 1;
            decision.TakeActionsBetweenDecisions = false;
            agentObject.SetActive(true);
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            Debug.Log("VISIONRL_SCENE_CREATED " + ScenePath);
        }

        static Material Material(string name, Color color)
        {
            string path = $"Assets/VisionRL/Materials/{name}.mat";
            var existing = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (existing != null) return existing;
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) throw new Exception("URP Unlit shader missing");
            var material = new Material(shader) { name = name };
            material.SetColor("_BaseColor", color);
            AssetDatabase.CreateAsset(material, path);
            return material;
        }

        static GameObject Cube(string name, Vector3 position, Vector3 scale, Material material)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.position = position;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = material;
            return go;
        }

        [MenuItem("VisionRL/Validate and Build macOS")]
        public static void ValidateAndBuild()
        {
            CreateScene();
            Validate();
            Directory.CreateDirectory("Builds");
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
                scenes = new[] {ScenePath}, locationPathName = "Builds/VisionRL.app",
                target = BuildTarget.StandaloneOSX, options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("VisionRL build failed: " + report.summary.result);
            Debug.Log("VISIONRL_BUILD_PASSED");
        }

        [MenuItem("VisionRL/Validate Environment")]
        public static void Validate()
        {
            var w = new GridWorld();
            int assertions = 0;
            void Check(bool condition) { assertions++; if (!condition) throw new Exception("Grid test failed " + assertions); }
            for (int x=0; x<7; x++) for (int z=0; z<7; z++) for (int a=0; a<4; a++)
            {
                w.Set(x,z,(x+3)%7,(z+3)%7);
                w.Step(a);
                Check(w.X == Mathf.Clamp(x + (a==2?-1:a==3?1:0),0,6));
                Check(w.Z == Mathf.Clamp(z + (a==1?-1:a==0?1:0),0,6));
                Check(w.Steps == 1);
            }
            var random = new System.Random(12);
            for (int i=0;i<1000;i++) { w.Reset(random); Check(!w.Success && w.Steps==0); }
            w.Set(0,0,1,0); Check(w.Step(3)>0 && w.Success && w.Done);
            w.Set(0,0,6,6); for(int i=0;i<64;i++) w.Step(2); Check(w.Done && !w.Success);
            var agent = UnityEngine.Object.FindAnyObjectByType<GridAgent>();
            Check(agent != null && agent.target != null);
            var b = agent.GetComponent<BehaviorParameters>();
            Check(b.BrainParameters.VectorObservationSize == 0);
            Check(b.BrainParameters.ActionSpec.NumContinuousActions == 0);
            Check(b.BrainParameters.ActionSpec.BranchSizes.Length == 1 && b.BrainParameters.ActionSpec.BranchSizes[0] == 4);
            var sensors = agent.GetComponents<SensorComponent>();
            Check(sensors.Length == 1);
            var c = agent.GetComponent<CameraSensorComponent>();
            Check(c.Width == 84 && c.Height == 84 && !c.Grayscale && c.Camera != null);
            Directory.CreateDirectory("Reports");
            File.WriteAllText("Reports/unity-validation.txt", $"PASS: {assertions} assertions; Unity {Application.unityVersion}; {SystemInfo.processorType}\n");
            Debug.Log($"VISIONRL_VALIDATION_PASSED {assertions} assertions");
        }
    }
}
